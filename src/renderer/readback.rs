use super::*;

fn copy_padded_readback_rows(
    data: &[u8],
    height: u32,
    unpadded_bytes_per_row: u32,
    padded_bytes_per_row: u32,
    output: &mut Vec<u8>,
) {
    let output_size = (unpadded_bytes_per_row * height) as usize;
    output.resize(output_size, 0);

    if padded_bytes_per_row == unpadded_bytes_per_row {
        output.copy_from_slice(data);
        return;
    }

    for row in 0..height {
        let padded_offset = (row * padded_bytes_per_row) as usize;
        let unpadded_offset = (row * unpadded_bytes_per_row) as usize;
        let row_data = &data[padded_offset..padded_offset + unpadded_bytes_per_row as usize];
        output[unpadded_offset..unpadded_offset + unpadded_bytes_per_row as usize]
            .copy_from_slice(row_data);
    }
}

impl<'a> Renderer<'a> {
    fn map_readback_buffer_into(
        device: &wgpu::Device,
        buffer: &wgpu::Buffer,
        mapped_bytes: &mut Vec<u8>,
    ) {
        mapped_bytes.clear();

        let buffer_slice = buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            if sender.send(result).is_err() {
                log::warn!("Failed to send map_async result from callback");
            }
        });

        let _ = device.poll(wgpu::MaintainBase::Wait);

        let map_result = match receiver.recv() {
            Ok(result) => result,
            Err(error) => {
                log::warn!("Failed to receive mapped buffer result: {}", error);
                return;
            }
        };

        if let Err(error) = map_result {
            log::warn!("Failed to map readback buffer: {:?}", error);
            return;
        }

        let mapped_range = buffer_slice.get_mapped_range();
        mapped_bytes.extend_from_slice(&mapped_range);
        drop(mapped_range);
        buffer.unmap();
    }

    pub fn render_to_buffer(&mut self, buffer: &mut Vec<u8>) {
        #[cfg(feature = "render_metrics")]
        let frame_render_loop_started_at = std::time::Instant::now();

        self.prepare_render();

        let (width, height) = self.physical_size;

        let size_changed = self.rtb_cached_width != width || self.rtb_cached_height != height;
        if size_changed {
            self.rtb_cached_width = width;
            self.rtb_cached_height = height;
        }

        if size_changed || self.rtb_offscreen_texture.is_none() {
            self.rtb_offscreen_texture = Some(create_offscreen_color_texture(
                &self.device,
                (width, height),
                self.config.format,
            ));
        }

        let texture_view = self
            .rtb_offscreen_texture
            .as_ref()
            .unwrap()
            .create_view(&wgpu::TextureViewDescriptor::default());

        let output_texture = self.rtb_offscreen_texture.take();
        self.render_to_texture_view(&texture_view, output_texture.as_ref());
        self.rtb_offscreen_texture = output_texture;

        let (unpadded_bytes_per_row, padded_bytes_per_row) = compute_padded_bytes_per_row(width, 4);

        let buffer_size = (padded_bytes_per_row * height) as u64;
        if size_changed
            || self
                .rtb_readback_buffer
                .as_ref()
                .map(|existing_buffer| existing_buffer.size() < buffer_size)
                .unwrap_or(true)
        {
            self.rtb_readback_buffer = Some(create_readback_buffer(
                &self.device,
                Some("rtb_readback_buffer"),
                buffer_size,
            ));
        }

        let output_buffer = self.rtb_readback_buffer.as_ref().unwrap();

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("copy_texture_encoder"),
            });

        encode_copy_texture_to_buffer(
            &mut encoder,
            self.rtb_offscreen_texture.as_ref().unwrap(),
            output_buffer,
            width,
            height,
            padded_bytes_per_row,
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        let mut readback_bytes = std::mem::take(&mut self.scratch.readback_bytes);
        Self::map_readback_buffer_into(&self.device, output_buffer, &mut readback_bytes);
        let required_readback_len = (height as usize).saturating_mul(padded_bytes_per_row as usize);
        if readback_bytes.is_empty() || readback_bytes.len() < required_readback_len {
            self.scratch.readback_bytes = readback_bytes;
            return;
        }
        copy_padded_readback_rows(
            &readback_bytes,
            height,
            unpadded_bytes_per_row,
            padded_bytes_per_row,
            buffer,
        );

        self.scratch.readback_bytes = readback_bytes;

        #[cfg(feature = "render_metrics")]
        {
            let frame_presented_at = std::time::Instant::now();
            self.render_loop_metrics_tracker
                .record_presented_frame(frame_render_loop_started_at, frame_presented_at);
        }
    }

    pub fn render_to_argb32(&mut self, out_pixels: &mut [u32]) {
        #[cfg(feature = "render_metrics")]
        let frame_render_loop_started_at = std::time::Instant::now();

        self.prepare_render();

        let (width, height) = self.physical_size;
        let needed_len = (width as usize) * (height as usize);
        if out_pixels.len() < needed_len {
            warn!(
                "render_to_argb32: output slice too small: {} < {}",
                out_pixels.len(),
                needed_len
            );
            return;
        }

        let size_changed = self.argb_cached_width != width || self.argb_cached_height != height;
        if size_changed {
            self.argb_cached_width = width;
            self.argb_cached_height = height;
        }

        if size_changed || self.argb_offscreen_texture.is_none() {
            self.argb_offscreen_texture = Some(create_offscreen_color_texture(
                &self.device,
                (width, height),
                self.config.format,
            ));
        }

        let texture_view = self
            .argb_offscreen_texture
            .as_ref()
            .unwrap()
            .create_view(&wgpu::TextureViewDescriptor::default());

        let output_texture = self.argb_offscreen_texture.take();
        self.render_to_texture_view(&texture_view, output_texture.as_ref());
        self.argb_offscreen_texture = output_texture;

        let (_, padded_bytes_per_row) = compute_padded_bytes_per_row(width, 4);
        let input_buffer_size = (padded_bytes_per_row as u64) * (height as u64);
        if size_changed
            || self.argb_input_buffer.is_none()
            || self.argb_input_buffer_size < input_buffer_size
        {
            self.argb_input_buffer = Some(create_storage_input_buffer(
                &self.device,
                Some("argb_input_padded_bytes"),
                input_buffer_size,
            ));
            self.argb_input_buffer_size = input_buffer_size;
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("argb_copy_encoder"),
            });

        encode_copy_texture_to_buffer(
            &mut encoder,
            self.argb_offscreen_texture.as_ref().unwrap(),
            self.argb_input_buffer.as_ref().unwrap(),
            width,
            height,
            padded_bytes_per_row,
        );

        let output_words = (width as u64) * (height as u64);
        let output_buffer_size = output_words * 4;
        if size_changed
            || self.argb_output_storage_buffer.is_none()
            || self.argb_output_buffer_size < output_buffer_size
        {
            self.argb_output_storage_buffer = Some(create_storage_output_buffer(
                &self.device,
                Some("argb_output_u32_storage"),
                output_buffer_size,
            ));
            self.argb_output_buffer_size = output_buffer_size;
            self.argb_readback_buffer = Some(create_readback_buffer(
                &self.device,
                Some("argb_output_u32_readback"),
                output_buffer_size,
            ));
        }

        if self.argb_cs_pipeline.is_none() {
            let (bind_group_layout, pipeline) = create_argb_swizzle_pipeline(&self.device);
            self.argb_cs_bgl = Some(bind_group_layout);
            self.argb_cs_pipeline = Some(pipeline);
        }

        let params = ArgbParams {
            width,
            height,
            padded_bpr: padded_bytes_per_row,
            _pad: 0,
        };
        let needs_new_params = self.argb_params_buffer.is_none();
        if needs_new_params {
            self.argb_params_buffer = Some(crate::pipeline::create_argb_params_buffer(
                &self.device,
                &params,
            ));
        } else {
            self.queue.write_buffer(
                self.argb_params_buffer.as_ref().unwrap(),
                0,
                bytemuck::bytes_of(&params),
            );
        }

        if size_changed || self.argb_swizzle_bind_group.is_none() || needs_new_params {
            self.argb_swizzle_bind_group = Some(create_argb_swizzle_bind_group(
                &self.device,
                self.argb_cs_bgl.as_ref().unwrap(),
                self.argb_input_buffer.as_ref().unwrap(),
                self.argb_output_storage_buffer.as_ref().unwrap(),
                self.argb_params_buffer.as_ref().unwrap(),
            ));
        }

        self.queue.submit(std::iter::once(encoder.finish()));

        let mut compute_encoder =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("argb_compute_encoder"),
                });
        {
            let mut pass = compute_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("argb_swizzle_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(self.argb_cs_pipeline.as_ref().unwrap());
            pass.set_bind_group(0, self.argb_swizzle_bind_group.as_ref().unwrap(), &[]);
            let workgroup_x = width.div_ceil(16);
            let workgroup_y = height.div_ceil(16);
            pass.dispatch_workgroups(workgroup_x, workgroup_y, 1);
        }
        self.queue.submit(std::iter::once(compute_encoder.finish()));

        let mut readback_encoder =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("argb_readback_copy_encoder"),
                });
        readback_encoder.copy_buffer_to_buffer(
            self.argb_output_storage_buffer.as_ref().unwrap(),
            0,
            self.argb_readback_buffer.as_ref().unwrap(),
            0,
            output_buffer_size,
        );
        self.queue
            .submit(std::iter::once(readback_encoder.finish()));

        let mut readback_bytes = std::mem::take(&mut self.scratch.readback_bytes);
        Self::map_readback_buffer_into(
            &self.device,
            self.argb_readback_buffer.as_ref().unwrap(),
            &mut readback_bytes,
        );
        if readback_bytes.is_empty() {
            self.scratch.readback_bytes = readback_bytes;
            return;
        }

        let src_words: &[u32] = bytemuck::cast_slice(&readback_bytes);
        out_pixels[..needed_len].copy_from_slice(&src_words[..needed_len]);
        self.scratch.readback_bytes = readback_bytes;

        #[cfg(feature = "render_metrics")]
        {
            let frame_presented_at = std::time::Instant::now();
            self.render_loop_metrics_tracker
                .record_presented_frame(frame_render_loop_started_at, frame_presented_at);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::copy_padded_readback_rows;

    #[test]
    fn copy_padded_readback_rows_handles_unpadded_data() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let mut output = Vec::new();

        copy_padded_readback_rows(&data, 2, 4, 4, &mut output);
        assert_eq!(output, data);
    }

    #[test]
    fn copy_padded_readback_rows_strips_padding() {
        let data = vec![1, 2, 3, 4, 9, 9, 9, 9, 5, 6, 7, 8, 8, 8, 8, 8];
        let mut output = Vec::new();

        copy_padded_readback_rows(&data, 2, 4, 8, &mut output);
        assert_eq!(output, vec![1, 2, 3, 4, 5, 6, 7, 8]);
    }
}
