use super::*;

impl<'a> Renderer<'a> {
    fn map_readback_buffer_to_vec(&self, buffer: &wgpu::Buffer) -> Vec<u8> {
        let buffer_slice = buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            if sender.send(result).is_err() {
                log::warn!("Failed to send map_async result from callback");
            }
        });

        let _ = self.device.poll(wgpu::MaintainBase::Wait);

        let map_result = match receiver.recv() {
            Ok(result) => result,
            Err(error) => {
                log::warn!("Failed to receive mapped buffer result: {}", error);
                return Vec::new();
            }
        };

        if let Err(error) = map_result {
            log::warn!("Failed to map readback buffer: {:?}", error);
            return Vec::new();
        }

        let mapped_range = buffer_slice.get_mapped_range();
        let mapped_bytes = mapped_range.to_vec();
        drop(mapped_range);
        buffer.unmap();
        mapped_bytes
    }

    pub fn render_to_buffer(&mut self, buffer: &mut Vec<u8>) {
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

        let data = self.map_readback_buffer_to_vec(output_buffer);
        let output_size = (unpadded_bytes_per_row * height) as usize;
        buffer.resize(output_size, 0);

        if padded_bytes_per_row == unpadded_bytes_per_row {
            buffer.copy_from_slice(&data);
        } else {
            for row in 0..height {
                let padded_offset = (row * padded_bytes_per_row) as usize;
                let unpadded_offset = (row * unpadded_bytes_per_row) as usize;
                let row_data =
                    &data[padded_offset..padded_offset + unpadded_bytes_per_row as usize];
                buffer[unpadded_offset..unpadded_offset + unpadded_bytes_per_row as usize]
                    .copy_from_slice(row_data);
            }
        }
    }

    pub fn render_to_argb32(&mut self, out_pixels: &mut [u32]) {
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

        let data = self.map_readback_buffer_to_vec(self.argb_readback_buffer.as_ref().unwrap());
        if data.is_empty() {
            return;
        }

        let src_words: &[u32] = bytemuck::cast_slice(&data);
        out_pixels[..needed_len].copy_from_slice(&src_words[..needed_len]);
    }
}
