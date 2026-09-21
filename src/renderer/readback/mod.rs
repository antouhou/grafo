use super::Renderer;
use crate::pipeline::{
    compute_padded_bytes_per_row, create_argb_row_packing_bind_group,
    create_argb_row_packing_params_buffer, create_argb_row_packing_pipeline,
    create_offscreen_color_texture, create_readback_buffer, encode_copy_texture_to_buffer,
    ArgbRowPackingParams,
};
#[cfg(feature = "render_metrics")]
use crate::renderer::metrics::PhaseTimings;
use crate::renderer::types::GeometryBufferError;
#[cfg(feature = "render_metrics")]
use std::time::{Duration, Instant};
use std::{iter, sync::mpsc};
use thiserror::Error;
use wgpu::{
    BindGroup, BindGroupLayout, Buffer, BufferAsyncError, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, Device, MapMode, PollError,
    PollType, Texture, TextureFormat, TextureViewDescriptor,
};

/// An offscreen render could not prepare geometry or read its pixels back from the GPU.
#[derive(Error, Debug)]
pub enum ReadbackError {
    #[error(transparent)]
    GeometryBuffer(#[from] GeometryBufferError),
    #[error("Output buffer needs {required_pixels} pixels, but has {provided_pixels}")]
    OutputTooSmall {
        required_pixels: usize,
        provided_pixels: usize,
    },
    #[error("Failed to wait for GPU readback: {0}")]
    GpuWait(#[from] PollError),
    #[error("Failed to map the readback buffer: {0}")]
    BufferMap(#[from] BufferAsyncError),
    #[error("Readback mapping callback was dropped before reporting a result")]
    MapCallbackDropped,
}

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

pub(super) struct BgraReadbackResources {
    pub(super) texture: Texture,
    pub(super) buffer: Buffer,
}

impl BgraReadbackResources {
    fn new(device: &Device, physical_size: (u32, u32), format: TextureFormat) -> Self {
        let (_, padded_bytes_per_row) = compute_padded_bytes_per_row(physical_size.0, 4);
        Self {
            texture: create_offscreen_color_texture(device, physical_size, format),
            buffer: create_readback_buffer(
                device,
                Some("rtb_readback_buffer"),
                u64::from(padded_bytes_per_row) * u64::from(physical_size.1),
            ),
        }
    }
}

/// The texture, buffers, and bindings for one ARGB readback size.
pub(super) struct ArgbReadbackTarget {
    pub(super) texture: Texture,
    pub(super) input_buffer: Buffer,
    pub(super) output_buffer: Buffer,
    pub(super) readback_buffer: Buffer,
    pub(super) params_buffer: Buffer,
    bind_group: BindGroup,
}

impl ArgbReadbackTarget {
    fn new(
        device: &Device,
        bind_group_layout: &BindGroupLayout,
        physical_size: (u32, u32),
        format: TextureFormat,
    ) -> Self {
        let (width, height) = physical_size;
        let (_, padded_bytes_per_row) = compute_padded_bytes_per_row(width, 4);
        let input_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("argb_input_padded_bytes"),
            size: u64::from(padded_bytes_per_row) * u64::from(height),
            usage: BufferUsages::COPY_DST | BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let output_buffer_size = u64::from(width) * u64::from(height) * 4;
        let output_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("argb_output_u32_storage"),
            size: output_buffer_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback_buffer =
            create_readback_buffer(device, Some("argb_output_u32_readback"), output_buffer_size);
        let params_buffer = create_argb_row_packing_params_buffer(
            device,
            &ArgbRowPackingParams {
                width,
                height,
                padded_bytes_per_row,
                _pad: 0,
            },
        );
        let bind_group = create_argb_row_packing_bind_group(
            device,
            bind_group_layout,
            &input_buffer,
            &output_buffer,
            &params_buffer,
        );
        Self {
            texture: create_offscreen_color_texture(device, physical_size, format),
            input_buffer,
            output_buffer,
            readback_buffer,
            params_buffer,
            bind_group,
        }
    }
}

pub(super) struct ArgbReadbackResources {
    pipeline: ComputePipeline,
    bind_group_layout: BindGroupLayout,
    pub(super) target: ArgbReadbackTarget,
}

impl ArgbReadbackResources {
    fn new(device: &Device, physical_size: (u32, u32), format: TextureFormat) -> Self {
        let (bind_group_layout, pipeline) = create_argb_row_packing_pipeline(device);
        let target = ArgbReadbackTarget::new(device, &bind_group_layout, physical_size, format);
        Self {
            pipeline,
            bind_group_layout,
            target,
        }
    }

    fn resize(&mut self, device: &Device, physical_size: (u32, u32), format: TextureFormat) {
        if (self.target.texture.width(), self.target.texture.height()) != physical_size {
            self.target =
                ArgbReadbackTarget::new(device, &self.bind_group_layout, physical_size, format);
        }
    }
}

impl<'a> Renderer<'a> {
    fn map_readback_buffer_into(
        device: &Device,
        buffer: &Buffer,
        mapped_bytes: &mut Vec<u8>,
    ) -> Result<(), ReadbackError> {
        mapped_bytes.clear();

        let buffer_slice = buffer.slice(..);
        let (sender, receiver) = mpsc::channel();
        buffer_slice.map_async(MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        if let Err(error) = device.poll(PollType::Wait) {
            buffer.unmap();
            return Err(error.into());
        }
        receiver
            .recv()
            .map_err(|_| ReadbackError::MapCallbackDropped)??;

        let mapped_range = buffer_slice.get_mapped_range();
        mapped_bytes.extend_from_slice(&mapped_range);
        drop(mapped_range);
        buffer.unmap();
        Ok(())
    }

    /// Reads tightly packed BGRA pixels into `buffer`, resizing it to the viewport.
    /// Returns an error if geometry preparation or GPU readback fails.
    /// On error, `buffer` retains its previous contents.
    pub fn render_to_buffer(&mut self, buffer: &mut Vec<u8>) -> Result<(), ReadbackError> {
        #[cfg(feature = "render_metrics")]
        let frame_render_loop_started_at = Instant::now();

        self.prepare_render()?;

        #[cfg(feature = "render_metrics")]
        let after_prepare = Instant::now();

        let (width, height) = self.state.physical_size;

        let physical_size = (width, height);
        let resources = match self.bgra_readback.take() {
            Some(resources)
                if (resources.texture.width(), resources.texture.height()) == physical_size =>
            {
                resources
            }
            _ => BgraReadbackResources::new(&self.device, physical_size, self.config.format),
        };
        let texture_view = resources
            .texture
            .create_view(&TextureViewDescriptor::default());
        self.render_to_texture_view(&texture_view, Some(&resources.texture));

        let (unpadded_bytes_per_row, padded_bytes_per_row) = compute_padded_bytes_per_row(width, 4);

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("copy_texture_encoder"),
            });

        encode_copy_texture_to_buffer(
            &mut encoder,
            &resources.texture,
            &resources.buffer,
            width,
            height,
            padded_bytes_per_row,
        );

        self.queue.submit(iter::once(encoder.finish()));

        #[cfg(feature = "render_metrics")]
        let after_submit = Instant::now();

        let readback_bytes = &mut self.state.scratch.readback_bytes;
        let readback_result =
            Self::map_readback_buffer_into(&self.device, &resources.buffer, readback_bytes);
        self.bgra_readback = Some(resources);
        readback_result?;
        copy_padded_readback_rows(
            readback_bytes,
            height,
            unpadded_bytes_per_row,
            padded_bytes_per_row,
            buffer,
        );

        #[cfg(feature = "render_metrics")]
        {
            let frame_presented_at = Instant::now();
            let prepare_dur = after_prepare.saturating_duration_since(frame_render_loop_started_at);
            let encode_submit_dur = after_submit.saturating_duration_since(after_prepare);
            let readback_dur = frame_presented_at.saturating_duration_since(after_submit);
            let total_dur =
                frame_presented_at.saturating_duration_since(frame_render_loop_started_at);
            self.last_phase_timings = PhaseTimings {
                prepare: prepare_dur,
                encode_and_submit: encode_submit_dur,
                present_or_readback: readback_dur,
                gpu_wait: Duration::ZERO, // GPU wait is included in readback time.
                total: total_dur,
            };
            self.render_loop_metrics_tracker
                .record_presented_frame(frame_render_loop_started_at, frame_presented_at);
        }
        Ok(())
    }

    /// Reads ARGB pixels into the first viewport-sized portion of `out_pixels`.
    /// Returns an error if the output is too small, geometry preparation fails,
    /// or GPU readback fails. On error, `out_pixels` retains its previous contents.
    pub fn render_to_argb32(&mut self, out_pixels: &mut [u32]) -> Result<(), ReadbackError> {
        let (width, height) = self.state.physical_size;
        let needed_len = (width as usize) * (height as usize);
        if out_pixels.len() < needed_len {
            return Err(ReadbackError::OutputTooSmall {
                required_pixels: needed_len,
                provided_pixels: out_pixels.len(),
            });
        }

        #[cfg(feature = "render_metrics")]
        let frame_render_loop_started_at = Instant::now();

        self.prepare_render()?;

        #[cfg(feature = "render_metrics")]
        let after_prepare = Instant::now();

        let mut resources = self.argb_readback.take().unwrap_or_else(|| {
            ArgbReadbackResources::new(&self.device, (width, height), self.config.format)
        });
        resources.resize(&self.device, (width, height), self.config.format);
        let target = &resources.target;
        let texture_view = target
            .texture
            .create_view(&TextureViewDescriptor::default());
        self.render_to_texture_view(&texture_view, Some(&target.texture));

        let (_, padded_bytes_per_row) = compute_padded_bytes_per_row(width, 4);
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("argb_copy_encoder"),
            });
        encode_copy_texture_to_buffer(
            &mut encoder,
            &target.texture,
            &target.input_buffer,
            width,
            height,
            padded_bytes_per_row,
        );

        self.queue.submit(iter::once(encoder.finish()));

        let mut compute_encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("argb_compute_encoder"),
            });
        {
            let mut pass = compute_encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("argb_row_packing_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&resources.pipeline);
            pass.set_bind_group(0, &target.bind_group, &[]);
            let workgroup_x = width.div_ceil(16);
            let workgroup_y = height.div_ceil(16);
            pass.dispatch_workgroups(workgroup_x, workgroup_y, 1);
        }
        self.queue.submit(iter::once(compute_encoder.finish()));

        let mut readback_encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("argb_readback_copy_encoder"),
            });
        readback_encoder.copy_buffer_to_buffer(
            &target.output_buffer,
            0,
            &target.readback_buffer,
            0,
            target.output_buffer.size(),
        );
        self.queue.submit(iter::once(readback_encoder.finish()));

        #[cfg(feature = "render_metrics")]
        let after_submit = Instant::now();

        let readback_bytes = &mut self.state.scratch.readback_bytes;
        let readback_result =
            Self::map_readback_buffer_into(&self.device, &target.readback_buffer, readback_bytes);
        self.argb_readback = Some(resources);
        readback_result?;

        let src_words: &[u32] = bytemuck::cast_slice(readback_bytes);
        out_pixels[..needed_len].copy_from_slice(&src_words[..needed_len]);

        #[cfg(feature = "render_metrics")]
        {
            let frame_presented_at = Instant::now();
            let prepare_dur = after_prepare.saturating_duration_since(frame_render_loop_started_at);
            let encode_submit_dur = after_submit.saturating_duration_since(after_prepare);
            let readback_dur = frame_presented_at.saturating_duration_since(after_submit);
            let total_dur =
                frame_presented_at.saturating_duration_since(frame_render_loop_started_at);
            self.last_phase_timings = PhaseTimings {
                prepare: prepare_dur,
                encode_and_submit: encode_submit_dur,
                present_or_readback: readback_dur,
                gpu_wait: Duration::ZERO, // GPU wait is included in readback time.
                total: total_dur,
            };
            self.render_loop_metrics_tracker
                .record_presented_frame(frame_render_loop_started_at, frame_presented_at);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
