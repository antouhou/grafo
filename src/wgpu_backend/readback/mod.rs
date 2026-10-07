use super::WgpuBackend;
use crate::commands::RenderPlan;
use crate::core::linear_to_srgb_u8;
use crate::render_backend::render_target::{PixelFormat, PixmapMut};
#[cfg(feature = "render_metrics")]
use crate::wgpu_backend::metrics::PhaseTimings;
use crate::wgpu_backend::pipeline::{
    compute_padded_bytes_per_row, create_argb_row_packing_bind_group,
    create_argb_row_packing_params_buffer, create_argb_row_packing_pipeline,
    create_readback_buffer, encode_copy_texture_to_buffer, ArgbRowPackingParams,
};
use mapping::ReadbackMapping;
use std::iter;
#[cfg(feature = "render_metrics")]
use std::time::{Duration, Instant};
use thiserror::Error;
use wgpu::{
    BindGroup, BindGroupLayout, Buffer, BufferAsyncError, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, Device, MapMode, PollError,
    PollType, TextureFormat,
};

mod mapping;

/// An offscreen render could not read its pixels into the output buffer.
#[derive(Error, Debug)]
pub enum ReadbackError {
    #[error("Texture format {0:?} cannot be read into an eight-bit pixel surface")]
    UnsupportedFormat(TextureFormat),
    #[error("Failed to wait for GPU readback: {0}")]
    GpuWait(#[from] PollError),
    #[error("Failed to map the readback buffer: {0}")]
    BufferMap(#[from] BufferAsyncError),
    #[error("Readback mapping callback was dropped before reporting a result")]
    MapCallbackDropped,
}

fn validate_readback_format(format: TextureFormat) -> Result<(), ReadbackError> {
    if matches!(
        format,
        TextureFormat::Bgra8UnormSrgb
            | TextureFormat::Rgba8UnormSrgb
            | TextureFormat::Bgra8Unorm
            | TextureFormat::Rgba8Unorm
    ) {
        Ok(())
    } else {
        Err(ReadbackError::UnsupportedFormat(format))
    }
}

fn copy_readback_rows(
    data: &[u8],
    source_stride: usize,
    source_format: TextureFormat,
    output: &mut PixmapMut<'_>,
) -> Result<(), ReadbackError> {
    validate_readback_format(source_format)?;
    let layout = output.layout();
    let row_bytes = layout.size().0 as usize * 4;
    let is_bgra = matches!(
        source_format,
        TextureFormat::Bgra8UnormSrgb | TextureFormat::Bgra8Unorm
    );
    let is_srgb = source_format.is_srgb();
    let matches_layout = is_srgb
        && match layout.format() {
            PixelFormat::Bgra8 => is_bgra,
            PixelFormat::Rgba8 => !is_bgra,
            PixelFormat::Argb32 => is_bgra && cfg!(target_endian = "little"),
        };
    let output = output.pixels_mut();
    if matches_layout && source_stride == row_bytes && layout.stride() == row_bytes {
        output[..layout.byte_len()].copy_from_slice(&data[..layout.byte_len()]);
        return Ok(());
    }
    for row in 0..layout.size().1 as usize {
        let source = &data[row * source_stride..row * source_stride + row_bytes];
        let destination = &mut output[row * layout.stride()..row * layout.stride() + row_bytes];
        if matches_layout {
            destination.copy_from_slice(source);
            continue;
        }
        for (source, destination) in source
            .as_chunks::<4>()
            .0
            .iter()
            .zip(destination.as_chunks_mut::<4>().0.iter_mut())
        {
            let (mut red, mut green, mut blue, alpha) = if is_bgra {
                (source[2], source[1], source[0], source[3])
            } else {
                (source[0], source[1], source[2], source[3])
            };
            if !is_srgb {
                red = linear_to_srgb_u8(red as f32 / 255.0);
                green = linear_to_srgb_u8(green as f32 / 255.0);
                blue = linear_to_srgb_u8(blue as f32 / 255.0);
            }
            let pixel = match layout.format() {
                PixelFormat::Bgra8 => [blue, green, red, alpha],
                PixelFormat::Rgba8 => [red, green, blue, alpha],
                PixelFormat::Argb32 => u32::from_be_bytes([alpha, red, green, blue]).to_ne_bytes(),
            };
            destination.copy_from_slice(&pixel);
        }
    }
    Ok(())
}

pub(in crate::wgpu_backend) struct ByteReadbackResources {
    physical_size: (u32, u32),
    mapping: ReadbackMapping,
    pub(in crate::wgpu_backend) buffer: Buffer,
}

impl ByteReadbackResources {
    fn new(device: &Device, physical_size: (u32, u32)) -> Self {
        let (_, padded_bytes_per_row) = compute_padded_bytes_per_row(physical_size.0, 4);
        Self {
            physical_size,
            mapping: ReadbackMapping::new(),
            buffer: create_readback_buffer(
                device,
                Some("rtb_readback_buffer"),
                u64::from(padded_bytes_per_row) * u64::from(physical_size.1),
            ),
        }
    }
}

/// Buffers and bindings for one ARGB readback size.
pub(in crate::wgpu_backend) struct ArgbReadbackBuffers {
    physical_size: (u32, u32),
    mapping: ReadbackMapping,
    pub(in crate::wgpu_backend) input_buffer: Buffer,
    pub(in crate::wgpu_backend) output_buffer: Buffer,
    pub(in crate::wgpu_backend) readback_buffer: Buffer,
    #[cfg(feature = "render_metrics")]
    pub(in crate::wgpu_backend) params_buffer: Buffer,
    bind_group: BindGroup,
}

impl ArgbReadbackBuffers {
    fn new(
        device: &Device,
        bind_group_layout: &BindGroupLayout,
        physical_size: (u32, u32),
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
            physical_size,
            mapping: ReadbackMapping::new(),
            input_buffer,
            output_buffer,
            readback_buffer,
            #[cfg(feature = "render_metrics")]
            params_buffer,
            bind_group,
        }
    }
}

pub(in crate::wgpu_backend) struct ArgbReadbackResources {
    pipeline: ComputePipeline,
    bind_group_layout: BindGroupLayout,
    pub(in crate::wgpu_backend) buffers: ArgbReadbackBuffers,
}

impl ArgbReadbackResources {
    fn new(device: &Device, physical_size: (u32, u32)) -> Self {
        let (bind_group_layout, pipeline) = create_argb_row_packing_pipeline(device);
        let buffers = ArgbReadbackBuffers::new(device, &bind_group_layout, physical_size);
        Self {
            pipeline,
            bind_group_layout,
            buffers,
        }
    }

    fn resize(&mut self, device: &Device, physical_size: (u32, u32)) {
        if self.buffers.physical_size != physical_size {
            self.buffers = ArgbReadbackBuffers::new(device, &self.bind_group_layout, physical_size);
        }
    }
}

impl WgpuBackend {
    #[cfg(feature = "render_metrics")]
    fn record_readback_metrics(
        &mut self,
        render_started_at: Instant,
        preparation_finished_at: Instant,
        submission_finished_at: Instant,
    ) {
        let readback_finished_at = Instant::now();
        self.last_phase_timings = PhaseTimings {
            prepare: preparation_finished_at.saturating_duration_since(render_started_at),
            encode_and_submit: submission_finished_at
                .saturating_duration_since(preparation_finished_at),
            present_or_readback: readback_finished_at
                .saturating_duration_since(submission_finished_at),
            gpu_wait: Duration::ZERO, // GPU wait is included in readback time.
            total: readback_finished_at.saturating_duration_since(render_started_at),
        };
    }

    fn map_readback_buffer_into(
        device: &Device,
        buffer: &Buffer,
        mapping: &ReadbackMapping,
        mapped_bytes: &mut Vec<u8>,
    ) -> Result<(), ReadbackError> {
        mapped_bytes.clear();

        let buffer_slice = buffer.slice(..);
        let completion = mapping.completion();
        buffer_slice.map_async(MapMode::Read, move |result| {
            completion.finish(result);
        });

        // A failed target is dropped so its late callback cannot reach a later mapping.
        if let Err(error) = device.poll(PollType::Wait) {
            buffer.unmap();
            return Err(error.into());
        }
        mapping.wait()?;

        let mapped_range = buffer_slice.get_mapped_range();
        mapped_bytes.extend_from_slice(&mapped_range);
        drop(mapped_range);
        buffer.unmap();
        Ok(())
    }

    pub(in crate::wgpu_backend) fn render_byte_pixels(
        &mut self,
        commands: &RenderPlan,
        output: &mut PixmapMut<'_>,
    ) -> Result<(), ReadbackError> {
        validate_readback_format(self.format)?;
        #[cfg(feature = "render_metrics")]
        let render_started_at = Instant::now();

        self.prepare_resources();

        #[cfg(feature = "render_metrics")]
        let preparation_finished_at = Instant::now();

        let (width, height) = self.viewport.physical_size;

        let physical_size = (width, height);
        let resources = match self.byte_readback.take() {
            Some(resources) if resources.physical_size == physical_size => resources,
            _ => ByteReadbackResources::new(&self.device, physical_size),
        };
        let _ = self.update_retained_output(commands);
        let retained = self
            .retained_output
            .as_ref()
            .expect("retained output was initialized before readback");

        let (_, padded_bytes_per_row) = compute_padded_bytes_per_row(width, 4);

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("copy_texture_encoder"),
            });

        encode_copy_texture_to_buffer(
            &mut encoder,
            &retained.texture,
            &resources.buffer,
            width,
            height,
            padded_bytes_per_row,
        );

        self.queue.submit(iter::once(encoder.finish()));

        #[cfg(feature = "render_metrics")]
        let submission_finished_at = Instant::now();

        let readback_bytes = &mut self.readback_bytes;
        Self::map_readback_buffer_into(
            &self.device,
            &resources.buffer,
            &resources.mapping,
            readback_bytes,
        )?;
        self.byte_readback = Some(resources);
        copy_readback_rows(
            readback_bytes,
            padded_bytes_per_row as usize,
            self.format,
            output,
        )?;

        #[cfg(feature = "render_metrics")]
        self.record_readback_metrics(
            render_started_at,
            preparation_finished_at,
            submission_finished_at,
        );
        Ok(())
    }

    pub(in crate::wgpu_backend) fn render_argb_pixels(
        &mut self,
        commands: &RenderPlan,
        output: &mut PixmapMut<'_>,
    ) -> Result<(), ReadbackError> {
        validate_readback_format(self.format)?;
        let (width, height) = self.viewport.physical_size;
        #[cfg(feature = "render_metrics")]
        let render_started_at = Instant::now();

        self.prepare_resources();

        #[cfg(feature = "render_metrics")]
        let preparation_finished_at = Instant::now();

        let mut resources = self
            .argb_readback
            .take()
            .unwrap_or_else(|| ArgbReadbackResources::new(&self.device, (width, height)));
        resources.resize(&self.device, (width, height));
        let _ = self.update_retained_output(commands);
        let retained = self
            .retained_output
            .as_ref()
            .expect("retained output was initialized before readback");
        let buffers = &resources.buffers;

        let (_, padded_bytes_per_row) = compute_padded_bytes_per_row(width, 4);
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("argb_copy_encoder"),
            });
        encode_copy_texture_to_buffer(
            &mut encoder,
            &retained.texture,
            &buffers.input_buffer,
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
            pass.set_bind_group(0, &buffers.bind_group, &[]);
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
            &buffers.output_buffer,
            0,
            &buffers.readback_buffer,
            0,
            buffers.output_buffer.size(),
        );
        self.queue.submit(iter::once(readback_encoder.finish()));

        #[cfg(feature = "render_metrics")]
        let submission_finished_at = Instant::now();

        let readback_bytes = &mut self.readback_bytes;
        Self::map_readback_buffer_into(
            &self.device,
            &buffers.readback_buffer,
            &buffers.mapping,
            readback_bytes,
        )?;
        self.argb_readback = Some(resources);

        copy_readback_rows(readback_bytes, width as usize * 4, self.format, output)?;

        #[cfg(feature = "render_metrics")]
        self.record_readback_metrics(
            render_started_at,
            preparation_finished_at,
            submission_finished_at,
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests;
