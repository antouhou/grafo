#![allow(dead_code)]

use super::types::{GradientData, GradientKind};

/// GPU-side gradient-only parameters packed into a uniform-friendly struct.
/// Matches the WGSL `GradientColorParams` struct in shader.wgsl.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[allow(dead_code)]
pub(crate) struct GpuGradientColorParams {
    // gradient_type: 1=linear, 2=radial, 3=conic, 0=none
    pub gradient_type: u32,
    // spread_mode: 0=pad, 1=repeat
    pub spread_mode: u32,
    // units: 0=local, 1=canvas
    pub units: u32,
    pub is_constant: u32,

    // Constant color (for degenerate cases)
    pub constant_color: [f32; 4],

    // Linear params: start_x, start_y, end_x, end_y
    pub linear_start: [f32; 2],
    pub linear_end: [f32; 2],

    // Radial params: center_x, center_y, radius_x, radius_y
    pub radial_center: [f32; 2],
    pub radial_radius: [f32; 2],

    // Conic params: center_x, center_y, start_angle
    pub conic_center: [f32; 2],
    pub conic_start_angle: f32,

    // Period info for repeating
    pub period_start: f32,
    pub period_len: f32,

    // Ramp domain: the t range the ramp texture covers [ramp_start, ramp_end]
    pub ramp_start: f32,
    pub ramp_end: f32,

    pub _padding: f32,
}

impl GpuGradientColorParams {
    pub fn from_gradient_data(data: &GradientData) -> Self {
        let gradient_type = match data.kind {
            GradientKind::Linear => 1u32,
            GradientKind::Radial => 2u32,
            GradientKind::Conic => 3u32,
        };
        let spread_mode = match data.spread {
            super::types::SpreadMode::Pad => 0u32,
            super::types::SpreadMode::Repeat => 1u32,
        };
        let units = match data.units {
            super::types::GradientUnits::Local => 0u32,
            super::types::GradientUnits::Canvas => 1u32,
        };

        let linear_start = data.linear_line.map(|l| l.start).unwrap_or([0.0, 0.0]);
        let linear_end = data.linear_line.map(|l| l.end).unwrap_or([0.0, 0.0]);
        let radial_center = data.radial_center.unwrap_or([0.0, 0.0]);
        let radial_radius = data.radial_radius.unwrap_or([0.0, 0.0]);
        let conic_center = data.conic_center.unwrap_or([0.0, 0.0]);
        let conic_start_angle = data.conic_start_angle.unwrap_or(0.0);

        // The shader normalises conic angles to turns [0, 1] (angle / TAU),
        // but the CPU normalises conic stop positions in radians [0, TAU].
        // Convert period to turns so both sides use the same domain.
        let is_conic = data.kind == GradientKind::Conic;
        let (period_start, period_len) = if is_conic {
            let tau = std::f32::consts::TAU;
            (data.period_start / tau, data.period_len / tau)
        } else {
            (data.period_start, data.period_len)
        };

        GpuGradientColorParams {
            gradient_type,
            spread_mode,
            units,
            is_constant: data.is_constant as u32,
            constant_color: data.constant_color,
            linear_start,
            linear_end,
            radial_center,
            radial_radius,
            conic_center,
            conic_start_angle,
            period_start,
            period_len,
            ramp_start: period_start,
            ramp_end: period_start + period_len,
            _padding: 0.0,
        }
    }

    pub fn none() -> Self {
        Self {
            gradient_type: 0,
            spread_mode: 0,
            units: 0,
            is_constant: 0,
            constant_color: [0.0; 4],
            linear_start: [0.0; 2],
            linear_end: [0.0; 2],
            radial_center: [0.0; 2],
            radial_radius: [0.0; 2],
            conic_center: [0.0; 2],
            conic_start_angle: 0.0,
            period_start: 0.0,
            period_len: 0.0,
            ramp_start: 0.0,
            ramp_end: 1.0,
            _padding: 0.0,
        }
    }
}

/// GPU-side backdrop sampling metadata reused by both solid and gradient backdrop draws.
/// Matches the WGSL `BackdropSamplingParams` struct in shader.wgsl.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct GpuBackdropSamplingParams {
    pub capture_origin: [f32; 2],
    pub inverse_capture_size: [f32; 2],
}

impl Default for GpuBackdropSamplingParams {
    fn default() -> Self {
        Self {
            capture_origin: [0.0, 0.0],
            inverse_capture_size: [1.0, 1.0],
        }
    }
}

impl GpuBackdropSamplingParams {
    pub fn from_sampling_uniform(
        sampling_uniform: crate::pipeline::BackdropSamplingUniform,
    ) -> Self {
        Self {
            capture_origin: sampling_uniform.capture_origin,
            inverse_capture_size: sampling_uniform.inverse_capture_size,
        }
    }
}

/// GPU-side material parameters bound at group 3 binding 0.
///
/// This uniform layout is shared across regular gradient fills and backdrop-capable pipelines.
/// Solid backdrop draws leave `gradient` in its inert `none()` state and only populate
/// `backdrop_sampling`.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct GpuMaterialParams {
    pub gradient: GpuGradientColorParams,
    pub backdrop_sampling: GpuBackdropSamplingParams,
}

impl Default for GpuMaterialParams {
    fn default() -> Self {
        Self {
            gradient: GpuGradientColorParams::none(),
            backdrop_sampling: GpuBackdropSamplingParams::default(),
        }
    }
}

impl GpuMaterialParams {
    pub fn from_gradient_data(data: &GradientData) -> Self {
        Self {
            gradient: GpuGradientColorParams::from_gradient_data(data),
            backdrop_sampling: GpuBackdropSamplingParams::default(),
        }
    }

    pub fn with_backdrop_sampling(
        mut self,
        sampling_uniform: crate::pipeline::BackdropSamplingUniform,
    ) -> Self {
        self.backdrop_sampling = GpuBackdropSamplingParams::from_sampling_uniform(sampling_uniform);
        self
    }

    pub fn for_backdrop_sampling(
        sampling_uniform: crate::pipeline::BackdropSamplingUniform,
    ) -> Self {
        Self::default().with_backdrop_sampling(sampling_uniform)
    }
}

/// Creates a 1D ramp texture from the baked ramp data.
/// Returns (texture, texture_view).
pub(crate) fn create_ramp_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    ramp: &[[f32; 4]],
) -> (wgpu::Texture, wgpu::TextureView) {
    let fallback = [[0.0_f32; 4]];
    let ramp = if ramp.is_empty() { &fallback[..] } else { ramp };
    let width = ramp.len() as u32;
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("gradient_ramp_texture"),
        size: wgpu::Extent3d {
            width,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D1,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(ramp),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width * 16), // 4 × f32 = 16 bytes per texel
            rows_per_image: None,
        },
        wgpu::Extent3d {
            width,
            height: 1,
            depth_or_array_layers: 1,
        },
    );

    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D1),
        ..Default::default()
    });
    (texture, view)
}

/// Creates a default (transparent) 1D ramp texture (single texel).
pub(crate) fn create_default_ramp_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> (wgpu::Texture, wgpu::TextureView) {
    create_ramp_texture(device, queue, &[[0.0, 0.0, 0.0, 0.0]])
}
