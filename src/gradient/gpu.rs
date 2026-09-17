use super::types::{GradientData, GradientGeometry, GradientUnits, SpreadMode};
use crate::pipeline::BackdropSamplingUniform;
use std::f32::consts::TAU;

/// GPU-side gradient-only parameters packed into a uniform-friendly struct.
/// Matches the WGSL `GradientColorParams` struct in shader.wgsl.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
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
        let spread_mode = match data.spread {
            SpreadMode::Pad => 0u32,
            SpreadMode::Repeat => 1u32,
        };
        let units = match data.units {
            GradientUnits::Local => 0u32,
            GradientUnits::Canvas => 1u32,
        };

        let mut params = Self {
            spread_mode,
            units,
            is_constant: data.is_constant as u32,
            constant_color: data.constant_color,
            period_start: data.period_start,
            period_len: data.period_len,
            ..Self::none()
        };
        match data.geometry {
            GradientGeometry::Linear(line) => {
                params.gradient_type = 1;
                params.linear_start = line.start;
                params.linear_end = line.end;
            }
            GradientGeometry::Radial { center, radius } => {
                params.gradient_type = 2;
                params.radial_center = center;
                params.radial_radius = radius;
            }
            GradientGeometry::Conic {
                center,
                start_angle,
            } => {
                params.gradient_type = 3;
                params.conic_center = center;
                params.conic_start_angle = start_angle;
                // The shader evaluates conic stops in turns; normalization uses radians.
                params.period_start /= TAU;
                params.period_len /= TAU;
            }
        }
        params.ramp_start = params.period_start;
        params.ramp_end = params.period_start + params.period_len;
        params
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

/// GPU-side material parameters bound at group 3 binding 0.
///
/// This uniform layout is shared across regular gradient fills and backdrop-capable pipelines.
/// Solid backdrop draws leave `gradient` in its inert `none()` state and only populate
/// `backdrop_sampling`.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct GpuMaterialParams {
    pub gradient: GpuGradientColorParams,
    pub backdrop_sampling: BackdropSamplingUniform,
}

impl Default for GpuMaterialParams {
    fn default() -> Self {
        Self {
            gradient: GpuGradientColorParams::none(),
            backdrop_sampling: BackdropSamplingUniform::default(),
        }
    }
}

impl GpuMaterialParams {
    pub fn from_gradient_data(data: &GradientData) -> Self {
        Self {
            gradient: GpuGradientColorParams::from_gradient_data(data),
            backdrop_sampling: BackdropSamplingUniform::default(),
        }
    }

    pub fn with_backdrop_sampling(mut self, sampling_uniform: BackdropSamplingUniform) -> Self {
        self.backdrop_sampling = sampling_uniform;
        self
    }

    pub fn for_backdrop_sampling(sampling_uniform: BackdropSamplingUniform) -> Self {
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
        // Use half floats so the ramp stays high precision while remaining filterable.
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    let upload_ramp: Vec<[u16; 4]> = ramp
        .iter()
        .map(|texel| texel.map(|channel| half::f16::from_f32(channel).to_bits()))
        .collect();

    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(&upload_ramp),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width * 8), // 4 × f16 = 8 bytes per texel
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
