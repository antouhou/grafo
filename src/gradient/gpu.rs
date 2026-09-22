use super::sampling::bake_gradient_ramp;
use super::types::{
    GradientData, GradientGeometry, GradientRamp, GradientRampCacheKey, GradientUnits, SpreadMode,
};
use crate::renderer::TextureSamplingUniform;
use lru::LruCache;
use std::f32::consts::TAU;
use std::num::NonZeroUsize;
use std::sync::Arc;
use wgpu::util::{BufferInitDescriptor, DeviceExt};

const MAX_GRADIENT_RAMP_CACHE_SIZE: usize = 256;
const MAX_GRADIENT_MATERIAL_CACHE_SIZE: usize = 1024;

/// Gradient parameters laid out for a GPU uniform buffer.
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

    // Constant color for degenerate gradients.
    pub constant_color: [f32; 4],

    pub linear_start: [f32; 2],
    pub linear_end: [f32; 2],

    pub radial_center: [f32; 2],
    pub radial_radius: [f32; 2],

    pub conic_center: [f32; 2],
    pub conic_start_angle: f32,

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
            period_start: data.period_start,
            period_len: data.period_len,
            ..Self::none()
        };
        if let GradientRamp::Constant(color) = &data.ramp {
            params.is_constant = 1;
            params.constant_color = *color;
        }
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
/// Gradient fills and under-fill textures share this uniform layout.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct GpuMaterialParams {
    pub gradient: GpuGradientColorParams,
    pub texture_sampling: TextureSamplingUniform,
}

impl Default for GpuMaterialParams {
    fn default() -> Self {
        Self {
            gradient: GpuGradientColorParams::none(),
            texture_sampling: TextureSamplingUniform::default(),
        }
    }
}

impl GpuMaterialParams {
    pub fn from_gradient_data(data: &GradientData) -> Self {
        Self {
            gradient: GpuGradientColorParams::from_gradient_data(data),
            texture_sampling: TextureSamplingUniform::default(),
        }
    }

    pub fn with_texture_sampling(mut self, sampling_uniform: TextureSamplingUniform) -> Self {
        self.texture_sampling = sampling_uniform;
        self
    }

    pub fn for_texture_sampling(sampling_uniform: TextureSamplingUniform) -> Self {
        Self::default().with_texture_sampling(sampling_uniform)
    }
}

/// Creates a 1D ramp texture from the baked ramp data.
/// Returns `(texture, texture_view)`.
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
            bytes_per_row: Some(width * 8), // Four f16 channels take eight bytes per texel.
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

/// Creates a default 1D ramp texture with one transparent texel.
pub(crate) fn create_default_ramp_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> (wgpu::Texture, wgpu::TextureView) {
    create_ramp_texture(device, queue, &[[0.0, 0.0, 0.0, 0.0]])
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct GpuGradientColorParamsKey {
    gradient_type: u32,
    spread_mode: u32,
    units: u32,
    is_constant: u32,
    constant_color: [u32; 4],
    linear_start: [u32; 2],
    linear_end: [u32; 2],
    radial_center: [u32; 2],
    radial_radius: [u32; 2],
    conic_center: [u32; 2],
    conic_start_angle: u32,
    period_start: u32,
    period_len: u32,
    ramp_start: u32,
    ramp_end: u32,
}

impl GpuGradientColorParamsKey {
    fn from_params(params: GpuGradientColorParams) -> Self {
        Self {
            gradient_type: params.gradient_type,
            spread_mode: params.spread_mode,
            units: params.units,
            is_constant: params.is_constant,
            constant_color: params.constant_color.map(f32::to_bits),
            linear_start: params.linear_start.map(f32::to_bits),
            linear_end: params.linear_end.map(f32::to_bits),
            radial_center: params.radial_center.map(f32::to_bits),
            radial_radius: params.radial_radius.map(f32::to_bits),
            conic_center: params.conic_center.map(f32::to_bits),
            conic_start_angle: params.conic_start_angle.to_bits(),
            period_start: params.period_start.to_bits(),
            period_len: params.period_len.to_bits(),
            ramp_start: params.ramp_start.to_bits(),
            ramp_end: params.ramp_end.to_bits(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct GradientMaterialCacheKey {
    params: GpuGradientColorParamsKey,
    ramp_key: GradientRampCacheKey,
}

struct CachedGradientRampTexture {
    _texture: wgpu::Texture,
    view: Arc<wgpu::TextureView>,
}

#[derive(Debug)]
pub(crate) struct GradientMaterial {
    pub(crate) bind_group: wgpu::BindGroup,
    pub(crate) ramp_view: Arc<wgpu::TextureView>,
}

pub(crate) struct GradientCache {
    ramps: LruCache<GradientRampCacheKey, GradientRamp>,
    ramp_textures: LruCache<GradientRampCacheKey, Arc<CachedGradientRampTexture>>,
    materials: LruCache<GradientMaterialCacheKey, Arc<GradientMaterial>>,
    default_ramp_texture: Option<Arc<CachedGradientRampTexture>>,
}

impl GradientCache {
    pub(crate) fn new() -> Self {
        Self {
            ramps: LruCache::new(
                NonZeroUsize::new(MAX_GRADIENT_RAMP_CACHE_SIZE)
                    .expect("gradient ramp cache size must be greater than 0"),
            ),
            ramp_textures: LruCache::new(
                NonZeroUsize::new(MAX_GRADIENT_RAMP_CACHE_SIZE)
                    .expect("gradient ramp cache size must be greater than 0"),
            ),
            materials: LruCache::new(
                NonZeroUsize::new(MAX_GRADIENT_MATERIAL_CACHE_SIZE)
                    .expect("gradient material cache size must be greater than 0"),
            ),
            default_ramp_texture: None,
        }
    }

    pub(crate) fn clear_materials(&mut self) {
        self.materials.clear();
    }

    fn get_or_create_default_ramp_texture(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Arc<CachedGradientRampTexture> {
        if let Some(default_ramp_texture) = &self.default_ramp_texture {
            return default_ramp_texture.clone();
        }

        let (texture, view) = create_default_ramp_texture(device, queue);
        let default_ramp_texture = Arc::new(CachedGradientRampTexture {
            _texture: texture,
            view: Arc::new(view),
        });
        self.default_ramp_texture = Some(default_ramp_texture.clone());
        default_ramp_texture
    }

    pub(super) fn get_or_create_ramp(&mut self, gradient_data: &mut GradientData) -> GradientRamp {
        let GradientRamp::Pending(ramp_source) = &gradient_data.ramp else {
            return gradient_data.ramp.clone();
        };

        if let Some(ramp) = self.ramps.get(&gradient_data.ramp_cache_key).cloned() {
            gradient_data.ramp = ramp.clone();
            return ramp;
        }

        let baked_ramp = bake_gradient_ramp(ramp_source);

        self.ramps
            .put(gradient_data.ramp_cache_key.clone(), baked_ramp.clone());
        gradient_data.ramp = baked_ramp.clone();
        baked_ramp
    }

    fn get_or_create_ramp_texture(
        &mut self,
        gradient_data: &mut GradientData,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Arc<CachedGradientRampTexture> {
        if let Some(ramp_texture) = self.ramp_textures.get(&gradient_data.ramp_cache_key) {
            return ramp_texture.clone();
        }

        let ramp = self.get_or_create_ramp(gradient_data);
        let (texture, view) = create_ramp_texture(device, queue, ramp.as_slice());
        let ramp_texture = Arc::new(CachedGradientRampTexture {
            _texture: texture,
            view: Arc::new(view),
        });
        self.ramp_textures
            .put(gradient_data.ramp_cache_key.clone(), ramp_texture.clone());
        ramp_texture
    }

    pub(crate) fn get_or_create_material(
        &mut self,
        gradient_data: &mut GradientData,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
        sampler: &wgpu::Sampler,
    ) -> Arc<GradientMaterial> {
        let material_params = GpuMaterialParams::from_gradient_data(gradient_data);
        let cache_key = GradientMaterialCacheKey {
            params: GpuGradientColorParamsKey::from_params(material_params.gradient),
            ramp_key: gradient_data.ramp_cache_key.clone(),
        };

        if let Some(material) = self.materials.get(&cache_key) {
            return material.clone();
        }

        let ramp_texture = if matches!(gradient_data.ramp, GradientRamp::Constant(_)) {
            self.get_or_create_default_ramp_texture(device, queue)
        } else {
            self.get_or_create_ramp_texture(gradient_data, device, queue)
        };

        let params_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Material Params Buffer"),
            contents: bytemuck::cast_slice(&[material_params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Gradient Bind Group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(ramp_texture.view.as_ref()),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });
        let material = Arc::new(GradientMaterial {
            bind_group,
            ramp_view: Arc::clone(&ramp_texture.view),
        });
        self.materials.put(cache_key, material.clone());
        material
    }

    pub(crate) fn print_sizes(&self) {
        println!("Gradient ramps: {}", self.ramps.len());
        println!("Gradient ramp textures: {}", self.ramp_textures.len());
        println!("Gradient materials: {}", self.materials.len());
    }
}
