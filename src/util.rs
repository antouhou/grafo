use crate::cache::Cache;
use crate::gradient::gpu::{create_default_ramp_texture, create_ramp_texture, GpuGradientParams};
use crate::gradient::sampling::bake_gradient_ramp;
use crate::gradient::types::{GradientData, GradientRamp, GradientRampCacheKey};
use crate::shape::AaFringeScratch;
use crate::vertex::CustomVertex;
use lru::LruCache;
use lyon::tessellation::VertexBuffers;
use std::num::NonZeroUsize;
use std::sync::Arc;

const MAX_LYON_VERTEX_BUFFER_POOL_SIZE: usize = 256;
const MAX_GRADIENT_RAMP_CACHE_SIZE: usize = 256;
const MAX_GRADIENT_BIND_GROUP_CACHE_SIZE: usize = 1024;

pub fn normalize_rgba_color(color: &[u8; 4]) -> [f32; 4] {
    [
        srgb_u8_to_linear(color[0]),
        srgb_u8_to_linear(color[1]),
        srgb_u8_to_linear(color[2]),
        color[3] as f32 / 255.0, // alpha is linear, not gamma-encoded
    ]
}

/// Converts a single sRGB u8 channel value (0–255) to linear f32 (0.0–1.0).
///
/// This mirrors the GPU-side `to_linear` function but is done on the CPU so the
/// fragment shader can skip the expensive per-fragment `pow()` call.
fn srgb_u8_to_linear(value: u8) -> f32 {
    let normalized = value as f32 / 255.0;
    if normalized <= 0.04045 {
        normalized / 12.92
    } else {
        ((normalized + 0.055) / 1.055).powf(2.4)
    }
}

pub struct LyonVertexBuffersPool {
    vertex_buffers: Vec<VertexBuffers<CustomVertex, u16>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct GradientBindGroupCacheKey {
    layout_epoch: u64,
    params: GpuGradientParamsKey,
    ramp_key: GradientRampCacheKey,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct GpuGradientParamsKey {
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

impl GpuGradientParamsKey {
    fn from_params(params: GpuGradientParams) -> Self {
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

struct CachedGradientRampTexture {
    _texture: wgpu::Texture,
    view: Arc<wgpu::TextureView>,
}

pub(crate) struct GradientCache {
    ramps: LruCache<GradientRampCacheKey, GradientRamp>,
    ramp_textures: LruCache<GradientRampCacheKey, Arc<CachedGradientRampTexture>>,
    bind_groups: LruCache<GradientBindGroupCacheKey, Arc<wgpu::BindGroup>>,
    default_ramp_texture: Option<Arc<CachedGradientRampTexture>>,
}

impl GradientCache {
    fn new() -> Self {
        Self {
            ramps: LruCache::new(
                NonZeroUsize::new(MAX_GRADIENT_RAMP_CACHE_SIZE)
                    .expect("gradient ramp cache size must be greater than 0"),
            ),
            ramp_textures: LruCache::new(
                NonZeroUsize::new(MAX_GRADIENT_RAMP_CACHE_SIZE)
                    .expect("gradient ramp cache size must be greater than 0"),
            ),
            bind_groups: LruCache::new(
                NonZeroUsize::new(MAX_GRADIENT_BIND_GROUP_CACHE_SIZE)
                    .expect("gradient bind group cache size must be greater than 0"),
            ),
            default_ramp_texture: None,
        }
    }

    pub(crate) fn clear_bind_groups(&mut self) {
        self.bind_groups.clear();
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

    fn get_or_create_ramp(&mut self, gradient_data: &mut GradientData) -> GradientRamp {
        match &gradient_data.ramp {
            GradientRamp::Constant(_) | GradientRamp::Sampled(_) => {
                return gradient_data.ramp.clone();
            }
            GradientRamp::Pending(_) => {}
        }

        if let Some(ramp) = self.ramps.get(&gradient_data.ramp_cache_key).cloned() {
            gradient_data.ramp = ramp.clone();
            return ramp;
        }

        let baked_ramp = match &gradient_data.ramp {
            GradientRamp::Pending(ramp_source) => bake_gradient_ramp(ramp_source),
            GradientRamp::Constant(_) | GradientRamp::Sampled(_) => unreachable!(),
        };

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

    pub(crate) fn get_or_create_bind_group(
        &mut self,
        gradient_data: &mut GradientData,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
        sampler: &wgpu::Sampler,
        layout_epoch: u64,
    ) -> Arc<wgpu::BindGroup> {
        let params = GpuGradientParams::from_gradient_data(gradient_data);
        let cache_key = GradientBindGroupCacheKey {
            layout_epoch,
            params: GpuGradientParamsKey::from_params(params),
            ramp_key: gradient_data.ramp_cache_key.clone(),
        };

        if let Some(bind_group) = self.bind_groups.get(&cache_key) {
            return bind_group.clone();
        }

        let ramp_texture = if gradient_data.is_constant {
            self.get_or_create_default_ramp_texture(device, queue)
        } else {
            self.get_or_create_ramp_texture(gradient_data, device, queue)
        };

        let params_buffer = crate::pipeline::create_buffer_init(
            device,
            Some("Gradient Params Buffer"),
            bytemuck::cast_slice(&[params]),
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );

        let bind_group = Arc::new(device.create_bind_group(&wgpu::BindGroupDescriptor {
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
        }));

        self.bind_groups.put(cache_key, bind_group.clone());
        bind_group
    }

    fn trim(&mut self) {}

    fn print_sizes(&self) {
        println!("Gradient ramps: {}", self.ramps.len());
        println!("Gradient ramp textures: {}", self.ramp_textures.len());
        println!("Gradient bind groups: {}", self.bind_groups.len());
    }
}

impl LyonVertexBuffersPool {
    pub fn new() -> Self {
        Self {
            vertex_buffers: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.vertex_buffers.len()
    }

    pub fn get_vertex_buffers(&mut self) -> VertexBuffers<CustomVertex, u16> {
        if let Some(mut vertex_buffers) = self.vertex_buffers.pop() {
            vertex_buffers.vertices.clear();
            vertex_buffers.indices.clear();
            vertex_buffers
        } else {
            VertexBuffers::new()
        }
    }

    pub fn return_vertex_buffers(&mut self, mut vertex_buffers: VertexBuffers<CustomVertex, u16>) {
        vertex_buffers.vertices.clear();
        vertex_buffers.indices.clear();
        if self.vertex_buffers.len() < MAX_LYON_VERTEX_BUFFER_POOL_SIZE {
            self.vertex_buffers.push(vertex_buffers);
        }
    }
}

pub(crate) struct PoolManager {
    pub lyon_vertex_buffers_pool: LyonVertexBuffersPool,
    pub tessellation_cache: Cache,
    pub aa_fringe_scratch: AaFringeScratch,
    pub gradient_cache: GradientCache,
}

impl PoolManager {
    pub(crate) fn new(tesselation_cache_size: NonZeroUsize) -> Self {
        Self {
            lyon_vertex_buffers_pool: LyonVertexBuffersPool::new(),
            tessellation_cache: Cache::new(tesselation_cache_size),
            aa_fringe_scratch: AaFringeScratch::new(),
            gradient_cache: GradientCache::new(),
        }
    }

    pub(crate) fn trim(&mut self) {
        self.aa_fringe_scratch.trim();
        self.gradient_cache.trim();
    }

    pub fn print_sizes(&self) {
        println!("Pool sizes:");
        println!("Vertex buffers: {}", self.lyon_vertex_buffers_pool.len());
        println!("Index buffers: {}", self.tessellation_cache.len());
        self.gradient_cache.print_sizes();
    }
}

#[inline(always)]
pub fn to_logical(physical_size: (u32, u32), scale_factor: f64) -> (f32, f32) {
    let (physical_width, physical_height) = physical_size;
    let logical_width = physical_width as f64 / scale_factor;
    let logical_height = physical_height as f64 / scale_factor;
    (logical_width as f32, logical_height as f32)
}

#[cfg(test)]
mod tests {
    use super::{GradientCache, GradientRamp, LyonVertexBuffersPool};
    use crate::gradient::types::{
        ColorInterpolation, Gradient, GradientColor, GradientCommonDesc, GradientStop,
        GradientStopOffset, GradientStopPositions, GradientUnits, LinearGradientDesc,
        LinearGradientLine, SpreadMode,
    };
    use std::sync::Arc;

    #[test]
    fn gradient_cache_reuses_sampled_ramps_without_globals() {
        let common = GradientCommonDesc {
            units: GradientUnits::Local,
            spread: SpreadMode::Pad,
            interpolation: ColorInterpolation::SrgbLinear,
            stops: vec![
                GradientStop {
                    positions: GradientStopPositions::Single(GradientStopOffset::LinearRadial(0.0)),
                    color: GradientColor::Srgb {
                        red: 1.0,
                        green: 0.0,
                        blue: 0.0,
                        alpha: 1.0,
                    },
                    hint_to_next_segment: None,
                },
                GradientStop {
                    positions: GradientStopPositions::Single(GradientStopOffset::LinearRadial(1.0)),
                    color: GradientColor::Srgb {
                        red: 0.0,
                        green: 0.0,
                        blue: 1.0,
                        alpha: 1.0,
                    },
                    hint_to_next_segment: None,
                },
            ]
            .into(),
        };

        let mut first = Gradient::linear(LinearGradientDesc {
            common: common.clone(),
            line: LinearGradientLine {
                start: [0.0, 0.0],
                end: [10.0, 0.0],
            },
        })
        .unwrap();
        let mut second = Gradient::linear(LinearGradientDesc {
            common,
            line: LinearGradientLine {
                start: [0.0, 0.0],
                end: [10.0, 0.0],
            },
        })
        .unwrap();

        let mut gradient_cache = GradientCache::new();
        let first_ramp = gradient_cache.get_or_create_ramp(&mut first.data);
        let second_ramp = gradient_cache.get_or_create_ramp(&mut second.data);

        let GradientRamp::Sampled(first_ramp) = first_ramp else {
            panic!("expected sampled ramp");
        };
        let GradientRamp::Sampled(second_ramp) = second_ramp else {
            panic!("expected sampled ramp");
        };

        assert!(Arc::ptr_eq(&first_ramp, &second_ramp));
    }
}
