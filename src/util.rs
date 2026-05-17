use crate::cache::{Cache, CachedTessellation};
use crate::gradient::gpu::{
    create_default_ramp_texture, create_ramp_texture, GpuGradientColorParams, GpuMaterialParams,
};
use crate::gradient::sampling::bake_gradient_ramp;
use crate::gradient::types::{GradientData, GradientRamp, GradientRampCacheKey};
use crate::shape::AaFringeScratch;
use crate::vertex::CustomVertex;
use lru::LruCache;
use lyon::tessellation::VertexBuffers;
use std::collections::HashSet;
use std::hash::{BuildHasher, Hash};
use std::num::NonZeroUsize;
use std::sync::Arc;

// const MAX_LYON_VERTEX_BUFFER_POOL_SIZE: usize = 256;
const MAX_GRADIENT_RAMP_CACHE_SIZE: usize = 256;
const MAX_GRADIENT_BIND_GROUP_CACHE_SIZE: usize = 1024;

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct MemoryUsage {
    pub(crate) cpu_bytes: u64,
    pub(crate) gpu_buffer_bytes: u64,
    pub(crate) gpu_texture_bytes: u64,
}

impl MemoryUsage {
    pub(crate) fn total_bytes(self) -> u64 {
        self.cpu_bytes
            .saturating_add(self.gpu_buffer_bytes)
            .saturating_add(self.gpu_texture_bytes)
    }

    pub(crate) fn add(&mut self, other: Self) {
        self.cpu_bytes = self.cpu_bytes.saturating_add(other.cpu_bytes);
        self.gpu_buffer_bytes = self.gpu_buffer_bytes.saturating_add(other.gpu_buffer_bytes);
        self.gpu_texture_bytes = self
            .gpu_texture_bytes
            .saturating_add(other.gpu_texture_bytes);
    }
}

pub(crate) fn vector_capacity_bytes<T>(values: &Vec<T>) -> u64 {
    (values.capacity() as u64).saturating_mul(std::mem::size_of::<T>() as u64)
}

pub(crate) fn hash_map_capacity_bytes<K, V, S>(values: &std::collections::HashMap<K, V, S>) -> u64
where
    S: BuildHasher,
{
    (values.capacity() as u64).saturating_mul(
        std::mem::size_of::<K>()
            .saturating_add(std::mem::size_of::<V>())
            .saturating_add(std::mem::size_of::<usize>())
            .saturating_add(1) as u64,
    )
}

pub(crate) fn lru_cache_capacity_bytes<K, V, S>(values: &lru::LruCache<K, V, S>) -> u64
where
    K: Eq + Hash,
    S: BuildHasher,
{
    let hash_map_slots = (values.cap().get() as u64).saturating_mul(
        std::mem::size_of::<usize>()
            .saturating_add(std::mem::size_of::<*const ()>())
            .saturating_add(1) as u64,
    );
    let linked_entries = (values.len() as u64).saturating_mul(
        std::mem::size_of::<K>()
            .saturating_add(std::mem::size_of::<V>())
            .saturating_add(std::mem::size_of::<*const ()>() * 2) as u64,
    );

    hash_map_slots.saturating_add(linked_entries)
}

pub(crate) fn texture_memory_size(
    size: wgpu::Extent3d,
    format: wgpu::TextureFormat,
    sample_count: u32,
    mip_level_count: u32,
) -> u64 {
    let bytes_per_block = texture_format_bytes_per_block(format);
    let (block_width, block_height) = format.block_dimensions();
    let mut total = 0_u64;
    let mut width = size.width.max(1);
    let mut height = size.height.max(1);
    let mut depth_or_layers = size.depth_or_array_layers.max(1);

    for _ in 0..mip_level_count.max(1) {
        let blocks_wide = width.div_ceil(block_width.max(1)) as u64;
        let blocks_high = height.div_ceil(block_height.max(1)) as u64;
        total = total.saturating_add(
            blocks_wide
                .saturating_mul(blocks_high)
                .saturating_mul(depth_or_layers as u64)
                .saturating_mul(bytes_per_block as u64)
                .saturating_mul(sample_count.max(1) as u64),
        );

        width = (width / 2).max(1);
        height = (height / 2).max(1);
        depth_or_layers = (depth_or_layers / 2).max(1);
    }

    total
}

fn texture_format_bytes_per_block(format: wgpu::TextureFormat) -> u32 {
    format.block_copy_size(None).unwrap_or(match format {
        // WebGPU leaves these as implementation-defined. Current desktop backends store
        // Depth24Plus as a 32-bit depth allocation and Depth24PlusStencil8 as D24S8.
        wgpu::TextureFormat::Depth24Plus | wgpu::TextureFormat::Depth24PlusStencil8 => 4,
        wgpu::TextureFormat::Depth32FloatStencil8 => 8,
        _ => 4,
    })
}

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
    params: GpuGradientColorParamsKey,
    ramp_key: GradientRampCacheKey,
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

struct CachedGradientRampTexture {
    _texture: wgpu::Texture,
    view: Arc<wgpu::TextureView>,
}

impl CachedGradientRampTexture {
    fn memory_usage(&self) -> MemoryUsage {
        MemoryUsage {
            cpu_bytes: 0,
            gpu_buffer_bytes: 0,
            gpu_texture_bytes: texture_memory_size(
                self._texture.size(),
                wgpu::TextureFormat::Rgba32Float,
                1,
                1,
            ),
        }
    }
}

struct CachedGradientBindGroup {
    bind_group: Arc<wgpu::BindGroup>,
    params_buffer: wgpu::Buffer,
}

pub(crate) struct GradientCache {
    ramps: LruCache<GradientRampCacheKey, GradientRamp>,
    ramp_textures: LruCache<GradientRampCacheKey, Arc<CachedGradientRampTexture>>,
    bind_groups: LruCache<GradientBindGroupCacheKey, Arc<CachedGradientBindGroup>>,
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
        let material_params = GpuMaterialParams::from_gradient_data(gradient_data);
        let cache_key = GradientBindGroupCacheKey {
            layout_epoch,
            params: GpuGradientColorParamsKey::from_params(material_params.gradient),
            ramp_key: gradient_data.ramp_cache_key.clone(),
        };

        if let Some(bind_group) = self.bind_groups.get(&cache_key) {
            return bind_group.bind_group.clone();
        }

        let ramp_texture = if gradient_data.is_constant {
            self.get_or_create_default_ramp_texture(device, queue)
        } else {
            self.get_or_create_ramp_texture(gradient_data, device, queue)
        };

        let params_buffer = crate::pipeline::create_buffer_init(
            device,
            Some("Material Params Buffer"),
            bytemuck::cast_slice(&[material_params]),
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
        let cached_bind_group = Arc::new(CachedGradientBindGroup {
            bind_group: bind_group.clone(),
            params_buffer,
        });

        self.bind_groups.put(cache_key, cached_bind_group);
        bind_group
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn create_backdrop_gradient_bind_group(
        &mut self,
        gradient_data: &mut GradientData,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
        material_params_buffer: &wgpu::Buffer,
        gradient_sampler: &wgpu::Sampler,
        backdrop_view: &wgpu::TextureView,
        backdrop_sampler: &wgpu::Sampler,
    ) -> wgpu::BindGroup {
        let ramp_texture = if gradient_data.is_constant {
            self.get_or_create_default_ramp_texture(device, queue)
        } else {
            self.get_or_create_ramp_texture(gradient_data, device, queue)
        };

        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Backdrop Gradient Material Bind Group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: material_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(ramp_texture.view.as_ref()),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(gradient_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(backdrop_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(backdrop_sampler),
                },
            ],
        })
    }

    fn trim(&mut self) {}

    pub(crate) fn memory_usage(&self) -> MemoryUsage {
        let mut usage = MemoryUsage {
            cpu_bytes: lru_cache_capacity_bytes(&self.ramps)
                .saturating_add(lru_cache_capacity_bytes(&self.ramp_textures))
                .saturating_add(lru_cache_capacity_bytes(&self.bind_groups)),
            gpu_buffer_bytes: 0,
            gpu_texture_bytes: 0,
        };

        for (_, ramp_texture) in self.ramp_textures.iter() {
            usage.add(ramp_texture.memory_usage());
        }
        if let Some(default_ramp_texture) = &self.default_ramp_texture {
            usage.add(default_ramp_texture.memory_usage());
        }
        for (_, cached_bind_group) in self.bind_groups.iter() {
            usage.gpu_buffer_bytes = usage
                .gpu_buffer_bytes
                .saturating_add(cached_bind_group.params_buffer.size());
        }

        usage
    }

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

    fn memory_usage(&self) -> MemoryUsage {
        let mut usage = MemoryUsage {
            cpu_bytes: vector_capacity_bytes(&self.vertex_buffers),
            gpu_buffer_bytes: 0,
            gpu_texture_bytes: 0,
        };

        for vertex_buffers in &self.vertex_buffers {
            usage.cpu_bytes = usage
                .cpu_bytes
                .saturating_add(vector_capacity_bytes(&vertex_buffers.vertices))
                .saturating_add(vector_capacity_bytes(&vertex_buffers.indices));
        }

        usage
    }

    // pub fn return_vertex_buffers(&mut self, mut vertex_buffers: VertexBuffers<CustomVertex, u16>) {
    //     vertex_buffers.vertices.clear();
    //     vertex_buffers.indices.clear();
    //     if self.vertex_buffers.len() < MAX_LYON_VERTEX_BUFFER_POOL_SIZE {
    //         self.vertex_buffers.push(vertex_buffers);
    //     }
    // }
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

    pub(crate) fn memory_usage(
        &self,
        visited_tessellations: &mut HashSet<*const CachedTessellation>,
    ) -> MemoryUsage {
        let mut usage = self.lyon_vertex_buffers_pool.memory_usage();
        usage.add(self.tessellation_cache.memory_usage(visited_tessellations));
        usage.add(self.aa_fringe_scratch.memory_usage());
        usage.add(self.gradient_cache.memory_usage());
        usage
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
    use super::{GradientCache, GradientRamp};
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
