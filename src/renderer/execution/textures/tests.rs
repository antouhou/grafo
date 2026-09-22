use super::{ShapeEffectCacheKey, ShapeEffectMaskCacheKey};
use crate::cache::CachedTessellation;
use crate::vertex::CustomVertex;
use lyon::tessellation::VertexBuffers;
use std::sync::Arc;
use wgpu::TextureFormat;

fn tessellation() -> Arc<CachedTessellation> {
    Arc::new(CachedTessellation {
        vertex_buffers: Arc::new(VertexBuffers::<CustomVertex, u16>::new()),
        local_bounds: [(0.0, 0.0), (10.0, 10.0)],
        texture_mapping_size: [10.0, 10.0],
    })
}

fn cache_key(tessellation: Arc<CachedTessellation>, params: Arc<[u8]>) -> ShapeEffectCacheKey {
    ShapeEffectCacheKey {
        mask_key: ShapeEffectMaskCacheKey {
            tessellation,
            local_raster_origin: [-1, -1],
            raster_size: [12, 12],
            scale_factor_bits: 1.0f64.to_bits(),
            fringe_width_bits: 0.75f32.to_bits(),
            downsample_bits: 1.0f32.to_bits(),
            texture_format: TextureFormat::Bgra8UnormSrgb,
        },
        effect_id: 7,
        params,
    }
}

#[test]
fn cache_key_uses_tessellation_identity_and_exact_parameter_bytes() {
    let shared_tessellation = tessellation();
    let first_key = cache_key(Arc::clone(&shared_tessellation), Arc::from([1u8, 2, 3, 4]));
    let equal_key = cache_key(Arc::clone(&shared_tessellation), Arc::from([1u8, 2, 3, 4]));
    let different_params = cache_key(Arc::clone(&shared_tessellation), Arc::from([1u8, 2, 3, 5]));
    let different_tessellation = cache_key(tessellation(), Arc::from([1u8, 2, 3, 4]));

    assert!(first_key == equal_key);
    assert!(first_key != different_params);
    assert!(first_key != different_tessellation);
}

#[test]
fn cache_key_differs_when_only_downsample_changes() {
    let shared_tessellation = tessellation();
    let full_resolution_key =
        cache_key(Arc::clone(&shared_tessellation), Arc::from([1u8, 2, 3, 4]));
    let mut downsampled_key = full_resolution_key.clone();
    downsampled_key.mask_key.downsample_bits = 0.5f32.to_bits();

    assert!(full_resolution_key != downsampled_key);
}
