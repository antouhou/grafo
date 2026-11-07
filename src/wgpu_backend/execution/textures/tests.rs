use super::{ShapeMaskCacheKey, ShapeMaskEffectCacheKey};
use crate::core::cache::{CachedTessellation, FrameCache};
use crate::core::vertex::CustomVertex;
use std::sync::Arc;
use wgpu::TextureFormat;

fn tessellation() -> Arc<CachedTessellation> {
    let mut tessellation = CachedTessellation::empty();
    tessellation.local_bounds = [(0.0, 0.0), (10.0, 10.0)];
    tessellation.texture_mapping_size = [10.0, 10.0];
    Arc::new(tessellation)
}

fn cache_key(
    tessellation: Arc<CachedTessellation>,
    parameters_hash: u64,
) -> ShapeMaskEffectCacheKey {
    ShapeMaskEffectCacheKey {
        mask_key: ShapeMaskCacheKey {
            tessellation: Arc::downgrade(&tessellation),
            local_raster_origin: [-1, -1],
            raster_size: [12, 12],
            scale_factor_bits: 1.0f64.to_bits(),
            downsample_bits: 1.0f32.to_bits(),
            texture_format: TextureFormat::Bgra8UnormSrgb,
        },
        effect_id: 7,
        parameters_hash,
    }
}

#[test]
fn cache_key_uses_tessellation_identity_and_parameter_hash() {
    let shared_tessellation = tessellation();
    let first_key = cache_key(Arc::clone(&shared_tessellation), 1);
    let equal_key = cache_key(Arc::clone(&shared_tessellation), 1);
    let different_params = cache_key(Arc::clone(&shared_tessellation), 2);
    let different_tessellation = cache_key(tessellation(), 1);

    assert!(first_key == equal_key);
    assert!(first_key != different_params);
    assert!(first_key != different_tessellation);
}

#[test]
fn mask_keys_release_geometry_and_reject_republished_buffers() {
    let mut geometry = tessellation();
    Arc::get_mut(&mut geometry)
        .unwrap()
        .vertex_buffers
        .vertices
        .push(CustomVertex {
            position: [1.0, 2.0],
            tex_coords: [0.0; 2],
        });
    let vertex_storage = geometry.vertex_buffers.vertices.as_ptr();
    let stale_key = cache_key(Arc::clone(&geometry), 1);
    let mut cache = FrameCache::new();
    cache.insert(stale_key.clone(), "prepared");

    let released = Arc::try_unwrap(geometry).expect("mask cache keys must not retain geometry");
    let republished = Arc::new(released);
    assert_eq!(republished.vertex_buffers.vertices.as_ptr(), vertex_storage);
    let replacement_key = cache_key(republished, 1);
    assert!(stale_key != replacement_key);
    assert!(cache.get(&replacement_key).is_none());
}

#[test]
fn cache_key_differs_when_only_downsample_changes() {
    let shared_tessellation = tessellation();
    let full_resolution_key = cache_key(Arc::clone(&shared_tessellation), 1);
    let mut downsampled_key = full_resolution_key.clone();
    downsampled_key.mask_key.downsample_bits = 0.5f32.to_bits();

    assert!(full_resolution_key != downsampled_key);
}
