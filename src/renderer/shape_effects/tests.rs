use super::{
    compute_shape_effect_raster_rect, shape_effect_quad_transform, ShapeEffectCacheKey,
    ShapeEffectMaskCacheKey,
};
use crate::cache::CachedTessellation;
use crate::effect::ShapeEffectConfig;
use crate::vertex::{CustomVertex, InstanceTransform};
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
fn raster_rect_rounds_outward_and_adds_fringe_guard() {
    let raster_rect = compute_shape_effect_raster_rect(
        [(1.25, 2.75), (10.1, 20.2)],
        ShapeEffectConfig::new().outsets(1.0, 2.0, 3.0, 4.0),
        2.0,
        0.75,
    )
    .unwrap();

    assert_eq!(raster_rect.local_physical_origin, [-1, 0]);
    assert_eq!(raster_rect.texture_size, [29, 50]);
    assert_eq!(raster_rect.local_bounds, [(-0.5, 0.0), (14.0, 25.0)]);
}

#[test]
fn raster_rect_downsample_shrinks_texture_but_not_coverage() {
    let full_resolution = compute_shape_effect_raster_rect(
        [(1.25, 2.75), (10.1, 20.2)],
        ShapeEffectConfig::new().outsets(1.0, 2.0, 3.0, 4.0),
        2.0,
        0.75,
    )
    .unwrap();
    let downsampled = compute_shape_effect_raster_rect(
        [(1.25, 2.75), (10.1, 20.2)],
        ShapeEffectConfig::new()
            .outsets(1.0, 2.0, 3.0, 4.0)
            .downsample(0.5),
        2.0,
        0.75,
    )
    .unwrap();

    assert_eq!(downsampled.texture_size, [15, 25]);
    assert_eq!(
        downsampled.local_physical_origin,
        full_resolution.local_physical_origin
    );
    assert_eq!(downsampled.local_bounds, full_resolution.local_bounds);
}

#[test]
fn raster_rect_downsample_keeps_at_least_one_texel() {
    let raster_rect = compute_shape_effect_raster_rect(
        [(0.0, 0.0), (1.0, 1.0)],
        ShapeEffectConfig::new().downsample(0.1),
        1.0,
        0.75,
    )
    .unwrap();

    assert!(raster_rect.texture_size[0] >= 1);
    assert!(raster_rect.texture_size[1] >= 1);
}

#[test]
fn raster_rect_rejects_out_of_range_downsample() {
    for downsample in [0.0, -0.5, f32::NAN, 1.5] {
        assert!(compute_shape_effect_raster_rect(
            [(0.0, 0.0), (10.0, 10.0)],
            ShapeEffectConfig::new().downsample(downsample),
            1.0,
            0.75,
        )
        .is_none());
    }
}

#[test]
fn raster_rect_rejects_non_finite_inputs() {
    assert!(compute_shape_effect_raster_rect(
        [(0.0, 0.0), (f32::NAN, 10.0)],
        ShapeEffectConfig::default(),
        1.0,
        0.75,
    )
    .is_none());
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
    let mut downsampled_key =
        cache_key(Arc::clone(&shared_tessellation), Arc::from([1u8, 2, 3, 4]));
    downsampled_key.mask_key.downsample_bits = 0.5f32.to_bits();

    assert!(full_resolution_key != downsampled_key);
}

#[test]
fn shape_effect_quad_transform_maps_unit_quad_before_source_transform() {
    let transform = shape_effect_quad_transform(
        [(-3.0, -4.0), (11.0, 15.0)],
        Some(InstanceTransform::translation(5.0, 7.0)),
    );

    assert_eq!(transform.col0, [14.0, 0.0, 0.0, 0.0]);
    assert_eq!(transform.col1, [0.0, 19.0, 0.0, 0.0]);
    assert_eq!(transform.col3, [2.0, 3.0, 0.0, 1.0]);
}
