use super::{compute_shape_effect_raster_rect, shape_effect_quad_transform};
use crate::effect::ShapeEffectConfig;
use crate::vertex::InstanceTransform;

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
fn shape_effect_quad_transform_maps_unit_quad_before_source_transform() {
    let transform = shape_effect_quad_transform(
        [(-3.0, -4.0), (11.0, 15.0)],
        Some(InstanceTransform::translation(5.0, 7.0)),
    );

    assert_eq!(transform.col0, [14.0, 0.0, 0.0, 0.0]);
    assert_eq!(transform.col1, [0.0, 19.0, 0.0, 0.0]);
    assert_eq!(transform.col3, [2.0, 3.0, 0.0, 1.0]);
}
