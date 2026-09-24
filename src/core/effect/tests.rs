use super::{
    compute_shape_effect_raster_rect, resolve_capture_region_to_viewport, ShapeEffectConfig,
};
use crate::core::{PhysicalRect, Size, UnsignedPhysicalPoint, UnsignedPhysicalRect};
use lyon::geom::Point;

#[test]
fn capture_region_offsets_visible_copy_into_transparent_texture() {
    let region = resolve_capture_region_to_viewport(
        PhysicalRect::new(Point::new(-10, 5), Point::new(30, 25)),
        Size::new(100, 100),
    );

    assert_eq!(region.bounds.min, Point::new(-10, 5));
    assert_eq!(
        region.copy_destination_origin,
        UnsignedPhysicalPoint::new(10, 0)
    );
    assert_eq!(region.bounds.size(), Size::new(40, 20).to_i32());
    assert_eq!(
        region.source_rect,
        Some(UnsignedPhysicalRect::new(
            Point::new(0, 5),
            Point::new(30, 25)
        ))
    );
}

#[test]
fn capture_region_skips_copy_when_fully_offscreen() {
    let offscreen_right = resolve_capture_region_to_viewport(
        PhysicalRect::new(Point::new(110, 5), Point::new(130, 25)),
        Size::new(100, 100),
    );
    let offscreen_bottom = resolve_capture_region_to_viewport(
        PhysicalRect::new(Point::new(5, 110), Point::new(25, 130)),
        Size::new(100, 100),
    );

    assert_eq!(offscreen_right.source_rect, None);
    assert_eq!(offscreen_bottom.source_rect, None);
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
