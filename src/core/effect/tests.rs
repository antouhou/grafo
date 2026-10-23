use super::{ShapeMaskEffectBounds, ShapeMaskEffectConfig};
use crate::core::{effect, PhysicalRect, Size, UnsignedPhysicalPoint, UnsignedPhysicalRect};
use lyon::geom::Point;

#[test]
fn capture_region_offsets_visible_copy_into_transparent_texture() {
    let bounds = PhysicalRect::new(Point::new(-10, 5), Point::new(30, 25));
    let region = effect::resolve_capture_region_to_viewport(bounds, bounds, Size::new(100, 100));

    assert_eq!(region.bounds.min, Point::new(-10, 5));
    assert_eq!(
        region
            .copy_from(PhysicalRect::from_size((100, 100).into()))
            .unwrap()
            .1,
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
    let right_bounds = PhysicalRect::new(Point::new(110, 5), Point::new(130, 25));
    let bottom_bounds = PhysicalRect::new(Point::new(5, 110), Point::new(25, 130));
    let offscreen_right =
        effect::resolve_capture_region_to_viewport(right_bounds, right_bounds, Size::new(100, 100));
    let offscreen_bottom = effect::resolve_capture_region_to_viewport(
        bottom_bounds,
        bottom_bounds,
        Size::new(100, 100),
    );

    assert_eq!(offscreen_right.source_rect, None);
    assert_eq!(offscreen_bottom.source_rect, None);
}

#[test]
fn bounded_captures_translate_source_pixels_and_keep_missing_input_transparent() {
    let bounds = PhysicalRect::new((-10, 5).into(), (50, 45).into());
    let capture = effect::resolve_capture_region_to_viewport(bounds, bounds, Size::new(100, 100));
    let source = PhysicalRect::new((12, 10).into(), (36, 30).into());
    assert_eq!(
        capture.copy_from(source),
        Some((
            UnsignedPhysicalRect::from_size(Size::new(24, 20)),
            (22, 5).into()
        ))
    );
    assert_eq!(
        capture.copy_from(PhysicalRect::new((60, 10).into(), (80, 30).into())),
        None
    );

    let padded_source = PhysicalRect::new((-20, -10).into(), (60, 60).into());
    assert_eq!(
        capture.copy_from(padded_source),
        Some((
            UnsignedPhysicalRect::new((20, 15).into(), (70, 55).into()),
            (10, 0).into(),
        ))
    );
}

#[test]
fn target_scissors_clip_input_and_preserve_transparent_outsets() {
    let input = PhysicalRect::new((-5, 20).into(), (40, 60).into());
    let output = PhysicalRect::new((-15, 10).into(), (50, 70).into());
    let region = effect::resolve_capture_region_to_viewport(input, output, Size::new(100, 100));
    assert_eq!(
        region.scissor(UnsignedPhysicalRect::from_size(Size::new(100, 100))),
        UnsignedPhysicalRect::new((15, 10).into(), (55, 50).into()),
    );
    assert_eq!(
        region.scissor(UnsignedPhysicalRect::new((10, 30).into(), (80, 80).into())),
        UnsignedPhysicalRect::new((25, 20).into(), (55, 50).into()),
    );
    assert_eq!(
        region.scissor(UnsignedPhysicalRect::new((60, 30).into(), (80, 80).into())),
        UnsignedPhysicalRect::zero(),
    );
}

#[test]
fn shape_mask_effect_bounds_round_outward_and_add_filtering_border() {
    let bounds = ShapeMaskEffectBounds::new(
        [(1.25, 2.75), (10.1, 20.2)],
        ShapeMaskEffectConfig::new().outsets(1.0, 2.0, 3.0, 4.0),
        None,
        2.0,
    )
    .unwrap();

    assert_eq!(bounds.local_physical_origin, [-1, 0]);
    assert_eq!(bounds.texture_size, [29, 50]);
    assert_eq!(bounds.local_bounds, [(-0.5, 0.0), (14.0, 25.0)]);
}

#[test]
fn shape_mask_effect_bounds_downsample_shrinks_texture_but_not_coverage() {
    let full_resolution = ShapeMaskEffectBounds::new(
        [(1.25, 2.75), (10.1, 20.2)],
        ShapeMaskEffectConfig::new().outsets(1.0, 2.0, 3.0, 4.0),
        None,
        2.0,
    )
    .unwrap();
    let downsampled = ShapeMaskEffectBounds::new(
        [(1.25, 2.75), (10.1, 20.2)],
        ShapeMaskEffectConfig::new()
            .outsets(1.0, 2.0, 3.0, 4.0)
            .downsample(0.5),
        None,
        2.0,
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
fn shape_mask_effect_bounds_downsample_keeps_at_least_one_texel() {
    let bounds = ShapeMaskEffectBounds::new(
        [(0.0, 0.0), (1.0, 1.0)],
        ShapeMaskEffectConfig::new().downsample(0.1),
        None,
        1.0,
    )
    .unwrap();

    assert!(bounds.texture_size[0] >= 1);
    assert!(bounds.texture_size[1] >= 1);
}

#[test]
fn shape_mask_effect_bounds_reject_out_of_range_downsample() {
    for downsample in [0.0, -0.5, f32::NAN, 1.5] {
        assert!(ShapeMaskEffectBounds::new(
            [(0.0, 0.0), (10.0, 10.0)],
            ShapeMaskEffectConfig::new().downsample(downsample),
            None,
            1.0,
        )
        .is_none());
    }
}

#[test]
fn shape_mask_effect_bounds_reject_non_finite_inputs() {
    assert!(ShapeMaskEffectBounds::new(
        [(0.0, 0.0), (f32::NAN, 10.0)],
        ShapeMaskEffectConfig::default(),
        None,
        1.0,
    )
    .is_none());
}
