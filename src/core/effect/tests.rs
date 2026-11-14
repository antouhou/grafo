use super::{ShapeMaskEffectBounds, ShapeMaskEffectConfig};

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
