use super::{encode_ramp_texel_f16, GpuGradientColorParams, GradientCache};
use crate::core::gradient::types::{
    ColorInterpolation, ConicGradientDesc, Gradient, GradientRamp, GradientStop,
    GradientStopOffset, GradientUnits, HueInterpolationMethod, LinearGradientDesc,
    LinearGradientLine, RadialGradientDesc, RadialGradientSize, SpreadMode,
};
use crate::core::Color;
use half::f16;
use std::f32::consts::{FRAC_PI_2, PI, TAU};
use std::sync::Arc;

#[test]
fn conic_periods_resolve_in_core_before_gpu_parameter_packing() {
    for (stops, expected) in [
        (
            vec![
                GradientStop::at_position(GradientStopOffset::conic_radians(0.0), Color::BLACK),
                GradientStop::at_position(
                    GradientStopOffset::conic_radians(f32::from_bits(1)),
                    Color::WHITE,
                ),
            ],
            [0.21404114, 0.21404114, 0.21404114, 1.0],
        ),
        (
            vec![
                GradientStop::at_position(GradientStopOffset::conic_radians(0.0), Color::BLACK),
                GradientStop::at_position(
                    GradientStopOffset::conic_radians(f32::from_bits(1)),
                    Color::BLACK,
                ),
                GradientStop::at_position(
                    GradientStopOffset::conic_radians(f32::from_bits(3)),
                    Color::WHITE,
                ),
            ],
            [0.05087609, 0.05087609, 0.05087609, 1.0],
        ),
        (
            vec![
                GradientStop::at_position(GradientStopOffset::conic_radians(0.0), Color::BLACK),
                GradientStop::at_position(GradientStopOffset::conic_radians(1e-40), Color::WHITE),
            ],
            [0.21404114, 0.21404114, 0.21404114, 1.0],
        ),
        (
            vec![
                GradientStop::at_position(GradientStopOffset::conic_radians(0.0), Color::BLACK),
                GradientStop::at_position(
                    GradientStopOffset::conic_radians(f32::from_bits(1 << 17)),
                    Color::BLACK,
                ),
                GradientStop::at_position(
                    GradientStopOffset::conic_radians(f32::from_bits(1 << 19)),
                    Color::WHITE,
                ),
            ],
            [0.116016135, 0.116016135, 0.116016135, 1.0],
        ),
        (
            vec![
                GradientStop::at_position(GradientStopOffset::conic_radians(0.0), Color::BLACK),
                GradientStop::at_position(GradientStopOffset::conic_radians(0.0), Color::BLACK),
                GradientStop::at_position(
                    GradientStopOffset::conic_radians(f32::from_bits(1)),
                    Color::WHITE,
                ),
            ],
            [0.05087609, 0.05087609, 0.05087609, 1.0],
        ),
    ] {
        let gradient = Gradient::conic(
            ConicGradientDesc::new([0.0, 0.0], 0.0, stops).with_spread(SpreadMode::Repeat),
        )
        .unwrap();
        let GradientRamp::Constant(color) = &gradient.data().ramp else {
            panic!("core must resolve degenerate conic periods before sampling");
        };
        assert_eq!(*color, expected);
        let params = GpuGradientColorParams::from_gradient_data(gradient.data());
        assert_eq!(params.period_len, gradient.data().period_len);
        assert_eq!(params.is_constant, 1);
        assert_eq!(params.constant_color, expected);
    }
}

#[test]
fn repeating_endpoint_hints_use_interval_limits_and_distinct_cache_entries() {
    let red = Color::rgb(255, 0, 0);
    let blue = Color::rgb(0, 0, 255);
    let pink = Color::rgb_f32(1.5, 0.5, 0.5);
    for (interpolation, start, end, hint, expected) in [
        (
            ColorInterpolation::SrgbLinear,
            red,
            blue,
            Some(0.0),
            blue.normalize(),
        ),
        (
            ColorInterpolation::SrgbLinear,
            red,
            blue,
            Some(1.0),
            red.normalize(),
        ),
        (
            ColorInterpolation::Hsl {
                hue: HueInterpolationMethod::Shorter,
            },
            red,
            pink,
            Some(0.0),
            [1.0; 4],
        ),
        (
            ColorInterpolation::Hsl {
                hue: HueInterpolationMethod::Shorter,
            },
            pink,
            blue,
            Some(1.0),
            [1.0; 4],
        ),
        (
            ColorInterpolation::Hsl {
                hue: HueInterpolationMethod::Shorter,
            },
            pink,
            pink,
            None,
            [1.0; 4],
        ),
    ] {
        let make_gradient = |spread| {
            let mut first =
                GradientStop::at_position(GradientStopOffset::linear_radial(0.0), start);
            first.hint_to_next_segment = hint.map(GradientStopOffset::linear_radial);
            Gradient::linear(
                LinearGradientDesc::new(
                    LinearGradientLine {
                        start: [0.0, 0.0],
                        end: [100.0, 0.0],
                    },
                    [
                        first,
                        GradientStop::at_position(GradientStopOffset::linear_radial(1.0), end),
                    ],
                )
                .with_spread(spread)
                .with_interpolation(interpolation),
            )
            .unwrap()
        };
        let mut cache = GradientCache::new();
        let pad = cache.get_or_create_ramp(make_gradient(SpreadMode::Pad).data());
        let repeat = cache.get_or_create_ramp(make_gradient(SpreadMode::Repeat).data());
        assert_eq!(pad.as_slice()[0], start.to_linear_premultiplied_rgba());
        assert_eq!(pad.as_slice()[1023], end.to_linear_premultiplied_rgba());
        for sample in repeat.as_slice() {
            assert_eq!(*sample, expected);
        }
        let (GradientRamp::Sampled(pad), GradientRamp::Sampled(repeat)) = (pad, repeat) else {
            panic!("sampled ramps required");
        };
        assert!(!Arc::ptr_eq(&pad, &repeat));
    }
}

#[test]
fn non_padding_ramps_discard_duplicate_outer_stop_colors_before_caching() {
    let make_gradient = |spread| {
        Gradient::linear(
            LinearGradientDesc::new(
                LinearGradientLine {
                    start: [0.0, 0.0],
                    end: [100.0, 0.0],
                },
                [
                    GradientStop::at_position(
                        GradientStopOffset::linear_radial(0.0),
                        Color::rgb(255, 0, 0),
                    ),
                    GradientStop::at_position(GradientStopOffset::linear_radial(0.0), Color::BLACK),
                    GradientStop::at_position(GradientStopOffset::linear_radial(1.0), Color::BLACK),
                    GradientStop::at_position(
                        GradientStopOffset::linear_radial(1.0),
                        Color::rgb(0, 0, 255),
                    ),
                ],
            )
            .with_spread(spread),
        )
        .unwrap()
    };
    let mut cache = GradientCache::new();
    let pad = cache.get_or_create_ramp(make_gradient(SpreadMode::Pad).data());
    let repeat = cache.get_or_create_ramp(make_gradient(SpreadMode::Repeat).data());
    assert_eq!(pad.as_slice()[0], Color::rgb(255, 0, 0).normalize());
    assert_eq!(pad.as_slice()[1023], Color::rgb(0, 0, 255).normalize());
    assert!(repeat
        .as_slice()
        .iter()
        .all(|texel| *texel == Color::BLACK.normalize()));
    let (GradientRamp::Sampled(pad), GradientRamp::Sampled(repeat)) = (pad, repeat) else {
        panic!("sampled ramps required");
    };
    assert!(!Arc::ptr_eq(&pad, &repeat));
}

#[test]
fn hsl_ramp_upload_preserves_finite_half_float_values_for_both_signs() {
    let gradient = Gradient::linear(
        LinearGradientDesc::new(
            LinearGradientLine {
                start: [0.0, 0.0],
                end: [100.0, 0.0],
            },
            [
                GradientStop::auto(Color::rgb_f32(0.349, -0.351, -0.351)),
                GradientStop::auto(Color::rgb_f32(1.349, 0.649, 0.649)),
            ],
        )
        .with_interpolation(ColorInterpolation::Hsl {
            hue: HueInterpolationMethod::Shorter,
        }),
    )
    .unwrap();
    let GradientRamp::Pending(source) = &gradient.data().ramp else {
        panic!("expected sampled ramp source");
    };
    let ramp = source.bake();
    let mut positive_overflow = false;
    let mut negative_overflow = false;
    for texel in ramp.as_slice() {
        let packed = encode_ramp_texel_f16(*texel).map(f16::from_bits);
        for (original, packed) in texel.iter().copied().zip(packed) {
            assert!(original.is_finite());
            assert!(
                packed.is_finite(),
                "finite ramp channel {original} became {packed}"
            );
            if original > f16::MAX.to_f32() {
                positive_overflow = true;
                assert_eq!(packed, f16::MAX);
            } else if original < f16::MIN.to_f32() {
                negative_overflow = true;
                assert_eq!(packed, f16::MIN);
            } else {
                assert_eq!(packed, f16::from_f32(original));
            }
        }
    }
    assert!(positive_overflow && negative_overflow);
    assert_eq!(
        encode_ramp_texel_f16([2.0, -2.0, 0.25, 1.0]),
        [0x4000, 0xc000, 0x3400, 0x3c00]
    );
}

#[test]
fn gradient_ramp_caches_reuse_equal_stops_and_preserve_distinct_stops() {
    let descriptor = LinearGradientDesc::new(
        LinearGradientLine {
            start: [0.0, 0.0],
            end: [10.0, 0.0],
        },
        [
            GradientStop::auto(Color::rgb(255, 0, 0)),
            GradientStop::auto(Color::rgb(0, 0, 255)),
        ],
    )
    .with_interpolation(ColorInterpolation::SrgbLinear);
    let first = Gradient::linear(descriptor.clone()).unwrap();
    let second = Gradient::linear(descriptor.clone()).unwrap();
    let distinct = Gradient::linear(
        LinearGradientDesc::new(
            descriptor.line,
            [
                GradientStop::auto(Color::rgb(255, 0, 0)),
                GradientStop::auto(Color::rgb(0, 255, 0)),
            ],
        )
        .with_interpolation(ColorInterpolation::SrgbLinear),
    )
    .unwrap();
    let independent = Gradient::linear(descriptor).unwrap();
    let mut cache = GradientCache::new();
    let mut independent_cache = GradientCache::new();

    let GradientRamp::Sampled(first_ramp) = cache.get_or_create_ramp(first.data()) else {
        panic!("expected sampled ramp");
    };
    let GradientRamp::Sampled(second_ramp) = cache.get_or_create_ramp(second.data()) else {
        panic!("expected sampled ramp");
    };
    let GradientRamp::Sampled(independent_ramp) =
        independent_cache.get_or_create_ramp(independent.data())
    else {
        panic!("expected sampled ramp");
    };

    assert!(Arc::ptr_eq(&first_ramp, &second_ramp));
    assert!(!Arc::ptr_eq(&first_ramp, &independent_ramp));
    assert_eq!(first_ramp, independent_ramp);
    let distinct_ramp = cache.get_or_create_ramp(distinct.data());
    assert_eq!(distinct_ramp.as_slice().last(), Some(&[0.0, 1.0, 0.0, 1.0]));
    assert_eq!(
        cache.get_or_create_ramp(first.data()).as_slice().last(),
        Some(&[0.0, 0.0, 1.0, 1.0])
    );
}

#[test]
fn gradient_cache_distinguishes_float_channels_and_interpolation() {
    let make_gradient = |start_color, interpolation| {
        Gradient::linear(
            LinearGradientDesc::new(
                LinearGradientLine {
                    start: [0.0, 0.0],
                    end: [10.0, 0.0],
                },
                [
                    GradientStop::auto(start_color),
                    GradientStop::auto(Color::WHITE),
                ],
            )
            .with_interpolation(interpolation),
        )
        .unwrap()
    };
    let base_color = Color::rgba_f32(0.5, 0.3, 0.6, 0.75);
    let reference = make_gradient(base_color, ColorInterpolation::Srgb);
    let mut cache = GradientCache::new();
    let GradientRamp::Sampled(reference_ramp) = cache.get_or_create_ramp(reference.data()) else {
        panic!("expected sampled ramp");
    };

    for channel_index in 0..4 {
        let mut channels = base_color.to_array_f32();
        channels[channel_index] += 0.0001;
        let distinct_color = Color(channels);
        assert_eq!(distinct_color.to_array(), base_color.to_array());
        let distinct = make_gradient(distinct_color, ColorInterpolation::Srgb);
        let GradientRamp::Sampled(distinct_ramp) = cache.get_or_create_ramp(distinct.data()) else {
            panic!("expected sampled ramp");
        };
        assert!(!Arc::ptr_eq(&reference_ramp, &distinct_ramp));
        assert_ne!(reference_ramp[0], distinct_ramp[0]);
    }

    for (first, second) in [
        (ColorInterpolation::Srgb, ColorInterpolation::SrgbLinear),
        (
            ColorInterpolation::Hsl {
                hue: HueInterpolationMethod::Shorter,
            },
            ColorInterpolation::Hsl {
                hue: HueInterpolationMethod::Longer,
            },
        ),
        (
            ColorInterpolation::Hwb {
                hue: HueInterpolationMethod::Increasing,
            },
            ColorInterpolation::Hwb {
                hue: HueInterpolationMethod::Decreasing,
            },
        ),
    ] {
        let first = make_gradient(base_color, first);
        let second = make_gradient(base_color, second);
        let GradientRamp::Sampled(first_ramp) = cache.get_or_create_ramp(first.data()) else {
            panic!("expected sampled ramp");
        };
        let GradientRamp::Sampled(second_ramp) = cache.get_or_create_ramp(second.data()) else {
            panic!("expected sampled ramp");
        };
        assert!(!Arc::ptr_eq(&first_ramp, &second_ramp));
    }
}

#[test]
fn degenerate_linear_gradient_uses_the_last_stop_color() {
    let gradient = Gradient::linear(LinearGradientDesc::new(
        LinearGradientLine {
            start: [2.0, 3.0],
            end: [2.0, 3.0],
        },
        [
            GradientStop::auto(Color::rgba_f32(1.0, 0.0, 0.0, 1.0)),
            GradientStop::auto(Color::rgba_f32(0.0, 1.0, 0.0, 0.25)),
        ],
    ))
    .unwrap();

    let GradientRamp::Constant(color) = gradient.data().ramp else {
        panic!("a degenerate gradient must have a constant ramp");
    };
    assert_eq!(color, [0.0, 0.25, 0.0, 0.25]);
    let params = GpuGradientColorParams::from_gradient_data(gradient.data());
    assert_eq!(params.is_constant, 1);
    assert_eq!(params.constant_color, color);
}

#[test]
fn conic_gpu_parameters_use_turns_for_stops_and_radians_for_the_start_angle() {
    let gradient = Gradient::conic(
        ConicGradientDesc::new(
            [12.0, 24.0],
            FRAC_PI_2,
            [
                GradientStop::at_position(
                    GradientStopOffset::conic_radians(PI),
                    Color::rgb(255, 0, 0),
                ),
                GradientStop::at_position(
                    GradientStopOffset::conic_radians(TAU),
                    Color::rgb(0, 0, 255),
                ),
            ],
        )
        .with_units(GradientUnits::Canvas)
        .with_spread(SpreadMode::Repeat),
    )
    .unwrap();

    let params = GpuGradientColorParams::from_gradient_data(gradient.data());
    assert_eq!(gradient.data().period_start, 0.5);
    assert_eq!(gradient.data().period_len, 0.5);
    assert_eq!(params.gradient_type, 3);
    assert_eq!(params.units, 1);
    assert_eq!(params.spread_mode, 1);
    assert_eq!(params.conic_center, [12.0, 24.0]);
    assert_eq!(params.conic_start_angle, FRAC_PI_2);
    assert_eq!(params.period_start, 0.5);
    assert_eq!(params.period_len, 0.5);
    assert_eq!(params.ramp_start, 0.5);
    assert_eq!(params.ramp_end, 1.0);
}

#[test]
fn degenerate_radial_gradient_uses_the_last_stop_color() {
    for size in [
        RadialGradientSize::ExplicitCircleRadius(0.0),
        RadialGradientSize::ExplicitEllipseRadii {
            radius_x: 0.0,
            radius_y: 20.0,
        },
        RadialGradientSize::ExplicitEllipseRadii {
            radius_x: 20.0,
            radius_y: 0.0,
        },
    ] {
        let gradient = Gradient::radial(RadialGradientDesc::new(
            [50.0, 50.0],
            size,
            [
                GradientStop::auto(Color::rgb(255, 0, 0)),
                GradientStop::auto(Color::rgba_f32(0.0, 1.0, 0.0, 0.25)),
            ],
        ))
        .unwrap();

        let GradientRamp::Constant(color) = gradient.data().ramp else {
            panic!("a degenerate radial gradient must have a constant ramp");
        };
        assert_eq!(color, [0.0, 0.25, 0.0, 0.25]);
        let params = GpuGradientColorParams::from_gradient_data(gradient.data());
        assert_eq!(params.is_constant, 1);
        assert_eq!(params.constant_color, color);
    }
}
