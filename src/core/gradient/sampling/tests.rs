use super::{
    bake_gradient_ramp, color_to_final_linear_premultiplied, hsl_to_srgb, linear_rgb_to_oklab,
    linear_to_srgb, oklab_to_linear_rgb, prepare_cylindrical_interpolation, srgb_to_linear,
    CylSpace,
};
use crate::core::gradient::normalize::NormalizedGradient;
use crate::core::gradient::types::{
    ColorInterpolation, GradientColor, GradientCommonDesc, GradientKind, GradientRampSource,
    GradientStop, GradientStopOffset, GradientStopPositions, GradientUnits, HueComponent,
    HueInterpolationMethod, SpreadMode, RAMP_RESOLUTION, RESOLVED_DEGENERATE_EPSILON,
};

fn srgb_color(red: f32, green: f32, blue: f32) -> GradientColor {
    GradientColor::Srgb {
        red,
        green,
        blue,
        alpha: 1.0,
    }
}

fn mixed_color_ramp_source(interpolation: ColorInterpolation) -> GradientRampSource {
    let common = GradientCommonDesc::new([
        GradientStop::at_position(
            GradientStopOffset::linear_radial(0.0),
            GradientColor::Oklab {
                l: 0.6,
                a: 0.1,
                b: -0.1,
                alpha: 0.4,
            },
        )
        .with_hint_to_next_segment(GradientStopOffset::linear_radial(0.1)),
        GradientStop::at_position(
            GradientStopOffset::linear_radial(0.3),
            GradientColor::SrgbLinear {
                red: 0.2,
                green: 0.7,
                blue: 0.3,
                alpha: 1.4,
            },
        ),
        GradientStop::at_position(
            GradientStopOffset::linear_radial(0.3),
            GradientColor::Hsl {
                hue: HueComponent::Missing,
                saturation: 0.8,
                lightness: 0.4,
                alpha: 0.7,
            },
        ),
        GradientStop::at_position(
            GradientStopOffset::linear_radial(0.6),
            GradientColor::Hwb {
                hue: HueComponent::Degrees(240.0),
                whiteness: 0.8,
                blackness: 0.4,
                alpha: -0.2,
            },
        ),
        GradientStop::at_position(
            GradientStopOffset::linear_radial(1.0),
            GradientColor::Srgb {
                red: 0.1,
                green: 0.3,
                blue: 0.9,
                alpha: 0.8,
            },
        ),
    ])
    .with_interpolation(interpolation);
    GradientRampSource {
        interpolation,
        normalized: NormalizedGradient::from_common(&common, GradientKind::Linear),
    }
}

#[test]
fn mixed_color_ramps_preserve_interpolation_output() {
    // Fixed output baseline added with the sampling refactor in a28d97a.
    // This detects numerical changes; it is not an independent color reference.
    // Review intentional output changes against the color equations before updating it.
    const CHANNEL_TOLERANCE: f32 = 2e-6;
    // Positions are index / 1023: inside the hinted segment, just after the
    // hard stop at 0.3, and inside each of the two remaining segments.
    const SAMPLE_TEXEL_INDICES: [usize; 4] = [137, 307, 512, 767];

    let interpolations = [
        ColorInterpolation::Srgb,
        ColorInterpolation::SrgbLinear,
        ColorInterpolation::Oklab,
        ColorInterpolation::Hsl {
            hue: HueInterpolationMethod::Shorter,
        },
        ColorInterpolation::Hsl {
            hue: HueInterpolationMethod::Longer,
        },
        ColorInterpolation::Hsl {
            hue: HueInterpolationMethod::Increasing,
        },
        ColorInterpolation::Hsl {
            hue: HueInterpolationMethod::Decreasing,
        },
        ColorInterpolation::Hwb {
            hue: HueInterpolationMethod::Shorter,
        },
        ColorInterpolation::Hwb {
            hue: HueInterpolationMethod::Longer,
        },
        ColorInterpolation::Hwb {
            hue: HueInterpolationMethod::Increasing,
        },
        ColorInterpolation::Hwb {
            hue: HueInterpolationMethod::Decreasing,
        },
    ];
    let expected_samples = [
        [
            [0.1729172, 0.40541703, 0.25489584, 0.76070446],
            [0.33379114, 0.0050344444, 0.0050344444, 0.6997719],
            [0.11075602, 0.0016704901, 0.0016704901, 0.23219293],
            [0.0030019488, 0.021935893, 0.23583883, 0.29951122],
        ],
        [
            [0.17592524, 0.4407735, 0.25822103, 0.76070446],
            [0.33379108, 0.0050344444, 0.0050344444, 0.6997719],
            [0.110755995, 0.0016704907, 0.0016704907, 0.23219293],
            [0.0030019488, 0.021935893, 0.23583886, 0.29951122],
        ],
        [
            [0.19306707, 0.41357756, 0.2678221, 0.76070446],
            [0.333791, 0.0050344802, 0.005034423, 0.6997719],
            [0.11075597, 0.0016704889, 0.0016704871, 0.23219293],
            [0.0030019623, 0.02193589, 0.23583879, 0.29951122],
        ],
        [
            [0.18444388, 0.5204905, 0.66794825, 1.0],
            [0.005030864, 0.005030864, 0.3337446, 0.69970673],
            [-0.00025836608, -0.00025836608, 0.0011545114, 0.09853381],
            [-0.003118001, -0.0006021938, 0.13839178, 0.17438902],
        ],
        [
            [0.66794825, 0.569362, 0.18444388, 1.0],
            [0.0051435283, 0.005030864, 0.3337446, 0.69970673],
            [-0.00025836608, 0.0011545114, -0.00024760913, 0.09853381],
            [0.13839178, 0.00023479709, -0.003118001, 0.17438902],
        ],
        [
            [0.66794825, 0.569362, 0.18444388, 1.0],
            [0.005030864, 0.005030864, 0.3337446, 0.69970673],
            [-0.00025836608, -0.00025836608, 0.0011545114, 0.09853381],
            [0.13839178, 0.00023479709, -0.003118001, 0.17438902],
        ],
        [
            [0.18444388, 0.5204905, 0.66794825, 1.0],
            [0.005030864, 0.0051435423, 0.3337446, 0.69970673],
            [0.0011545114, -0.00025836608, -0.00024760913, 0.09853381],
            [-0.003118001, -0.0006021938, 0.13839178, 0.17438902],
        ],
        [
            [0.18667893, 0.51835436, 0.66323996, 1.0],
            [0.3337652, 0.0050290865, 0.005066458, 0.69970673],
            [0.006493173, -0.04639187, 0.05822707, 0.09853381],
            [-0.013347357, -0.0046479045, 0.20233947, 0.17438902],
        ],
        [
            [0.66323996, 0.56640154, 0.18667893, 1.0],
            [0.3337652, 0.0051040268, 0.0050290865, 0.69970673],
            [-0.04639187, 0.05822707, 0.0071900864, 0.09853381],
            [0.20233947, -0.0016603572, -0.013347357, 0.17438902],
        ],
        [
            [0.66323996, 0.56640154, 0.18667893, 1.0],
            [0.3337652, 0.0051040268, 0.0050290865, 0.69970673],
            [-0.04639187, 0.05822707, 0.0071900864, 0.09853381],
            [0.20233947, -0.0016603572, -0.013347357, 0.17438902],
        ],
        [
            [0.18667893, 0.51835436, 0.66323996, 1.0],
            [0.3337652, 0.0050290865, 0.005066458, 0.69970673],
            [0.006493173, -0.04639187, 0.05822707, 0.09853381],
            [-0.013347357, -0.0046479045, 0.20233947, 0.17438902],
        ],
    ];
    for (interpolation, expected_samples) in interpolations.into_iter().zip(expected_samples) {
        let ramp = bake_gradient_ramp(&mixed_color_ramp_source(interpolation));
        for (index, expected) in SAMPLE_TEXEL_INDICES.into_iter().zip(expected_samples) {
            let actual = ramp.as_slice()[index];
            for (actual_channel, expected_channel) in actual.into_iter().zip(expected) {
                assert!(
                    (actual_channel - expected_channel).abs() < CHANNEL_TOLERANCE,
                    "{interpolation:?} texel {index}: expected {expected:?}, got {actual:?}",
                );
            }
        }
    }
}

#[test]
fn sampled_hard_stop_uses_the_last_coincident_color() {
    let boundary_index = RAMP_RESOLUTION / 3;
    let boundary_position = boundary_index as f32 / (RAMP_RESOLUTION - 1) as f32;
    let common = GradientCommonDesc::new([
        GradientStop::at_position(
            GradientStopOffset::linear_radial(0.0),
            srgb_color(1.0, 0.0, 0.0),
        ),
        GradientStop::at_position(
            GradientStopOffset::linear_radial(boundary_position),
            srgb_color(1.0, 0.0, 0.0),
        ),
        GradientStop::at_position(
            GradientStopOffset::linear_radial(boundary_position),
            srgb_color(0.0, 0.0, 1.0),
        ),
        GradientStop::at_position(
            GradientStopOffset::linear_radial(1.0),
            srgb_color(0.0, 0.0, 1.0),
        ),
    ]);
    let ramp = bake_gradient_ramp(&GradientRampSource {
        interpolation: common.interpolation,
        normalized: NormalizedGradient::from_common(&common, GradientKind::Linear),
    });

    assert_eq!(ramp.as_slice()[boundary_index - 1], [1.0, 0.0, 0.0, 1.0]);
    assert_eq!(ramp.as_slice()[boundary_index], [0.0, 0.0, 1.0, 1.0]);
    assert_eq!(ramp.as_slice()[RAMP_RESOLUTION - 1], [0.0, 0.0, 1.0, 1.0]);
}

#[test]
fn srgb_linear_conversion_roundtrip_preserves_channel_values() {
    for channel in [0.0, 0.04045, 0.5, 1.0, -0.5] {
        let linear = srgb_to_linear(channel);
        let restored = linear_to_srgb(linear);
        assert!(
            (channel - restored).abs() < 1e-5,
            "roundtrip failed for {channel}: got {restored}"
        );
    }
}

#[test]
fn oklab_conversion_roundtrip_preserves_linear_rgb() {
    let [lightness, a, b] = linear_rgb_to_oklab(0.5, 0.3, 0.1);
    let [red, green, blue] = oklab_to_linear_rgb(lightness, a, b);
    assert!((red - 0.5).abs() < 1e-4);
    assert!((green - 0.3).abs() < 1e-4);
    assert!((blue - 0.1).abs() < 1e-4);
}

#[test]
fn hsl_green_converts_to_srgb_green() {
    let (red, green, blue) = hsl_to_srgb(120.0, 1.0, 0.5);
    assert!((red - 0.0).abs() < 1e-5);
    assert!((green - 1.0).abs() < 1e-5);
    assert!((blue - 0.0).abs() < 1e-5);
}

#[test]
fn bake_gradient_ramp_preserves_degenerate_hard_stop_boundary() {
    let common = GradientCommonDesc {
        units: GradientUnits::Local,
        spread: SpreadMode::Pad,
        interpolation: ColorInterpolation::Srgb,
        stops: vec![
            GradientStop {
                positions: GradientStopPositions::Single(GradientStopOffset::LinearRadial(0.5)),
                color: srgb_color(1.0, 0.0, 0.0),
                hint_to_next_segment: None,
            },
            GradientStop {
                positions: GradientStopPositions::Single(GradientStopOffset::LinearRadial(0.5)),
                color: srgb_color(0.0, 0.0, 1.0),
                hint_to_next_segment: None,
            },
        ]
        .into(),
    };

    let normalized = NormalizedGradient::from_common(&common, GradientKind::Linear);
    let ramp = bake_gradient_ramp(&GradientRampSource {
        interpolation: common.interpolation,
        normalized,
    });
    let ramp = ramp.as_slice();
    let transition_index = RAMP_RESOLUTION / 2;

    assert_eq!(
        ramp[transition_index - 1],
        color_to_final_linear_premultiplied(&srgb_color(1.0, 0.0, 0.0))
    );
    assert_eq!(
        ramp[transition_index],
        color_to_final_linear_premultiplied(&srgb_color(0.0, 0.0, 1.0))
    );
}

#[test]
fn bake_gradient_ramp_does_not_create_hard_stop_for_near_degenerate_span() {
    let common = GradientCommonDesc {
        units: GradientUnits::Local,
        spread: SpreadMode::Pad,
        interpolation: ColorInterpolation::Srgb,
        stops: vec![
            GradientStop {
                positions: GradientStopPositions::Single(GradientStopOffset::LinearRadial(0.0)),
                color: srgb_color(1.0, 0.0, 0.0),
                hint_to_next_segment: None,
            },
            GradientStop {
                positions: GradientStopPositions::Single(GradientStopOffset::LinearRadial(
                    RESOLVED_DEGENERATE_EPSILON * 0.5,
                )),
                color: srgb_color(0.0, 0.0, 1.0),
                hint_to_next_segment: None,
            },
        ]
        .into(),
    };

    let normalized = NormalizedGradient::from_common(&common, GradientKind::Linear);
    let ramp = bake_gradient_ramp(&GradientRampSource {
        interpolation: common.interpolation,
        normalized,
    });
    let ramp = ramp.as_slice();
    let expected = color_to_final_linear_premultiplied(&srgb_color(0.0, 0.0, 1.0));

    assert!(ramp.iter().all(|texel| *texel == expected));
}

#[test]
fn cylindrical_interpolation_premultiplies_non_hue_channels() {
    let color_a = GradientColor::Hsl {
        hue: HueComponent::Degrees(0.0),
        saturation: 1.0,
        lightness: 0.5,
        alpha: 1.0,
    };
    let color_b = GradientColor::Hsl {
        hue: HueComponent::Degrees(120.0),
        saturation: 0.0,
        lightness: 1.0,
        alpha: 0.0,
    };

    let interpolate = prepare_cylindrical_interpolation(
        &color_a,
        &color_b,
        CylSpace::Hsl,
        HueInterpolationMethod::Shorter,
    );
    let interpolated = interpolate(0.5);

    let expected = color_to_final_linear_premultiplied(&GradientColor::Hsl {
        hue: HueComponent::Degrees(0.0),
        saturation: 1.0,
        lightness: 0.5,
        alpha: 0.5,
    });

    assert_eq!(interpolated, expected);
}
