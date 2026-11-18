use grafo::core::gradient::types::GradientRamp;
use grafo::{
    core::color, Color, ColorInterpolation, Gradient, GradientStop, GradientStopOffset,
    HueInterpolationMethod, LinearGradientDesc, LinearGradientLine,
};

struct LinearRgba([f32; 4]);

impl From<LinearRgba> for Color {
    fn from(value: LinearRgba) -> Self {
        let [red, green, blue, alpha] = value.0;
        Color::rgba_f32(
            color::linear_to_srgb(red),
            color::linear_to_srgb(green),
            color::linear_to_srgb(blue),
            alpha,
        )
    }
}

#[test]
fn rgba8_premultiplication_matches_css_transfer_rounding() {
    // CSS Color 4's double-precision transfer and CSS Values 4's nearest-integer rounding.
    // https://www.w3.org/TR/2026/CRD-css-color-4-20261112/#color-conversion-code
    // https://www.w3.org/TR/css-values-4/#integers
    for channel in u8::MIN..=u8::MAX {
        let encoded = f64::from(channel) / 255.0;
        let linear = if encoded <= 0.04045 {
            encoded / 12.92
        } else {
            ((encoded + 0.055) / 1.055).powf(2.4)
        };
        for alpha in u8::MIN..=u8::MAX {
            let premultiplied = linear * (f64::from(alpha) / 255.0);
            let encoded_premultiplied = if premultiplied <= 0.0031308 {
                premultiplied * 12.92
            } else {
                1.055 * premultiplied.powf(1.0 / 2.4) - 0.055
            };
            let expected = (encoded_premultiplied * 255.0).round() as u8;
            let mut pixel = [channel, channel, channel, alpha];
            color::premultiply_rgba8_srgb_inplace(&mut pixel);
            assert_eq!(
                pixel,
                [expected, expected, expected, alpha],
                "{channel}, {alpha}"
            );
        }
    }
}

#[test]
fn linear_colors_keep_css_encoding_before_hsl_interpolation() {
    let other_channels = f32::from_bits(0x3ec0_73be);
    let color = Color::from(LinearRgba([2.0, other_channels, other_channels, 1.0]));
    // CSS's encoder rounded once to Color's f32 storage.
    assert_eq!(
        color.to_array_f32().map(f32::to_bits),
        [0x3fad_377e, 0x3f25_9254, 0x3f25_9254, 0x3f80_0000]
    );
    assert_eq!(color::linear_to_srgb(1.0), 1.0);
    let gradient = Gradient::linear(
        LinearGradientDesc::new(
            LinearGradientLine {
                start: [0.0, 0.0],
                end: [100.0, 0.0],
            },
            [GradientStop::auto(color), GradientStop::auto(color)],
        )
        .with_interpolation(ColorInterpolation::Hsl {
            hue: HueInterpolationMethod::Shorter,
        }),
    )
    .unwrap();
    let GradientRamp::Pending(source) = &gradient.data().ramp else {
        panic!("expected a sampled gradient source");
    };
    let ramp = source.bake();
    let midpoint = ramp.as_slice()[ramp.as_slice().len() / 2];
    assert!(midpoint[0] > midpoint[1]);
}

#[test]
fn float_colors_preserve_byte_roundtrips_and_nearest_rounding() {
    for channel in u8::MIN..=u8::MAX {
        let bytes = [channel, u8::MAX - channel, channel / 2, channel];
        let color = Color::rgba(bytes[0], bytes[1], bytes[2], bytes[3]);
        assert_eq!(color.to_array(), bytes);
    }
    let below_half_byte = f32::from_bits(0x3f01_0101);
    assert_eq!(
        Color::rgba_f32(
            below_half_byte,
            below_half_byte,
            below_half_byte,
            below_half_byte
        )
        .to_array(),
        [128; 4]
    );
}

#[test]
fn extended_colors_decode_without_clipping_or_premultiplication() {
    let encoded = [-0.5, 0.04045, 1.5, 0.25];
    let color = Color::rgba_f32(encoded[0], encoded[1], encoded[2], encoded[3]);
    assert_eq!(color.to_array_f32(), encoded);

    // Reference values from the extended sRGB transfer function.
    let expected = [-0.214_041_14, 0.003_130_805, 2.537_155_4, 0.25];
    for (actual, expected) in color.normalize().into_iter().zip(expected) {
        assert!((actual - expected).abs() < 1e-6);
    }
    let premultiplied = [-0.053_510_286, 0.000_782_701_25, 0.634_288_85, 0.25];
    for (actual, expected) in color
        .to_linear_premultiplied_rgba()
        .into_iter()
        .zip(premultiplied)
    {
        assert!((actual - expected).abs() < 1e-6);
    }
    assert_eq!(color.to_array(), [0, 10, 255, 64]);
}

#[test]
fn premultiplication_preserves_representable_results_for_very_large_rgb() {
    for sign in [-1.0, 1.0] {
        let transparent = Color::rgba_f32(sign * 1e17, 0.0, 0.0, 0.0);
        assert_eq!(transparent.to_linear_premultiplied_rgba(), [0.0; 4]);
        let translucent = Color::rgba_f32(sign * 1e17, 0.0, 0.0, 1e-30);
        // Extended sRGB decoding and multiplication in double precision, rounded to f32.
        let expected = sign * f32::from_bits(0x514e_b4df);
        assert_eq!(
            translucent.to_linear_premultiplied_rgba(),
            [expected, 0.0, 0.0, 1e-30]
        );
    }
}

#[test]
fn solid_and_gradient_decoding_preserve_finite_values_near_the_linear_limit() {
    for sign in [-1.0, 1.0] {
        let encoded = sign * f32::from_bits(0x5a2a_23b6);
        let color = Color::rgb_f32(encoded, 0.0, 0.0);
        let expected = sign * 3.4028121e38;
        assert_eq!(color::srgb_to_linear(encoded), expected);
        assert_eq!(color.normalize(), [expected, 0.0, 0.0, 1.0]);
        assert_eq!(color.normalize(), color.to_linear_premultiplied_rgba());
    }
}

#[test]
fn extended_srgb_decoding_stays_monotonic_at_the_linear_limit() {
    let lower = f32::from_bits(0x5a2a_23b5);
    let upper = f32::from_bits(0x5a2a_23b6);
    assert!(color::srgb_to_linear(lower) < color::srgb_to_linear(upper));
    assert!(color::srgb_to_linear(-upper) < color::srgb_to_linear(-lower));
    // CSS double-precision transfer results, rounded to f32.
    for (encoded_bits, linear_bits) in [(0x5a2a_23b5, 0x7f7f_ffc3), (0x5a2a_23b6, 0x7f7f_ffc7)] {
        for sign in [-1.0, 1.0] {
            let encoded = sign * f32::from_bits(encoded_bits);
            let expected = sign * f32::from_bits(linear_bits);
            assert_eq!(color::srgb_to_linear(encoded), expected);
            assert_eq!(Color::rgb_f32(encoded, 0.0, 0.0).normalize()[0], expected);
        }
    }
}

#[test]
fn sampled_gradients_handle_extreme_rgb_with_zero_or_small_alpha() {
    for interpolation in [
        ColorInterpolation::Srgb,
        ColorInterpolation::SrgbLinear,
        ColorInterpolation::Oklab,
        ColorInterpolation::Hsl {
            hue: HueInterpolationMethod::Shorter,
        },
        ColorInterpolation::Hwb {
            hue: HueInterpolationMethod::Shorter,
        },
    ] {
        for alpha in [0.0, 1e-30] {
            let start = Color::rgba_f32(1e17, 0.0, 0.0, alpha);
            let gradient = Gradient::linear(
                LinearGradientDesc::new(
                    LinearGradientLine {
                        start: [0.0, 0.0],
                        end: [100.0, 0.0],
                    },
                    [GradientStop::auto(start), GradientStop::auto(Color::WHITE)],
                )
                .with_interpolation(interpolation),
            )
            .unwrap();
            let GradientRamp::Pending(source) = &gradient.data().ramp else {
                panic!("expected sampled ramp source");
            };
            let ramp = source.bake();
            assert_eq!(ramp.as_slice()[0], start.to_linear_premultiplied_rgba());
            assert!(
                ramp.as_slice()
                    .iter()
                    .flatten()
                    .all(|channel| channel.is_finite()),
                "{interpolation:?}, alpha {alpha}"
            );
            if alpha == 0.0 {
                let midpoint = ramp.as_slice()[ramp.as_slice().len() / 2];
                for channel in &midpoint[..3] {
                    assert!(
                        (channel - midpoint[3]).abs() < 2e-6,
                        "{interpolation:?}: {midpoint:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn hwb_hdr_gradients_preserve_a_constant_blue_hue() {
    for hue in [
        HueInterpolationMethod::Shorter,
        HueInterpolationMethod::Increasing,
        HueInterpolationMethod::Decreasing,
    ] {
        let gradient = Gradient::linear(
            LinearGradientDesc::new(
                LinearGradientLine {
                    start: [0.0, 0.0],
                    end: [100.0, 0.0],
                },
                [
                    GradientStop::auto(Color::rgb_f32(0.0, 0.0, 1e15)),
                    GradientStop::auto(Color::rgb_f32(0.0, 0.0, 2e15)),
                ],
            )
            .with_interpolation(ColorInterpolation::Hwb { hue }),
        )
        .unwrap();
        let GradientRamp::Pending(source) = &gradient.data().ramp else {
            panic!("expected a sampled gradient source");
        };
        let ramp = source.bake();
        for (index, texel) in ramp.as_slice().iter().enumerate() {
            assert_eq!([texel[0], texel[1]], [0.0, 0.0], "{hue:?}, texel {index}");
            assert!(texel[2].is_finite() && texel[2] > 0.0);
        }
    }
}

#[test]
fn srgb_ramps_combine_signed_channels_before_dividing_weights() {
    let unit = 2.0_f32.powi(49);
    for (sign, hint) in [
        (-1.0, None),
        (1.0, None),
        (-1.0, Some(0.5)),
        (1.0, Some(0.5)),
    ] {
        let start = GradientStop::auto(Color::rgb_f32(sign * 12.0 * unit, 0.0, 0.0));
        let start = match hint {
            Some(position) => {
                start.with_hint_to_next_segment(GradientStopOffset::linear_radial(position))
            }
            None => start,
        };
        let gradient = Gradient::linear(
            LinearGradientDesc::new(
                LinearGradientLine {
                    start: [0.0, 0.0],
                    end: [100.0, 0.0],
                },
                [
                    start,
                    GradientStop::auto(Color::rgb_f32(sign * -19.0 * unit, 0.0, 0.0)),
                ],
            )
            .with_interpolation(ColorInterpolation::Srgb),
        )
        .unwrap();
        let GradientRamp::Pending(source) = &gradient.data().ramp else {
            panic!("expected a sampled gradient source");
        };
        let ramp = source.bake();
        assert_eq!(
            ramp.as_slice()[396],
            [0.0, 0.0, 0.0, 1.0],
            "{sign}, {hint:?}"
        );
    }
}

#[test]
fn sampled_positions_do_not_round_onto_lossy_hsl_stops() {
    let pink = Color::rgb_f32(1.5, 0.5, 0.5);
    let stops = [2.0_f32.powi(-80), 1.0 / 1024.0, 1023.0 / 1024.0].map(|position| {
        GradientStop::at_position(GradientStopOffset::linear_radial(position), pink)
    });
    let gradient = Gradient::linear(
        LinearGradientDesc::new(
            LinearGradientLine {
                start: [0.0, 0.0],
                end: [100.0, 0.0],
            },
            stops,
        )
        .with_interpolation(ColorInterpolation::Hsl {
            hue: HueInterpolationMethod::Shorter,
        }),
    )
    .unwrap();
    let GradientRamp::Pending(source) = &gradient.data().ramp else {
        panic!("expected a sampled gradient source");
    };
    let ramp = source.bake();
    assert_eq!(ramp.as_slice()[1], [1.0; 4]);
}

#[test]
fn sampled_positions_keep_the_correct_side_of_hard_stops() {
    let stops = [
        (-2.0_f32.powi(-80), Color::BLACK),
        (1.0 / 1024.0, Color::BLACK),
        (1.0 / 1024.0, Color::WHITE),
        (1023.0 / 1024.0, Color::WHITE),
    ]
    .map(|(position, color)| {
        GradientStop::at_position(GradientStopOffset::linear_radial(position), color)
    });
    let gradient = Gradient::linear(
        LinearGradientDesc::new(
            LinearGradientLine {
                start: [0.0, 0.0],
                end: [100.0, 0.0],
            },
            stops,
        )
        .with_interpolation(ColorInterpolation::Srgb),
    )
    .unwrap();
    let GradientRamp::Pending(source) = &gradient.data().ramp else {
        panic!("expected a sampled gradient source");
    };
    let ramp = source.bake();
    assert_eq!(ramp.as_slice()[1], [0.0, 0.0, 0.0, 1.0]);
}

#[test]
fn caller_owned_colors_bake_through_every_stop_constructor() {
    let gradient = Gradient::linear(
        LinearGradientDesc::new(
            LinearGradientLine {
                start: [0.0, 0.0],
                end: [100.0, 0.0],
            },
            [
                GradientStop::auto(LinearRgba([1.5, -0.25, 0.0, 0.25])),
                GradientStop::at_position(
                    GradientStopOffset::linear_radial(0.4),
                    LinearRgba([0.0, 0.5, 1.0, 0.5]),
                ),
                GradientStop::between_positions(
                    GradientStopOffset::linear_radial(0.7),
                    GradientStopOffset::linear_radial(1.0),
                    LinearRgba([0.25, 0.25, 0.25, 1.0]),
                ),
            ],
        )
        .with_interpolation(ColorInterpolation::SrgbLinear),
    )
    .unwrap();
    let GradientRamp::Pending(source) = &gradient.data().ramp else {
        panic!("expected a sampled gradient source");
    };
    let ramp = source.bake();
    let samples = ramp.as_slice();
    let expected = [
        (0, [0.375, -0.0625, 0.0, 0.25]),
        (341, [0.0625, 0.197_916_67, 0.416_666_67, 0.458_333_34]),
        (682, [0.222_222_22, 0.25, 0.277_777_8, 0.944_444_4]),
        (1023, [0.25, 0.25, 0.25, 1.0]),
    ];
    for (index, expected) in expected {
        for (actual, expected) in samples[index].into_iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6, "sample {index}");
        }
    }
}
