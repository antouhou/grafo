use super::{bake_gradient_ramp, bake_segments, prepare_cylindrical_interpolation};
use crate::core::color::CylindricalColorSpace;
use crate::core::gradient::normalize::NormalizedGradient;
use crate::core::gradient::types::{
    ColorInterpolation, GradientCommonDesc, GradientKind, GradientRampSource, GradientStop,
    GradientStopOffset, GradientStopPositions, GradientUnits, HueInterpolationMethod, SpreadMode,
    RAMP_RESOLUTION, RESOLVED_DEGENERATE_EPSILON,
};
use crate::core::{color, Color};
use half::f16;
use std::cell::Cell;

fn canonical_color_ramp_source(interpolation: ColorInterpolation) -> GradientRampSource {
    let common = GradientCommonDesc::new([
        GradientStop::at_position(
            GradientStopOffset::linear_radial(0.0),
            Color::rgba_f32(0.3, 0.1, 0.8, 0.4),
        )
        .with_hint_to_next_segment(GradientStopOffset::linear_radial(0.1)),
        GradientStop::at_position(
            GradientStopOffset::linear_radial(0.3),
            Color::rgba_f32(0.2, 0.7, 0.3, 1.0),
        ),
        GradientStop::at_position(
            GradientStopOffset::linear_radial(0.3),
            Color::rgba_f32(0.8, 0.2, 0.4, 0.7),
        ),
        GradientStop::at_position(
            GradientStopOffset::linear_radial(0.6),
            Color::rgba_f32(0.4, 0.5, 0.6, 0.2),
        ),
        GradientStop::at_position(
            GradientStopOffset::linear_radial(1.0),
            Color::rgba_f32(0.1, 0.3, 0.9, 0.8),
        ),
    ])
    .with_interpolation(interpolation);
    GradientRampSource {
        interpolation,
        normalized: NormalizedGradient::from_common(&common, GradientKind::Linear),
    }
}

#[test]
fn canonical_color_ramps_preserve_interpolation_output() {
    // Fixed sRGB-input samples cover hints, hard stops, and varying alpha in every mode.
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
            [0.03044274, 0.22001682, 0.10368237, 0.76070446],
            [0.42253664, 0.023174033, 0.09299532, 0.6998371],
            [0.14088109, 0.028566856, 0.06949506, 0.3658521],
            [0.012570953, 0.044961173, 0.2648352, 0.42463344],
        ],
        [
            [0.031585567, 0.27091795, 0.14035818, 0.76070446],
            [0.4225501, 0.023179732, 0.09299829, 0.6998371],
            [0.15796351, 0.036295254, 0.0734278, 0.3658521],
            [0.019626727, 0.048717193, 0.27569613, 0.42463344],
        ],
        [
            [0.027757622, 0.2536937, 0.13759063, 0.76070446],
            [0.4225475, 0.023180656, 0.09299563, 0.6998371],
            [0.15449432, 0.037514467, 0.0698964, 0.3658521],
            [0.011996892, 0.050960653, 0.2683851, 0.42463344],
        ],
        [
            [0.020495351, 0.34482792, 0.36394837, 0.76070446],
            [0.42255867, 0.023172002, 0.09319709, 0.6998371],
            [0.041813605, 0.022172298, 0.17828147, 0.3658521],
            [0.012570961, 0.069625944, 0.2648352, 0.42463344],
        ],
        [
            [0.36394837, 0.1811819, 0.020495351, 0.76070446],
            [0.42255867, 0.023172002, 0.092622064, 0.6998371],
            [0.022172298, 0.17828147, 0.042863045, 0.3658521],
            [0.13289209, 0.2648352, 0.012570961, 0.42463344],
        ],
        [
            [0.36394837, 0.1811819, 0.020495351, 0.76070446],
            [0.42255867, 0.023172002, 0.092622064, 0.6998371],
            [0.022172298, 0.17828147, 0.042863045, 0.3658521],
            [0.012570961, 0.069625944, 0.2648352, 0.42463344],
        ],
        [
            [0.020495351, 0.34482792, 0.36394837, 0.76070446],
            [0.42255867, 0.023172002, 0.09319709, 0.6998371],
            [0.041813605, 0.022172298, 0.17828147, 0.3658521],
            [0.13289209, 0.2648352, 0.012570961, 0.42463344],
        ],
        [
            [0.020495359, 0.34482798, 0.36394837, 0.76070446],
            [0.42255875, 0.023172002, 0.09319709, 0.6998371],
            [0.041813616, 0.02217231, 0.17828147, 0.3658521],
            [0.012570953, 0.069625944, 0.26483527, 0.42463344],
        ],
        [
            [0.36394837, 0.18118194, 0.020495359, 0.76070446],
            [0.42255875, 0.023172002, 0.092622064, 0.6998371],
            [0.02217231, 0.17828147, 0.04286306, 0.3658521],
            [0.13289209, 0.26483527, 0.012570953, 0.42463344],
        ],
        [
            [0.36394837, 0.18118194, 0.020495359, 0.76070446],
            [0.42255875, 0.023172002, 0.092622064, 0.6998371],
            [0.02217231, 0.17828147, 0.04286306, 0.3658521],
            [0.012570953, 0.069625944, 0.26483527, 0.42463344],
        ],
        [
            [0.020495359, 0.34482798, 0.36394837, 0.76070446],
            [0.42255875, 0.023172002, 0.09319709, 0.6998371],
            [0.041813616, 0.02217231, 0.17828147, 0.3658521],
            [0.13289209, 0.26483527, 0.012570953, 0.42463344],
        ],
    ];
    for (interpolation, expected_samples) in interpolations.into_iter().zip(expected_samples) {
        let ramp = bake_gradient_ramp(&canonical_color_ramp_source(interpolation));
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
fn rectangular_interpolation_keeps_tiny_hdr_endpoint_contributions() {
    let common = GradientCommonDesc::new([
        GradientStop::at_position(
            GradientStopOffset::linear_radial(-1.0),
            Color::rgb_f32(1e10, 0.0, 0.0),
        ),
        GradientStop::at_position(GradientStopOffset::linear_radial(1e-20), Color::BLACK),
        GradientStop::at_position(GradientStopOffset::linear_radial(2.0), Color::BLACK),
    ])
    .with_interpolation(ColorInterpolation::SrgbLinear);
    let ramp = bake_gradient_ramp(&GradientRampSource {
        interpolation: common.interpolation,
        normalized: NormalizedGradient::from_common(&common, GradientKind::Linear),
    });
    assert_eq!(ramp.as_slice()[341], [8794.154, 0.0, 0.0, 1.0]);
}

#[test]
fn sampled_hard_stops_preserve_exact_dyadic_texel_alignment() {
    let boundary = 105.0 / 4096.0;
    let common = GradientCommonDesc::new([
        GradientStop::at_position(GradientStopOffset::linear_radial(0.0), Color::BLACK),
        GradientStop::at_position(GradientStopOffset::linear_radial(boundary), Color::BLACK),
        GradientStop::at_position(GradientStopOffset::linear_radial(boundary), Color::WHITE),
        GradientStop::at_position(
            GradientStopOffset::linear_radial(3069.0 / 4096.0),
            Color::WHITE,
        ),
    ])
    .with_interpolation(ColorInterpolation::SrgbLinear);
    let ramp = bake_gradient_ramp(&GradientRampSource {
        interpolation: common.interpolation,
        normalized: NormalizedGradient::from_common(&common, GradientKind::Linear),
    });
    assert_eq!(ramp.as_slice()[34], Color::BLACK.normalize());
    assert_eq!(ramp.as_slice()[35], Color::WHITE.normalize());
    assert_eq!(ramp.as_slice()[36], Color::WHITE.normalize());
}

#[test]
fn sampled_hard_stop_uses_the_last_coincident_color() {
    let boundary_index = RAMP_RESOLUTION / 3;
    let boundary_position = boundary_index as f32;
    let common = GradientCommonDesc::new([
        GradientStop::at_position(
            GradientStopOffset::linear_radial(0.0),
            Color::rgb_f32(1.0, 0.0, 0.0),
        ),
        GradientStop::at_position(
            GradientStopOffset::linear_radial(boundary_position),
            Color::rgb_f32(1.0, 0.0, 0.0),
        ),
        GradientStop::at_position(
            GradientStopOffset::linear_radial(boundary_position),
            Color::rgb_f32(0.0, 0.0, 1.0),
        ),
        GradientStop::at_position(
            GradientStopOffset::linear_radial((RAMP_RESOLUTION - 1) as f32),
            Color::rgb_f32(0.0, 0.0, 1.0),
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
fn sampled_hsl_ramp_keeps_the_last_coincident_stop_when_positions_round_down() {
    let common = GradientCommonDesc::new([
        GradientStop::at_position(GradientStopOffset::linear_radial(-1.0), Color::BLACK),
        GradientStop::at_position(
            GradientStopOffset::linear_radial(0.01),
            Color::rgb(0, 255, 0),
        ),
        GradientStop::at_position(
            GradientStopOffset::linear_radial(0.01),
            Color::rgb_f32(1.5, 0.5, 0.5),
        ),
    ])
    .with_interpolation(ColorInterpolation::Hsl {
        hue: HueInterpolationMethod::Shorter,
    });
    let ramp = bake_gradient_ramp(&GradientRampSource {
        interpolation: common.interpolation,
        normalized: NormalizedGradient::from_common(&common, GradientKind::Linear),
    });
    assert_eq!(
        ramp.as_slice()[RAMP_RESOLUTION - 1],
        Color::rgb_f32(1.5, 0.5, 0.5).to_linear_premultiplied_rgba()
    );
}

#[test]
fn hsl_short_segments_keep_high_saturation_interpolation() {
    let end_color = Color::rgb_f32(0.55, 0.45, 0.45);
    let common = GradientCommonDesc::new([
        GradientStop::at_position(
            GradientStopOffset::linear_radial(0.0),
            Color::rgb_f32(10.0, -9.999979, -9.999979),
        ),
        GradientStop::at_position(GradientStopOffset::linear_radial(5e-7), end_color),
        GradientStop::at_position(
            GradientStopOffset::linear_radial(1.8e-6),
            Color::rgb(0, 255, 0),
        ),
    ])
    .with_interpolation(ColorInterpolation::Hsl {
        hue: HueInterpolationMethod::Shorter,
    });
    let ramp = bake_gradient_ramp(&GradientRampSource {
        interpolation: common.interpolation,
        normalized: NormalizedGradient::from_common(&common, GradientKind::Linear),
    });
    // CSS HSL and interval weighting in double precision at the first interior sample.
    assert_eq!(
        ramp.as_slice()[1],
        [48504016.0, -48503772.0, -48503772.0, 1.0]
    );
}

#[test]
fn hinted_hsl_segments_keep_high_saturation_contributions_before_the_end_stop() {
    let end_color = Color::rgb_f32(0.55, 0.45, 0.45);
    let common = GradientCommonDesc::new([
        GradientStop::at_position(
            GradientStopOffset::linear_radial(0.0),
            Color::rgb_f32(50.0, -49.999973, -49.999973),
        )
        .with_hint_to_next_segment(GradientStopOffset::linear_radial(0.01)),
        GradientStop::at_position(GradientStopOffset::linear_radial(0.33333337), end_color),
        GradientStop::at_position(
            GradientStopOffset::linear_radial(1.0),
            Color::rgb(0, 255, 0),
        ),
    ])
    .with_interpolation(ColorInterpolation::Hsl {
        hue: HueInterpolationMethod::Shorter,
    });
    let ramp = bake_gradient_ramp(&GradientRampSource {
        interpolation: common.interpolation,
        normalized: NormalizedGradient::from_common(&common, GradientKind::Linear),
    });
    // CSS hint weighting and HSL evaluated in double precision at 341/1023.
    assert_eq!(
        ramp.as_slice()[341],
        [0.31173036, 0.13702439, 0.13702439, 1.0]
    );
}

#[test]
fn hinted_rectangular_segments_preserve_small_hdr_contributions() {
    for (interpolation, red, expected) in [
        (ColorInterpolation::Srgb, 1e8, 7.2719884),
        (ColorInterpolation::SrgbLinear, 1000.0, 0.3284777),
    ] {
        let common = GradientCommonDesc::new([
            GradientStop::at_position(
                GradientStopOffset::linear_radial(0.0),
                Color::rgb_f32(red, 0.0, 0.0),
            )
            .with_hint_to_next_segment(GradientStopOffset::linear_radial(0.01)),
            GradientStop::at_position(GradientStopOffset::linear_radial(0.33333337), Color::BLACK),
            GradientStop::at_position(GradientStopOffset::linear_radial(1.0), Color::BLACK),
        ])
        .with_interpolation(interpolation);
        let ramp = bake_gradient_ramp(&GradientRampSource {
            interpolation,
            normalized: NormalizedGradient::from_common(&common, GradientKind::Linear),
        });
        assert_eq!(ramp.as_slice()[341], [expected, 0.0, 0.0, 1.0]);
    }
}

#[test]
fn hsl_ramps_preserve_authored_colors_at_stops() {
    let first = Color::rgb_f32(1.5, 0.5, 0.5);
    let middle = Color::rgb_f32(2.0, 1.0, 0.0);
    let last = Color::rgb_f32(0.5, 1.5, 0.5);
    let middle_index = RAMP_RESOLUTION / 3;
    // Integer offsets place this authored interior stop exactly on its sampled texel.
    let middle_position = middle_index as f32;
    let common = GradientCommonDesc::new([
        GradientStop::at_position(GradientStopOffset::linear_radial(0.0), first),
        GradientStop::at_position(GradientStopOffset::linear_radial(middle_position), middle),
        GradientStop::at_position(
            GradientStopOffset::linear_radial((RAMP_RESOLUTION - 1) as f32),
            last,
        ),
    ])
    .with_interpolation(ColorInterpolation::Hsl {
        hue: HueInterpolationMethod::Shorter,
    });
    let ramp = bake_gradient_ramp(&GradientRampSource {
        interpolation: common.interpolation,
        normalized: NormalizedGradient::from_common(&common, GradientKind::Linear),
    });
    for (index, color) in [
        (0, first),
        (middle_index, middle),
        (RAMP_RESOLUTION - 1, last),
    ] {
        assert_eq!(
            ramp.as_slice()[index],
            color.to_linear_premultiplied_rgba(),
            "sample {index}"
        );
    }
    assert_eq!(ramp.as_slice()[middle_index - 1], [1.0; 4]);
    assert_eq!(ramp.as_slice()[middle_index + 1], [1.0; 4]);
}

#[test]
fn hwb_powerless_stops_borrow_the_other_color_hue() {
    for (neutral, other, midpoint) in [
        (
            Color::rgb_f32(0.0, 0.0, 1e-8),
            Color::rgb(255, 0, 0),
            [0.21404114, 0.0, 0.0, 1.0],
        ),
        (
            Color::rgb_f32(0.500002, 0.5, 0.5),
            Color::rgb(0, 0, 255),
            [0.05087609, 0.05087609, 0.52252156, 1.0],
        ),
        (
            // Computing B = 1 - W rounds W + B below one; hue must stay missing.
            Color::rgb_f32(-2.0_f32.powi(-53), -2.0_f32.powi(-53), 1e-8),
            Color::rgb(0, 0, 255),
            [-4.2965287e-18, -4.2965287e-18, 0.21404114, 1.0],
        ),
        (
            // Exact gray still has missing hue when W + B loses its unit offset.
            Color::rgba_f32(1e17, 1e17, 1e17, 1e-18),
            Color::rgb(0, 0, 255),
            [0.0050114132, 0.0050114132, 0.62138516, 0.5],
        ),
    ] {
        for (first, last) in [(neutral, other), (other, neutral)] {
            let interpolate = prepare_cylindrical_interpolation(
                &first,
                &last,
                CylindricalColorSpace::Hwb,
                HueInterpolationMethod::Shorter,
            );
            // CSS transfer-function references evaluated in double precision and rounded to f32.
            assert_eq!(interpolate(0.5, 0.5), midpoint);
        }
    }
}

#[test]
fn short_segments_prepare_colors_only_when_the_segment_changes() {
    let first = Color::rgb(255, 0, 0);
    let middle = Color::rgb(0, 0, 255);
    let last = Color::rgb(0, 255, 0);
    let common = GradientCommonDesc::new([
        GradientStop::at_position(GradientStopOffset::linear_radial(0.0), first),
        GradientStop::at_position(GradientStopOffset::linear_radial(0.9e-6), middle),
        GradientStop::at_position(GradientStopOffset::linear_radial(1.8e-6), last),
    ]);
    let normalized = NormalizedGradient::from_common(&common, GradientKind::Linear);
    let preparations = Cell::new(0);
    let ramp = bake_segments(&normalized, |_, end| {
        preparations.set(preparations.get() + 1);
        let color = end.to_linear_premultiplied_rgba();
        move |_, _| color
    });
    assert!(
        preparations.get() <= normalized.segments.len(),
        "{} preparations for {} segments",
        preparations.get(),
        normalized.segments.len()
    );
    assert_eq!(ramp.as_slice()[0], first.normalize());
    assert_eq!(ramp.as_slice()[1], middle.normalize());
    assert_eq!(ramp.as_slice()[RAMP_RESOLUTION / 2], last.normalize());
}

#[test]
fn srgb_linear_conversion_roundtrip_preserves_channel_values() {
    for channel in [0.0, 0.04045, 0.5, 1.0, -0.5] {
        let linear = color::srgb_to_linear(channel);
        let restored = color::linear_to_srgb(linear);
        assert!(
            (channel - restored).abs() < 1e-5,
            "roundtrip failed for {channel}: got {restored}"
        );
    }
}

#[test]
fn oklab_conversion_roundtrip_preserves_linear_rgb() {
    let [lightness, a, b] = color::linear_rgb_to_oklab(0.5, 0.3, 0.1);
    let [red, green, blue] = color::oklab_to_linear_rgb(lightness, a, b);
    assert!((red - 0.5).abs() < 1e-4);
    assert!((green - 0.3).abs() < 1e-4);
    assert!((blue - 0.1).abs() < 1e-4);
}

#[test]
fn hsl_green_converts_to_srgb_green() {
    let (red, green, blue) = color::hsl_to_srgb(120.0, 1.0, 0.5);
    assert!((red - 0.0).abs() < 1e-5);
    assert!((green - 1.0).abs() < 1e-5);
    assert!((blue - 0.0).abs() < 1e-5);
}

#[test]
fn polar_interpolation_preserves_borrowed_hues_and_opposite_hue_directions() {
    let red = Color::rgb_f32(1.0, 0.0, 0.0);
    let cyan = Color::rgb_f32(0.0, 1.0, 1.0);
    let gray = Color::rgb_f32(0.5, 0.5, 0.5);
    let blue = Color::rgb_f32(0.0, 0.0, 1.0);
    let cases = [
        (
            red,
            cyan,
            HueInterpolationMethod::Shorter,
            Color::rgb_f32(0.5, 1.0, 0.0),
        ),
        (
            cyan,
            red,
            HueInterpolationMethod::Shorter,
            Color::rgb_f32(0.5, 1.0, 0.0),
        ),
        (
            red,
            cyan,
            HueInterpolationMethod::Longer,
            Color::rgb_f32(0.5, 1.0, 0.0),
        ),
        (
            cyan,
            red,
            HueInterpolationMethod::Longer,
            Color::rgb_f32(0.5, 1.0, 0.0),
        ),
        (
            red,
            gray,
            HueInterpolationMethod::Decreasing,
            Color::rgb_f32(0.75, 0.25, 0.25),
        ),
        (
            gray,
            blue,
            HueInterpolationMethod::Decreasing,
            Color::rgb_f32(0.25, 0.25, 0.75),
        ),
    ];

    for space in [CylindricalColorSpace::Hsl, CylindricalColorSpace::Hwb] {
        for (start, end, method, expected) in cases {
            let interpolate = prepare_cylindrical_interpolation(&start, &end, space, method);
            let actual = interpolate(0.5, 0.5);
            let expected = expected.to_linear_premultiplied_rgba();
            for (actual, expected) in actual.into_iter().zip(expected) {
                assert!((actual - expected).abs() < 1e-6, "{space:?}, {method:?}");
            }
        }
    }
}

#[test]
fn interior_gray_stops_borrow_hue_independently_on_each_side() {
    for interpolation in [
        ColorInterpolation::Hsl {
            hue: HueInterpolationMethod::Shorter,
        },
        ColorInterpolation::Hwb {
            hue: HueInterpolationMethod::Shorter,
        },
    ] {
        let common = GradientCommonDesc::new([
            GradientStop::auto(Color::rgb_f32(1.0, 0.0, 0.0)),
            GradientStop::auto(Color::rgb_f32(0.5, 0.5, 0.5)),
            GradientStop::auto(Color::rgb_f32(0.0, 0.0, 1.0)),
        ])
        .with_interpolation(interpolation);
        let ramp = bake_gradient_ramp(&GradientRampSource {
            interpolation,
            normalized: NormalizedGradient::from_common(&common, GradientKind::Linear),
        });
        for (index, expected) in [
            (341, Color::rgb_f32(2.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)),
            (682, Color::rgb_f32(1.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0)),
        ] {
            for (actual, expected) in ramp.as_slice()[index].into_iter().zip(expected.normalize()) {
                assert!(
                    (actual - expected).abs() < 1e-6,
                    "{interpolation:?}, sample {index}"
                );
            }
        }
    }
}

#[test]
fn rectangular_ramps_preserve_extended_rgb_with_varying_alpha() {
    let opaque = Color::rgba_f32(1.5, -0.5, 0.25, 1.0);
    let translucent = Color::rgba_f32(1.5, -0.5, 0.25, 0.5);
    let [red, green, blue, _] = opaque.normalize();
    for interpolation in [
        ColorInterpolation::Srgb,
        ColorInterpolation::SrgbLinear,
        ColorInterpolation::Oklab,
    ] {
        let common =
            GradientCommonDesc::new([GradientStop::auto(opaque), GradientStop::auto(translucent)])
                .with_interpolation(interpolation);
        let ramp = bake_gradient_ramp(&GradientRampSource {
            interpolation,
            normalized: NormalizedGradient::from_common(&common, GradientKind::Linear),
        });
        for index in [0, 341, 682, RAMP_RESOLUTION - 1] {
            let alpha = 1.0 - 0.5 * index as f32 / (RAMP_RESOLUTION - 1) as f32;
            let expected = [red * alpha, green * alpha, blue * alpha, alpha];
            for (actual, expected) in ramp.as_slice()[index].into_iter().zip(expected) {
                assert!(
                    (actual - expected).abs() < 1e-5,
                    "{interpolation:?}, sample {index}"
                );
            }
        }
    }
}

#[test]
fn oklab_ramps_preserve_hdr_green_without_visible_channel_leakage() {
    for (start, end, expected_green) in [
        (80.0, 100.0, [39_309.844, 46_967.71]),
        (4000.0, 5000.0, [469223584.0, 560692032.0]),
    ] {
        let common = GradientCommonDesc::new([
            GradientStop::auto(Color::rgb_f32(0.0, start, 0.0)),
            GradientStop::auto(Color::rgb_f32(0.0, end, 0.0)),
        ])
        .with_interpolation(ColorInterpolation::Oklab);
        let ramp = bake_gradient_ramp(&GradientRampSource {
            interpolation: common.interpolation,
            normalized: NormalizedGradient::from_common(&common, GradientKind::Linear),
        });
        for (index, texel) in ramp.as_slice().iter().enumerate() {
            let leakage = Color::rgb_f32(
                color::linear_to_srgb(texel[0].abs()),
                0.0,
                color::linear_to_srgb(texel[2].abs()),
            );
            assert_eq!(leakage.to_array(), [0, 0, 0, 255], "sample {index}");
            assert_eq!(texel[3], 1.0);
        }
        // CSS Oklab interpolation on a pure-green ray cubes the interpolated cube-root intensity.
        for (index, expected) in [341, 682].into_iter().zip(expected_green) {
            assert_eq!(ramp.as_slice()[index][1], expected, "sample {index}");
        }
    }
}

#[test]
fn hwb_ramps_preserve_extended_neutrals_and_extremal_channels() {
    let magnitude = 2.0_f32.powi(53);
    let midpoint_index = RAMP_RESOLUTION / 3;
    let midpoint_position = midpoint_index as f32;
    for (start, end, expected) in [
        (
            Color::rgb_f32(magnitude, magnitude, magnitude),
            Color::rgb_f32(-magnitude, -magnitude, -magnitude),
            [0.0, 0.0, 0.0, 1.0],
        ),
        (
            Color::rgb_f32(1.0, -magnitude, -magnitude),
            Color::rgb_f32(2.0, -magnitude, -magnitude),
            [2.5371552, -1.7187375e38, -1.7187375e38, 1.0],
        ),
    ] {
        let common = GradientCommonDesc::new([
            GradientStop::auto(start),
            GradientStop::at_position(
                GradientStopOffset::linear_radial(2.0 * midpoint_position),
                end,
            ),
            GradientStop::at_position(
                GradientStopOffset::linear_radial((RAMP_RESOLUTION - 1) as f32),
                Color::BLACK,
            ),
        ])
        .with_interpolation(ColorInterpolation::Hwb {
            hue: HueInterpolationMethod::Shorter,
        });
        let ramp = bake_gradient_ramp(&GradientRampSource {
            interpolation: common.interpolation,
            normalized: NormalizedGradient::from_common(&common, GradientKind::Linear),
        });
        assert_eq!(ramp.as_slice()[midpoint_index], expected);
    }
}

#[test]
fn hsl_ramps_follow_css_achromatic_conversion_rules() {
    let bake_ramp = |start_color, end_color| {
        let common = GradientCommonDesc::new([
            GradientStop::auto(start_color),
            GradientStop::auto(end_color),
        ])
        .with_interpolation(ColorInterpolation::Hsl {
            hue: HueInterpolationMethod::Shorter,
        });
        bake_gradient_ramp(&GradientRampSource {
            interpolation: common.interpolation,
            normalized: NormalizedGradient::from_common(&common, GradientKind::Linear),
        })
    };

    for (start_color, neutral_color) in [
        (Color::rgb_f32(2.0, 1.0, 0.0), Color::rgb_f32(1.0, 1.0, 1.0)),
        (
            Color::rgb_f32(-1.0, 0.0, 1.0),
            Color::rgb_f32(0.0, 0.0, 0.0),
        ),
        (
            Color::rgb_f32(1.5, 0.500001, 0.500001),
            Color::rgb_f32(1.0000005, 1.0000005, 1.0000005),
        ),
        (
            Color::rgb_f32(1.5, 0.499999, 0.499999),
            Color::rgb_f32(0.9999995, 0.9999995, 0.9999995),
        ),
        (
            Color::rgb_f32(1.5, 0.5000199, 0.5000199),
            Color::rgb_f32(1.00001, 1.00001, 1.00001),
        ),
        (
            Color::rgb_f32(0.5, -0.499999, -0.499999),
            Color::rgb_f32(0.0000005066395, 0.0000005066395, 0.0000005066395),
        ),
        (
            Color::rgb_f32(0.5, -0.500001, -0.500001),
            Color::rgb_f32(-0.0000005066395, -0.0000005066395, -0.0000005066395),
        ),
        (
            Color::rgb_f32(0.500002, 0.5, 0.5),
            Color::rgb_f32(0.500001, 0.500001, 0.500001),
        ),
    ] {
        for (actual, expected) in [
            (
                bake_ramp(start_color, start_color),
                bake_ramp(neutral_color, neutral_color),
            ),
            (
                bake_ramp(start_color, Color::rgb(0, 255, 0)),
                bake_ramp(neutral_color, Color::rgb(0, 255, 0)),
            ),
            (
                bake_ramp(Color::rgb(0, 255, 0), start_color),
                bake_ramp(Color::rgb(0, 255, 0), neutral_color),
            ),
        ] {
            for (index, (actual, expected)) in actual
                .as_slice()
                .iter()
                .zip(expected.as_slice())
                .enumerate()
                .skip(1)
                .take(RAMP_RESOLUTION - 2)
            {
                for (actual, expected) in actual.iter().zip(expected) {
                    assert!(
                        (actual - expected).abs() < 2e-6,
                        "color {start_color:?}, sample {index}: {actual} != {expected}"
                    );
                    assert!(
                        f16::from_f32(*actual).is_finite(),
                        "color {start_color:?}, sample {index}"
                    );
                }
            }
        }
    }
}

#[test]
fn hsl_ramps_preserve_authored_endpoints_despite_lossy_interpolation() {
    for (color, expected_channel) in [
        (Color::rgba_f32(1.5, 0.5, 0.5, 0.5), 0.5),
        (Color::rgba_f32(-0.5, 0.5, 0.5, 0.5), 0.0),
    ] {
        let common = GradientCommonDesc::new([
            GradientStop::at_position(GradientStopOffset::linear_radial(0.25), color),
            GradientStop::at_position(GradientStopOffset::linear_radial(0.75), color),
        ])
        .with_interpolation(ColorInterpolation::Hsl {
            hue: HueInterpolationMethod::Shorter,
        });
        let ramp = bake_gradient_ramp(&GradientRampSource {
            interpolation: common.interpolation,
            normalized: NormalizedGradient::from_common(&common, GradientKind::Linear),
        });
        for index in [0, RAMP_RESOLUTION - 1] {
            assert_eq!(
                ramp.as_slice()[index],
                color.to_linear_premultiplied_rgba(),
                "color {color:?}, sample {index}"
            );
        }
        assert_eq!(
            ramp.as_slice()[RAMP_RESOLUTION / 2],
            [expected_channel, expected_channel, expected_channel, 0.5]
        );
    }
}

#[test]
fn hsl_ramps_preserve_constant_extended_rgb_with_large_saturation() {
    let opaque = Color::rgb_f32(1.95, 0.050025, 0.050025);
    let [red, green, blue, _] = opaque.normalize();
    for start_alpha in [1.0, 0.25] {
        let common = GradientCommonDesc::new([
            GradientStop::auto(Color::rgba_f32(1.95, 0.050025, 0.050025, start_alpha)),
            GradientStop::auto(opaque),
        ])
        .with_interpolation(ColorInterpolation::Hsl {
            hue: HueInterpolationMethod::Shorter,
        });
        let ramp = bake_gradient_ramp(&GradientRampSource {
            interpolation: common.interpolation,
            normalized: NormalizedGradient::from_common(&common, GradientKind::Linear),
        });
        for (index, actual) in ramp.as_slice().iter().enumerate() {
            let weight = index as f32 / (RAMP_RESOLUTION - 1) as f32;
            let alpha = start_alpha * (1.0 - weight) + weight;
            let expected = [red * alpha, green * alpha, blue * alpha, alpha];
            for (actual, expected) in actual.iter().zip(expected) {
                assert!(
                    (actual - expected).abs() < 2e-6,
                    "alpha {start_alpha}, sample {index}: {actual} != {expected}"
                );
            }
        }
    }
}

#[test]
fn hsl_corrects_negative_saturation_without_changing_hwb_hues() {
    let start = Color::rgb_f32(2.0, 1.5, 1.5);
    let end = Color::rgb_f32(0.0, 0.0, 1.0);
    for hue_method in [
        HueInterpolationMethod::Shorter,
        HueInterpolationMethod::Longer,
        HueInterpolationMethod::Increasing,
        HueInterpolationMethod::Decreasing,
    ] {
        let follows_shorter_arc = matches!(
            hue_method,
            HueInterpolationMethod::Shorter | HueInterpolationMethod::Increasing
        );
        for (space, expected) in [
            (
                CylindricalColorSpace::Hsl,
                if follows_shorter_arc {
                    Color::rgb_f32(29.0 / 24.0, 9.0 / 8.0, 25.0 / 24.0)
                } else {
                    Color::rgb_f32(25.0 / 24.0, 9.0 / 8.0, 29.0 / 24.0)
                },
            ),
            (
                CylindricalColorSpace::Hwb,
                if matches!(
                    hue_method,
                    HueInterpolationMethod::Shorter | HueInterpolationMethod::Decreasing
                ) {
                    Color::rgb_f32(1.5, 0.75, 1.5)
                } else {
                    Color::rgb_f32(0.75, 1.5, 0.75)
                },
            ),
        ] {
            let interpolate = prepare_cylindrical_interpolation(&start, &end, space, hue_method);
            for (position, expected) in [(0.0, start), (0.5, expected), (1.0, end)] {
                let actual = interpolate(1.0 - position, position);
                let expected = expected.to_linear_premultiplied_rgba();
                for (actual, expected) in actual.into_iter().zip(expected) {
                    assert!(
                        (actual - expected).abs() < 2e-6,
                        "{space:?}, {hue_method:?}, {position}: {actual} != {expected}"
                    );
                }
            }
        }
    }
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
                color: Color::rgb_f32(1.0, 0.0, 0.0),
                hint_to_next_segment: None,
            },
            GradientStop {
                positions: GradientStopPositions::Single(GradientStopOffset::LinearRadial(0.5)),
                color: Color::rgb_f32(0.0, 0.0, 1.0),
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
        Color::rgb_f32(1.0, 0.0, 0.0).to_linear_premultiplied_rgba()
    );
    assert_eq!(
        ramp[transition_index],
        Color::rgb_f32(0.0, 0.0, 1.0).to_linear_premultiplied_rgba()
    );
}

#[test]
fn bake_gradient_ramp_interpolates_positive_spans() {
    let common = GradientCommonDesc {
        units: GradientUnits::Local,
        spread: SpreadMode::Pad,
        interpolation: ColorInterpolation::Srgb,
        stops: vec![
            GradientStop {
                positions: GradientStopPositions::Single(GradientStopOffset::LinearRadial(0.0)),
                color: Color::rgb_f32(1.0, 0.0, 0.0),
                hint_to_next_segment: None,
            },
            GradientStop {
                positions: GradientStopPositions::Single(GradientStopOffset::LinearRadial(
                    RESOLVED_DEGENERATE_EPSILON * 0.5,
                )),
                color: Color::rgb_f32(0.0, 0.0, 1.0),
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
    assert_eq!(ramp[0], [1.0, 0.0, 0.0, 1.0]);
    assert_eq!(ramp[RAMP_RESOLUTION - 1], [0.0, 0.0, 1.0, 1.0]);
    assert!(ramp[RAMP_RESOLUTION / 2][0] > 0.0);
    assert!(ramp[RAMP_RESOLUTION / 2][2] > 0.0);
}

#[test]
fn cylindrical_interpolation_premultiplies_non_hue_channels() {
    let color_a = Color::rgb_f32(1.0, 0.0, 0.0);
    let color_b = Color::rgba_f32(1.0, 1.0, 1.0, 0.0);

    let interpolate = prepare_cylindrical_interpolation(
        &color_a,
        &color_b,
        CylindricalColorSpace::Hsl,
        HueInterpolationMethod::Shorter,
    );
    let interpolated = interpolate(0.5, 0.5);

    let expected = Color::rgba_f32(1.0, 0.0, 0.0, 0.5).to_linear_premultiplied_rgba();

    assert_eq!(interpolated, expected);
}
