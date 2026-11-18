use super::errors::GradientError;
use super::normalize::NormalizedGradient;
use super::sampling::bake_gradient_ramp;
use super::types::{
    ColorInterpolation, ConicGradientDesc, Fill, Gradient, GradientCommonDesc, GradientKind,
    GradientRamp, GradientRampSource, GradientStop, GradientStopOffset, LinearGradientDesc,
    LinearGradientLine, RadialGradientDesc, RadialGradientSize, SpreadMode,
};
use crate::core::Color;
use std::f32::consts::TAU;

fn single_stop_common() -> GradientCommonDesc {
    GradientCommonDesc::new([GradientStop::auto(Color::rgb(255, 0, 0))])
        .with_interpolation(ColorInterpolation::SrgbLinear)
}

#[test]
fn degenerate_repeating_gradients_use_css_average_colors() {
    let stops = [
        GradientStop::at_position(GradientStopOffset::linear_radial(0.0), Color::BLACK),
        GradientStop::at_position(GradientStopOffset::linear_radial(1.0), Color::WHITE),
        GradientStop::at_position(
            GradientStopOffset::linear_radial(1.0),
            Color::rgb(255, 0, 0),
        ),
    ];
    let linear = Gradient::linear(
        LinearGradientDesc::new(
            LinearGradientLine {
                start: [0.0, 0.0],
                end: [0.0, 0.0],
            },
            stops.clone(),
        )
        .with_spread(SpreadMode::Repeat),
    )
    .unwrap();
    let radial = Gradient::radial(
        RadialGradientDesc::new(
            [0.0, 0.0],
            RadialGradientSize::ExplicitCircleRadius(0.0),
            stops,
        )
        .with_spread(SpreadMode::Repeat),
    )
    .unwrap();
    for gradient in [linear, radial] {
        assert_eq!(
            gradient.data().ramp.as_slice(),
            &[[0.21404114, 0.21404114, 0.21404114, 1.0]]
        );
    }
    let transparent = Gradient::linear(
        LinearGradientDesc::new(
            LinearGradientLine {
                start: [0.0, 0.0],
                end: [0.0, 0.0],
            },
            [
                GradientStop::auto(Color::from_srgba_f32(1.0, 0.0, 0.0, 0.0)),
                GradientStop::auto(Color::rgb(0, 0, 255)),
            ],
        )
        .with_spread(SpreadMode::Repeat),
    )
    .unwrap();
    assert_eq!(transparent.data().ramp.as_slice(), &[[0.0, 0.0, 0.5, 0.5]]);
    let conic = Gradient::conic(
        ConicGradientDesc::new(
            [0.0, 0.0],
            0.0,
            [
                GradientStop::at_position(
                    GradientStopOffset::conic_radians(0.0),
                    Color::rgb(255, 0, 0),
                ),
                GradientStop::at_position(GradientStopOffset::conic_radians(0.0), Color::WHITE),
                GradientStop::at_position(
                    GradientStopOffset::conic_radians(0.0),
                    Color::rgb(0, 0, 255),
                ),
            ],
        )
        .with_spread(SpreadMode::Repeat),
    )
    .unwrap();
    assert_eq!(
        conic.data().ramp.as_slice(),
        &[[0.52252156, 0.21404114, 0.52252156, 1.0]]
    );
}

#[test]
fn degenerate_repeat_average_retains_signed_hdr_cancellation() {
    let extreme = 2.0_f32.powi(53);
    for sign in [-1.0, 1.0] {
        let stops = [extreme, sign, -extreme].map(|red| {
            GradientStop::at_position(
                GradientStopOffset::linear_radial(0.0),
                Color::from_srgb_f32(red, 0.0, 0.0),
            )
        });
        let gradient = Gradient::linear(
            LinearGradientDesc::new(
                LinearGradientLine {
                    start: [0.0, 0.0],
                    end: [1.0, 0.0],
                },
                stops,
            )
            .with_spread(SpreadMode::Repeat),
        )
        .unwrap();

        // CSS Images 3 gives encoded red 0.25*extreme + 0.5*sign - 0.25*extreme.
        assert_eq!(
            gradient.data().ramp.as_slice(),
            &[[0.21404114 * sign, 0.0, 0.0, 1.0]]
        );
    }
}

#[test]
fn degenerate_repeat_average_preserves_interval_ratios_before_division() {
    let extreme = 2.0_f32.powi(54);
    for sign in [-1.0, 1.0] {
        let stops = [(0.0, -5.0 * extreme), (1.0, 0.0), (6.0, extreme)].map(|(position, red)| {
            GradientStop::at_position(
                GradientStopOffset::linear_radial(position),
                Color::from_srgb_f32(red * sign, 0.0, 0.0),
            )
        });
        let gradient = Gradient::linear(
            LinearGradientDesc::new(
                LinearGradientLine {
                    start: [0.0, 0.0],
                    end: [0.0, 0.0],
                },
                stops,
            )
            .with_spread(SpreadMode::Repeat),
        )
        .unwrap();
        assert_eq!(gradient.data().ramp.as_slice(), &[[0.0, 0.0, 0.0, 1.0]]);
    }
}

#[test]
fn repeating_conic_gradients_resolve_skia_sweep_cutoff_before_sampling() {
    // Adjacent f32 radians straddle Skia's 1 / 32768 degree sweep cutoff.
    for (start, end, is_constant) in [
        (0.0, f32::from_bits(0x350e_fa35), true),
        (0.0, f32::from_bits(0x350e_fa36), false),
        (2.0, f32::from_bits(0x4000_0002), true),
        (
            f32::from_bits(0x3e80_001b),
            f32::from_bits(0x3e80_002d),
            false,
        ),
        (0.25, 0.250002, false),
    ] {
        let gradient = Gradient::conic(
            ConicGradientDesc::new(
                [0.0, 0.0],
                0.0,
                [
                    GradientStop::at_position(
                        GradientStopOffset::conic_radians(start),
                        Color::rgb(255, 0, 0),
                    ),
                    GradientStop::at_position(
                        GradientStopOffset::conic_radians(end),
                        Color::rgb(0, 0, 255),
                    ),
                ],
            )
            .with_spread(SpreadMode::Repeat)
            .with_interpolation(ColorInterpolation::SrgbLinear),
        )
        .unwrap();
        assert_eq!(gradient.data().period_start, start / TAU);
        assert_eq!(
            gradient.data().period_len,
            ((f64::from(end) - f64::from(start)) / f64::from(TAU)) as f32
        );
        if is_constant {
            assert!(matches!(gradient.data().ramp, GradientRamp::Constant(..)));
            assert_eq!(
                gradient.data().ramp.as_slice(),
                &[[0.21404114, 0.0, 0.21404114, 1.0]]
            );
        } else {
            let GradientRamp::Pending(source) = &gradient.data().ramp else {
                panic!("a usable conic interval must retain its interpolation source");
            };
            let ramp = source.bake();
            assert_eq!(ramp.as_slice().first().unwrap(), &[1.0, 0.0, 0.0, 1.0]);
            assert_eq!(ramp.as_slice().last().unwrap(), &[0.0, 0.0, 1.0, 1.0]);
        }
    }
}

#[test]
fn conic_ramp_baking_preserves_distinct_authored_angles() {
    for (start_bits, end_bits) in [(0x40c9_0fdc, 0x40c9_0fde), (0x4049_0fdc, 0x4049_0fdf)] {
        let gradient = Gradient::conic(
            ConicGradientDesc::new(
                [0.0, 0.0],
                0.0,
                [
                    (start_bits, Color::BLACK),
                    (start_bits + 1, Color::WHITE),
                    (end_bits, Color::WHITE),
                ]
                .map(|(bits, color)| {
                    GradientStop::at_position(
                        GradientStopOffset::conic_radians(f32::from_bits(bits)),
                        color,
                    )
                }),
            )
            .with_spread(SpreadMode::Repeat)
            .with_interpolation(ColorInterpolation::SrgbLinear),
        )
        .unwrap();
        let GradientRamp::Pending(source) = &gradient.data().ramp else {
            panic!("the conic period exceeds Skia's sweep cutoff");
        };
        let ramp = source.bake();
        assert_eq!(ramp.as_slice()[0], [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(ramp.as_slice()[682], [1.0; 4]);
    }
}

#[test]
fn single_authored_color_stops_preserve_linear_premultiplied_color() {
    let color = Color::from_srgba_f32(1.0, 0.0, 0.0, 0.5);
    for stop in [
        GradientStop::auto(color),
        GradientStop::at_position(GradientStopOffset::LinearRadial(0.5), color),
        GradientStop::between_positions(
            GradientStopOffset::LinearRadial(0.2),
            GradientStopOffset::LinearRadial(0.8),
            color,
        ),
    ] {
        let common =
            GradientCommonDesc::new([stop]).with_interpolation(ColorInterpolation::SrgbLinear);
        let normalized = NormalizedGradient::from_common(&common, GradientKind::Linear);
        assert_eq!(normalized.degenerate_constant_color(), [0.5, 0.0, 0.0, 0.5]);

        let ramp = bake_gradient_ramp(&GradientRampSource {
            interpolation: common.interpolation,
            normalized,
        });
        for texel in ramp.as_slice() {
            assert_eq!(*texel, [0.5, 0.0, 0.0, 0.5]);
        }
    }
}

#[test]
fn float_color_stops_preserve_the_solid_fill_components() {
    let color = Color::from_srgba_f32(0.123_456, -0.5, 1.5, 0.25);
    let [red, green, blue, alpha] = Fill::Solid(color).to_solid_linear_rgba().unwrap();
    let gradient = Gradient::linear(LinearGradientDesc::new(
        LinearGradientLine {
            start: [0.0, 0.0],
            end: [10.0, 0.0],
        },
        [GradientStop::auto(color)],
    ))
    .unwrap();

    assert_eq!(
        gradient.data().ramp.as_slice(),
        &[[red * alpha, green * alpha, blue * alpha, alpha]]
    );
}

#[test]
fn transparent_solid_accessor_preserves_straight_rgb() {
    let fill = Fill::Solid(Color::rgba(255, 0, 0, 0));
    assert_eq!(fill.to_solid_linear_rgba(), Some([1.0, 0.0, 0.0, 0.0]));
}

#[test]
fn gradients_reject_nonfinite_color_channels() {
    for (channel_index, component) in ["red", "green", "blue", "alpha"].into_iter().enumerate() {
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut channels = [0.2, 0.3, 0.4, 1.0];
            channels[channel_index] = invalid;
            let [red, green, blue, alpha] = channels;
            let color = Color::from_srgba_f32(red, green, blue, alpha);
            let gradient = Gradient::linear(LinearGradientDesc::new(
                LinearGradientLine {
                    start: [0.0, 0.0],
                    end: [10.0, 0.0],
                },
                [GradientStop::auto(color)],
            ));
            assert!(matches!(
                gradient,
                Err(GradientError::NonFiniteColorComponent {
                    stop_index: 0,
                    component: actual,
                }) if actual == component
            ));
        }
    }
}

#[test]
fn radial_rejects_negative_circle_radius() {
    let gradient = Gradient::radial(RadialGradientDesc {
        common: single_stop_common(),
        center: [50.0, 50.0],
        size: RadialGradientSize::ExplicitCircleRadius(-1.0),
    });

    assert!(matches!(
        gradient,
        Err(GradientError::InvalidRadialDefinition)
    ));
}

#[test]
fn radial_rejects_negative_ellipse_radius() {
    let gradient = Gradient::radial(RadialGradientDesc {
        common: single_stop_common(),
        center: [50.0, 50.0],
        size: RadialGradientSize::ExplicitEllipseRadii {
            radius_x: 20.0,
            radius_y: -1.0,
        },
    });

    assert!(matches!(
        gradient,
        Err(GradientError::InvalidRadialDefinition)
    ));
}

#[test]
fn nonconstant_gradients_start_with_pending_ramp() {
    let gradient = Gradient::linear(
        LinearGradientDesc::new(
            LinearGradientLine {
                start: [0.0, 0.0],
                end: [10.0, 0.0],
            },
            [
                GradientStop::at_position(
                    GradientStopOffset::linear_radial(0.0),
                    Color::rgb(255, 0, 0),
                ),
                GradientStop::at_position(
                    GradientStopOffset::linear_radial(1.0),
                    Color::rgb(0, 0, 255),
                ),
            ],
        )
        .with_interpolation(ColorInterpolation::SrgbLinear),
    )
    .unwrap();

    assert!(matches!(gradient.data().ramp, GradientRamp::Pending(_)));
}
