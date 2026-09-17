use super::gpu::GpuGradientColorParams;
use super::normalize::NormalizedGradient;
use super::sampling::bake_gradient_ramp;
use super::types::{
    ConicGradientDesc, Gradient, GradientColor, GradientCommonDesc, GradientKind, GradientRamp,
    GradientRampSource, GradientStop, GradientStopOffset, GradientUnits, LinearGradientDesc,
    LinearGradientLine, SpreadMode,
};
use crate::Color;
use std::f32::consts::{FRAC_PI_2, PI, TAU};

#[test]
fn single_stop_bakes_a_constant_premultiplied_color_for_any_position() {
    let color = GradientColor::Srgb {
        red: 1.0,
        green: 0.0,
        blue: 0.0,
        alpha: 0.5,
    };
    for stop in [
        GradientStop::auto(color),
        GradientStop::at_position(GradientStopOffset::LinearRadial(0.5), color),
        GradientStop::between_positions(
            GradientStopOffset::LinearRadial(0.2),
            GradientStopOffset::LinearRadial(0.8),
            color,
        ),
    ] {
        let common = GradientCommonDesc::new([stop]);
        let normalized = NormalizedGradient::from_common(&common, GradientKind::Linear);
        assert_eq!(normalized.degenerate_constant_color(), [0.5, 0.0, 0.0, 0.5]);

        let ramp = bake_gradient_ramp(&GradientRampSource {
            interpolation: common.interpolation,
            normalized,
        });
        assert_eq!(ramp.as_slice(), &[[0.5, 0.0, 0.0, 0.5]]);
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
            GradientStop::auto(GradientColor::Srgb {
                red: 1.0,
                green: 0.0,
                blue: 0.0,
                alpha: 1.0,
            }),
            GradientStop::auto(GradientColor::Srgb {
                red: 0.0,
                green: 1.0,
                blue: 0.0,
                alpha: 0.25,
            }),
        ],
    ))
    .unwrap();

    let GradientRamp::Constant(color) = gradient.data.ramp else {
        panic!("a degenerate gradient must have a constant ramp");
    };
    assert_eq!(color, [0.0, 0.25, 0.0, 0.25]);
    assert_eq!(gradient.data.constant_color, color);
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

    let params = GpuGradientColorParams::from_gradient_data(&gradient.data);
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
