use super::{GpuGradientColorParams, GradientCache};
use crate::core::gradient::types::{
    ColorInterpolation, ConicGradientDesc, Gradient, GradientColor, GradientRamp, GradientStop,
    GradientStopOffset, GradientUnits, LinearGradientDesc, LinearGradientLine, RadialGradientDesc,
    RadialGradientSize, SpreadMode,
};
use crate::core::Color;
use std::f32::consts::{FRAC_PI_2, PI, TAU};
use std::sync::Arc;

#[test]
fn gradient_ramps_are_reused_within_each_cache() {
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
    let mut first = Gradient::linear(descriptor.clone()).unwrap();
    let mut second = Gradient::linear(descriptor.clone()).unwrap();
    let mut independent = Gradient::linear(descriptor).unwrap();
    let mut cache = GradientCache::new();
    let mut independent_cache = GradientCache::new();

    let GradientRamp::Sampled(first_ramp) = cache.get_or_create_ramp(&mut first.data) else {
        panic!("expected sampled ramp");
    };
    let GradientRamp::Sampled(second_ramp) = cache.get_or_create_ramp(&mut second.data) else {
        panic!("expected sampled ramp");
    };
    let GradientRamp::Sampled(independent_ramp) =
        independent_cache.get_or_create_ramp(&mut independent.data)
    else {
        panic!("expected sampled ramp");
    };

    assert!(Arc::ptr_eq(&first_ramp, &second_ramp));
    assert!(!Arc::ptr_eq(&first_ramp, &independent_ramp));
    assert_eq!(first_ramp, independent_ramp);
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
    let params = GpuGradientColorParams::from_gradient_data(&gradient.data);
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
            panic!("a degenerate radial gradient must have a constant ramp");
        };
        assert_eq!(color, [0.0, 0.25, 0.0, 0.25]);
        let params = GpuGradientColorParams::from_gradient_data(&gradient.data);
        assert_eq!(params.is_constant, 1);
        assert_eq!(params.constant_color, color);
    }
}
