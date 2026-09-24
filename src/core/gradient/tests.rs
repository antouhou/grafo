use super::errors::GradientError;
use super::normalize::NormalizedGradient;
use super::sampling::bake_gradient_ramp;
use super::types::{
    ColorInterpolation, Gradient, GradientColor, GradientCommonDesc, GradientKind, GradientRamp,
    GradientRampSource, GradientStop, GradientStopOffset, LinearGradientDesc, LinearGradientLine,
    RadialGradientDesc, RadialGradientSize,
};
use crate::core::Color;

fn single_stop_common() -> GradientCommonDesc {
    GradientCommonDesc::new([GradientStop::auto(Color::rgb(255, 0, 0))])
        .with_interpolation(ColorInterpolation::SrgbLinear)
}

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

    assert!(matches!(gradient.data.ramp, GradientRamp::Pending(_)));
}
