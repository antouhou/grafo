use grafo::core::gradient::types::GradientRamp;
use grafo::{
    Color, ColorError, ColorInterpolation, Fill, Gradient, GradientStop, LinearGradientDesc,
    LinearGradientLine,
};

type ColorConstructor = fn([f32; 4]) -> Result<Color, ColorError>;

fn assert_channels_close(actual: [f32; 4], expected: [f32; 4]) {
    for (actual, expected) in actual.into_iter().zip(expected) {
        let tolerance = 1e-6 * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() < tolerance,
            "{actual} != {expected}"
        );
    }
}

#[test]
fn source_colors_convert_to_linear_rgb_without_premultiplying_or_gamut_clipping() {
    assert_channels_close(
        Color::srgb([0.5, 0.5, 0.5, 0.25]).unwrap().linear_rgba(),
        [0.214_041_14, 0.214_041_14, 0.214_041_14, 0.25],
    );
    assert_channels_close(
        Color::rgba(128, 128, 128, 64).linear_rgba(),
        [0.215_860_5, 0.215_860_5, 0.215_860_5, 64.0 / 255.0],
    );
    assert_channels_close(
        Color::display_p3([1.0, 0.0, 0.0, 0.5])
            .unwrap()
            .linear_rgba(),
        [1.224_940_2, -0.042_056_955, -0.019_637_555, 0.5],
    );
    assert_channels_close(
        Color::display_p3([0.5, 0.5, 0.5, 0.25])
            .unwrap()
            .linear_rgba(),
        [0.214_041_14, 0.214_041_14, 0.214_041_14, 0.25],
    );
    assert_channels_close(
        Color::srgb([-0.5, 0.0, 2.0, 1.0]).unwrap().linear_rgba(),
        [-0.214_041_14, 0.0, 4.953_846, 1.0],
    );
}

#[test]
fn linear_colors_reach_solid_and_gradient_paints_without_encoding_or_clipping() {
    let rgba = [4.0, -0.125, 0.25, 0.5];
    let color = Color::linear_srgb(rgba).unwrap();
    assert_eq!(Fill::Solid(color).solid_linear_rgba(), Some(rgba));

    let line = LinearGradientLine {
        start: [0.0, 0.0],
        end: [10.0, 0.0],
    };
    let constant =
        Gradient::linear(LinearGradientDesc::new(line, [GradientStop::auto(color)])).unwrap();
    assert_eq!(
        constant.data().ramp.as_slice(),
        &[[2.0, -0.0625, 0.125, 0.5]]
    );

    let end = Color::linear_srgb([-2.0, 1.0, 3.0, 0.75]).unwrap();
    let gradient = Gradient::linear(
        LinearGradientDesc::new(line, [GradientStop::auto(color), GradientStop::auto(end)])
            .with_interpolation(ColorInterpolation::SrgbLinear),
    )
    .unwrap();
    let GradientRamp::Pending(source) = &gradient.data().ramp else {
        panic!("two different colors must produce a sampled ramp");
    };
    let ramp = source.bake();
    assert_eq!(ramp.as_slice()[0], [2.0, -0.0625, 0.125, 0.5]);
    assert_eq!(ramp.as_slice().last(), Some(&[-1.5, 0.75, 2.25, 0.75]));
}

#[test]
fn float_constructors_reject_nonfinite_rgb_and_invalid_alpha() {
    let constructors: [ColorConstructor; 3] = [Color::srgb, Color::display_p3, Color::linear_srgb];
    for constructor in constructors {
        for channel in 0..3 {
            for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                let mut rgba = [0.5, 0.5, 0.5, 1.0];
                rgba[channel] = invalid;
                assert_eq!(constructor(rgba), Err(ColorError::NonFiniteRgb));
            }
        }
        for alpha in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.1, 1.1] {
            assert_eq!(
                constructor([0.5, 0.5, 0.5, alpha]),
                Err(ColorError::InvalidAlpha),
            );
        }
    }
    assert_eq!(
        Color::srgb([f32::MAX, 0.0, 0.0, 1.0]),
        Err(ColorError::NonFiniteRgb),
    );
    assert_eq!(
        Color::display_p3([f32::MAX, 0.0, 0.0, 1.0]),
        Err(ColorError::NonFiniteRgb),
    );
}
