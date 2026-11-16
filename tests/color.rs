use grafo::core::gradient::types::GradientRamp;
use grafo::{
    Color, ColorError, ColorInterpolation, Fill, Gradient, GradientColor, GradientStop,
    HueComponent, HueInterpolationMethod, LinearGradientDesc, LinearGradientLine,
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

fn authored_hue_colors(degrees: f32) -> [GradientColor; 2] {
    [
        GradientColor::Hsl {
            hue: HueComponent::Degrees(degrees),
            saturation: 1.0,
            lightness: 0.5,
            alpha: 1.0,
        },
        GradientColor::Hwb {
            hue: HueComponent::Degrees(degrees),
            whiteness: 0.0,
            blackness: 0.0,
            alpha: 1.0,
        },
    ]
}

fn gradient_midpoint(colors: [GradientColor; 2], interpolation: ColorInterpolation) -> [f32; 4] {
    let gradient = Gradient::linear(
        LinearGradientDesc::new(
            LinearGradientLine {
                start: [0.0, 0.0],
                end: [1.0, 0.0],
            },
            colors.map(GradientStop::auto),
        )
        .with_interpolation(interpolation),
    )
    .unwrap();
    let ramp = match &gradient.data().ramp {
        GradientRamp::Pending(source) => source.bake(),
        ramp => ramp.clone(),
    };
    let samples = ramp.as_slice();
    let before = samples[(samples.len() - 1) / 2];
    let after = samples[samples.len() / 2];
    [
        (before[0] + after[0]) * 0.5,
        (before[1] + after[1]) * 0.5,
        (before[2] + after[2]) * 0.5,
        (before[3] + after[3]) * 0.5,
    ]
}

fn assert_gradient_midpoint(
    colors: [GradientColor; 2],
    interpolation: ColorInterpolation,
    expected: [f32; 4],
) {
    let actual = gradient_midpoint(colors, interpolation);
    // Saturated cusps fall between the two midpoint ramp texels.
    let channels_match = actual
        .into_iter()
        .zip(expected)
        .all(|(actual, expected)| (actual - expected).abs() < 0.004);
    assert!(
        channels_match,
        "{colors:?}, {interpolation:?}: {actual:?} != {expected:?}"
    );
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

#[test]
fn coincident_rgb_hues_do_not_make_directed_full_turns() {
    let encoded = [
        GradientColor::Srgb {
            red: 147.0 / 255.0,
            green: 187.0 / 255.0,
            blue: 227.0 / 255.0,
            alpha: 1.0,
        },
        GradientColor::Srgb {
            red: 19.0 / 255.0,
            green: 70.0 / 255.0,
            blue: 121.0 / 255.0,
            alpha: 1.0,
        },
    ];
    let linear = [
        Color::rgb(147, 187, 227).into(),
        Color::rgb(19, 70, 121).into(),
    ];
    for method in [
        HueInterpolationMethod::Increasing,
        HueInterpolationMethod::Decreasing,
    ] {
        for interpolation in [
            ColorInterpolation::Hsl { hue: method },
            ColorInterpolation::Hwb { hue: method },
        ] {
            let expected = gradient_midpoint(encoded, interpolation);
            assert!(
                expected[2] > expected[0] + 0.2,
                "{interpolation:?}: {expected:?}"
            );
            assert_channels_close(gradient_midpoint(linear, interpolation), expected);
        }
    }
}

#[test]
fn authored_hues_keep_their_arcs_near_coincident_and_antipodal_angles() {
    let [hsl_start, hwb_start] = authored_hue_colors(0.0);
    for (end_hue, method, expected_srgb) in [
        (
            180.0005,
            HueInterpolationMethod::Shorter,
            [0.5, 0.0, 1.0, 1.0],
        ),
        (
            180.0005,
            HueInterpolationMethod::Longer,
            [0.5, 1.0, 0.0, 1.0],
        ),
        (
            179.9995,
            HueInterpolationMethod::Shorter,
            [0.5, 1.0, 0.0, 1.0],
        ),
        (
            179.9995,
            HueInterpolationMethod::Longer,
            [0.5, 0.0, 1.0, 1.0],
        ),
        (
            0.0005,
            HueInterpolationMethod::Increasing,
            [1.0, 0.0, 0.0, 1.0],
        ),
        (
            0.0005,
            HueInterpolationMethod::Decreasing,
            [0.0, 1.0, 1.0, 1.0],
        ),
        (
            359.9995,
            HueInterpolationMethod::Increasing,
            [0.0, 1.0, 1.0, 1.0],
        ),
        (
            359.9995,
            HueInterpolationMethod::Decreasing,
            [1.0, 0.0, 0.0, 1.0],
        ),
    ] {
        let expected = Color::srgb(expected_srgb).unwrap().linear_rgba();
        let reversed_method = match method {
            HueInterpolationMethod::Increasing => HueInterpolationMethod::Decreasing,
            HueInterpolationMethod::Decreasing => HueInterpolationMethod::Increasing,
            method => method,
        };
        let [hsl_end, hwb_end] = authored_hue_colors(end_hue);
        let rgb_start = Color::rgb(255, 0, 0).into();
        for colors in [
            [hsl_start, hsl_end],
            [hsl_start, hwb_end],
            [hwb_start, hsl_end],
            [hwb_start, hwb_end],
            [rgb_start, hsl_end],
            [rgb_start, hwb_end],
        ] {
            for (colors, method) in [(colors, method), ([colors[1], colors[0]], reversed_method)] {
                assert_gradient_midpoint(colors, ColorInterpolation::Hsl { hue: method }, expected);
                assert_gradient_midpoint(colors, ColorInterpolation::Hwb { hue: method }, expected);
            }
        }
    }
}
