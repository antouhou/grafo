use super::NormalizedGradient;
use crate::gradient::types::{
    ColorInterpolation, GradientColor, GradientCommonDesc, GradientKind, GradientStop,
    GradientStopOffset, GradientStopPositions, GradientUnits, SpreadMode,
};

fn srgb_color(red: f32, green: f32, blue: f32) -> GradientColor {
    GradientColor::Srgb {
        red,
        green,
        blue,
        alpha: 1.0,
    }
}

fn make_stop(color: GradientColor, position: Option<f32>) -> GradientStop {
    GradientStop {
        positions: match position {
            Some(position) => {
                GradientStopPositions::Single(GradientStopOffset::LinearRadial(position))
            }
            None => GradientStopPositions::Auto,
        },
        color,
        hint_to_next_segment: None,
    }
}

#[test]
fn automatic_endpoint_stops_span_the_full_gradient() {
    let common = GradientCommonDesc {
        units: GradientUnits::Local,
        spread: SpreadMode::Pad,
        interpolation: ColorInterpolation::SrgbLinear,
        stops: vec![
            make_stop(srgb_color(1.0, 0.0, 0.0), None),
            make_stop(srgb_color(0.0, 0.0, 1.0), None),
        ]
        .into(),
    };

    let normalized = NormalizedGradient::from_common(&common, GradientKind::Linear);
    assert_eq!(normalized.stops.len(), 2);
    assert!((normalized.stops[0].position - 0.0).abs() < 1e-6);
    assert!((normalized.stops[1].position - 1.0).abs() < 1e-6);
    assert_eq!(normalized.segments.len(), 1);
}

#[test]
fn two_stop_normalization_keeps_storage_inline() {
    let common = GradientCommonDesc {
        units: GradientUnits::Local,
        spread: SpreadMode::Pad,
        interpolation: ColorInterpolation::SrgbLinear,
        stops: vec![
            make_stop(srgb_color(1.0, 0.0, 0.0), None),
            make_stop(srgb_color(0.0, 0.0, 1.0), None),
        ]
        .into(),
    };

    let normalized = NormalizedGradient::from_common(&common, GradientKind::Linear);
    assert!(!normalized.stops.spilled());
    assert!(!normalized.segments.spilled());
}

#[test]
fn implicit_interior_stops_are_evenly_spaced() {
    let common = GradientCommonDesc {
        units: GradientUnits::Local,
        spread: SpreadMode::Pad,
        interpolation: ColorInterpolation::SrgbLinear,
        stops: vec![
            make_stop(srgb_color(1.0, 0.0, 0.0), Some(0.0)),
            make_stop(srgb_color(0.0, 1.0, 0.0), None),
            make_stop(srgb_color(0.0, 0.0, 1.0), Some(1.0)),
        ]
        .into(),
    };

    let normalized = NormalizedGradient::from_common(&common, GradientKind::Linear);
    assert_eq!(normalized.stops.len(), 3);
    assert!((normalized.stops[1].position - 0.5).abs() < 1e-6);
}

#[test]
fn decreasing_stop_positions_are_clamped_to_the_previous_stop() {
    let common = GradientCommonDesc {
        units: GradientUnits::Local,
        spread: SpreadMode::Pad,
        interpolation: ColorInterpolation::SrgbLinear,
        stops: vec![
            make_stop(srgb_color(1.0, 0.0, 0.0), Some(0.5)),
            make_stop(srgb_color(0.0, 1.0, 0.0), Some(0.2)),
            make_stop(srgb_color(0.0, 0.0, 1.0), Some(1.0)),
        ]
        .into(),
    };

    let normalized = NormalizedGradient::from_common(&common, GradientKind::Linear);
    // The second stop cannot precede the first stop at 0.5.
    assert!((normalized.stops[1].position - 0.5).abs() < 1e-6);
}

#[test]
fn double_position_stops_expand_into_two_normalized_stops() {
    let common = GradientCommonDesc {
        units: GradientUnits::Local,
        spread: SpreadMode::Pad,
        interpolation: ColorInterpolation::SrgbLinear,
        stops: vec![
            GradientStop {
                positions: GradientStopPositions::Double(
                    GradientStopOffset::LinearRadial(0.2),
                    GradientStopOffset::LinearRadial(0.5),
                ),
                color: srgb_color(1.0, 0.0, 0.0),
                hint_to_next_segment: None,
            },
            make_stop(srgb_color(0.0, 0.0, 1.0), Some(1.0)),
        ]
        .into(),
    };

    let normalized = NormalizedGradient::from_common(&common, GradientKind::Linear);
    assert_eq!(normalized.stops.len(), 3);
    assert!((normalized.stops[0].position - 0.2).abs() < 1e-6);
    assert!((normalized.stops[1].position - 0.5).abs() < 1e-6);
}
