use super::NormalizedGradient;
use crate::core::gradient::sampling;
use crate::core::gradient::types::{
    ColorInterpolation, GradientCommonDesc, GradientKind, GradientRampSource, GradientStop,
    GradientStopOffset, GradientStopPositions, GradientUnits, HueInterpolationMethod, SpreadMode,
    RAMP_RESOLUTION,
};
use crate::core::Color;

fn make_stop(color: Color, position: Option<f32>) -> GradientStop {
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
fn transition_hints_keep_normalized_color_stops_inline() {
    let stops = [0, 1, 2, 3, 4].map(|index| {
        let position = index as f32 / 4.0;
        let stop =
            GradientStop::at_position(GradientStopOffset::linear_radial(position), Color::WHITE);
        if index < 4 {
            stop.with_hint_to_next_segment(GradientStopOffset::linear_radial(position + 0.125))
        } else {
            stop
        }
    });
    let normalized =
        NormalizedGradient::from_common(&GradientCommonDesc::new(stops), GradientKind::Linear);
    assert_eq!(normalized.stops.len(), 5);
    assert!(!normalized.stops.spilled());
}

#[test]
fn single_double_position_stops_keep_longer_hue_interpolation() {
    for interpolation in [
        ColorInterpolation::Hsl {
            hue: HueInterpolationMethod::Longer,
        },
        ColorInterpolation::Hwb {
            hue: HueInterpolationMethod::Longer,
        },
    ] {
        let single = GradientCommonDesc::new([GradientStop::between_positions(
            GradientStopOffset::linear_radial(0.25),
            GradientStopOffset::linear_radial(0.75),
            Color::rgb(255, 0, 0),
        )])
        .with_interpolation(interpolation);
        let expanded = GradientCommonDesc::new([
            GradientStop::at_position(
                GradientStopOffset::linear_radial(0.25),
                Color::rgb(255, 0, 0),
            ),
            GradientStop::at_position(
                GradientStopOffset::linear_radial(0.75),
                Color::rgb(255, 0, 0),
            ),
        ])
        .with_interpolation(interpolation);
        let bake = |common: &GradientCommonDesc| {
            sampling::bake_gradient_ramp(&GradientRampSource {
                interpolation,
                normalized: NormalizedGradient::from_common(common, GradientKind::Linear),
            })
        };
        let single = bake(&single);
        let expanded = bake(&expanded);
        assert_eq!(single.as_slice().len(), expanded.as_slice().len());
        assert_eq!(single.as_slice(), expanded.as_slice());
        assert_eq!(single.as_slice()[RAMP_RESOLUTION / 2][2], 1.0);
    }
}

#[test]
fn positive_short_spans_preserve_both_endpoints_and_interpolation() {
    for (start, end) in [
        (0.25, 0.2500005),
        (0.5, f32::from_bits(0.5_f32.to_bits() + 1)),
    ] {
        let common = GradientCommonDesc::new([
            GradientStop::at_position(
                GradientStopOffset::linear_radial(start),
                Color::rgb(255, 0, 0),
            ),
            GradientStop::at_position(
                GradientStopOffset::linear_radial(end),
                Color::rgb(0, 0, 255),
            ),
        ])
        .with_interpolation(ColorInterpolation::SrgbLinear);
        let ramp = sampling::bake_gradient_ramp(&GradientRampSource {
            interpolation: common.interpolation,
            normalized: NormalizedGradient::from_common(&common, GradientKind::Linear),
        });
        let samples = ramp.as_slice();
        assert_eq!(samples[0], [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(samples[RAMP_RESOLUTION - 1], [0.0, 0.0, 1.0, 1.0]);
        assert!(samples[RAMP_RESOLUTION / 2][0] > 0.0);
        assert!(samples[RAMP_RESOLUTION / 2][2] > 0.0);
        assert_eq!(
            samples[RAMP_RESOLUTION / 3],
            [2.0 / 3.0, 0.0, 1.0 / 3.0, 1.0]
        );
    }
}

#[test]
fn hinted_intervals_with_tiny_endpoint_distances_remain_finite() {
    for spread in [SpreadMode::Pad, SpreadMode::Repeat] {
        let common = GradientCommonDesc::new([
            GradientStop::at_position(GradientStopOffset::linear_radial(-1.0), Color::BLACK)
                .with_hint_to_next_segment(GradientStopOffset::linear_radial(0.0)),
            GradientStop::auto(Color::WHITE),
            GradientStop::at_position(GradientStopOffset::linear_radial(2e-20), Color::WHITE),
        ])
        .with_spread(spread)
        .with_interpolation(ColorInterpolation::SrgbLinear);
        let ramp = sampling::bake_gradient_ramp(&GradientRampSource {
            interpolation: common.interpolation,
            normalized: NormalizedGradient::from_common(&common, GradientKind::Linear),
        });
        assert!(ramp
            .as_slice()
            .iter()
            .flatten()
            .all(|channel| channel.is_finite()));
        assert_eq!(
            ramp.as_slice()[RAMP_RESOLUTION / 2],
            Color::BLACK.normalize()
        );
        assert_eq!(
            ramp.as_slice()[RAMP_RESOLUTION - 1],
            Color::WHITE.normalize()
        );
    }
}

#[test]
fn hinted_intervals_preserve_the_half_color_at_the_authored_hint() {
    let common = GradientCommonDesc::new([
        GradientStop::at_position(GradientStopOffset::linear_radial(-1.0), Color::BLACK)
            .with_hint_to_next_segment(GradientStopOffset::linear_radial(0.0)),
        GradientStop::at_position(GradientStopOffset::linear_radial(1e-20), Color::WHITE),
        GradientStop::at_position(GradientStopOffset::linear_radial(2.0), Color::WHITE),
    ])
    .with_interpolation(ColorInterpolation::SrgbLinear);
    let ramp = sampling::bake_gradient_ramp(&GradientRampSource {
        interpolation: common.interpolation,
        normalized: NormalizedGradient::from_common(&common, GradientKind::Linear),
    });
    assert_eq!(ramp.as_slice()[341], [0.5, 0.5, 0.5, 1.0]);
}

#[test]
fn automatic_hsl_stops_preserve_exact_alignment_with_explicit_positions() {
    let pink = Color::rgb_f32(1.5, 0.5, 0.5);
    let mut stops = vec![GradientStop::at_position(
        GradientStopOffset::linear_radial(-63.0 / 128.0),
        Color::WHITE,
    )];
    stops.extend(
        (1..=9).map(|index| GradientStop::auto(if index == 7 { pink } else { Color::WHITE })),
    );
    stops.push(GradientStop::at_position(
        GradientStopOffset::linear_radial(27.0 / 128.0),
        Color::WHITE,
    ));
    stops.push(GradientStop::at_position(
        GradientStopOffset::linear_radial(126.0 / 128.0),
        Color::WHITE,
    ));
    let common = GradientCommonDesc::new(stops).with_interpolation(ColorInterpolation::Hsl {
        hue: HueInterpolationMethod::Shorter,
    });
    let normalized = NormalizedGradient::from_common(&common, GradientKind::Linear);
    assert_eq!(normalized.stops[7].position, 0.0);
    let ramp = sampling::bake_gradient_ramp(&GradientRampSource {
        interpolation: common.interpolation,
        normalized,
    });
    assert_eq!(ramp.as_slice()[341], pink.to_linear_premultiplied_rgba());
}

#[test]
fn strictly_interior_hints_keep_the_final_repeat_color() {
    let common = GradientCommonDesc::new([
        GradientStop::at_position(GradientStopOffset::linear_radial(-1.0), Color::BLACK)
            .with_hint_to_next_segment(GradientStopOffset::linear_radial(0.0)),
        GradientStop::at_position(GradientStopOffset::linear_radial(1e-20), Color::WHITE),
    ])
    .with_spread(SpreadMode::Repeat)
    .with_interpolation(ColorInterpolation::SrgbLinear);
    let normalized = NormalizedGradient::from_common(&common, GradientKind::Linear);
    assert_eq!(normalized.stops[0].hint, Some(0.0));
    let ramp = sampling::bake_gradient_ramp(&GradientRampSource {
        interpolation: common.interpolation,
        normalized,
    });
    assert_eq!(ramp.as_slice()[0], Color::BLACK.normalize());
    assert_eq!(
        ramp.as_slice()[RAMP_RESOLUTION - 1],
        Color::WHITE.normalize()
    );
}

#[test]
fn endpoint_hints_keep_their_limiting_interpolation_weights() {
    for (hint, expected) in [(0.0, Color::WHITE), (1.0, Color::BLACK)] {
        let common = GradientCommonDesc::new([
            GradientStop::at_position(GradientStopOffset::linear_radial(0.0), Color::BLACK)
                .with_hint_to_next_segment(GradientStopOffset::linear_radial(hint)),
            GradientStop::at_position(GradientStopOffset::linear_radial(1.0), Color::WHITE),
        ])
        .with_interpolation(ColorInterpolation::SrgbLinear);
        let ramp = sampling::bake_gradient_ramp(&GradientRampSource {
            interpolation: common.interpolation,
            normalized: NormalizedGradient::from_common(&common, GradientKind::Linear),
        });
        let samples = ramp.as_slice();
        assert_eq!(samples[0], Color::BLACK.normalize());
        assert_eq!(samples[RAMP_RESOLUTION - 1], Color::WHITE.normalize());
        assert!(samples[1..RAMP_RESOLUTION - 1]
            .iter()
            .all(|sample| *sample == expected.normalize()));
    }
}

#[test]
fn hints_participate_in_css_position_resolution_and_implicit_spacing() {
    let common = GradientCommonDesc::new([
        GradientStop::at_position(GradientStopOffset::linear_radial(0.0), Color::BLACK)
            .with_hint_to_next_segment(GradientStopOffset::linear_radial(0.6)),
        GradientStop::auto(Color::rgb(128, 128, 128)),
        GradientStop::at_position(GradientStopOffset::linear_radial(1.0), Color::WHITE),
    ]);
    let normalized = NormalizedGradient::from_common(&common, GradientKind::Linear);
    assert_eq!(normalized.stops[0].hint, Some(0.6));
    assert_eq!(normalized.stops[1].position, 0.8);

    let common = GradientCommonDesc::new([
        GradientStop::at_position(GradientStopOffset::linear_radial(0.0), Color::BLACK)
            .with_hint_to_next_segment(GradientStopOffset::linear_radial(1.5)),
        GradientStop::at_position(GradientStopOffset::linear_radial(1.0), Color::WHITE),
    ]);
    let normalized = NormalizedGradient::from_common(&common, GradientKind::Linear);
    assert_eq!(normalized.stops[0].hint, Some(1.5));
    assert_eq!(normalized.stops[1].position, 1.5);
}

#[test]
fn automatic_endpoint_stops_span_the_full_gradient() {
    let common = GradientCommonDesc {
        units: GradientUnits::Local,
        spread: SpreadMode::Pad,
        interpolation: ColorInterpolation::SrgbLinear,
        stops: vec![
            make_stop(Color::rgb_f32(1.0, 0.0, 0.0), None),
            make_stop(Color::rgb_f32(0.0, 0.0, 1.0), None),
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
            make_stop(Color::rgb_f32(1.0, 0.0, 0.0), None),
            make_stop(Color::rgb_f32(0.0, 0.0, 1.0), None),
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
            make_stop(Color::rgb_f32(1.0, 0.0, 0.0), Some(0.0)),
            make_stop(Color::rgb_f32(0.0, 1.0, 0.0), None),
            make_stop(Color::rgb_f32(0.0, 0.0, 1.0), Some(1.0)),
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
            make_stop(Color::rgb_f32(1.0, 0.0, 0.0), Some(0.5)),
            make_stop(Color::rgb_f32(0.0, 1.0, 0.0), Some(0.2)),
            make_stop(Color::rgb_f32(0.0, 0.0, 1.0), Some(1.0)),
        ]
        .into(),
    };

    let normalized = NormalizedGradient::from_common(&common, GradientKind::Linear);
    // The second stop cannot precede the first stop at 0.5.
    assert!((normalized.stops[1].position - 0.5).abs() < 1e-6);
}

#[test]
fn implicit_positions_stay_finite_between_extreme_finite_bounds() {
    let common = GradientCommonDesc::new([
        make_stop(Color::rgb(255, 0, 0), Some(-3e38)),
        make_stop(Color::rgb(0, 255, 0), None),
        make_stop(Color::rgb(0, 0, 255), Some(3e38)),
    ])
    .with_interpolation(ColorInterpolation::SrgbLinear);
    let normalized = NormalizedGradient::from_common(&common, GradientKind::Linear);
    assert_eq!(normalized.stops[1].position, 0.0);
    assert!(normalized
        .stops
        .iter()
        .all(|stop| stop.position.is_finite()));
    let ramp = sampling::bake_gradient_ramp(&GradientRampSource {
        interpolation: common.interpolation,
        normalized,
    });
    assert!(ramp
        .as_slice()
        .iter()
        .flatten()
        .all(|channel| channel.is_finite()));
    assert!(ramp.as_slice()[RAMP_RESOLUTION / 2][1] > 0.0);
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
                color: Color::rgb_f32(1.0, 0.0, 0.0),
                hint_to_next_segment: None,
            },
            make_stop(Color::rgb_f32(0.0, 0.0, 1.0), Some(1.0)),
        ]
        .into(),
    };

    let normalized = NormalizedGradient::from_common(&common, GradientKind::Linear);
    assert_eq!(normalized.stops.len(), 3);
    assert!((normalized.stops[0].position - 0.2).abs() < 1e-6);
    assert!((normalized.stops[1].position - 0.5).abs() < 1e-6);
}
