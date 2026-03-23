use std::f32::consts::TAU;

use super::sampling::color_to_final_linear_premultiplied;
use super::types::{
    GradientColor, GradientCommonDesc, GradientKind, GradientStopPositions,
};

/// A single normalized stop after CSS canonicalization.
#[derive(Debug, Clone)]
pub(crate) struct NormalizedStop {
    pub(crate) position: f32,
    pub(crate) color: GradientColor,
    /// If Some, the hint position between this stop and the next.
    pub(crate) hint: Option<f32>,
}

/// A pairwise interpolation segment between two normalized stops.
#[derive(Debug, Clone)]
pub(crate) struct NormalizedSegment {
    pub(crate) start_position: f32,
    pub(crate) end_position: f32,
    pub(crate) start_color: GradientColor,
    pub(crate) end_color: GradientColor,
    pub(crate) hint: Option<f32>,
}

/// The fully normalized gradient after CSS stop resolution.
#[derive(Debug, Clone)]
pub(crate) struct NormalizedGradient {
    pub(crate) stops: Vec<NormalizedStop>,
    pub(crate) segments: Vec<NormalizedSegment>,
    pub(crate) period_start: f32,
    pub(crate) period_len: f32,
    pub(crate) is_single_stop: bool,
    pub(crate) single_stop_color: Option<GradientColor>,
}

impl NormalizedGradient {
    pub(crate) fn from_common(common: &GradientCommonDesc, kind: GradientKind) -> Self {
        let is_conic = kind == GradientKind::Conic;

        // Single-stop extension
        if common.stops.len() == 1 {
            return NormalizedGradient {
                stops: vec![NormalizedStop {
                    position: 0.0,
                    color: common.stops[0].color,
                    hint: None,
                }],
                segments: Vec::new(),
                period_start: 0.0,
                period_len: 0.0,
                is_single_stop: true,
                single_stop_color: Some(common.stops[0].color),
            };
        }

        // Step 1-2: Expand double positions into paired stops
        struct AuthoredStop {
            color: GradientColor,
            raw_position: Option<f32>,
            hint_to_next_segment: Option<f32>,
        }

        let mut authored: Vec<AuthoredStop> = Vec::new();
        for stop in &common.stops {
            match &stop.positions {
                GradientStopPositions::Auto => {
                    authored.push(AuthoredStop {
                        color: stop.color,
                        raw_position: None,
                        hint_to_next_segment: stop
                            .hint_to_next_segment
                            .map(|h| h.value()),
                    });
                }
                GradientStopPositions::Single(offset) => {
                    authored.push(AuthoredStop {
                        color: stop.color,
                        raw_position: Some(offset.value()),
                        hint_to_next_segment: stop
                            .hint_to_next_segment
                            .map(|h| h.value()),
                    });
                }
                GradientStopPositions::Double(a, b) => {
                    // First stop of the pair: gets the hint
                    authored.push(AuthoredStop {
                        color: stop.color,
                        raw_position: Some(a.value()),
                        hint_to_next_segment: None,
                    });
                    // Second stop of the pair
                    authored.push(AuthoredStop {
                        color: stop.color,
                        raw_position: Some(b.value()),
                        hint_to_next_segment: stop
                            .hint_to_next_segment
                            .map(|h| h.value()),
                    });
                }
            }
        }

        // Step 4: Default first and last positions if omitted
        let default_end = if is_conic { TAU } else { 1.0 };

        if authored[0].raw_position.is_none() {
            authored[0].raw_position = Some(0.0);
        }
        let last_index = authored.len() - 1;
        if authored[last_index].raw_position.is_none() {
            authored[last_index].raw_position = Some(default_end);
        }

        // Step 5: Fill interior runs of omitted positions by even spacing
        let mut positions: Vec<Option<f32>> = authored.iter().map(|s| s.raw_position).collect();
        fill_implicit_positions(&mut positions);

        // Step 6: Monotonic fixup
        let mut resolved: Vec<f32> = positions.iter().map(|p| p.unwrap()).collect();
        for i in 1..resolved.len() {
            if resolved[i] < resolved[i - 1] {
                resolved[i] = resolved[i - 1];
            }
        }

        // Build normalized stops
        let mut stops: Vec<NormalizedStop> = resolved
            .iter()
            .enumerate()
            .map(|(i, &position)| NormalizedStop {
                position,
                color: authored[i].color,
                hint: None,
            })
            .collect();

        // Step 9: Validate and retain hints
        for i in 0..authored.len() {
            if let Some(hint_value) = authored[i].hint_to_next_segment {
                if i + 1 < stops.len() {
                    let p_i = stops[i].position;
                    let p_next = stops[i + 1].position;
                    if p_i < hint_value && hint_value < p_next {
                        stops[i].hint = Some(hint_value);
                    }
                    // Otherwise drop the hint
                }
            }
        }

        // Build segments
        let mut segments = Vec::with_capacity(stops.len().saturating_sub(1));
        for i in 0..stops.len() - 1 {
            segments.push(NormalizedSegment {
                start_position: stops[i].position,
                end_position: stops[i + 1].position,
                start_color: stops[i].color,
                end_color: stops[i + 1].color,
                hint: stops[i].hint,
            });
        }

        // Step 10: Derive repeating metadata
        let period_start = stops.first().unwrap().position;
        let period_end = stops.last().unwrap().position;
        let period_len = period_end - period_start;

        NormalizedGradient {
            stops,
            segments,
            period_start,
            period_len,
            is_single_stop: false,
            single_stop_color: None,
        }
    }

    /// The degenerate constant color: final linear premultiplied color of the last stop.
    pub(crate) fn degenerate_constant_color(&self) -> [f32; 4] {
        let color = if self.is_single_stop {
            self.single_stop_color.unwrap()
        } else {
            self.stops.last().unwrap().color
        };
        color_to_final_linear_premultiplied(&color)
    }
}

/// Fills contiguous runs of `None` positions by even spacing between the
/// nearest explicit neighbors.
fn fill_implicit_positions(positions: &mut [Option<f32>]) {
    let len = positions.len();
    let mut i = 0;
    while i < len {
        if positions[i].is_some() {
            i += 1;
            continue;
        }
        // Find the start of the run (the explicit position before it)
        let run_start = i;
        let left_value = positions[run_start - 1].unwrap();

        // Find the end of the run
        let mut run_end = run_start;
        while run_end < len && positions[run_end].is_none() {
            run_end += 1;
        }
        let right_value = positions[run_end].unwrap();

        let run_count = run_end - run_start;
        for j in 0..run_count {
            let fraction = (j + 1) as f32 / (run_count + 1) as f32;
            positions[run_start + j] = Some(left_value + fraction * (right_value - left_value));
        }

        i = run_end + 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gradient::types::{
        GradientColor, GradientCommonDesc, GradientStop, GradientStopOffset,
        GradientStopPositions, GradientUnits, SpreadMode, ColorInterpolation,
    };

    fn srgb_color(r: f32, g: f32, b: f32) -> GradientColor {
        GradientColor::Srgb {
            red: r,
            green: g,
            blue: b,
            alpha: 1.0,
        }
    }

    fn make_stop(color: GradientColor, position: Option<f32>) -> GradientStop {
        GradientStop {
            positions: match position {
                Some(p) => GradientStopPositions::Single(GradientStopOffset::LinearRadial(p)),
                None => GradientStopPositions::Auto,
            },
            color,
            hint_to_next_segment: None,
        }
    }

    #[test]
    fn test_two_stop_defaults() {
        let common = GradientCommonDesc {
            units: GradientUnits::Local,
            spread: SpreadMode::Pad,
            interpolation: ColorInterpolation::SrgbLinear,
            stops: vec![
                make_stop(srgb_color(1.0, 0.0, 0.0), None),
                make_stop(srgb_color(0.0, 0.0, 1.0), None),
            ],
        };

        let normalized = NormalizedGradient::from_common(&common, GradientKind::Linear);
        assert_eq!(normalized.stops.len(), 2);
        assert!((normalized.stops[0].position - 0.0).abs() < 1e-6);
        assert!((normalized.stops[1].position - 1.0).abs() < 1e-6);
        assert_eq!(normalized.segments.len(), 1);
    }

    #[test]
    fn test_implicit_interior_stops() {
        let common = GradientCommonDesc {
            units: GradientUnits::Local,
            spread: SpreadMode::Pad,
            interpolation: ColorInterpolation::SrgbLinear,
            stops: vec![
                make_stop(srgb_color(1.0, 0.0, 0.0), Some(0.0)),
                make_stop(srgb_color(0.0, 1.0, 0.0), None),
                make_stop(srgb_color(0.0, 0.0, 1.0), Some(1.0)),
            ],
        };

        let normalized = NormalizedGradient::from_common(&common, GradientKind::Linear);
        assert_eq!(normalized.stops.len(), 3);
        assert!((normalized.stops[1].position - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_monotonic_fixup() {
        let common = GradientCommonDesc {
            units: GradientUnits::Local,
            spread: SpreadMode::Pad,
            interpolation: ColorInterpolation::SrgbLinear,
            stops: vec![
                make_stop(srgb_color(1.0, 0.0, 0.0), Some(0.5)),
                make_stop(srgb_color(0.0, 1.0, 0.0), Some(0.2)),
                make_stop(srgb_color(0.0, 0.0, 1.0), Some(1.0)),
            ],
        };

        let normalized = NormalizedGradient::from_common(&common, GradientKind::Linear);
        // Second stop should be bumped to 0.5 (max with previous)
        assert!((normalized.stops[1].position - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_single_stop() {
        let common = GradientCommonDesc {
            units: GradientUnits::Local,
            spread: SpreadMode::Pad,
            interpolation: ColorInterpolation::SrgbLinear,
            stops: vec![make_stop(srgb_color(1.0, 0.0, 0.0), Some(0.5))],
        };

        let normalized = NormalizedGradient::from_common(&common, GradientKind::Linear);
        assert!(normalized.is_single_stop);
    }

    #[test]
    fn test_double_position_expansion() {
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
            ],
        };

        let normalized = NormalizedGradient::from_common(&common, GradientKind::Linear);
        assert_eq!(normalized.stops.len(), 3);
        assert!((normalized.stops[0].position - 0.2).abs() < 1e-6);
        assert!((normalized.stops[1].position - 0.5).abs() < 1e-6);
    }
}
