use std::f32::consts::TAU;

use smallvec::{smallvec, SmallVec};

use super::sampling::color_to_final_linear_premultiplied;
use super::types::{GradientColor, GradientCommonDesc, GradientKind, GradientStopPositions};

const NORMALIZED_INLINE_STOP_CAPACITY: usize = 8;
const NORMALIZED_INLINE_SEGMENT_CAPACITY: usize = 8;

#[derive(Debug, Clone, Copy)]
struct AuthoredStop {
    color: GradientColor,
    raw_position: Option<f32>,
    hint_to_next_segment: Option<f32>,
}

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
    pub(crate) stops: SmallVec<[NormalizedStop; NORMALIZED_INLINE_STOP_CAPACITY]>,
    pub(crate) segments: SmallVec<[NormalizedSegment; NORMALIZED_INLINE_SEGMENT_CAPACITY]>,
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
                stops: smallvec![NormalizedStop {
                    position: 0.0,
                    color: common.stops[0].color,
                    hint: None,
                }],
                segments: SmallVec::new(),
                period_start: 0.0,
                period_len: 0.0,
                is_single_stop: true,
                single_stop_color: Some(common.stops[0].color),
            };
        }

        // Step 1-2: Expand double positions into paired stops
        let mut authored: SmallVec<[AuthoredStop; NORMALIZED_INLINE_STOP_CAPACITY]> =
            SmallVec::with_capacity(common.stops.len() * 2);
        for stop in &common.stops {
            match &stop.positions {
                GradientStopPositions::Auto => {
                    authored.push(AuthoredStop {
                        color: stop.color,
                        raw_position: None,
                        hint_to_next_segment: stop.hint_to_next_segment.map(|h| h.value()),
                    });
                }
                GradientStopPositions::Single(offset) => {
                    authored.push(AuthoredStop {
                        color: stop.color,
                        raw_position: Some(offset.value()),
                        hint_to_next_segment: stop.hint_to_next_segment.map(|h| h.value()),
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
                        hint_to_next_segment: stop.hint_to_next_segment.map(|h| h.value()),
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

        // Step 5: Fill interior runs of omitted positions.
        fill_implicit_positions(&mut authored);

        // Build normalized stops
        let mut previous_position: Option<f32> = None;
        let mut stops: SmallVec<[NormalizedStop; NORMALIZED_INLINE_STOP_CAPACITY]> =
            SmallVec::with_capacity(authored.len());
        for authored_stop in &authored {
            let mut position = authored_stop
                .raw_position
                .expect("gradient stop positions should be resolved before normalization");
            if let Some(previous_position) = previous_position {
                if position < previous_position {
                    position = previous_position;
                }
            }

            stops.push(NormalizedStop {
                position,
                color: authored_stop.color,
                hint: None,
            });
            previous_position = Some(position);
        }

        // Step 9: Validate and retain hints
        for (stop_index, authored_stop) in authored.iter().enumerate() {
            if let Some(hint_value) = authored_stop.hint_to_next_segment {
                if stop_index + 1 < stops.len() {
                    let current_position = stops[stop_index].position;
                    let next_position = stops[stop_index + 1].position;
                    if current_position < hint_value && hint_value < next_position {
                        stops[stop_index].hint = Some(hint_value);
                    }
                    // Otherwise drop the hint
                }
            }
        }

        // Build segments
        let mut segments: SmallVec<[NormalizedSegment; NORMALIZED_INLINE_SEGMENT_CAPACITY]> =
            SmallVec::with_capacity(stops.len().saturating_sub(1));
        for (start_stop, end_stop) in stops.iter().zip(stops.iter().skip(1)) {
            segments.push(NormalizedSegment {
                start_position: start_stop.position,
                end_position: end_stop.position,
                start_color: start_stop.color,
                end_color: end_stop.color,
                hint: start_stop.hint,
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
fn fill_implicit_positions(authored_stops: &mut [AuthoredStop]) {
    let len = authored_stops.len();
    let mut i = 0;
    while i < len {
        if authored_stops[i].raw_position.is_some() {
            i += 1;
            continue;
        }
        // Find the start of the run (the explicit position before it)
        let run_start = i;
        let left_value = authored_stops[run_start - 1]
            .raw_position
            .expect("implicit position runs must have an explicit left bound");

        // Find the end of the run
        let mut run_end = run_start;
        while run_end < len && authored_stops[run_end].raw_position.is_none() {
            run_end += 1;
        }
        let right_value = authored_stops[run_end]
            .raw_position
            .expect("implicit position runs must have an explicit right bound");

        let run_count = run_end - run_start;
        for j in 0..run_count {
            let fraction = (j + 1) as f32 / (run_count + 1) as f32;
            authored_stops[run_start + j].raw_position =
                Some(left_value + fraction * (right_value - left_value));
        }

        i = run_end + 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gradient::types::{
        ColorInterpolation, GradientColor, GradientCommonDesc, GradientStop, GradientStopOffset,
        GradientStopPositions, GradientUnits, SpreadMode,
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
    fn test_two_stop_defaults_stay_inline() {
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
        assert!(!normalized.stops.spilled());
        assert!(!normalized.segments.spilled());
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
