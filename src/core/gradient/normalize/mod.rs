use super::types::{GradientCommonDesc, GradientKind, GradientStopPositions, SpreadMode};
use crate::core::{color, Color};
use smallvec::SmallVec;
use std::f32::consts::TAU;

const NORMALIZED_INLINE_STOP_CAPACITY: usize = 8;
const NORMALIZED_INLINE_SEGMENT_CAPACITY: usize = 8;

#[derive(Debug, Clone, Copy)]
struct AuthoredGradientItem {
    color: Option<Color>,
    raw_position: Option<f32>,
}

/// A gradient stop after resolving its CSS position.
#[derive(Debug, Clone)]
pub(crate) struct NormalizedStop {
    pub(crate) position: f32,
    pub(crate) color: Color,
    /// If Some, the hint position between this stop and the next.
    pub(crate) hint: Option<f32>,
}

/// An interpolation segment between two normalized stops.
#[derive(Debug, Clone)]
pub(crate) struct NormalizedSegment {
    pub(crate) start_position: f32,
    pub(crate) end_position: f32,
    pub(crate) start_color: Color,
    pub(crate) end_color: Color,
    pub(crate) hint: Option<f32>,
}

fn add_compensated_rgba(sums: &mut [f64; 4], corrections: &mut [f64; 4], contribution: [f64; 4]) {
    for ((sum, correction), channel) in sums.iter_mut().zip(corrections).zip(contribution) {
        let next_sum = *sum + channel;
        *correction += if sum.abs() >= channel.abs() {
            (*sum - next_sum) + channel
        } else {
            (channel - next_sum) + *sum
        };
        *sum = next_sum;
    }
}

/// A gradient with resolved CSS stop positions.
#[derive(Debug, Clone)]
pub(crate) struct NormalizedGradient {
    pub(crate) spread: SpreadMode,
    pub(crate) stops: SmallVec<[NormalizedStop; NORMALIZED_INLINE_STOP_CAPACITY]>,
    pub(crate) segments: SmallVec<[NormalizedSegment; NORMALIZED_INLINE_SEGMENT_CAPACITY]>,
    pub(crate) period_start: f32,
    pub(crate) period_len: f32,
    zero_period_color: Option<[f32; 4]>,
}

impl NormalizedGradient {
    fn average_color(stops: &[NormalizedStop], span: f64) -> [f32; 4] {
        if stops.len() == 1 {
            return stops.last().unwrap().color.to_linear_premultiplied_rgba();
        }
        // CSS Images 3 averages intervals in premultiplied encoded sRGB. A zero period
        // uses equally spaced stops before averaging, preserving alpha and zero-width weights.
        // https://www.w3.org/TR/css-images-3/#gradient-average-color
        // Keep the common divisor until after summation so rounded interval ratios cannot
        // alter signed HDR cancellation.
        let denominator = if span == 0.0 {
            2.0 * (stops.len() - 1) as f64
        } else {
            2.0 * span
        };
        let mut encoded_premultiplied = [0.0_f64; 4];
        let mut summation_corrections = [0.0_f64; 4];
        for pair in stops.windows(2) {
            let interval_width = if span == 0.0 {
                1.0
            } else {
                f64::from(pair[1].position) - f64::from(pair[0].position)
            };
            for stop in pair {
                let [red, green, blue, alpha] = stop.color.to_array_f32().map(f64::from);
                let alpha = alpha.clamp(0.0, 1.0);
                // Retain small CSS sum terms when signed HDR contributions later cancel.
                add_compensated_rgba(
                    &mut encoded_premultiplied,
                    &mut summation_corrections,
                    [red * alpha, green * alpha, blue * alpha, alpha]
                        .map(|channel| channel * interval_width),
                );
            }
        }
        for (sum, correction) in encoded_premultiplied.iter_mut().zip(summation_corrections) {
            *sum += correction;
        }
        let weighted_alpha = encoded_premultiplied[3];
        if weighted_alpha == 0.0 {
            return [0.0; 4];
        }
        let alpha = weighted_alpha / denominator;
        let decode = |channel| (color::srgb_to_linear_f64(channel / weighted_alpha) * alpha) as f32;
        [
            decode(encoded_premultiplied[0]),
            decode(encoded_premultiplied[1]),
            decode(encoded_premultiplied[2]),
            alpha as f32,
        ]
    }

    pub(crate) fn from_common(common: &GradientCommonDesc, kind: GradientKind) -> Self {
        let is_conic = kind == GradientKind::Conic;

        // CSS Images 4 section 3.5.1 expands double positions before considering constants.
        // Hints participate in the same position fixup and implicit spacing as color stops.
        // https://www.w3.org/TR/css-images-4/#color-stop-syntax
        // https://www.w3.org/TR/css-images-4/#color-stop-fixup
        let authored_item_count = common
            .stops
            .iter()
            .enumerate()
            .map(|(index, stop)| {
                1 + usize::from(matches!(stop.positions, GradientStopPositions::Double(..)))
                    + usize::from(
                        index + 1 < common.stops.len() && stop.hint_to_next_segment.is_some(),
                    )
            })
            .sum();
        let mut authored: SmallVec<[AuthoredGradientItem; NORMALIZED_INLINE_STOP_CAPACITY]> =
            SmallVec::with_capacity(authored_item_count);
        for (stop_index, stop) in common.stops.iter().enumerate() {
            match &stop.positions {
                GradientStopPositions::Auto => {
                    authored.push(AuthoredGradientItem {
                        color: Some(stop.color),
                        raw_position: None,
                    });
                }
                GradientStopPositions::Single(offset) => {
                    authored.push(AuthoredGradientItem {
                        color: Some(stop.color),
                        raw_position: Some(offset.value()),
                    });
                }
                GradientStopPositions::Double(a, b) => {
                    authored.push(AuthoredGradientItem {
                        color: Some(stop.color),
                        raw_position: Some(a.value()),
                    });
                    // The hint applies after the second position of the double stop.
                    authored.push(AuthoredGradientItem {
                        color: Some(stop.color),
                        raw_position: Some(b.value()),
                    });
                }
            }
            if stop_index + 1 < common.stops.len() {
                if let Some(hint) = stop.hint_to_next_segment {
                    authored.push(AuthoredGradientItem {
                        color: None,
                        raw_position: Some(hint.value()),
                    });
                }
            }
        }

        // Omitted endpoints span one turn for conic gradients and 0..1 otherwise.
        let default_end = if is_conic { TAU } else { 1.0 };

        if authored[0].raw_position.is_none() {
            authored[0].raw_position = Some(0.0);
        }
        let last_index = authored.len() - 1;
        if authored[last_index].raw_position.is_none() {
            authored[last_index].raw_position = Some(default_end);
        }

        // Resolve decreasing color/hint positions before spacing omitted color stops.
        let mut previous_position = f32::NEG_INFINITY;
        for position in authored
            .iter_mut()
            .filter_map(|stop| stop.raw_position.as_mut())
        {
            *position = position.max(previous_position);
            previous_position = *position;
        }

        fill_implicit_positions(&mut authored);

        let color_stop_count = authored.iter().filter(|item| item.color.is_some()).count();
        let mut stops: SmallVec<[NormalizedStop; NORMALIZED_INLINE_STOP_CAPACITY]> =
            SmallVec::with_capacity(color_stop_count);
        for authored_item in &authored {
            let position = authored_item
                .raw_position
                .expect("gradient stop positions should be resolved before normalization");

            if let Some(color) = authored_item.color {
                stops.push(NormalizedStop {
                    position,
                    color,
                    hint: None,
                });
            } else if let Some(previous) = stops.last_mut() {
                previous.hint = Some(position);
            }
        }

        // Skia removes duplicate outer colors used only by clamp padding for non-clamp modes.
        // Do this before segments and cache keys so a repeating seam cannot inherit padding colors.
        // https://skia.googlesource.com/skia/+/d2b9e48baf1697760afc1dc8ea3ad40110b8cacc/src/shaders/gradients/SkGradientBaseShader.cpp#281
        let first_position = stops.first().unwrap().position;
        let last_position = stops.last().unwrap().position;
        let period_len = last_position - first_position;
        // A conic period can round to zero in turns. CSS's equally spaced average must
        // include the authored colors before Skia's non-clamp seam simplification.
        // https://www.w3.org/TR/css-images-3/#repeating-gradients
        let zero_period_color = if is_conic
            && common.spread == SpreadMode::Repeat
            && period_len > 0.0
            && period_len / TAU == 0.0
        {
            Some(Self::average_color(&stops, 0.0))
        } else {
            None
        };
        if common.spread == SpreadMode::Repeat && first_position < last_position {
            let first_kept = stops.partition_point(|stop| stop.position == first_position) - 1;
            let last_kept = stops.partition_point(|stop| stop.position < last_position);
            stops.truncate(last_kept + 1);
            drop(stops.drain(..first_kept));
        }

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

        NormalizedGradient {
            spread: common.spread,
            stops,
            segments,
            period_start: first_position,
            period_len,
            zero_period_color,
        }
    }

    /// Averages equally spaced stops when a period is zero in the destination representation.
    pub(crate) fn zero_period_constant_color(&self) -> [f32; 4] {
        self.zero_period_color
            .unwrap_or_else(|| Self::average_color(&self.stops, 0.0))
    }

    /// Resolves degenerate padding to the final color and repeats to their average.
    pub(crate) fn degenerate_constant_color(&self) -> [f32; 4] {
        if self.spread == SpreadMode::Pad {
            return self
                .stops
                .last()
                .unwrap()
                .color
                .to_linear_premultiplied_rgba();
        }
        let span = f64::from(self.stops.last().unwrap().position)
            - f64::from(self.stops.first().unwrap().position);
        Self::average_color(&self.stops, span)
    }

    /// Returns a constant color for a single stop or a zero-period repeating gradient.
    pub(crate) fn constant_color(&self) -> Option<[f32; 4]> {
        if self.stops.len() == 1 || (self.spread == SpreadMode::Repeat && self.period_len == 0.0) {
            Some(self.degenerate_constant_color())
        } else {
            None
        }
    }
}

/// Fills contiguous runs of `None` positions by even spacing between the
/// nearest explicit neighbors.
fn fill_implicit_positions(authored_items: &mut [AuthoredGradientItem]) {
    let len = authored_items.len();
    let mut i = 0;
    while i < len {
        if authored_items[i].raw_position.is_some() {
            i += 1;
            continue;
        }
        let run_start = i;
        let left_value = f64::from(
            authored_items[run_start - 1]
                .raw_position
                .expect("implicit position runs must have an explicit left bound"),
        );

        let mut run_end = run_start;
        while run_end < len && authored_items[run_end].raw_position.is_none() {
            run_end += 1;
        }
        let right_value = f64::from(
            authored_items[run_end]
                .raw_position
                .expect("implicit position runs must have an explicit right bound"),
        );

        let run_count = run_end - run_start;
        for j in 0..run_count {
            // Finite f32 bounds can have a difference outside f32's representable range.
            authored_items[run_start + j].raw_position = Some(
                ((left_value * (run_count - j) as f64 + right_value * (j + 1) as f64)
                    / (run_count + 1) as f64) as f32,
            );
        }

        i = run_end + 1;
    }
}

#[cfg(test)]
mod tests;
