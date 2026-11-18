use super::normalize::NormalizedGradient;
use super::types::{
    ColorInterpolation, GradientRamp, GradientRampSource, HueInterpolationMethod, SpreadMode,
    RAMP_RESOLUTION,
};
use crate::core::color::{CylindricalColorSpace, RectangularColorSpace};
use crate::core::{color, Color};
use std::sync::Arc;

#[derive(Clone, Copy)]
struct SamplePosition {
    numerator: f64,
    rounding_residual: f64,
}

impl SamplePosition {
    fn from_sum(start_contribution: f64, end_contribution: f64) -> Self {
        // CSS selects intervals and authored stops using the exact line position.
        // Retain the rounding residual of the two exactly representable scaled endpoint terms.
        // https://www.w3.org/TR/css-images-4/#coloring-gradient-line
        let numerator = start_contribution + end_contribution;
        let recovered_end = numerator - start_contribution;
        let start_residual = start_contribution - (numerator - recovered_end);
        let end_residual = end_contribution - recovered_end;
        Self {
            numerator,
            rounding_residual: start_residual + end_residual,
        }
    }

    fn distance_from(self, scaled_position: f64) -> f64 {
        (self.numerator - scaled_position) + self.rounding_residual
    }
}

/// Keeps unhinted CSS quantities on a common texel scale until channels are combined.
/// https://www.w3.org/TR/css-images-4/#coloring-gradient-line
/// https://www.w3.org/TR/2026/CRD-css-color-4-20261112/#interpolation
fn segment_interpolation_weights(
    position: SamplePosition,
    position_denominator: f64,
    hint: Option<f32>,
    segment_start: f32,
    segment_end: f32,
) -> (f64, f64) {
    let scaled_start = f64::from(segment_start) * position_denominator;
    let scaled_end = f64::from(segment_end) * position_denominator;
    let left_distance = position.distance_from(scaled_start);
    let right_distance = -position.distance_from(scaled_end);
    let Some(hint_position) = hint else {
        return (right_distance, left_distance);
    };
    // These are the exact endpoint limits of the CSS exponent, also handled by Blink.
    // https://chromium.googlesource.com/chromium/src/+/052831f0220b79fe0c3343b49f6d2863ea6de05d/third_party/blink/renderer/core/css/css_gradient_value.cc#259
    if hint_position == segment_start {
        return (0.0, 1.0);
    }
    if hint_position == segment_end {
        return (1.0, 0.0);
    }
    if left_distance == 0.0 {
        return (1.0, 0.0);
    }
    if right_distance == 0.0 {
        return (0.0, 1.0);
    }
    let scaled_hint = f64::from(hint_position) * position_denominator;
    let hint_left = scaled_hint - scaled_start;
    let hint_right = scaled_end - scaled_hint;
    // CSS requires an exactly centered hint to produce ordinary linear interpolation.
    if hint_left == hint_right {
        return (right_distance, left_distance);
    }
    // CSS C = P^(ln(0.5)/ln(H)). With P = L/(L+R), ln(P) = -ln1p(R/L).
    // expm1 retains 1-C when C rounds to one; both identities preserve the same CSS equation.
    let log_position = -(right_distance / left_distance).ln_1p();
    let log_hint = -(hint_right / hint_left).ln_1p();
    let log_end_weight = (log_position / log_hint) * 0.5_f64.ln();
    (-log_end_weight.exp_m1(), log_end_weight.exp())
}

fn bake_segments<Interpolator: Fn(f64, f64) -> [f32; 4]>(
    normalized: &NormalizedGradient,
    prepare_colors: impl Fn(&Color, &Color) -> Interpolator,
) -> GradientRamp {
    if let Some(color) = normalized.constant_color() {
        return GradientRamp::Constant(color);
    }

    let first_pos = f64::from(normalized.stops.first().unwrap().position);
    let last_pos = f64::from(normalized.stops.last().unwrap().position);
    let span = last_pos - first_pos;

    if span == 0.0 {
        return GradientRamp::Sampled(Arc::new(bake_degenerate_hard_stop_ramp(normalized)));
    }

    // CSS Images 4 preserves authored stop and padding colors; conversion affects only intervals.
    // https://www.w3.org/TR/css-images-4/#coloring-gradient-line
    // A finite uniform ramp can preserve an interior stop exactly only when a texel samples it.
    let first_color = normalized
        .stops
        .first()
        .unwrap()
        .color
        .to_linear_premultiplied_rgba();
    let last_color = normalized
        .stops
        .last()
        .unwrap()
        .color
        .to_linear_premultiplied_rgba();
    let mut segment_index = 0;
    let mut segment = &normalized.segments[segment_index];
    let mut interpolate = prepare_colors(&segment.start_color, &segment.end_color);
    let mut start_color = segment.start_color.to_linear_premultiplied_rgba();
    let uses_padding = normalized.spread == SpreadMode::Pad;
    let position_denominator = (RAMP_RESOLUTION - 1) as f64;
    let mut ramp = [[0.0; 4]; RAMP_RESOLUTION];
    for (index, texel) in ramp.iter_mut().enumerate() {
        // Keep CSS's common position denominator through segment lookup and channel sums.
        let position = SamplePosition::from_sum(
            first_pos * (RAMP_RESOLUTION - 1 - index) as f64,
            last_pos * index as f64,
        );

        if uses_padding && index == 0 {
            *texel = first_color;
            continue;
        }
        if uses_padding
            && (index == RAMP_RESOLUTION - 1
                || position.distance_from(last_pos * position_denominator) >= 0.0)
        {
            *texel = last_color;
            continue;
        }

        // Samples advance through half-open segments, including coincident hard stops.
        while normalized
            .segments
            .get(segment_index + 1)
            .is_some_and(|segment| {
                position.distance_from(f64::from(segment.start_position) * position_denominator)
                    >= 0.0
            })
        {
            segment_index += 1;
            segment = &normalized.segments[segment_index];
            interpolate = prepare_colors(&segment.start_color, &segment.end_color);
            start_color = segment.start_color.to_linear_premultiplied_rgba();
        }

        *texel = if index != 0
            && index != RAMP_RESOLUTION - 1
            && position.distance_from(f64::from(segment.start_position) * position_denominator)
                == 0.0
        {
            start_color
        } else {
            // Repeats have no padding, so their endpoints use the same-segment limits.
            // https://skia.googlesource.com/skia/+/d2b9e48baf1697760afc1dc8ea3ad40110b8cacc/src/shaders/gradients/SkGradientBaseShader.cpp#891
            let (start_weight, end_weight) = segment_interpolation_weights(
                position,
                position_denominator,
                segment.hint,
                segment.start_position,
                segment.end_position,
            );
            interpolate(start_weight, end_weight)
        };
    }
    GradientRamp::Sampled(Arc::new(ramp))
}

/// Bakes linear premultiplied RGBA texels from the first normalized stop to the last.
/// Constant gradients use one texel; sampled ramps use RAMP_RESOLUTION texels.
pub(crate) fn bake_gradient_ramp(ramp_source: &GradientRampSource) -> GradientRamp {
    let normalized = &ramp_source.normalized;
    match ramp_source.interpolation {
        ColorInterpolation::Srgb => bake_segments(normalized, |start, end| {
            prepare_rectangular_interpolation(start, end, RectangularColorSpace::Srgb)
        }),
        ColorInterpolation::SrgbLinear => bake_segments(normalized, |start, end| {
            prepare_rectangular_interpolation(start, end, RectangularColorSpace::SrgbLinear)
        }),
        ColorInterpolation::Oklab => bake_segments(normalized, |start, end| {
            prepare_rectangular_interpolation(start, end, RectangularColorSpace::Oklab)
        }),
        ColorInterpolation::Hsl { hue } => bake_segments(normalized, |start, end| {
            prepare_cylindrical_interpolation(start, end, CylindricalColorSpace::Hsl, hue)
        }),
        ColorInterpolation::Hwb { hue } => bake_segments(normalized, |start, end| {
            prepare_cylindrical_interpolation(start, end, CylindricalColorSpace::Hwb, hue)
        }),
    }
}

fn bake_degenerate_hard_stop_ramp(normalized: &NormalizedGradient) -> [[f32; 4]; RAMP_RESOLUTION] {
    let first_stop = normalized.stops.first().unwrap();
    let last_stop = normalized.stops.last().unwrap();

    let first_color = first_stop.color.to_linear_premultiplied_rgba();
    let last_color = last_stop.color.to_linear_premultiplied_rgba();
    let transition_index = RAMP_RESOLUTION / 2;

    let mut ramp = [[last_color[0], last_color[1], last_color[2], last_color[3]]; RAMP_RESOLUTION];
    ramp[..transition_index].fill(first_color);
    ramp
}

fn prepare_rectangular_interpolation(
    color_a: &Color,
    color_b: &Color,
    space: RectangularColorSpace,
) -> impl Fn(f64, f64) -> [f32; 4] {
    let [ra, ga, ba, aa] = color_a.to_rectangular_space(space);
    let [rb, gb, bb, ab] = color_b.to_rectangular_space(space);

    let (pra, pga, pba) = (ra * aa, ga * aa, ba * aa);
    let (prb, pgb, pbb) = (rb * ab, gb * ab, bb * ab);

    move |start_weight: f64, end_weight: f64| {
        let pr = pra * start_weight + prb * end_weight;
        let pg = pga * start_weight + pgb * end_weight;
        let pb = pba * start_weight + pbb * end_weight;
        let weighted_alpha_sum = aa * start_weight + ab * end_weight;
        let interpolated_alpha = weighted_alpha_sum / (start_weight + end_weight);

        // Color-space conversion needs unpremultiplied channels.
        let (ur, ug, ub) = if weighted_alpha_sum > 0.0 {
            (
                pr / weighted_alpha_sum,
                pg / weighted_alpha_sum,
                pb / weighted_alpha_sum,
            )
        } else {
            (0.0, 0.0, 0.0)
        };

        let [lr, lg, lb] = rect_to_linear(ur, ug, ub, space);

        [
            (lr * interpolated_alpha) as f32,
            (lg * interpolated_alpha) as f32,
            (lb * interpolated_alpha) as f32,
            interpolated_alpha as f32,
        ]
    }
}

fn prepare_cylindrical_interpolation(
    color_a: &Color,
    color_b: &Color,
    space: CylindricalColorSpace,
    hue_method: HueInterpolationMethod,
) -> impl Fn(f64, f64) -> [f32; 4] {
    let (h0, c1_a, c2_a, a_a, h0_powerless) = color_a.to_cylindrical_space(space);
    let (h1, c1_b, c2_b, a_b, h1_powerless) = color_b.to_cylindrical_space(space);
    let (hue_start, hue_end) = resolve_hue_pair(h0, h0_powerless, h1, h1_powerless);
    let hue_delta = compute_hue_delta(hue_start, hue_end, hue_method);
    let adjusted_hue_end = hue_start + hue_delta;
    let c1_a_p = c1_a * a_a;
    let c1_b_p = c1_b * a_b;
    let c2_a_p = c2_a * a_a;
    let c2_b_p = c2_b * a_b;

    move |start_weight: f64, end_weight: f64| {
        let total_weight = start_weight + end_weight;
        // Hue follows the selected angular path; only the other channels are premultiplied.
        // Skia's difference-based slope keeps identical components exact. Preserve that
        // zero-slope result before HDR conversion can amplify angle roundoff into color.
        // https://skia.googlesource.com/skia/+/d2b9e48baf1697760afc1dc8ea3ad40110b8cacc/src/shaders/gradients/SkGradientBaseShader.cpp#400
        let h_interp = if hue_delta == 0.0 {
            hue_start
        } else {
            (hue_start * start_weight + adjusted_hue_end * end_weight) / total_weight
        }
        .rem_euclid(360.0);
        // CSS Color 4 section 13.2 weights endpoints directly, preserving their channels at p=0/1.
        // https://www.w3.org/TR/2026/CRD-css-color-4-20261112/#interpolation
        let weighted_alpha_sum = a_a * start_weight + a_b * end_weight;
        let c1_p = c1_a_p * start_weight + c1_b_p * end_weight;
        let c2_p = c2_a_p * start_weight + c2_b_p * end_weight;
        let (c1_interp, c2_interp) = if weighted_alpha_sum > 0.0 {
            (c1_p / weighted_alpha_sum, c2_p / weighted_alpha_sum)
        } else {
            (0.0, 0.0)
        };

        let [lr, lg, lb] = cylindrical_to_linear(h_interp, c1_interp, c2_interp, space);
        let interpolated_alpha = (weighted_alpha_sum / total_weight).clamp(0.0, 1.0);
        [
            (lr * interpolated_alpha) as f32,
            (lg * interpolated_alpha) as f32,
            (lb * interpolated_alpha) as f32,
            interpolated_alpha as f32,
        ]
    }
}

/// Converts from a rectangular interpolation space back to linear sRGB.
fn rect_to_linear(c0: f64, c1: f64, c2: f64, space: RectangularColorSpace) -> [f64; 3] {
    match space {
        RectangularColorSpace::Srgb => [
            color::srgb_to_linear_f64(c0),
            color::srgb_to_linear_f64(c1),
            color::srgb_to_linear_f64(c2),
        ],
        RectangularColorSpace::SrgbLinear => [c0, c1, c2],
        RectangularColorSpace::Oklab => color::oklab_to_linear_rgb(c0, c1, c2),
    }
}

fn cylindrical_to_linear(hue: f64, c1: f64, c2: f64, space: CylindricalColorSpace) -> [f64; 3] {
    let (sr, sg, sb) = match space {
        CylindricalColorSpace::Hsl => color::hsl_to_srgb(hue, c1, c2),
        CylindricalColorSpace::Hwb => color::hwb_interpolation_to_srgb(hue, c1, c2),
    };
    [
        color::srgb_to_linear_f64(sr),
        color::srgb_to_linear_f64(sg),
        color::srgb_to_linear_f64(sb),
    ]
}

fn resolve_hue_pair(h0: f64, h0_powerless: bool, h1: f64, h1_powerless: bool) -> (f64, f64) {
    match (h0_powerless, h1_powerless) {
        (false, false) => (h0, h1),
        (true, false) => (h1, h1),
        (false, true) => (h0, h0),
        (true, true) => (0.0, 0.0),
    }
}

fn compute_hue_delta(start_hue: f64, end_hue: f64, method: HueInterpolationMethod) -> f64 {
    let delta = end_hue - start_hue;
    match method {
        HueInterpolationMethod::Shorter => {
            if delta > 180.0 {
                delta - 360.0
            } else if delta < -180.0 {
                delta + 360.0
            } else {
                delta
            }
        }
        HueInterpolationMethod::Longer => {
            if delta > 0.0 && delta < 180.0 {
                delta - 360.0
            } else if delta > -180.0 && delta <= 0.0 {
                delta + 360.0
            } else {
                delta
            }
        }
        HueInterpolationMethod::Increasing => {
            if delta < 0.0 {
                delta + 360.0
            } else {
                delta
            }
        }
        HueInterpolationMethod::Decreasing => {
            if delta > 0.0 {
                delta - 360.0
            } else {
                delta
            }
        }
    }
}

#[cfg(test)]
mod tests;
