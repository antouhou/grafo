use super::normalize::{NormalizedGradient, NormalizedSegment};
use super::types::{
    ColorInterpolation, GradientColor, HueComponent, HueInterpolationMethod, RAMP_RESOLUTION,
    RESOLVED_DEGENERATE_EPSILON,
};

/// Bakes a gradient ramp: RAMP_RESOLUTION texels of final linear premultiplied RGBA.
/// The texels span from `period_start` to `period_start + period_len` (non-repeating)
/// or from first stop to last stop (pad mode).
pub(crate) fn bake_gradient_ramp(
    normalized: &NormalizedGradient,
    interpolation: &ColorInterpolation,
) -> Vec<[f32; 4]> {
    if normalized.is_single_stop {
        let color = color_to_final_linear_premultiplied(&normalized.single_stop_color.unwrap());
        return vec![color; RAMP_RESOLUTION];
    }

    let first_pos = normalized.stops.first().unwrap().position;
    let last_pos = normalized.stops.last().unwrap().position;
    let span = last_pos - first_pos;

    if span <= RESOLVED_DEGENERATE_EPSILON {
        if let Some(ramp) = bake_degenerate_hard_stop_ramp(normalized) {
            return ramp;
        }

        let color = color_to_final_linear_premultiplied(&normalized.stops.last().unwrap().color);
        return vec![color; RAMP_RESOLUTION];
    }

    let mut ramp = Vec::with_capacity(RAMP_RESOLUTION);
    for i in 0..RAMP_RESOLUTION {
        let t_normalized = i as f32 / (RAMP_RESOLUTION - 1) as f32;
        let u = first_pos + t_normalized * span;

        let color = evaluate_at_scalar(u, &normalized.segments, &normalized.stops, interpolation);
        ramp.push(color);
    }
    ramp
}

fn bake_degenerate_hard_stop_ramp(normalized: &NormalizedGradient) -> Option<Vec<[f32; 4]>> {
    let first_stop = normalized.stops.first()?;
    let last_stop = normalized.stops.last()?;
    let has_coincident_stops = normalized.stops.len() > 1
        && normalized
            .stops
            .windows(2)
            .any(|pair| (pair[1].position - pair[0].position).abs() <= RESOLVED_DEGENERATE_EPSILON);

    if !has_coincident_stops {
        return None;
    }

    let first_color = color_to_final_linear_premultiplied(&first_stop.color);
    let last_color = color_to_final_linear_premultiplied(&last_stop.color);
    let transition_index = RAMP_RESOLUTION / 2;

    let mut ramp = vec![last_color; RAMP_RESOLUTION];
    ramp[..transition_index].fill(first_color);
    Some(ramp)
}

/// Evaluates the gradient at scalar `u` (after spread-mode folding).
/// Returns final linear premultiplied RGBA.
fn evaluate_at_scalar(
    u: f32,
    segments: &[NormalizedSegment],
    stops: &[super::normalize::NormalizedStop],
    interpolation: &ColorInterpolation,
) -> [f32; 4] {
    if segments.is_empty() {
        return color_to_final_linear_premultiplied(&stops.last().unwrap().color);
    }

    // Find the segment containing u
    // CSS rule: half-open [p_i, p_{i+1}), last stop takes final stop color
    let last_stop_pos = stops.last().unwrap().position;
    if u >= last_stop_pos {
        return color_to_final_linear_premultiplied(&stops.last().unwrap().color);
    }

    // Find segment: last segment where start_position <= u
    let mut segment_index = 0;
    for (i, seg) in segments.iter().enumerate() {
        if u >= seg.start_position {
            segment_index = i;
        } else {
            break;
        }
    }

    let segment = &segments[segment_index];

    // If the segment has zero length (hard stop), use last stop in coincident run
    let seg_len = segment.end_position - segment.start_position;
    if seg_len <= RESOLVED_DEGENERATE_EPSILON {
        return color_to_final_linear_premultiplied(&segment.end_color);
    }

    // Compute interpolation parameter with hint reparameterization
    let x = (u - segment.start_position) / seg_len;
    let p = apply_hint_reparameterization(x, segment.hint, seg_len, segment.start_position);

    interpolate_colors(&segment.start_color, &segment.end_color, p, interpolation)
}

/// Applies the CSS gradient hint reparameterization.
fn apply_hint_reparameterization(
    x: f32,
    hint: Option<f32>,
    segment_len: f32,
    segment_start: f32,
) -> f32 {
    let hint_value = match hint {
        None => return x.clamp(0.0, 1.0),
        Some(h) => h,
    };

    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }

    let m = (hint_value - segment_start) / segment_len;

    if (x - m).abs() < 1e-10 {
        return 0.5;
    }

    if x < m {
        let k0 = (0.5_f32).ln() / m.ln();
        0.5 * (x / m).powf(k0)
    } else {
        let k1 = (0.5_f32).ln() / (1.0 - m).ln();
        1.0 - 0.5 * ((1.0 - x) / (1.0 - m)).powf(k1)
    }
}

// ── Color interpolation ──────────────────────────────────────────────────────

/// Interpolates between two gradient colors at parameter p (0..1).
/// Returns final linear premultiplied RGBA.
fn interpolate_colors(
    color_a: &GradientColor,
    color_b: &GradientColor,
    p: f32,
    interpolation: &ColorInterpolation,
) -> [f32; 4] {
    match interpolation {
        ColorInterpolation::Srgb => interpolate_rectangular(color_a, color_b, p, RectSpace::Srgb),
        ColorInterpolation::SrgbLinear => {
            interpolate_rectangular(color_a, color_b, p, RectSpace::SrgbLinear)
        }
        ColorInterpolation::Oklab => interpolate_rectangular(color_a, color_b, p, RectSpace::Oklab),
        ColorInterpolation::Hsl { hue } => {
            interpolate_cylindrical(color_a, color_b, p, CylSpace::Hsl, *hue)
        }
        ColorInterpolation::Hwb { hue } => {
            interpolate_cylindrical(color_a, color_b, p, CylSpace::Hwb, *hue)
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum RectSpace {
    Srgb,
    SrgbLinear,
    Oklab,
}

#[derive(Debug, Clone, Copy)]
enum CylSpace {
    Hsl,
    Hwb,
}

/// Rectangular (non-hue) interpolation pipeline as specified in the plan.
fn interpolate_rectangular(
    color_a: &GradientColor,
    color_b: &GradientColor,
    p: f32,
    space: RectSpace,
) -> [f32; 4] {
    // Step 1-2: Convert both colors into the interpolation space
    let [ra, ga, ba, aa] = to_rect_space(color_a, space);
    let [rb, gb, bb, ab] = to_rect_space(color_b, space);

    // Step 3: Premultiply
    let (pra, pga, pba) = (ra * aa, ga * aa, ba * aa);
    let (prb, pgb, pbb) = (rb * ab, gb * ab, bb * ab);

    // Step 4: Interpolate premultiplied channels and alpha
    let pr = pra + (prb - pra) * p;
    let pg = pga + (pgb - pga) * p;
    let pb = pba + (pbb - pba) * p;
    let alpha_p = aa + (ab - aa) * p;

    // Step 5-6: Unpremultiply if alpha > 0
    let (ur, ug, ub) = if alpha_p > 0.0 {
        (pr / alpha_p, pg / alpha_p, pb / alpha_p)
    } else {
        (0.0, 0.0, 0.0)
    };

    // Step 7: Convert to final linear output space
    let [lr, lg, lb] = rect_to_linear(ur, ug, ub, space);

    // Step 8: Premultiply with alpha_p for final output
    [lr * alpha_p, lg * alpha_p, lb * alpha_p, alpha_p]
}

fn interpolate_cylindrical(
    color_a: &GradientColor,
    color_b: &GradientColor,
    p: f32,
    space: CylSpace,
    hue_method: HueInterpolationMethod,
) -> [f32; 4] {
    // Step 1: Convert to HSL/HWB
    let (h0, c1_a, c2_a, a_a, h0_powerless) = to_cylindrical(color_a, space);
    let (h1, c1_b, c2_b, a_b, h1_powerless) = to_cylindrical(color_b, space);

    // Step 2: Resolve missing/powerless hue
    let (rh0, rh1) = resolve_hue_pair(h0, h0_powerless, h1, h1_powerless);

    // Step 3: Choose hue path
    let delta = compute_hue_delta(rh0, rh1, hue_method);

    // Step 4: Interpolate
    let h_interp = rem_euclid_f32(rh0 + delta * p, 360.0);
    let c1_interp = c1_a + (c1_b - c1_a) * p;
    let c2_interp = c2_a + (c2_b - c2_a) * p;
    let alpha_interp = a_a + (a_b - a_a) * p;

    // Step 5: Convert to final linear output space and premultiply
    let [lr, lg, lb] = cylindrical_to_linear(h_interp, c1_interp, c2_interp, space);
    let alpha_clamped = alpha_interp.clamp(0.0, 1.0);
    [
        lr * alpha_clamped,
        lg * alpha_clamped,
        lb * alpha_clamped,
        alpha_clamped,
    ]
}

// ── Color conversion helpers ─────────────────────────────────────────────────

/// Converts a GradientColor to the specified rectangular interpolation space.
/// Returns [channel0, channel1, channel2, alpha] with alpha clamped to [0,1].
fn to_rect_space(color: &GradientColor, space: RectSpace) -> [f32; 4] {
    // First get the color as [r, g, b, alpha] in sRGB space, handling
    // missing/powerless hue for HSL/HWB.
    let (srgb_r, srgb_g, srgb_b, alpha) = gradient_color_to_srgb(color);
    let alpha = alpha.clamp(0.0, 1.0);

    match space {
        RectSpace::Srgb => [srgb_r, srgb_g, srgb_b, alpha],
        RectSpace::SrgbLinear => {
            let lr = srgb_to_linear(srgb_r);
            let lg = srgb_to_linear(srgb_g);
            let lb = srgb_to_linear(srgb_b);
            [lr, lg, lb, alpha]
        }
        RectSpace::Oklab => {
            let lr = srgb_to_linear(srgb_r);
            let lg = srgb_to_linear(srgb_g);
            let lb = srgb_to_linear(srgb_b);
            let [ol, oa, ob] = linear_rgb_to_oklab(lr, lg, lb);
            [ol, oa, ob, alpha]
        }
    }
}

/// Converts from a rectangular interpolation space back to linear sRGB.
fn rect_to_linear(c0: f32, c1: f32, c2: f32, space: RectSpace) -> [f32; 3] {
    match space {
        RectSpace::Srgb => [srgb_to_linear(c0), srgb_to_linear(c1), srgb_to_linear(c2)],
        RectSpace::SrgbLinear => [c0, c1, c2],
        RectSpace::Oklab => oklab_to_linear_rgb(c0, c1, c2),
    }
}

/// Converts a GradientColor into (hue_degrees, component1, component2, alpha, is_powerless).
fn to_cylindrical(color: &GradientColor, space: CylSpace) -> (f32, f32, f32, f32, bool) {
    match (color, space) {
        (
            GradientColor::Hsl {
                hue,
                saturation,
                lightness,
                alpha,
            },
            CylSpace::Hsl,
        ) => {
            let s_clamped = saturation.clamp(0.0, 1.0);
            let l_clamped = lightness.clamp(0.0, 1.0);
            let (h_deg, is_powerless) = match hue {
                HueComponent::Degrees(deg) => {
                    let h = rem_euclid_f32(*deg, 360.0);
                    let powerless = s_clamped == 0.0 || l_clamped == 0.0 || l_clamped == 1.0;
                    (h, powerless)
                }
                HueComponent::Missing => (0.0, true),
            };
            (h_deg, s_clamped, l_clamped, *alpha, is_powerless)
        }
        (
            GradientColor::Hwb {
                hue,
                whiteness,
                blackness,
                alpha,
            },
            CylSpace::Hwb,
        ) => {
            let mut w = whiteness.max(0.0);
            let mut b = blackness.max(0.0);
            if w + b > 1.0 {
                let sum = w + b;
                w /= sum;
                b /= sum;
            }
            let (h_deg, is_powerless) = match hue {
                HueComponent::Degrees(deg) => {
                    let h = rem_euclid_f32(*deg, 360.0);
                    let powerless = w + b >= 1.0;
                    (h, powerless)
                }
                HueComponent::Missing => (0.0, true),
            };
            (h_deg, w, b, *alpha, is_powerless)
        }
        // Convert any other color space to HSL or HWB through sRGB
        (_, cyl_space) => {
            let (srgb_r, srgb_g, srgb_b, alpha) = gradient_color_to_srgb(color);
            match cyl_space {
                CylSpace::Hsl => {
                    let (h, s, l) = srgb_to_hsl(srgb_r, srgb_g, srgb_b);
                    let powerless = s == 0.0 || l == 0.0 || l == 1.0;
                    (h, s, l, alpha, powerless)
                }
                CylSpace::Hwb => {
                    let (h, w, b) = srgb_to_hwb(srgb_r, srgb_g, srgb_b);
                    let powerless = w + b >= 1.0;
                    (h, w, b, alpha, powerless)
                }
            }
        }
    }
}

fn cylindrical_to_linear(hue: f32, c1: f32, c2: f32, space: CylSpace) -> [f32; 3] {
    let (sr, sg, sb) = match space {
        CylSpace::Hsl => hsl_to_srgb(hue, c1, c2),
        CylSpace::Hwb => hwb_to_srgb(hue, c1, c2),
    };
    [srgb_to_linear(sr), srgb_to_linear(sg), srgb_to_linear(sb)]
}

fn resolve_hue_pair(h0: f32, h0_powerless: bool, h1: f32, h1_powerless: bool) -> (f32, f32) {
    match (h0_powerless, h1_powerless) {
        (false, false) => (h0, h1),
        (true, false) => (h1, h1),
        (false, true) => (h0, h0),
        (true, true) => (0.0, 0.0),
    }
}

fn compute_hue_delta(h0: f32, h1: f32, method: HueInterpolationMethod) -> f32 {
    match method {
        HueInterpolationMethod::Shorter => {
            let mut delta = rem_euclid_f32(h1 - h0 + 180.0, 360.0) - 180.0;
            if delta == -180.0 {
                delta = 180.0;
            }
            delta
        }
        HueInterpolationMethod::Longer => {
            let mut shorter = rem_euclid_f32(h1 - h0 + 180.0, 360.0) - 180.0;
            if shorter == -180.0 {
                shorter = 180.0;
            }
            if shorter == 180.0 {
                -180.0
            } else if shorter > 0.0 {
                shorter - 360.0
            } else {
                shorter + 360.0
            }
        }
        HueInterpolationMethod::Increasing => rem_euclid_f32(h1 - h0, 360.0),
        HueInterpolationMethod::Decreasing => rem_euclid_f32(h1 - h0, 360.0) - 360.0,
    }
}

// ── Color space conversion functions ─────────────────────────────────────────

/// Converts any GradientColor to sRGB (r, g, b, alpha).
/// Missing hue for HSL/HWB is treated as 0 for the purpose of conversion.
fn gradient_color_to_srgb(color: &GradientColor) -> (f32, f32, f32, f32) {
    match color {
        GradientColor::Srgb {
            red,
            green,
            blue,
            alpha,
        } => (*red, *green, *blue, *alpha),
        GradientColor::SrgbLinear {
            red,
            green,
            blue,
            alpha,
        } => (
            linear_to_srgb(*red),
            linear_to_srgb(*green),
            linear_to_srgb(*blue),
            *alpha,
        ),
        GradientColor::Oklab { l, a, b, alpha } => {
            let [lr, lg, lb] = oklab_to_linear_rgb(*l, *a, *b);
            (
                linear_to_srgb(lr),
                linear_to_srgb(lg),
                linear_to_srgb(lb),
                *alpha,
            )
        }
        GradientColor::Hsl {
            hue,
            saturation,
            lightness,
            alpha,
        } => {
            let s_clamped = saturation.clamp(0.0, 1.0);
            let l_clamped = lightness.clamp(0.0, 1.0);
            let h_deg = match hue {
                HueComponent::Degrees(deg) => rem_euclid_f32(*deg, 360.0),
                HueComponent::Missing => 0.0,
            };
            let (r, g, b) = hsl_to_srgb(h_deg, s_clamped, l_clamped);
            (r, g, b, *alpha)
        }
        GradientColor::Hwb {
            hue,
            whiteness,
            blackness,
            alpha,
        } => {
            let mut w = whiteness.max(0.0);
            let mut bk = blackness.max(0.0);
            if w + bk > 1.0 {
                let sum = w + bk;
                w /= sum;
                bk /= sum;
            }
            let h_deg = match hue {
                HueComponent::Degrees(deg) => rem_euclid_f32(*deg, 360.0),
                HueComponent::Missing => 0.0,
            };
            let (r, g, b) = hwb_to_srgb(h_deg, w, bk);
            (r, g, b, *alpha)
        }
    }
}

/// Convert a single authored GradientColor to final linear premultiplied RGBA.
pub(crate) fn color_to_final_linear_premultiplied(color: &GradientColor) -> [f32; 4] {
    let (sr, sg, sb, alpha) = gradient_color_to_srgb(color);
    let alpha = alpha.clamp(0.0, 1.0);
    let lr = srgb_to_linear(sr);
    let lg = srgb_to_linear(sg);
    let lb = srgb_to_linear(sb);
    [lr * alpha, lg * alpha, lb * alpha, alpha]
}

// ── sRGB transfer functions (extended) ───────────────────────────────────────

fn srgb_to_linear(c: f32) -> f32 {
    if c.abs() <= 0.04045 {
        c / 12.92
    } else {
        let sign = c.signum();
        sign * ((c.abs() + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb(c: f32) -> f32 {
    if c.abs() <= 0.0031308 {
        c * 12.92
    } else {
        let sign = c.signum();
        sign * (1.055 * c.abs().powf(1.0 / 2.4) - 0.055)
    }
}

// ── Oklab conversion ─────────────────────────────────────────────────────────

#[allow(clippy::excessive_precision)]
fn linear_rgb_to_oklab(r: f32, g: f32, b: f32) -> [f32; 3] {
    let l_ = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b;
    let m_ = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b;
    let s_ = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b;

    let l_c = cbrt(l_);
    let m_c = cbrt(m_);
    let s_c = cbrt(s_);

    [
        0.2104542553 * l_c + 0.7936177850 * m_c - 0.0040720468 * s_c,
        1.9779984951 * l_c - 2.4285922050 * m_c + 0.4505937099 * s_c,
        0.0259040371 * l_c + 0.7827717662 * m_c - 0.8086757660 * s_c,
    ]
}

#[allow(clippy::excessive_precision)]
fn oklab_to_linear_rgb(l: f32, a: f32, b: f32) -> [f32; 3] {
    let l_ = l + 0.3963377774 * a + 0.2158037573 * b;
    let m_ = l - 0.1055613458 * a - 0.0638541728 * b;
    let s_ = l - 0.0894841775 * a - 1.2914855480 * b;

    let l3 = l_ * l_ * l_;
    let m3 = m_ * m_ * m_;
    let s3 = s_ * s_ * s_;

    [
        4.0767416621 * l3 - 3.3077115913 * m3 + 0.2309699292 * s3,
        -1.2684380046 * l3 + 2.6097574011 * m3 - 0.3413193965 * s3,
        -0.0041960863 * l3 - 0.7034186147 * m3 + 1.7076147010 * s3,
    ]
}

fn cbrt(x: f32) -> f32 {
    if x >= 0.0 {
        x.powf(1.0 / 3.0)
    } else {
        -((-x).powf(1.0 / 3.0))
    }
}

// ── HSL/HWB conversions ──────────────────────────────────────────────────────

fn hsl_to_srgb(h: f32, s: f32, l: f32) -> (f32, f32, f32) {
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let h_prime = h / 60.0;
    let x = c * (1.0 - (h_prime % 2.0 - 1.0).abs());

    let (r1, g1, b1) = if h_prime < 1.0 {
        (c, x, 0.0)
    } else if h_prime < 2.0 {
        (x, c, 0.0)
    } else if h_prime < 3.0 {
        (0.0, c, x)
    } else if h_prime < 4.0 {
        (0.0, x, c)
    } else if h_prime < 5.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };

    let m = l - c / 2.0;
    (r1 + m, g1 + m, b1 + m)
}

fn hwb_to_srgb(h: f32, w: f32, b: f32) -> (f32, f32, f32) {
    // HWB to sRGB: first get the pure hue from HSL with S=1, L=0.5
    let (r, g, bl) = hsl_to_srgb(h, 1.0, 0.5);
    // Then mix with white and black
    let r = r * (1.0 - w - b) + w;
    let g = g * (1.0 - w - b) + w;
    let bl = bl * (1.0 - w - b) + w;
    (r, g, bl)
}

fn srgb_to_hsl(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let l = (max + min) / 2.0;

    if (max - min).abs() < 1e-10 {
        return (0.0, 0.0, l);
    }

    let d = max - min;
    let s = if l > 0.5 {
        d / (2.0 - max - min)
    } else {
        d / (max + min)
    };

    let h = if (max - r).abs() < 1e-10 {
        let mut h = (g - b) / d;
        if g < b {
            h += 6.0;
        }
        h
    } else if (max - g).abs() < 1e-10 {
        (b - r) / d + 2.0
    } else {
        (r - g) / d + 4.0
    };

    (rem_euclid_f32(h * 60.0, 360.0), s, l)
}

fn srgb_to_hwb(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let (h, _, _) = srgb_to_hsl(r, g, b);
    let w = r.min(g).min(b);
    let bk = 1.0 - r.max(g).max(b);
    (h, w, bk)
}

pub(crate) fn rem_euclid_f32(a: f32, b: f32) -> f32 {
    ((a % b) + b) % b
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gradient::normalize::NormalizedGradient;
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

    #[test]
    fn test_srgb_linear_roundtrip() {
        for v in [0.0, 0.04045, 0.5, 1.0, -0.5] {
            let linear = srgb_to_linear(v);
            let back = linear_to_srgb(linear);
            assert!(
                (v - back).abs() < 1e-5,
                "roundtrip failed for {v}: got {back}"
            );
        }
    }

    #[test]
    fn test_oklab_roundtrip() {
        let [ol, oa, ob] = linear_rgb_to_oklab(0.5, 0.3, 0.1);
        let [r, g, b] = oklab_to_linear_rgb(ol, oa, ob);
        assert!((r - 0.5).abs() < 1e-4);
        assert!((g - 0.3).abs() < 1e-4);
        assert!((b - 0.1).abs() < 1e-4);
    }

    #[test]
    fn test_hsl_srgb_roundtrip() {
        let (r, g, b) = hsl_to_srgb(120.0, 1.0, 0.5);
        assert!((r - 0.0).abs() < 1e-5);
        assert!((g - 1.0).abs() < 1e-5);
        assert!((b - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_rem_euclid() {
        assert!((rem_euclid_f32(-30.0, 360.0) - 330.0).abs() < 1e-5);
        assert!((rem_euclid_f32(370.0, 360.0) - 10.0).abs() < 1e-5);
    }

    #[test]
    fn bake_gradient_ramp_preserves_degenerate_hard_stop_boundary() {
        let common = GradientCommonDesc {
            units: GradientUnits::Local,
            spread: SpreadMode::Pad,
            interpolation: ColorInterpolation::Srgb,
            stops: vec![
                GradientStop {
                    positions: GradientStopPositions::Single(GradientStopOffset::LinearRadial(0.5)),
                    color: srgb_color(1.0, 0.0, 0.0),
                    hint_to_next_segment: None,
                },
                GradientStop {
                    positions: GradientStopPositions::Single(GradientStopOffset::LinearRadial(0.5)),
                    color: srgb_color(0.0, 0.0, 1.0),
                    hint_to_next_segment: None,
                },
            ],
        };

        let normalized = NormalizedGradient::from_common(&common, GradientKind::Linear);
        let ramp = bake_gradient_ramp(&normalized, &common.interpolation);
        let transition_index = RAMP_RESOLUTION / 2;

        assert_eq!(
            ramp[transition_index - 1],
            color_to_final_linear_premultiplied(&srgb_color(1.0, 0.0, 0.0))
        );
        assert_eq!(
            ramp[transition_index],
            color_to_final_linear_premultiplied(&srgb_color(0.0, 0.0, 1.0))
        );
    }
}
