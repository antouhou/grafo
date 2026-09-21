use super::normalize::NormalizedGradient;
use super::types::{
    ColorInterpolation, GradientColor, GradientRamp, GradientRampSource, HueComponent,
    HueInterpolationMethod, RAMP_RESOLUTION, RESOLVED_DEGENERATE_EPSILON,
};
use std::sync::Arc;

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

fn bake_segments<Interpolator: Fn(f32) -> [f32; 4]>(
    normalized: &NormalizedGradient,
    prepare_colors: impl Fn(&GradientColor, &GradientColor) -> Interpolator,
) -> GradientRamp {
    if let Some(color) = normalized.constant_color() {
        return GradientRamp::Constant(color);
    }

    let first_pos = normalized.stops.first().unwrap().position;
    let last_pos = normalized.stops.last().unwrap().position;
    let span = last_pos - first_pos;

    if span <= RESOLVED_DEGENERATE_EPSILON {
        return GradientRamp::Sampled(Arc::new(bake_degenerate_hard_stop_ramp(normalized)));
    }

    let last_color = color_to_final_linear_premultiplied(&normalized.stops.last().unwrap().color);
    let mut segment_index = 0;
    let mut segment = &normalized.segments[segment_index];
    let mut interpolate = prepare_colors(&segment.start_color, &segment.end_color);
    let mut end_color = color_to_final_linear_premultiplied(&segment.end_color);
    let mut ramp = [[0.0; 4]; RAMP_RESOLUTION];
    for (index, texel) in ramp.iter_mut().enumerate() {
        let t_normalized = index as f32 / (RAMP_RESOLUTION - 1) as f32;
        let u = first_pos + t_normalized * span;

        if u >= last_pos {
            *texel = last_color;
            continue;
        }

        // Samples advance through half-open segments, including coincident hard stops.
        while normalized
            .segments
            .get(segment_index + 1)
            .is_some_and(|segment| u >= segment.start_position)
        {
            segment_index += 1;
            segment = &normalized.segments[segment_index];
            interpolate = prepare_colors(&segment.start_color, &segment.end_color);
            end_color = color_to_final_linear_premultiplied(&segment.end_color);
        }

        let segment_len = segment.end_position - segment.start_position;
        *texel = if segment_len <= RESOLVED_DEGENERATE_EPSILON {
            end_color
        } else {
            let x = (u - segment.start_position) / segment_len;
            let p =
                apply_hint_reparameterization(x, segment.hint, segment_len, segment.start_position);
            interpolate(p)
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
            prepare_rectangular_interpolation(start, end, RectSpace::Srgb)
        }),
        ColorInterpolation::SrgbLinear => bake_segments(normalized, |start, end| {
            prepare_rectangular_interpolation(start, end, RectSpace::SrgbLinear)
        }),
        ColorInterpolation::Oklab => bake_segments(normalized, |start, end| {
            prepare_rectangular_interpolation(start, end, RectSpace::Oklab)
        }),
        ColorInterpolation::Hsl { hue } => bake_segments(normalized, |start, end| {
            prepare_cylindrical_interpolation(start, end, CylSpace::Hsl, hue)
        }),
        ColorInterpolation::Hwb { hue } => bake_segments(normalized, |start, end| {
            prepare_cylindrical_interpolation(start, end, CylSpace::Hwb, hue)
        }),
    }
}

fn bake_degenerate_hard_stop_ramp(normalized: &NormalizedGradient) -> [[f32; 4]; RAMP_RESOLUTION] {
    let first_stop = normalized.stops.first().unwrap();
    let last_stop = normalized.stops.last().unwrap();

    let first_color = color_to_final_linear_premultiplied(&first_stop.color);
    let last_color = color_to_final_linear_premultiplied(&last_stop.color);
    let transition_index = RAMP_RESOLUTION / 2;

    let mut ramp = [[last_color[0], last_color[1], last_color[2], last_color[3]]; RAMP_RESOLUTION];
    ramp[..transition_index].fill(first_color);
    ramp
}

/// Applies the CSS gradient hint reparameterization.
fn apply_hint_reparameterization(
    x: f32,
    hint: Option<f32>,
    segment_len: f32,
    segment_start: f32,
) -> f32 {
    let Some(hint_position) = hint else {
        return x.clamp(0.0, 1.0);
    };
    let hint_fraction = (hint_position - segment_start) / segment_len;
    x.clamp(0.0, 1.0).powf(0.5_f32.ln() / hint_fraction.ln())
}

fn prepare_rectangular_interpolation(
    color_a: &GradientColor,
    color_b: &GradientColor,
    space: RectSpace,
) -> impl Fn(f32) -> [f32; 4] {
    let [ra, ga, ba, aa] = to_rect_space(color_a, space);
    let [rb, gb, bb, ab] = to_rect_space(color_b, space);

    let (pra, pga, pba) = (ra * aa, ga * aa, ba * aa);
    let (prb, pgb, pbb) = (rb * ab, gb * ab, bb * ab);

    move |p| {
        let pr = pra + (prb - pra) * p;
        let pg = pga + (pgb - pga) * p;
        let pb = pba + (pbb - pba) * p;
        let alpha_p = aa + (ab - aa) * p;

        // Color-space conversion needs unpremultiplied channels.
        let (ur, ug, ub) = if alpha_p > 0.0 {
            (pr / alpha_p, pg / alpha_p, pb / alpha_p)
        } else {
            (0.0, 0.0, 0.0)
        };

        let [lr, lg, lb] = rect_to_linear(ur, ug, ub, space);

        [lr * alpha_p, lg * alpha_p, lb * alpha_p, alpha_p]
    }
}

fn prepare_cylindrical_interpolation(
    color_a: &GradientColor,
    color_b: &GradientColor,
    space: CylSpace,
    hue_method: HueInterpolationMethod,
) -> impl Fn(f32) -> [f32; 4] {
    let (h0, c1_a, c2_a, a_a, h0_powerless) = to_cylindrical(color_a, space);
    let (h1, c1_b, c2_b, a_b, h1_powerless) = to_cylindrical(color_b, space);
    let (hue_start, hue_end) = resolve_hue_pair(h0, h0_powerless, h1, h1_powerless);
    let hue_delta = compute_hue_delta(hue_start, hue_end, hue_method);
    let c1_a_p = c1_a * a_a;
    let c1_b_p = c1_b * a_b;
    let c2_a_p = c2_a * a_a;
    let c2_b_p = c2_b * a_b;

    move |p| {
        // Hue follows the selected angular path; only the other channels are premultiplied.
        let h_interp = (hue_start + hue_delta * p).rem_euclid(360.0);
        let alpha_interp = a_a + (a_b - a_a) * p;
        let c1_p = c1_a_p + (c1_b_p - c1_a_p) * p;
        let c2_p = c2_a_p + (c2_b_p - c2_a_p) * p;
        let (c1_interp, c2_interp) = if alpha_interp > 0.0 {
            (c1_p / alpha_interp, c2_p / alpha_interp)
        } else {
            (0.0, 0.0)
        };

        let [lr, lg, lb] = cylindrical_to_linear(h_interp, c1_interp, c2_interp, space);
        let alpha_clamped = alpha_interp.clamp(0.0, 1.0);
        [
            lr * alpha_clamped,
            lg * alpha_clamped,
            lb * alpha_clamped,
            alpha_clamped,
        ]
    }
}

/// Converts a GradientColor to the specified rectangular interpolation space.
/// Returns [channel0, channel1, channel2, alpha] with alpha clamped to [0,1].
fn to_rect_space(color: &GradientColor, space: RectSpace) -> [f32; 4] {
    match (color, space) {
        (
            GradientColor::SrgbLinear {
                red,
                green,
                blue,
                alpha,
            },
            RectSpace::SrgbLinear,
        ) => {
            return [*red, *green, *blue, alpha.clamp(0.0, 1.0)];
        }
        (GradientColor::Oklab { l, a, b, alpha }, RectSpace::Oklab) => {
            return [*l, *a, *b, alpha.clamp(0.0, 1.0)];
        }
        _ => {}
    }

    // Convert to sRGB, resolving missing or powerless hue for HSL and HWB.
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
                    let h = deg.rem_euclid(360.0);
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
                    let h = deg.rem_euclid(360.0);
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
            let mut delta = (h1 - h0 + 180.0).rem_euclid(360.0) - 180.0;
            if delta == -180.0 {
                delta = 180.0;
            }
            delta
        }
        HueInterpolationMethod::Longer => {
            let mut shorter = (h1 - h0 + 180.0).rem_euclid(360.0) - 180.0;
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
        HueInterpolationMethod::Increasing => (h1 - h0).rem_euclid(360.0),
        HueInterpolationMethod::Decreasing => (h1 - h0).rem_euclid(360.0) - 360.0,
    }
}

/// Converts a `GradientColor` to sRGB `(r, g, b, alpha)`.
/// Uses zero for missing HSL or HWB hue.
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
                HueComponent::Degrees(deg) => deg.rem_euclid(360.0),
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
                HueComponent::Degrees(deg) => deg.rem_euclid(360.0),
                HueComponent::Missing => 0.0,
            };
            let (r, g, b) = hwb_to_srgb(h_deg, w, bk);
            (r, g, b, *alpha)
        }
    }
}

/// Converts a `GradientColor` to linear premultiplied RGBA.
pub(crate) fn color_to_final_linear_premultiplied(color: &GradientColor) -> [f32; 4] {
    let (sr, sg, sb, alpha) = gradient_color_to_srgb(color);
    let alpha = alpha.clamp(0.0, 1.0);
    let lr = srgb_to_linear(sr);
    let lg = srgb_to_linear(sg);
    let lb = srgb_to_linear(sb);
    [lr * alpha, lg * alpha, lb * alpha, alpha]
}

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

#[allow(clippy::excessive_precision)]
fn linear_rgb_to_oklab(r: f32, g: f32, b: f32) -> [f32; 3] {
    let l_ = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b;
    let m_ = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b;
    let s_ = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b;

    let l_c = l_.cbrt();
    let m_c = m_.cbrt();
    let s_c = s_.cbrt();

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
    // HSL with S=1 and L=0.5 gives the pure hue for HWB conversion.
    let (r, g, bl) = hsl_to_srgb(h, 1.0, 0.5);
    // Mix the pure hue with white and black.
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

    ((h * 60.0).rem_euclid(360.0), s, l)
}

fn srgb_to_hwb(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let (h, _, _) = srgb_to_hsl(r, g, b);
    let w = r.min(g).min(b);
    let bk = 1.0 - r.max(g).max(b);
    (h, w, bk)
}

#[cfg(test)]
mod tests;
