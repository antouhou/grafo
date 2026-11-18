//! sRGB colors, channel conversions, and pixel-buffer preparation.

const HSL_EPSILON: f64 = 1e-5;
const HWB_POWERLESS_EPSILON: f64 = 1e-5;

#[derive(Debug, Clone, Copy)]
pub(super) enum RectangularColorSpace {
    Srgb,
    SrgbLinear,
    Oklab,
}

#[derive(Debug, Clone, Copy)]
pub(super) enum CylindricalColorSpace {
    Hsl,
    Hwb,
}

// CSS Color 4's extended transfer function, kept wide until after alpha multiplication.
// https://www.w3.org/TR/2026/CRD-css-color-4-20261112/#color-conversion-code
pub(crate) fn srgb_to_linear_f64(value: f64) -> f64 {
    if value.abs() <= 0.04045 {
        value / 12.92
    } else {
        value.signum() * ((value.abs() + 0.055) / 1.055).powf(2.4)
    }
}

/// Converts one encoded sRGB channel to linear sRGB.
///
/// Values outside `[0.0, 1.0]` are not clamped. Negative values use the
/// sign-preserving extension of the sRGB transfer function.
/// Apply this to RGB channels only; alpha stays linear.
pub fn srgb_to_linear(value: f32) -> f32 {
    // Evaluate extended channels with the CSS reference's double precision throughout.
    // Switching precision only after overflow can reverse adjacent HDR intensities.
    // https://www.w3.org/TR/2026/CRD-css-color-4-20261112/#color-conversion-code
    if value.abs() > 1.0 {
        return srgb_to_linear_f64(f64::from(value)) as f32;
    }
    if value.abs() <= 0.04045 {
        value / 12.92
    } else {
        value.signum() * ((value.abs() + 0.055) / 1.055).powf(2.4)
    }
}

/// Converts one linear sRGB channel to encoded sRGB.
///
/// Values outside `[0.0, 1.0]` are not clamped. Negative values use the
/// sign-preserving extension of the sRGB transfer function.
/// Apply this to RGB channels only; alpha stays linear.
pub fn linear_to_srgb(value: f32) -> f32 {
    // CSS's double-precision transfer avoids rounding across byte and powerless-hue boundaries.
    // https://www.w3.org/TR/2026/CRD-css-color-4-20261112/#color-conversion-code
    let value = f64::from(value);
    let encoded = if value.abs() <= 0.0031308 {
        value * 12.92
    } else {
        let sign = value.signum();
        sign * (1.055 * value.abs().powf(1.0 / 2.4) - 0.055)
    };
    encoded as f32
}

/// Converts an sRGB byte to a linear channel value between 0.0 and 1.0.
fn srgb_u8_to_linear(value: u8) -> f32 {
    srgb_to_linear(f32::from(value) / 255.0)
}

fn normalized_channel_to_u8(value: f32) -> u8 {
    // CSS Color 4 preserves authored precision until nearest-integer rounding.
    // f64 holds the exact product of an f32 channel and 255, avoiding artificial ties.
    // https://www.w3.org/TR/2026/CRD-css-color-4-20261112/#rgb-functions
    // https://drafts.csswg.org/css-values-4/#integers
    (f64::from(value.clamp(0.0, 1.0)) * 255.0).round() as u8
}

// CSS Color 4 section 19's 64-bit Oklab matrices, precomposed with sRGB/XYZ conversion.
// https://www.w3.org/TR/2026/CRD-css-color-4-20261112/#color-conversion-code
// Shorter coefficients leak visible RGB channels when extended intensities amplify roundoff.
pub(super) fn linear_rgb_to_oklab(red: f64, green: f64, blue: f64) -> [f64; 3] {
    let long_response =
        0.41222146947076305 * red + 0.5363325372617348 * green + 0.05144599326750221 * blue;
    let medium_response =
        0.21190349581782522 * red + 0.680_699_550_645_234_4 * green + 0.10739695353694055 * blue;
    let short_response =
        0.08830245919005643 * red + 0.28171883913612145 * green + 0.6299787016738222 * blue;

    let long_root = long_response.cbrt();
    let medium_root = medium_response.cbrt();
    let short_root = short_response.cbrt();

    [
        0.210_454_268_309_314 * long_root + 0.7936177747023054 * medium_root
            - 0.0040720430116193 * short_root,
        1.9779985324311684 * long_root - 2.428_592_242_048_58 * medium_root
            + 0.450_593_709_617_411 * short_root,
        0.0259040424655478 * long_root + 0.7827717124575296 * medium_root
            - 0.8086757549230774 * short_root,
    ]
}

pub(crate) fn oklab_to_linear_rgb(lightness: f64, opponent_a: f64, opponent_b: f64) -> [f64; 3] {
    let long_root = lightness + 0.3963377773761749 * opponent_a + 0.2158037573099136 * opponent_b;
    let medium_root = lightness - 0.1055613458156586 * opponent_a - 0.0638541728258133 * opponent_b;
    let short_root = lightness - 0.0894841775298119 * opponent_a - 1.2914855480194092 * opponent_b;

    let long_response = long_root * long_root * long_root;
    let medium_response = medium_root * medium_root * medium_root;
    let short_response = short_root * short_root * short_root;

    [
        4.076_741_636_075_958 * long_response - 3.3077115392580616 * medium_response
            + 0.23096990318210442 * short_response,
        -1.268_437_973_285_032 * long_response + 2.609_757_349_287_688 * medium_response
            - 0.34131937600265727 * short_response,
        -0.004196076138675566 * long_response - 0.7034186179359363 * medium_response
            + 1.7076146940746117 * short_response,
    ]
}

// CSS Color 4 section 7.1, with saturation and lightness in normalized units:
// https://www.w3.org/TR/2026/CRD-css-color-4-20261112/#hsl-to-rgb
pub(crate) fn hsl_to_srgb(hue: f64, saturation: f64, lightness: f64) -> (f64, f64, f64) {
    let amplitude = saturation * lightness.min(1.0 - lightness);
    let hue_phase = hue.rem_euclid(360.0) / 30.0;
    let channel = |offset: f64| {
        let phase = (offset + hue_phase).rem_euclid(12.0);
        lightness - amplitude * (phase - 3.0).min(9.0 - phase).clamp(-1.0, 1.0)
    };
    (channel(0.0), channel(8.0), channel(4.0))
}

pub(crate) fn hwb_interpolation_to_srgb(hue: f64, whiteness: f64, maximum: f64) -> (f64, f64, f64) {
    let (red, green, blue) = hsl_to_srgb(hue, 1.0, 0.5);
    let red = (1.0 - red) * whiteness + red * maximum;
    let green = (1.0 - green) * whiteness + green * maximum;
    let blue = (1.0 - blue) * whiteness + blue * maximum;
    (red, green, blue)
}

fn srgb_to_hue(red: f64, green: f64, blue: f64) -> Option<f64> {
    let maximum = red.max(green).max(blue);
    let minimum = red.min(green).min(blue);
    let chroma = maximum - minimum;
    if chroma == 0.0 {
        return None;
    }

    let hue_sector = if maximum == red {
        let mut sector = (green - blue) / chroma;
        if green < blue {
            sector += 6.0;
        }
        sector
    } else if maximum == green {
        (blue - red) / chroma + 2.0
    } else {
        (red - green) / chroma + 4.0
    };

    Some((hue_sector * 60.0).rem_euclid(360.0))
}

// CSS Color 4 (2026-11-12), section 7.2:
// https://www.w3.org/TR/2026/CRD-css-color-4-20261112/#rgb-to-hsl
// Its epsilon avoids singular divisions and powerless-hue noise by discarding chroma
// near lightness 0/1 or zero saturation. Extended RGB therefore may not round-trip.
// Keep double precision through the inverse: high saturation magnifies lightness roundoff.
fn srgb_to_hsl(red: f64, green: f64, blue: f64) -> (f64, f64, f64) {
    let maximum = red.max(green).max(blue);
    let minimum = red.min(green).min(blue);
    let lightness = (maximum + minimum) / 2.0;
    if maximum == minimum || lightness.abs() <= HSL_EPSILON || (1.0 - lightness).abs() < HSL_EPSILON
    {
        return (0.0, 0.0, lightness);
    }

    let mut saturation = (maximum - lightness) / lightness.min(1.0 - lightness);
    let mut hue = srgb_to_hue(red, green, blue).unwrap_or(0.0);
    if saturation < 0.0 {
        hue = (hue + 180.0).rem_euclid(360.0);
        saturation = -saturation;
    }
    if saturation <= HSL_EPSILON {
        return (0.0, 0.0, lightness);
    }
    (hue, saturation, lightness)
}

fn srgb_to_hwb_interpolation(red: f64, green: f64, blue: f64) -> (Option<f64>, f64, f64) {
    // HWB uses RGB hue directly, without HSL's negative-saturation hue correction.
    let whiteness = red.min(green).min(blue);
    let maximum = red.max(green).max(blue);
    // CSS Color 4 sections 8.2 and 4.4.1 remove near-neutral chroma before borrowing hue.
    // https://www.w3.org/TR/2026/CRD-css-color-4-20261112/#rgb-to-hwb
    // https://www.w3.org/TR/2026/CRD-css-color-4-20261112/#powerless
    // Store M = 1 - B instead of B. This affine change preserves premultiplied CSS
    // interpolation while avoiding cancellation of B's unit offset for large RGB.
    // W + B >= 1 - epsilon becomes M - W <= epsilon; B = 1 - W becomes M = W.
    if maximum - whiteness <= HWB_POWERLESS_EPSILON {
        return (None, whiteness, whiteness);
    }
    (srgb_to_hue(red, green, blue), whiteness, maximum)
}

/// An extended sRGB color with floating-point red, green, blue, and alpha channels.
/// RGB is encoded, not linear or premultiplied, and may be negative or greater than one.
/// Channels must be finite. Alpha ranges from zero for transparent to one for opaque.
///
/// # Examples
///
/// ```
/// use grafo::Color;
///
/// let red = Color::rgb(255, 0, 0);
/// let semi_blue = Color::rgba(0, 0, 255, 128);
///
/// assert_eq!(red.to_array(), [255, 0, 0, 255]);
/// assert_eq!(semi_blue.to_array(), [0, 0, 255, 128]);
/// ```
/// `Color` stores HDR values in extended sRGB. Implement [`From`] for your
/// own type to supply its conversion to encoded extended sRGB with straight alpha.
/// Shape, group, and gradient stop color builders accept these types through `Into<Color>`.
///
/// ```
/// use grafo::{core::color, Color, GradientStop, GroupDrawCommandOptions, ShapeDrawOptions};
///
/// struct LinearRgb([f32; 3]);
///
/// impl From<LinearRgb> for Color {
///     fn from(value: LinearRgb) -> Self {
///         let [red, green, blue] = value.0.map(color::linear_to_srgb);
///         Color::rgb_f32(red, green, blue)
///     }
/// }
///
/// # fn main() {
/// let shape = ShapeDrawOptions::new().color(LinearRgb([0.5, 0.25, 1.5]));
/// let group = GroupDrawCommandOptions::new().color(LinearRgb([0.1, 0.2, 0.3]));
/// let stop = GradientStop::auto(LinearRgb([0.25, 0.5, 1.0]));
/// # }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Color(pub [f32; 4]);

impl Color {
    /// Transparent black.
    pub const TRANSPARENT: Self = Self([0.0, 0.0, 0.0, 0.0]);
    /// Opaque black.
    pub const BLACK: Self = Self([0.0, 0.0, 0.0, 1.0]);
    /// Opaque white.
    pub const WHITE: Self = Self([1.0, 1.0, 1.0, 1.0]);

    /// Creates an opaque color from 8-bit sRGB channel values.
    pub fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self::rgba(r, g, b, 255)
    }

    /// Creates a color from 8-bit sRGB channel values and alpha.
    pub fn rgba(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self::from_rgba_array([r, g, b, a])
    }

    pub fn from_rgba_array(arr: [u8; 4]) -> Self {
        Self(arr.map(|channel| f32::from(channel) / 255.0))
    }

    /// Creates a color from floating-point encoded sRGB channel values and alpha.
    /// Channels must be finite. RGB may be negative or greater than one;
    /// alpha must be in `[0.0, 1.0]`. Values are stored without clamping.
    pub const fn rgba_f32(red: f32, green: f32, blue: f32, alpha: f32) -> Self {
        Self([red, green, blue, alpha])
    }

    /// Creates an opaque color from floating-point encoded sRGB channel values.
    /// RGB channels must be finite and may be negative or greater than one.
    pub const fn rgb_f32(red: f32, green: f32, blue: f32) -> Self {
        Self::rgba_f32(red, green, blue, 1.0)
    }

    /// Converts encoded sRGB channels to linear RGB and leaves alpha unchanged.
    /// RGB may be negative or greater than one. The result is not premultiplied.
    /// Decoded RGB that exceeds the `f32` range becomes infinite.
    ///
    /// # Examples
    ///
    /// ```
    /// use grafo::Color;
    ///
    /// let red = Color::rgba(255, 0, 0, 128);
    /// let normalized = red.normalize();
    /// assert_eq!(normalized, [1.0, 0.0, 0.0, 128.0 / 255.0]);
    /// ```
    pub fn normalize(&self) -> [f32; 4] {
        let [red, green, blue, alpha] = self.0;
        [
            srgb_to_linear(red),
            srgb_to_linear(green),
            srgb_to_linear(blue),
            alpha,
        ]
    }

    /// Converts to linear premultiplied `[red, green, blue, alpha]` components.
    /// RGB is not clamped. Alpha is clamped to `[0.0, 1.0]` before premultiplication.
    /// Zero alpha returns transparent black.
    pub fn to_linear_premultiplied_rgba(&self) -> [f32; 4] {
        let [red, green, blue, alpha] = self.0;
        let alpha = alpha.clamp(0.0, 1.0);
        if alpha == 0.0 {
            return [0.0; 4];
        }
        let premultiply = |encoded| {
            let linear = srgb_to_linear(encoded);
            if linear.is_finite() {
                linear * alpha
            } else {
                // Keep the CSS extended-sRGB product representable when decoding alone overflows f32.
                // https://www.w3.org/TR/2026/CRD-css-color-4-20261112/#color-conversion-code
                let linear = srgb_to_linear_f64(f64::from(encoded));
                (linear * f64::from(alpha)) as f32
            }
        };
        [
            premultiply(red),
            premultiply(green),
            premultiply(blue),
            alpha,
        ]
    }

    /// Returns `[red, green, blue, alpha]` as bytes.
    /// Clamps each channel to `[0.0, 1.0]` and rounds to the nearest byte value.
    pub fn to_array(&self) -> [u8; 4] {
        self.0.map(normalized_channel_to_u8)
    }

    /// Returns the encoded sRGB channels and alpha without clamping or quantization.
    pub const fn to_array_f32(&self) -> [f32; 4] {
        self.0
    }

    /// Converts to the selected interpolation space with alpha clamped to [0,1].
    pub(super) fn to_rectangular_space(self, space: RectangularColorSpace) -> [f64; 4] {
        let [red, green, blue, alpha] = self.to_array_f32().map(f64::from);
        let alpha = alpha.clamp(0.0, 1.0);

        match space {
            RectangularColorSpace::Srgb => [red, green, blue, alpha],
            RectangularColorSpace::SrgbLinear => {
                let [red, green, blue] = [red, green, blue].map(srgb_to_linear_f64);
                [red, green, blue, alpha]
            }
            RectangularColorSpace::Oklab => {
                let [red, green, blue] = [red, green, blue].map(srgb_to_linear_f64);
                let [lightness, opponent_a, opponent_b] = linear_rgb_to_oklab(red, green, blue);
                [lightness, opponent_a, opponent_b, alpha]
            }
        }
    }

    /// Returns hue, the other two polar components, alpha, and whether hue is powerless.
    pub(super) fn to_cylindrical_space(
        self,
        space: CylindricalColorSpace,
    ) -> (f64, f64, f64, f64, bool) {
        let [red, green, blue, alpha] = self.to_array_f32().map(f64::from);
        let alpha = alpha.clamp(0.0, 1.0);

        match space {
            CylindricalColorSpace::Hsl => {
                let (hue, saturation, lightness) = srgb_to_hsl(red, green, blue);
                (hue, saturation, lightness, alpha, saturation == 0.0)
            }
            CylindricalColorSpace::Hwb => {
                let (hue, whiteness, maximum) = srgb_to_hwb_interpolation(red, green, blue);
                (hue.unwrap_or(0.0), whiteness, maximum, alpha, hue.is_none())
            }
        }
    }
}

pub(crate) fn linear_to_srgb_u8(value: f32) -> u8 {
    let encoded = linear_to_srgb(value.clamp(0.0, 1.0));
    normalized_channel_to_u8(encoded)
}

/// Converts straight-alpha RGBA8 sRGB pixels to premultiplied alpha in place.
///
/// Multiplies RGB by alpha in linear space, then encodes it as sRGB. Alpha bytes are unchanged.
///
/// # Panics
///
/// Panics if the slice length is not a multiple of four.
pub fn premultiply_rgba8_srgb_inplace(pixels: &mut [u8]) {
    assert!(
        pixels.len().is_multiple_of(4),
        "RGBA8 data length must be multiple of 4"
    );
    for px in pixels.chunks_mut(4) {
        let r_lin = srgb_u8_to_linear(px[0]);
        let g_lin = srgb_u8_to_linear(px[1]);
        let b_lin = srgb_u8_to_linear(px[2]);
        let a = px[3] as f32 / 255.0;

        let r_pma = r_lin * a;
        let g_pma = g_lin * a;
        let b_pma = b_lin * a;

        px[0] = linear_to_srgb_u8(r_pma);
        px[1] = linear_to_srgb_u8(g_pma);
        px[2] = linear_to_srgb_u8(b_pma);
    }
}
