use thiserror::Error;

const DISPLAY_P3_TO_LINEAR_SRGB: [[f32; 3]; 3] = [
    [1.224_940_2, -0.224_940_18, 0.0],
    [-0.042_056_955, 1.042_057, 0.0],
    [-0.019_637_555, -0.078_636_05, 1.098_273_6],
];

pub(crate) fn srgb_to_linear(channel: f32) -> f32 {
    if channel.abs() <= 0.04045 {
        channel / 12.92
    } else {
        channel.signum() * ((channel.abs() + 0.055) / 1.055).powf(2.4)
    }
}

pub(crate) fn linear_to_srgb(channel: f32) -> f32 {
    if channel.abs() <= 0.0031308 {
        channel * 12.92
    } else {
        channel.signum() * (1.055 * channel.abs().powf(1.0 / 2.4) - 0.055)
    }
}

/// A color contains a nonfinite channel or invalid alpha.
#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum ColorError {
    #[error("RGB channels must be finite")]
    NonFiniteRgb,
    #[error("Alpha must be finite and between zero and one")]
    InvalidAlpha,
}

/// Straight-alpha linear sRGB with D65 white.
/// RGB channels may be negative or greater than one. Alpha is in `[0.0, 1.0]`.
/// Constructors convert source colors once; [`Self::linear_rgba`] returns those linear values.
///
/// # Examples
///
/// ```
/// use grafo::{Color, ColorError};
///
/// let red = Color::rgb(255, 0, 0);
/// let bright = Color::linear_srgb([4.0, 0.5, -0.1, 1.0])?;
///
/// assert_eq!(red.linear_rgba(), [1.0, 0.0, 0.0, 1.0]);
/// assert_eq!(bright.linear_rgba(), [4.0, 0.5, -0.1, 1.0]);
/// # Ok::<(), ColorError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Color {
    linear_rgba: [f32; 4],
}

impl Color {
    /// Transparent black.
    pub const TRANSPARENT: Self = Self {
        linear_rgba: [0.0, 0.0, 0.0, 0.0],
    };
    /// Opaque black.
    pub const BLACK: Self = Self {
        linear_rgba: [0.0, 0.0, 0.0, 1.0],
    };
    /// Opaque white.
    pub const WHITE: Self = Self {
        linear_rgba: [1.0, 1.0, 1.0, 1.0],
    };

    /// Creates an opaque color from sRGB channel bytes.
    pub fn rgb(red: u8, green: u8, blue: u8) -> Self {
        Self::rgba(red, green, blue, 255)
    }

    /// Creates a color from straight-alpha sRGB channel bytes.
    pub fn rgba(red: u8, green: u8, blue: u8, alpha: u8) -> Self {
        Self {
            linear_rgba: [
                srgb_to_linear(f32::from(red) / 255.0),
                srgb_to_linear(f32::from(green) / 255.0),
                srgb_to_linear(f32::from(blue) / 255.0),
                f32::from(alpha) / 255.0,
            ],
        }
    }

    /// Decodes straight-alpha sRGB channels. Nominal RGB values are in `[0.0, 1.0]`;
    /// negative and extended values use the signed sRGB transfer function.
    /// Returns an error if conversion produces nonfinite RGB or alpha is outside `[0.0, 1.0]`.
    pub fn srgb(rgba: [f32; 4]) -> Result<Self, ColorError> {
        let [red, green, blue, alpha] = rgba;
        Self::linear_srgb([
            srgb_to_linear(red),
            srgb_to_linear(green),
            srgb_to_linear(blue),
            alpha,
        ])
    }

    /// Converts straight-alpha Display P3 channels into linear sRGB without gamut clipping.
    /// Nominal RGB values are in `[0.0, 1.0]`. Alpha remains linear and unassociated.
    /// Returns an error if conversion produces nonfinite RGB or alpha is outside `[0.0, 1.0]`.
    pub fn display_p3(rgba: [f32; 4]) -> Result<Self, ColorError> {
        let [red, green, blue, alpha] = rgba;
        let linear_p3 = [red, green, blue].map(srgb_to_linear);
        let linear_srgb = DISPLAY_P3_TO_LINEAR_SRGB.map(|coefficients| {
            coefficients[0] * linear_p3[0]
                + coefficients[1] * linear_p3[1]
                + coefficients[2] * linear_p3[2]
        });
        Self::linear_srgb([linear_srgb[0], linear_srgb[1], linear_srgb[2], alpha])
    }

    /// Creates a color from straight-alpha linear sRGB without encoding or clipping RGB.
    /// Returns an error for nonfinite RGB or alpha outside `[0.0, 1.0]`.
    pub fn linear_srgb(rgba: [f32; 4]) -> Result<Self, ColorError> {
        if !rgba[..3].iter().all(|channel| channel.is_finite()) {
            return Err(ColorError::NonFiniteRgb);
        }
        if !(0.0..=1.0).contains(&rgba[3]) {
            return Err(ColorError::InvalidAlpha);
        }
        Ok(Self { linear_rgba: rgba })
    }

    /// Returns straight-alpha linear sRGB in `[red, green, blue, alpha]` order.
    pub fn linear_rgba(self) -> [f32; 4] {
        self.linear_rgba
    }
}

pub(crate) fn linear_to_srgb_u8(x: f32) -> u8 {
    let y = linear_to_srgb(x.clamp(0.0, 1.0));
    (y.clamp(0.0, 1.0) * 255.0 + 0.5).floor() as u8
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
    for pixel in pixels.chunks_mut(4) {
        let [red, green, blue, alpha] =
            Color::rgba(pixel[0], pixel[1], pixel[2], pixel[3]).linear_rgba();
        pixel[0] = linear_to_srgb_u8(red * alpha);
        pixel[1] = linear_to_srgb_u8(green * alpha);
        pixel[2] = linear_to_srgb_u8(blue * alpha);
    }
}
