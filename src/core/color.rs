use crate::core::util::{normalize_rgba_color, srgb_u8_to_linear};

/// An sRGB color with 8-bit red, green, blue, and alpha channels.
/// Alpha ranges from 0 for transparent to 255 for opaque. RGB is not premultiplied.
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
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Color(pub [u8; 4]);

impl Color {
    /// Transparent black.
    pub const TRANSPARENT: Self = Self([0, 0, 0, 0]);
    /// Opaque black.
    pub const BLACK: Self = Self([0, 0, 0, 255]);
    /// Opaque white.
    pub const WHITE: Self = Self([255, 255, 255, 255]);

    /// Creates an opaque color from sRGB channel values.
    pub fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self([r, g, b, 255])
    }

    /// Creates a color from sRGB channel values and alpha.
    pub fn rgba(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self([r, g, b, a])
    }

    /// Converts sRGB channels to linear RGB and divides alpha by 255.
    /// The returned channels are in `[0.0, 1.0]` and are not premultiplied.
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
        normalize_rgba_color(&self.0)
    }

    /// Returns the stored `[red, green, blue, alpha]` bytes.
    pub fn to_array(&self) -> [u8; 4] {
        self.0
    }
}

fn linear_to_srgb_u8(x: f32) -> u8 {
    let x = x.clamp(0.0, 1.0);
    let y = if x <= 0.0031308 {
        x * 12.92
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    };
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
