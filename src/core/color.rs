/// Decodes an extended sRGB channel without clamping.
pub(crate) fn srgb_to_linear(value: f32) -> f32 {
    if value.abs() <= 0.04045 {
        value / 12.92
    } else {
        value.signum() * ((value.abs() + 0.055) / 1.055).powf(2.4)
    }
}

/// Encodes a linear sRGB channel without clamping.
pub(crate) fn linear_to_srgb(value: f32) -> f32 {
    if value.abs() <= 0.0031308 {
        value * 12.92
    } else {
        let sign = value.signum();
        sign * (1.055 * value.abs().powf(1.0 / 2.4) - 0.055)
    }
}

/// Converts an sRGB byte to a linear channel value between 0.0 and 1.0.
fn srgb_u8_to_linear(value: u8) -> f32 {
    srgb_to_linear(f32::from(value) / 255.0)
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
        Self([r, g, b, a].map(|channel| f32::from(channel) / 255.0))
    }

    /// Creates an opaque color from floating-point encoded sRGB channel values.
    /// RGB channels must be finite and may be negative or greater than one.
    pub const fn rgb_f32(red: f32, green: f32, blue: f32) -> Self {
        Self::rgba_f32(red, green, blue, 1.0)
    }

    /// Creates a color from floating-point encoded sRGB channel values and alpha.
    /// Channels must be finite. RGB may be negative or greater than one;
    /// alpha must be in `[0.0, 1.0]`. Values are stored without clamping.
    pub const fn rgba_f32(red: f32, green: f32, blue: f32, alpha: f32) -> Self {
        Self([red, green, blue, alpha])
    }

    /// Converts encoded sRGB channels to linear RGB and leaves alpha unchanged.
    /// RGB may be negative or greater than one. The result is not premultiplied.
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

    /// Returns `[red, green, blue, alpha]` as bytes.
    /// Clamps each channel to `[0.0, 1.0]` and rounds to the nearest byte value.
    pub fn to_array(&self) -> [u8; 4] {
        self.0
            .map(|channel| (channel.clamp(0.0, 1.0) * 255.0).round() as u8)
    }

    /// Returns the encoded sRGB channels and alpha without clamping or quantization.
    pub const fn to_array_f32(&self) -> [f32; 4] {
        self.0
    }
}

pub(crate) fn linear_to_srgb_u8(value: f32) -> u8 {
    let encoded = linear_to_srgb(value.clamp(0.0, 1.0));
    (encoded.clamp(0.0, 1.0) * 255.0 + 0.5).floor() as u8
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
