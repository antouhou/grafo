use crate::util::normalize_rgba_color;

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
