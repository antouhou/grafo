use crate::util::normalize_rgba_color;

/// Represents a color in RGBA format.
///
/// This struct encapsulates color information using red, green, blue, and alpha (opacity) channels.
/// Each channel is an 8-bit unsigned integer.
///
/// # Examples
///
/// Creating and manipulating colors:
///
/// ```
/// use grafo::Color;
///
/// // Create a black color
/// let black = Color::BLACK;
///
/// // Create a red color with full opacity
/// let red = Color::rgb(255, 0, 0);
///
/// // Create a semi-transparent blue color
/// let semi_blue = Color::rgba(0, 0, 255, 128);
///
/// // Normalize the color values to [0.0, 1.0]
/// let normalized = red.normalize();
/// assert_eq!(normalized, [1.0, 0.0, 0.0, 1.0]);
///
/// // Convert the color to an array
/// let color_array = semi_blue.to_array();
/// assert_eq!(color_array, [0, 0, 255, 128]);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Color(pub [u8; 4]);

impl Color {
    /// A transparent color.
    ///
    /// All color channels are set to zero, making the color fully transparent.
    pub const TRANSPARENT: Self = Self([0, 0, 0, 0]);
    /// A black color.
    ///
    /// Red, green, and blue channels are set to zero, and alpha is fully opaque.
    pub const BLACK: Self = Self([0, 0, 0, 255]);
    /// A white color.
    ///
    /// Red, green, and blue channels are set to zero, and alpha is fully opaque.
    pub const WHITE: Self = Self([255, 255, 255, 255]);

    /// Creates a new color with the specified RGB values and full opacity.
    ///
    /// # Parameters
    ///
    /// - `r`: Red channel (0-255)
    /// - `g`: Green channel (0-255)
    /// - `b`: Blue channel (0-255)
    ///
    /// # Examples
    ///
    /// ```
    /// use grafo::Color;
    ///
    /// let green = Color::rgb(0, 255, 0);
    /// assert_eq!(green, Color([0, 255, 0, 255]));
    /// ```
    pub fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self([r, g, b, 255])
    }

    /// Creates a new color with the specified RGBA values.
    ///
    /// # Parameters
    ///
    /// - `r`: Red channel (0-255)
    /// - `g`: Green channel (0-255)
    /// - `b`: Blue channel (0-255)
    /// - `a`: Alpha channel (0-255), where 0 is fully transparent and 255 is fully opaque
    ///
    /// # Examples
    ///
    /// ```
    /// use grafo::Color;
    ///
    /// // Semi-transparent purple
    /// let purple = Color::rgba(128, 0, 128, 128);
    /// assert_eq!(purple, Color([128, 0, 128, 128]));
    /// ```
    pub fn rgba(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self([r, g, b, a])
    }

    /// Normalizes the color values to the range [0.0, 1.0].
    ///
    /// This is useful for graphics operations that require floating-point color values.
    ///
    /// # Examples
    ///
    /// ```
    /// use grafo::Color;
    ///
    /// let red = Color::rgb(255, 0, 0);
    /// let normalized = red.normalize();
    /// assert_eq!(normalized, [1.0, 0.0, 0.0, 1.0]);
    /// ```
    pub fn normalize(&self) -> [f32; 4] {
        normalize_rgba_color(&self.0)
    }

    /// Returns the color as an array of 4 `u8` values.
    ///
    /// # Examples
    ///
    /// ```
    /// use grafo::Color;
    ///
    /// let blue = Color::rgb(0, 0, 255);
    /// let array = blue.to_array();
    /// assert_eq!(array, [0, 0, 255, 255]);
    /// ```
    pub fn to_array(&self) -> [u8; 4] {
        self.0
    }
}
