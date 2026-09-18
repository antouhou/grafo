use crate::Color;

/// A shape's stroke width in pixels and its color.
///
/// The default has zero width and [`Color::TRANSPARENT`].
///
/// # Examples
///
/// ```
/// use grafo::{Color, Stroke};
///
/// let stroke = Stroke::new(2.0_f32, Color::rgb(255, 0, 0));
/// assert!(!stroke.is_empty());
/// assert!(Stroke::default().is_empty());
/// ```
#[derive(Clone, Debug, Copy, PartialEq, Default)]
pub struct Stroke {
    /// The width of the stroke in pixels.
    pub width: f32,
    /// The color of the stroke.
    pub color: Color,
}

impl Stroke {
    /// Creates a stroke with the given width in pixels and color.
    #[inline]
    pub fn new(width: impl Into<f32>, color: impl Into<Color>) -> Self {
        Self {
            width: width.into(),
            color: color.into(),
        }
    }

    /// Returns whether the width is at most zero or the color equals [`Color::TRANSPARENT`].
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.width <= 0.0 || self.color == Color::TRANSPARENT
    }
}
