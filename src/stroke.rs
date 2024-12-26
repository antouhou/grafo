//! Stroke properties for shapes in the Grafo library.
//!
//! This module defines the `Stroke` struct, which represents the stroke properties of a shape.
//!
//! # Examples
//!
//! Creating and using stroke properties:
///
/// ```
/// use grafo::Color;
/// use grafo::Stroke;
///
/// // Create a red stroke with a width of 2.0
/// let red_stroke = Stroke::new(2.0, Color::rgb(255, 0, 0));
///
/// // Create a transparent stroke using default values
/// let transparent_stroke = Stroke::default();
///
/// // Check if a stroke is empty
/// assert!(!red_stroke.is_empty());
/// assert!(transparent_stroke.is_empty());
/// ```
use crate::Color;

/// Represents the stroke properties of a shape.
///
/// The `Stroke` struct allows you to define the visual outline of shapes with specific width and color.
///
/// # Examples
///
/// ```
/// use grafo::Color;
/// use grafo::Stroke;
///
/// // Create a blue stroke with a width of 1.5
/// let blue_stroke = Stroke::new(1.5, Color::rgb(0, 0, 255));
///
/// // Check if the stroke is empty
/// assert!(!blue_stroke.is_empty());
///
/// // Create a stroke with zero width
/// let no_stroke = Stroke::new(0.0, Color::rgb(0, 0, 0));
/// assert!(no_stroke.is_empty());
/// ```
#[derive(Clone, Debug, Copy, PartialEq, Default)]
pub struct Stroke {
    /// The width of the stroke in pixels.
    pub width: f32,
    /// The color of the stroke.
    pub color: Color,
}

impl Stroke {
    /// Creates a new `Stroke` with the specified width and color.
    ///
    /// # Parameters
    ///
    /// - `width`: The width of the stroke in pixels. Must be a positive value for the stroke to be visible.
    /// - `color`: The color of the stroke. Use `Color::TRANSPARENT` for a transparent stroke.
    ///
    /// # Examples
    ///
    /// ```
    /// use grafo::Color;
    /// use grafo::Stroke;
    ///
    /// // Create an orange stroke with a width of 3.0
    /// let orange_stroke = Stroke::new(3.0, Color::rgba(255, 165, 0, 255));
    /// ```
    #[inline]
    pub fn new(width: impl Into<f32>, color: impl Into<Color>) -> Self {
        Self {
            width: width.into(),
            color: color.into(),
        }
    }

    /// Determines whether the stroke is empty.
    ///
    /// A stroke is considered empty if its width is zero or if its color is fully transparent.
    ///
    /// # Examples
    ///
    /// ```
    /// use grafo::Color;
    /// use grafo::Stroke;
    ///
    /// let empty_stroke = Stroke::default();
    /// assert!(empty_stroke.is_empty());
    ///
    /// let visible_stroke = Stroke::new(1.0, Color::BLACK);
    /// assert!(!visible_stroke.is_empty());
    /// ```
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.width <= 0.0 || self.color == Color::TRANSPARENT
    }
}
