use crate::wgpu_state::Color;

#[derive(Clone, Debug, Copy, PartialEq, Default)]
pub struct Stroke {
    pub width: f32,
    pub color: Color,
}

impl Stroke {
    #[inline]
    pub fn new(width: impl Into<f32>, color: impl Into<Color>) -> Self {
        Self {
            width: width.into(),
            color: color.into(),
        }
    }

    /// True if width is zero or color is transparent
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.width <= 0.0 || self.color == Color::TRANSPARENT
    }
}