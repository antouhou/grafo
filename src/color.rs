use crate::util::normalize_rgba_color;

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Color(pub [u8; 4]);

impl Color {
    pub const TRANSPARENT: Self = Self([0, 0, 0, 0]);
    pub const BLACK: Self = Self([0, 0, 0, 255]);

    pub fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self([r, g, b, 255])
    }

    pub fn rgba(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self([r, g, b, a])
    }

    pub fn normalize(&self) -> [f32; 4] {
        normalize_rgba_color(&self.0)
    }

    pub fn to_array(&self) -> [u8; 4] {
        self.0
    }
}
