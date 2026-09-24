use crate::core::cache::Cache;
use crate::core::shape::AaFringeScratch;

/// Converts an sRGB byte to a linear channel value between 0.0 and 1.0.
pub(crate) fn srgb_u8_to_linear(value: u8) -> f32 {
    let normalized = value as f32 / 255.0;
    if normalized <= 0.04045 {
        normalized / 12.92
    } else {
        ((normalized + 0.055) / 1.055).powf(2.4)
    }
}

pub fn normalize_rgba_color(color: &[u8; 4]) -> [f32; 4] {
    [
        srgb_u8_to_linear(color[0]),
        srgb_u8_to_linear(color[1]),
        srgb_u8_to_linear(color[2]),
        color[3] as f32 / 255.0, // alpha is linear, not gamma-encoded
    ]
}

pub(crate) struct ShapeResources {
    pub tessellation_cache: Cache,
    pub aa_fringe_scratch: AaFringeScratch,
}

impl ShapeResources {
    pub(crate) fn new() -> Self {
        Self {
            tessellation_cache: Cache::new(),
            aa_fringe_scratch: AaFringeScratch::new(),
        }
    }

    pub fn print_sizes(&self) {
        println!("Tessellations: {}", self.tessellation_cache.len());
    }
}

#[inline(always)]
pub fn to_logical(physical_size: (u32, u32), scale_factor: f64) -> (f32, f32) {
    let (physical_width, physical_height) = physical_size;
    let logical_width = physical_width as f64 / scale_factor;
    let logical_height = physical_height as f64 / scale_factor;
    (logical_width as f32, logical_height as f32)
}
