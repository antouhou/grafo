/// Decodes an extended sRGB channel without clamping.
pub(crate) fn srgb_to_linear(value: f32) -> f32 {
    if value.abs() <= 0.04045 {
        value / 12.92
    } else {
        value.signum() * ((value.abs() + 0.055) / 1.055).powf(2.4)
    }
}

/// Converts an sRGB byte to a linear channel value between 0.0 and 1.0.
pub(crate) fn srgb_u8_to_linear(value: u8) -> f32 {
    srgb_to_linear(f32::from(value) / 255.0)
}

pub fn normalize_rgba_color(color: &[f32; 4]) -> [f32; 4] {
    [
        srgb_to_linear(color[0]),
        srgb_to_linear(color[1]),
        srgb_to_linear(color[2]),
        color[3],
    ]
}

#[inline(always)]
pub fn to_logical(physical_size: (u32, u32), scale_factor: f64) -> (f32, f32) {
    let (physical_width, physical_height) = physical_size;
    let logical_width = physical_width as f64 / scale_factor;
    let logical_height = physical_height as f64 / scale_factor;
    (logical_width as f32, logical_height as f32)
}
