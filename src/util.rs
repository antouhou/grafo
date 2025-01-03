use crate::renderer::MathRect;
use lyon::geom::euclid::Point2D;

pub fn normalize_rgba_color(color: &[u8; 4]) -> [f32; 4] {
    [
        color[0] as f32 / 255.0,
        color[1] as f32 / 255.0,
        color[2] as f32 / 255.0,
        color[3] as f32 / 255.0,
    ]
}

#[inline(always)]
pub fn normalize_rect(
    logical_rect: &MathRect,
    canvas_physical_size: (u32, u32),
    scale_factor: f32,
) -> MathRect {
    let ndc_min_x = 2.0 * logical_rect.min.x * scale_factor / canvas_physical_size.0 as f32 - 1.0;
    let ndc_min_y = 1.0 - 2.0 * logical_rect.min.y * scale_factor / canvas_physical_size.1 as f32;
    let ndc_max_x = 2.0 * logical_rect.max.x * scale_factor / canvas_physical_size.0 as f32 - 1.0;
    let ndc_max_y = 1.0 - 2.0 * logical_rect.max.y * scale_factor / canvas_physical_size.1 as f32;

    MathRect {
        min: Point2D::new(ndc_min_x, ndc_min_y),
        max: Point2D::new(ndc_max_x, ndc_max_y),
    }
}

// #[inline(always)]
// fn srgb_to_linear(value: f32) -> f32 {
//     if value <= 0.04045 {
//         value / 12.92
//     } else {
//         ((value + 0.055) / 1.055).powf(2.4)
//     }
// }

// pub fn rgba_to_linear(rgba: [f32; 4]) -> [f32; 4] {
//     let [r, g, b, a] = rgba;
//     [
//         srgb_to_linear(r),
//         srgb_to_linear(g),
//         srgb_to_linear(b),
//         a, // Alpha channel remains unchanged
//     ]
// }

#[inline(always)]
pub fn to_logical(physical_size: (u32, u32), scale_factor: f64) -> (f32, f32) {
    let (physical_width, physical_height) = physical_size;
    let logical_width = physical_width as f64 / scale_factor;
    let logical_height = physical_height as f64 / scale_factor;
    (logical_width as f32, logical_height as f32)
}
