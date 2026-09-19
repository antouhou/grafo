use crate::effect::{BackdropCaptureArea, BackdropEffectConfig};
use crate::pipeline::BackdropSamplingUniform;
use crate::renderer::types::DrawCommand;
use crate::vertex::InstanceTransform;
use tracing::warn;

const MAX_BACKDROP_CAPTURE_VIEWPORT_TEXEL_MULTIPLIER: u64 = 4;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) struct BackdropCaptureRegion {
    pub(super) capture_origin: (i32, i32),
    pub(super) capture_size: (u32, u32),
    pub(super) copy_source_origin: Option<(u32, u32)>,
    pub(super) copy_destination_origin: (u32, u32),
    pub(super) copy_size: (u32, u32),
}

impl BackdropCaptureRegion {
    pub(super) fn sample_uniform(self) -> BackdropSamplingUniform {
        BackdropSamplingUniform::new(self.capture_origin, self.capture_size)
    }
}

fn transform_point_to_logical_screen(
    point: (f32, f32),
    transform: Option<InstanceTransform>,
) -> (f32, f32) {
    let transform = transform.unwrap_or_else(InstanceTransform::identity);
    let homogeneous_x =
        transform.col0[0] * point.0 + transform.col1[0] * point.1 + transform.col3[0];
    let homogeneous_y =
        transform.col0[1] * point.0 + transform.col1[1] * point.1 + transform.col3[1];
    let homogeneous_w =
        transform.col0[3] * point.0 + transform.col1[3] * point.1 + transform.col3[3];
    let clamped_w = homogeneous_w.signum() * homogeneous_w.abs().max(1e-6);
    let inverse_w = 1.0 / clamped_w;
    (homogeneous_x * inverse_w, homogeneous_y * inverse_w)
}

fn transformed_bounds_to_logical_screen_rect(
    local_bounds: [(f32, f32); 2],
    transform: Option<InstanceTransform>,
) -> [(f32, f32); 2] {
    let corners = [
        (local_bounds[0].0, local_bounds[0].1),
        (local_bounds[1].0, local_bounds[0].1),
        (local_bounds[1].0, local_bounds[1].1),
        (local_bounds[0].0, local_bounds[1].1),
    ];

    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;

    for corner in corners {
        let (x, y) = transform_point_to_logical_screen(corner, transform);
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }

    [(min_x, min_y), (max_x, max_y)]
}

fn inflate_logical_rect(logical_rect: [(f32, f32); 2], padding: f32) -> [(f32, f32); 2] {
    if padding <= 0.0 {
        return logical_rect;
    }

    let min_x = logical_rect[0].0.min(logical_rect[1].0) - padding;
    let min_y = logical_rect[0].1.min(logical_rect[1].1) - padding;
    let max_x = logical_rect[0].0.max(logical_rect[1].0) + padding;
    let max_y = logical_rect[0].1.max(logical_rect[1].1) + padding;

    [(min_x, min_y), (max_x, max_y)]
}

fn logical_rect_is_finite(logical_rect: [(f32, f32); 2]) -> bool {
    logical_rect[0].0.is_finite()
        && logical_rect[0].1.is_finite()
        && logical_rect[1].0.is_finite()
        && logical_rect[1].1.is_finite()
}

fn round_capture_coordinate(value: f32, scale_factor: f32, round_outward: bool) -> Option<i32> {
    let scaled_value = value * scale_factor;
    if !scaled_value.is_finite() {
        return None;
    }

    let rounded_value = if round_outward {
        scaled_value.ceil()
    } else {
        scaled_value.floor()
    };
    if !rounded_value.is_finite()
        || rounded_value < i32::MIN as f32
        || rounded_value > i32::MAX as f32
    {
        return None;
    }

    Some(rounded_value as i32)
}

fn logical_rect_to_physical_capture_rect(
    logical_rect: [(f32, f32); 2],
    scale_factor: f64,
) -> Option<(i32, i32, u32, u32)> {
    let scale_factor = scale_factor as f32;
    if !scale_factor.is_finite() || scale_factor <= 0.0 || !logical_rect_is_finite(logical_rect) {
        return None;
    }

    let min_x = logical_rect[0].0.min(logical_rect[1].0);
    let min_y = logical_rect[0].1.min(logical_rect[1].1);
    let max_x = logical_rect[0].0.max(logical_rect[1].0);
    let max_y = logical_rect[0].1.max(logical_rect[1].1);

    let physical_min_x = round_capture_coordinate(min_x, scale_factor, false)?;
    let physical_min_y = round_capture_coordinate(min_y, scale_factor, false)?;
    let physical_max_x = round_capture_coordinate(max_x, scale_factor, true)?;
    let physical_max_y = round_capture_coordinate(max_y, scale_factor, true)?;

    let width = physical_max_x.saturating_sub(physical_min_x) as u32;
    let height = physical_max_y.saturating_sub(physical_min_y) as u32;
    if width == 0 || height == 0 {
        return None;
    }

    Some((physical_min_x, physical_min_y, width, height))
}

fn capture_size_exceeds_limits(capture_size: (u32, u32), max_capture_dimension: u32) -> bool {
    capture_size.0 > max_capture_dimension || capture_size.1 > max_capture_dimension
}

fn max_backdrop_capture_texels(physical_size: (u32, u32)) -> u64 {
    u64::from(physical_size.0)
        .saturating_mul(u64::from(physical_size.1))
        .saturating_mul(MAX_BACKDROP_CAPTURE_VIEWPORT_TEXEL_MULTIPLIER)
}

fn capture_size_exceeds_budget(capture_size: (u32, u32), physical_size: (u32, u32)) -> bool {
    let capture_texels = u64::from(capture_size.0).saturating_mul(u64::from(capture_size.1));
    capture_texels > max_backdrop_capture_texels(physical_size)
}

fn resolve_capture_region_to_viewport(
    requested_rect: (i32, i32, u32, u32),
    physical_size: (u32, u32),
) -> BackdropCaptureRegion {
    let (capture_x, capture_y, capture_width, capture_height) = requested_rect;
    let capture_right = capture_x.saturating_add(capture_width as i32);
    let capture_bottom = capture_y.saturating_add(capture_height as i32);

    let overlap_left = capture_x.max(0);
    let overlap_top = capture_y.max(0);
    let overlap_right = capture_right.min(physical_size.0 as i32);
    let overlap_bottom = capture_bottom.min(physical_size.1 as i32);

    let overlap_width = overlap_right.saturating_sub(overlap_left).max(0) as u32;
    let overlap_height = overlap_bottom.saturating_sub(overlap_top).max(0) as u32;
    let has_overlap = overlap_width > 0 && overlap_height > 0;

    BackdropCaptureRegion {
        capture_origin: (capture_x, capture_y),
        capture_size: (capture_width, capture_height),
        copy_source_origin: has_overlap.then_some((overlap_left as u32, overlap_top as u32)),
        copy_destination_origin: (
            overlap_left.saturating_sub(capture_x) as u32,
            overlap_top.saturating_sub(capture_y) as u32,
        ),
        copy_size: (overlap_width, overlap_height),
    }
}

pub(super) fn compute_backdrop_capture_region(
    draw_command: &DrawCommand,
    backdrop_config: BackdropEffectConfig,
    scale_factor: f64,
    physical_size: (u32, u32),
    max_capture_dimension: u32,
) -> Option<BackdropCaptureRegion> {
    let logical_rect = match backdrop_config.capture_area {
        BackdropCaptureArea::NodeBounds => transformed_bounds_to_logical_screen_rect(
            draw_command.local_bounds(),
            draw_command.transform(),
        ),
        BackdropCaptureArea::FullScene => {
            let logical_width = physical_size.0 as f32 / scale_factor as f32;
            let logical_height = physical_size.1 as f32 / scale_factor as f32;
            [(0.0, 0.0), (logical_width, logical_height)]
        }
        BackdropCaptureArea::ScreenRect(rect) => rect,
    };

    let logical_rect = inflate_logical_rect(logical_rect, backdrop_config.padding);

    logical_rect_to_physical_capture_rect(logical_rect, scale_factor).and_then(|requested_rect| {
        let capture_size = (requested_rect.2, requested_rect.3);
        if capture_size_exceeds_limits(capture_size, max_capture_dimension) {
            warn!(
                requested_width = capture_size.0,
                requested_height = capture_size.1,
                max_capture_dimension,
                "Skipping backdrop capture that exceeds supported texture dimensions"
            );
            return None;
        }

        if capture_size_exceeds_budget(capture_size, physical_size) {
            warn!(
                requested_width = capture_size.0,
                requested_height = capture_size.1,
                requested_texels =
                    u64::from(capture_size.0).saturating_mul(u64::from(capture_size.1)),
                max_capture_texels = max_backdrop_capture_texels(physical_size),
                viewport_width = physical_size.0,
                viewport_height = physical_size.1,
                "Skipping backdrop capture that exceeds the per-viewport texel budget"
            );
            return None;
        }

        Some(resolve_capture_region_to_viewport(
            requested_rect,
            physical_size,
        ))
    })
}

#[cfg(test)]
mod tests;
