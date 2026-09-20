use super::types::DrawCommand;
use crate::effect::EffectInstance;
use crate::vertex::InstanceTransform;
use ahash::HashMap;

#[derive(Clone, Copy)]
pub(super) struct AxisAlignedRectTransform {
    pub(super) scale_x: f32,
    pub(super) scale_y: f32,
    pub(super) translate_x: f32,
    pub(super) translate_y: f32,
}

pub(super) fn extract_axis_aligned_rect_transform(
    transform: Option<InstanceTransform>,
) -> Option<AxisAlignedRectTransform> {
    let transform = transform.unwrap_or_else(InstanceTransform::identity);

    if transform.col0[3] != 0.0 || transform.col1[3] != 0.0 || transform.col3[3] != 1.0 {
        return None;
    }

    if transform.col0[1] != 0.0 || transform.col1[0] != 0.0 {
        return None;
    }

    Some(AxisAlignedRectTransform {
        scale_x: transform.col0[0],
        scale_y: transform.col1[1],
        translate_x: transform.col3[0],
        translate_y: transform.col3[1],
    })
}

pub(super) fn should_skip_visible_rect_draw(
    node_id: usize,
    draw_command: &DrawCommand,
    group_effects: &HashMap<usize, EffectInstance>,
    backdrop_effects: &HashMap<usize, EffectInstance>,
) -> bool {
    if !draw_command.is_rect() {
        return false;
    }

    if group_effects.contains_key(&node_id) || backdrop_effects.contains_key(&node_id) {
        return false;
    }

    if draw_command.texture_id(0).is_some() || draw_command.texture_id(1).is_some() {
        return false;
    }

    // Gradient-filled shapes are visually active even before GPU prep creates bind groups.
    if draw_command.has_gradient_fill() {
        return false;
    }

    if draw_command
        .instance_color_override()
        .is_some_and(|color| color[3] != 0.0)
    {
        return false;
    }

    extract_axis_aligned_rect_transform(draw_command.transform()).is_some()
}

/// Compute a screen-space scissor rect from a local-space axis-aligned rect and its transform.
pub(super) fn compute_scissor_rect(
    rect: [(f32, f32); 2],
    transform: Option<InstanceTransform>,
    scale_factor: f64,
    physical_size: (u32, u32),
) -> Option<(u32, u32, u32, u32)> {
    let axis_aligned_transform = extract_axis_aligned_rect_transform(transform)?;

    let x0 = rect[0].0 * axis_aligned_transform.scale_x + axis_aligned_transform.translate_x;
    let y0 = rect[0].1 * axis_aligned_transform.scale_y + axis_aligned_transform.translate_y;
    let x1 = rect[1].0 * axis_aligned_transform.scale_x + axis_aligned_transform.translate_x;
    let y1 = rect[1].1 * axis_aligned_transform.scale_y + axis_aligned_transform.translate_y;

    let min_x = x0.min(x1);
    let min_y = y0.min(y1);
    let max_x = x0.max(x1);
    let max_y = y0.max(y1);

    let scale_factor = scale_factor as f32;
    let px_min_x = ((min_x * scale_factor).floor().max(0.0) as u32).min(physical_size.0);
    let px_min_y = ((min_y * scale_factor).floor().max(0.0) as u32).min(physical_size.1);
    let px_max_x = (max_x * scale_factor).ceil().min(physical_size.0 as f32) as u32;
    let px_max_y = (max_y * scale_factor).ceil().min(physical_size.1 as f32) as u32;

    let width = px_max_x.saturating_sub(px_min_x);
    let height = px_max_y.saturating_sub(px_min_y);

    Some((px_min_x, px_min_y, width, height))
}

/// Intersect two scissor rects, returning the overlapping region.
/// If the rects don't overlap, returns a zero-size rect.
pub(super) fn intersect_scissor(
    a: (u32, u32, u32, u32),
    b: (u32, u32, u32, u32),
) -> (u32, u32, u32, u32) {
    let a_right = a.0 + a.2;
    let a_bottom = a.1 + a.3;
    let b_right = b.0 + b.2;
    let b_bottom = b.1 + b.3;

    let left = a.0.max(b.0);
    let top = a.1.max(b.1);
    let right = a_right.min(b_right);
    let bottom = a_bottom.min(b_bottom);

    let width = right.saturating_sub(left);
    let height = bottom.saturating_sub(top);

    (left, top, width, height)
}

/// Returns a scissor rect when the draw command is a rectangle whose transform
/// preserves axis alignment.
pub(super) fn try_scissor_for_rect(
    draw_command: &DrawCommand,
    scale_factor: f64,
    physical_size: (u32, u32),
) -> Option<(u32, u32, u32, u32)> {
    if !draw_command.is_rect() {
        return None;
    }
    let rect_bounds = draw_command.rect_bounds()?;
    let transform = draw_command.transform();
    compute_scissor_rect(rect_bounds, transform, scale_factor, physical_size)
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

pub(super) fn transformed_bounds_to_logical_screen_rect(
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

pub(super) fn inflate_logical_rect(logical_rect: [(f32, f32); 2], padding: f32) -> [(f32, f32); 2] {
    if padding <= 0.0 {
        return logical_rect;
    }

    let min_x = logical_rect[0].0.min(logical_rect[1].0) - padding;
    let min_y = logical_rect[0].1.min(logical_rect[1].1) - padding;
    let max_x = logical_rect[0].0.max(logical_rect[1].0) + padding;
    let max_y = logical_rect[0].1.max(logical_rect[1].1) + padding;

    [(min_x, min_y), (max_x, max_y)]
}

fn is_logical_rect_finite(logical_rect: [(f32, f32); 2]) -> bool {
    logical_rect[0].0.is_finite()
        && logical_rect[0].1.is_finite()
        && logical_rect[1].0.is_finite()
        && logical_rect[1].1.is_finite()
}

fn round_physical_coordinate(value: f32, scale_factor: f32, should_round_up: bool) -> Option<i32> {
    let scaled_value = value * scale_factor;
    if !scaled_value.is_finite() {
        return None;
    }

    let rounded_value = if should_round_up {
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

/// Rounds outward and retains signed offscreen coordinates without viewport clipping.
pub(super) fn logical_rect_to_physical_rect(
    logical_rect: [(f32, f32); 2],
    scale_factor: f64,
) -> Option<(i32, i32, u32, u32)> {
    let scale_factor = scale_factor as f32;
    if !scale_factor.is_finite() || scale_factor <= 0.0 || !is_logical_rect_finite(logical_rect) {
        return None;
    }

    let min_x = logical_rect[0].0.min(logical_rect[1].0);
    let min_y = logical_rect[0].1.min(logical_rect[1].1);
    let max_x = logical_rect[0].0.max(logical_rect[1].0);
    let max_y = logical_rect[0].1.max(logical_rect[1].1);

    let physical_min_x = round_physical_coordinate(min_x, scale_factor, false)?;
    let physical_min_y = round_physical_coordinate(min_y, scale_factor, false)?;
    let physical_max_x = round_physical_coordinate(max_x, scale_factor, true)?;
    let physical_max_y = round_physical_coordinate(max_y, scale_factor, true)?;

    let width = physical_max_x.saturating_sub(physical_min_x) as u32;
    let height = physical_max_y.saturating_sub(physical_min_y) as u32;
    if width == 0 || height == 0 {
        return None;
    }

    Some((physical_min_x, physical_min_y, width, height))
}

/// Rounds downsampled texture dimensions up and keeps at least one texel per axis.
pub(super) fn compute_downsampled_dimensions(
    full_resolution_size: (u32, u32),
    downsample: f32,
) -> (u32, u32) {
    (
        ((full_resolution_size.0 as f32) * downsample)
            .ceil()
            .max(1.0) as u32,
        ((full_resolution_size.1 as f32) * downsample)
            .ceil()
            .max(1.0) as u32,
    )
}

#[cfg(test)]
mod tests;
