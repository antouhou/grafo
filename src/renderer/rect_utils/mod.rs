use super::types::DrawCommand;
use crate::effect::{BackdropEffectInstance, EffectInstance};
use crate::vertex::InstanceTransform;
use crate::{MathRect, PhysicalRect, Size, UnsignedPhysicalRect};
use ahash::HashMap;
use lyon::geom::euclid::default::Transform3D;
use lyon::math::{Point, Transform};

pub(super) fn extract_axis_aligned_rect_transform(
    transform: Option<InstanceTransform>,
) -> Option<Transform> {
    let transform = transform.unwrap_or_else(InstanceTransform::identity);

    if transform.col0[3] != 0.0 || transform.col1[3] != 0.0 || transform.col3[3] != 1.0 {
        return None;
    }

    if transform.col0[1] != 0.0 || transform.col1[0] != 0.0 {
        return None;
    }

    Some(Transform::new(
        transform.col0[0],
        0.0,
        0.0,
        transform.col1[1],
        transform.col3[0],
        transform.col3[1],
    ))
}

pub(super) fn should_skip_visible_rect_draw(
    node_id: usize,
    draw_command: &DrawCommand,
    group_effects: &HashMap<usize, EffectInstance>,
    backdrop_effects: &HashMap<usize, BackdropEffectInstance>,
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

/// Resolves an axis-aligned rectangle to an outward-rounded viewport scissor.
pub(super) fn compute_scissor_rect(
    rect: MathRect,
    transform: Option<InstanceTransform>,
    scale_factor: f64,
    physical_size: Size,
) -> Option<UnsignedPhysicalRect> {
    let transform = extract_axis_aligned_rect_transform(transform)?;
    let scale_factor = scale_factor as f32;
    let physical_rect = transform
        .outer_transformed_box(&rect)
        .scale(scale_factor, scale_factor)
        .round_out();
    let viewport = MathRect::from_size(physical_size.to_f32());

    physical_rect
        .intersection(&viewport)
        .unwrap_or_else(MathRect::zero)
        .try_cast()
}

/// Returns a scissor rect when the draw command's transform preserves axis alignment.
pub(super) fn try_scissor_for_rect(
    draw_command: &DrawCommand,
    scale_factor: f64,
    physical_size: Size,
) -> Option<UnsignedPhysicalRect> {
    if !draw_command.is_rect() {
        return None;
    }
    let rect_bounds = draw_command.rect_bounds()?;
    let rect = MathRect::new(rect_bounds[0].into(), rect_bounds[1].into());
    compute_scissor_rect(rect, draw_command.transform(), scale_factor, physical_size)
}

fn transform_point_to_logical_screen(point: Point, transform: Option<InstanceTransform>) -> Point {
    let transform = Transform3D::from_arrays(
        transform
            .unwrap_or_else(InstanceTransform::identity)
            .as_cols(),
    );
    let homogeneous = transform.transform_point2d_homogeneous(point);
    let clamped_w = homogeneous.w.signum() * homogeneous.w.abs().max(1e-6);
    let inverse_w = 1.0 / clamped_w;
    Point::new(homogeneous.x * inverse_w, homogeneous.y * inverse_w)
}

pub(super) fn transformed_bounds_to_logical_screen_rect(
    local_bounds: MathRect,
    transform: Option<InstanceTransform>,
) -> MathRect {
    let corners = [
        local_bounds.min,
        Point::new(local_bounds.max.x, local_bounds.min.y),
        local_bounds.max,
        Point::new(local_bounds.min.x, local_bounds.max.y),
    ];
    MathRect::from_points(
        corners.map(|corner| transform_point_to_logical_screen(corner, transform)),
    )
}

/// Keeps signed offscreen coordinates and rejects bounds or extents that exceed i32.
pub(super) fn logical_rect_to_physical_rect(
    logical_rect: MathRect,
    scale_factor: f64,
) -> Option<PhysicalRect> {
    let scale_factor = scale_factor as f32;
    if !scale_factor.is_finite() || scale_factor <= 0.0 || !logical_rect.is_finite() {
        return None;
    }

    let physical_rect = logical_rect
        .scale(scale_factor, scale_factor)
        .round_out()
        .try_cast::<i32>()?;
    if physical_rect.is_empty() {
        return None;
    }
    // Check the extent without overflowing when the endpoints span most of i32.
    physical_rect.to_i64().size().try_cast::<i32>()?;
    Some(physical_rect)
}

/// Rounds downsampled texture dimensions up and keeps at least one texel per axis.
pub(super) fn compute_downsampled_dimensions(full_resolution_size: Size, downsample: f32) -> Size {
    let texture_size = (full_resolution_size.to_f32() * downsample)
        .ceil()
        .max(Size::new(1, 1).to_f32());
    Size::new(texture_size.width as u32, texture_size.height as u32)
}

#[cfg(test)]
mod tests;
