//! Geometry types and coordinate conversions backed by Lyon and Euclid.

use crate::core::vertex::InstanceTransform;
use lyon::geom::euclid::default::Transform3D;
use lyon::geom::{self, Box2D, Point};
use lyon::math;

/// Rectangle with floating-point coordinates.
pub type MathRect = math::Box2D;

/// Physical pixel bounds that can extend into negative offscreen coordinates.
pub type PhysicalRect = Box2D<i32>;

/// Physical pixel bounds with nonnegative coordinates.
pub type UnsignedPhysicalRect = Box2D<u32>;

/// Position in physical pixels with nonnegative coordinates.
pub type UnsignedPhysicalPoint = Point<u32>;

/// Width and height in physical pixels.
pub type Size = geom::Size<u32>;

/// Physical output dimensions and logical-to-physical coordinate scale.
#[derive(Clone, Copy, Debug)]
pub struct Viewport {
    pub physical_size: (u32, u32),
    pub scale_factor: f64,
}

/// Includes antialiasing coverage and clips logical bounds to the viewport.
pub(crate) fn logical_bounds_to_viewport_rect(
    logical_screen_bounds: MathRect,
    viewport: Viewport,
    fringe_width: f32,
) -> Option<UnsignedPhysicalRect> {
    let scale = viewport.scale_factor as f32;
    let bounds = logical_screen_bounds
        .scale(scale, scale)
        .inflate(fringe_width, fringe_width)
        .round_out();
    let viewport_bounds = UnsignedPhysicalRect::from_size(viewport.physical_size.into());
    if bounds.is_finite() {
        bounds
            .intersection(&viewport_bounds.to_f32())
            .map(|bounds| bounds.cast())
    } else {
        Some(viewport_bounds)
    }
}

pub(crate) fn is_axis_aligned_rect_transform(transform: Option<InstanceTransform>) -> bool {
    let transform = transform.unwrap_or_else(InstanceTransform::identity);

    if transform.col0[3] != 0.0 || transform.col1[3] != 0.0 || transform.col3[3] != 1.0 {
        return false;
    }

    if transform.col0[1] != 0.0 || transform.col1[0] != 0.0 {
        return false;
    }

    true
}

/// Resolves logical screen bounds to an outward-rounded viewport scissor.
pub(crate) fn logical_rect_to_scissor_rect(
    logical_screen_bounds: MathRect,
    scale_factor: f64,
    physical_size: Size,
) -> Option<UnsignedPhysicalRect> {
    let scale_factor = scale_factor as f32;
    let physical_rect = logical_screen_bounds
        .scale(scale_factor, scale_factor)
        .round_out();
    let viewport = MathRect::from_size(physical_size.to_f32());

    physical_rect
        .intersection(&viewport)
        .unwrap_or_else(MathRect::zero)
        .try_cast()
}

fn transform_point_to_logical_screen(
    point: Point<f32>,
    transform: Option<InstanceTransform>,
) -> Point<f32> {
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

pub(crate) fn transformed_bounds_to_logical_screen_rect(
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
pub(crate) fn logical_rect_to_physical_rect(
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
pub(crate) fn compute_downsampled_dimensions(full_resolution_size: Size, downsample: f32) -> Size {
    let texture_size = (full_resolution_size.to_f32() * downsample)
        .ceil()
        .max(Size::new(1, 1).to_f32());
    Size::new(texture_size.width as u32, texture_size.height as u32)
}

/// Maps a unit quad to local bounds before applying the source transform.
pub(crate) fn unit_quad_transform(
    local_bounds: [(f32, f32); 2],
    source_transform: Option<InstanceTransform>,
) -> InstanceTransform {
    let [(minimum_x, minimum_y), (maximum_x, maximum_y)] = local_bounds;
    let bounds_transform = InstanceTransform::affine_2d(
        maximum_x - minimum_x,
        0.0,
        0.0,
        maximum_y - minimum_y,
        minimum_x,
        minimum_y,
    );
    source_transform.map_or(bounds_transform, |transform| {
        bounds_transform.then(&transform)
    })
}

#[cfg(test)]
mod tests;
