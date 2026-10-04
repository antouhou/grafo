use crate::core::effect::ShapeEffectBounds;
use crate::core::geometry;
use crate::core::{MathRect, UnsignedPhysicalRect, Viewport};

pub(super) fn mark_dirty(
    dirty_bounds: &mut Option<UnsignedPhysicalRect>,
    logical_screen_bounds: MathRect,
    viewport: Viewport,
    fringe_width: f32,
) {
    mark_physical_dirty(
        dirty_bounds,
        geometry::logical_bounds_to_viewport_rect(logical_screen_bounds, viewport, fringe_width),
    );
}

pub(super) fn mark_physical_dirty(
    dirty_bounds: &mut Option<UnsignedPhysicalRect>,
    bounds: Option<UnsignedPhysicalRect>,
) {
    if let Some(bounds) = bounds {
        *dirty_bounds = Some(dirty_bounds.map_or(bounds, |dirty| dirty.union(&bounds)));
    }
}

pub(super) fn mark_shape_effect_dirty(
    dirty_bounds: &mut Option<UnsignedPhysicalRect>,
    bounds: ShapeEffectBounds,
    viewport: Viewport,
) {
    // The cached raster rectangle already includes the fringe guard.
    mark_dirty(dirty_bounds, bounds.logical_screen_bounds, viewport, 0.0);
}
