use crate::core::effect::ShapeMaskEffectBounds;
use crate::core::geometry;
use crate::core::{MathRect, UnsignedPhysicalRect, Viewport};
pub(super) use pending::PendingDamage;

mod pending;

pub(super) fn mark_physical_dirty(
    dirty_bounds: &mut Option<UnsignedPhysicalRect>,
    bounds: Option<UnsignedPhysicalRect>,
) {
    if let Some(bounds) = bounds {
        *dirty_bounds = Some(dirty_bounds.map_or(bounds, |dirty| dirty.union(&bounds)));
    }
}

pub(super) fn mark_dirty(
    dirty_bounds: &mut Option<UnsignedPhysicalRect>,
    logical_screen_bounds: MathRect,
    viewport: Viewport,
) {
    mark_physical_dirty(
        dirty_bounds,
        geometry::logical_bounds_to_viewport_rect(logical_screen_bounds, viewport),
    );
}

pub(super) fn mark_shape_mask_effect_dirty(
    dirty_bounds: &mut Option<UnsignedPhysicalRect>,
    bounds: ShapeMaskEffectBounds,
    viewport: Viewport,
) {
    // The cached raster rectangle already includes the mask filtering border.
    mark_dirty(dirty_bounds, bounds.logical_screen_bounds, viewport);
}
