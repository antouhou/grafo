use crate::core::effect::ShapeEffectBounds;
use crate::core::{MathRect, UnsignedPhysicalRect, Viewport};

pub(super) fn mark_dirty(
    dirty_bounds: &mut Option<UnsignedPhysicalRect>,
    logical_screen_bounds: MathRect,
    viewport: Viewport,
    fringe_width: f32,
) {
    let scale = viewport.scale_factor as f32;
    let bounds = logical_screen_bounds
        .scale(scale, scale)
        .inflate(fringe_width, fringe_width)
        .round_out();
    let viewport_bounds = UnsignedPhysicalRect::from_size(viewport.physical_size.into());
    let bounds = if bounds.is_finite() {
        let Some(bounds) = bounds.intersection(&viewport_bounds.to_f32()) else {
            return;
        };
        bounds.cast()
    } else {
        viewport_bounds
    };
    *dirty_bounds = Some(dirty_bounds.map_or(bounds, |dirty| dirty.union(&bounds)));
}

pub(super) fn mark_shape_effect_dirty(
    dirty_bounds: &mut Option<UnsignedPhysicalRect>,
    bounds: ShapeEffectBounds,
    viewport: Viewport,
) {
    // The cached raster rectangle already includes the fringe guard.
    mark_dirty(dirty_bounds, bounds.logical_screen_bounds, viewport, 0.0);
}
