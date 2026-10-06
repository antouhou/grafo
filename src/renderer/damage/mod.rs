use crate::core::effect::ShapeEffectBounds;
use crate::core::geometry;
use crate::core::{MathRect, UnsignedPhysicalRect, Viewport};
use crate::scene::types::DrawTreeNode;
use crate::scene::Scene;
use ahash::HashSet;

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

#[derive(Default)]
pub(super) struct PendingClipDamage {
    roots: HashSet<usize>,
    visited: HashSet<usize>,
    stack: Vec<usize>,
}

impl PendingClipDamage {
    pub(super) fn insert(&mut self, node_id: usize) {
        self.roots.insert(node_id);
    }

    pub(super) fn apply(
        &mut self,
        scene: &Scene,
        dirty_bounds: &mut Option<UnsignedPhysicalRect>,
        viewport: Viewport,
        fringe_width: f32,
    ) {
        if self.roots.is_empty() {
            return;
        }
        self.visited.clear();
        self.stack.clear();
        self.stack.extend(self.roots.drain());
        while let Some(node_id) = self.stack.pop() {
            if !self.visited.insert(node_id) {
                continue;
            }
            let Some(node) = scene.draw_tree.get(node_id) else {
                continue;
            };
            if matches!(node, DrawTreeNode::CachedShape(_)) {
                mark_dirty(
                    dirty_bounds,
                    node.logical_screen_bounds(),
                    viewport,
                    fringe_width,
                );
            }
            if let Some(effect) = scene.shape_effects.get(&node_id) {
                mark_shape_effect_dirty(dirty_bounds, effect.bounds, viewport);
            }
            self.stack
                .extend_from_slice(scene.draw_tree.children(node_id));
        }
    }
}
