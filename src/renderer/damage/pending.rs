use super::{mark_dirty, mark_shape_mask_effect_dirty};
use crate::core::UnsignedPhysicalRect;
use crate::scene::types::DrawTreeNode;
use crate::scene::Scene;
use ahash::HashSet;

#[derive(Default)]
pub(in crate::renderer) struct PendingDamage {
    roots: HashSet<usize>,
    visited: HashSet<usize>,
    stack: Vec<usize>,
}

impl PendingDamage {
    pub(in crate::renderer) fn insert(&mut self, node_id: usize) {
        self.roots.insert(node_id);
    }

    pub(in crate::renderer) fn apply(
        &mut self,
        scene: &Scene,
        dirty_bounds: &mut Option<UnsignedPhysicalRect>,
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
                    scene.viewport(),
                    scene.fringe_width(),
                );
            }
            if let Some(effect) = scene.shape_mask_effects.get(&node_id) {
                mark_shape_mask_effect_dirty(dirty_bounds, effect.bounds, scene.viewport());
            }
            self.stack
                .extend_from_slice(scene.draw_tree.children(node_id));
        }
    }
}
