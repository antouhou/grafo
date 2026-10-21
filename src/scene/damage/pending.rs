use super::{mark_dirty, mark_shape_mask_effect_dirty};
use crate::core::UnsignedPhysicalRect;
use crate::scene_state::types::DrawTreeNode;
use crate::scene_state::SceneState;
use ahash::HashSet;

#[derive(Default)]
pub(in crate::scene) struct PendingDamage {
    roots: HashSet<usize>,
    visited: HashSet<usize>,
    stack: Vec<usize>,
}

impl PendingDamage {
    pub(in crate::scene) fn insert(&mut self, node_id: usize) {
        self.roots.insert(node_id);
    }

    pub(in crate::scene) fn apply(
        &mut self,
        state: &SceneState,
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
            let Some(node) = state.draw_tree.get(node_id) else {
                continue;
            };
            if matches!(node, DrawTreeNode::CachedShape(_)) {
                mark_dirty(
                    dirty_bounds,
                    node.logical_screen_bounds(),
                    state.viewport(),
                    state.fringe_width(),
                );
            }
            if let Some(effect) = state.shape_mask_effects.get(&node_id) {
                mark_shape_mask_effect_dirty(dirty_bounds, effect.bounds, state.viewport());
            }
            self.stack
                .extend_from_slice(state.draw_tree.children(node_id));
        }
    }
}
