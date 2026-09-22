use super::plan::shape_effects::PreparedShapeEffectLeaf;
use super::types::TraversalEvent;
use super::*;
use crate::effect::BackdropEffectInstance;

#[derive(Default)]
pub(super) struct TraversalScratch {
    events: Vec<TraversalEvent>,
    substituted_subtree: Option<usize>,
    excluded_depth: usize,
}

impl TraversalScratch {
    pub(super) fn new() -> Self {
        Self::default()
    }

    pub(super) fn begin(&mut self) {
        self.events.clear();
        self.substituted_subtree = None;
        self.excluded_depth = 0;
    }

    pub(super) fn events(&self) -> &[TraversalEvent] {
        &self.events
    }
}

pub(super) fn subtree_has_backdrop_effects(
    tree: &easy_tree::Tree<DrawTreeNode>,
    backdrop_effects: &HashMap<usize, BackdropEffectInstance>,
    root_id: usize,
) -> bool {
    if backdrop_effects.is_empty() {
        return false;
    }
    if backdrop_effects.contains_key(&root_id) {
        return true;
    }

    fn scan(
        tree: &easy_tree::Tree<DrawTreeNode>,
        backdrop_effects: &HashMap<usize, BackdropEffectInstance>,
        node_id: usize,
    ) -> bool {
        for &child_id in tree.children(node_id) {
            if backdrop_effects.contains_key(&child_id) {
                return true;
            }
            if scan(tree, backdrop_effects, child_id) {
                return true;
            }
        }
        false
    }

    scan(tree, backdrop_effects, root_id)
}

pub(super) fn plan_traversal_in_place(
    draw_tree: &mut easy_tree::Tree<DrawTreeNode>,
    effect_results: &HashMap<usize, IntermediateTextureId>,
    prepared_shape_effect_leaves: &HashMap<usize, PreparedShapeEffectLeaf>,
    subtree_root: Option<usize>,
    exclude_subtree_id: Option<usize>,
    traversal_scratch: &mut TraversalScratch,
) {
    traversal_scratch.begin();

    let exclude_id = exclude_subtree_id;

    let pre_fn =
        |node_id: usize, _draw_tree_node: &mut DrawTreeNode, state: &mut TraversalScratch| {
            // Skip the excluded node and its descendants
            if state.excluded_depth > 0 {
                state.excluded_depth += 1;
                return;
            }
            if exclude_id == Some(node_id) {
                state.excluded_depth = 1;
                return;
            }

            if state.substituted_subtree.is_some() {
                return;
            }

            if effect_results.contains_key(&node_id) {
                state.substituted_subtree = Some(node_id);
            } else if prepared_shape_effect_leaves.contains_key(&node_id) {
                state.events.push(TraversalEvent::PreparedLeaf(node_id));
            }
            state.events.push(TraversalEvent::Pre(node_id));
        };

    let post_fn =
        |node_id: usize, _draw_tree_node: &mut DrawTreeNode, state: &mut TraversalScratch| {
            if state.excluded_depth > 0 {
                state.excluded_depth -= 1;
                return;
            }

            if state.substituted_subtree == Some(node_id) {
                state.substituted_subtree = None;
                state.events.push(TraversalEvent::Post(node_id));
                return;
            }

            if state.substituted_subtree.is_some() {
                return;
            }

            state.events.push(TraversalEvent::Post(node_id));
        };

    match subtree_root {
        Some(root_id) => {
            draw_tree.traverse_subtree_mut(root_id, pre_fn, post_fn, traversal_scratch);
        }
        None => {
            draw_tree.traverse_mut(pre_fn, post_fn, traversal_scratch);
        }
    }
}

pub(super) fn compute_node_depth(tree: &easy_tree::Tree<DrawTreeNode>, node_id: usize) -> usize {
    let mut depth = 0;
    let mut current = node_id;

    while let Some(parent) = tree.parent_index_unchecked(current) {
        depth += 1;
        current = parent;
    }

    depth
}

#[cfg(test)]
mod tests;
