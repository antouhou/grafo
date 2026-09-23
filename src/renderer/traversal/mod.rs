use crate::effect::BackdropEffectInstance;
use crate::renderer::types::DrawTreeNode;
use ahash::HashMap;
use easy_tree::Tree;

pub(super) fn subtree_has_backdrop_effects(
    tree: &Tree<DrawTreeNode>,
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
        tree: &Tree<DrawTreeNode>,
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

pub(super) fn compute_node_depth(tree: &Tree<DrawTreeNode>, node_id: usize) -> usize {
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
