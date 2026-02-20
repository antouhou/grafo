use super::types::TraversalEvent;
use super::*;

pub(super) fn subtree_has_backdrop_effects(
    tree: &easy_tree::Tree<DrawCommand>,
    backdrop_effects: &HashMap<usize, EffectInstance>,
    root_id: usize,
) -> bool {
    if backdrop_effects.is_empty() {
        return false;
    }

    fn scan(
        tree: &easy_tree::Tree<DrawCommand>,
        backdrop_effects: &HashMap<usize, EffectInstance>,
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

pub(super) fn plan_traversal(
    draw_tree: &mut easy_tree::Tree<DrawCommand>,
    effect_results: &HashMap<usize, wgpu::BindGroup>,
    subtree_root: Option<usize>,
) -> (
    Vec<TraversalEvent>,
    HashMap<usize, u32>,
    HashMap<usize, u32>,
) {
    let mut stencil_refs: HashMap<usize, u32> = HashMap::new();
    let mut parent_stencils: HashMap<usize, u32> = HashMap::new();

    let mut plan_state: (Vec<usize>, Vec<u32>, Vec<TraversalEvent>) =
        (Vec::new(), Vec::new(), Vec::new());

    let pre_fn = |node_id: usize,
                  _draw_command: &mut DrawCommand,
                  state: &mut (Vec<usize>, Vec<u32>, Vec<TraversalEvent>)| {
        let (skipped_stack, stencil_stack, events) = state;
        if effect_results.contains_key(&node_id) {
            skipped_stack.push(node_id);
        }

        if !skipped_stack.is_empty() && !effect_results.contains_key(&node_id) {
            return;
        }

        let parent_stencil = stencil_stack.last().copied().unwrap_or(0);
        let this_stencil = parent_stencil + 1;
        parent_stencils.insert(node_id, parent_stencil);
        stencil_refs.insert(node_id, this_stencil);
        stencil_stack.push(this_stencil);
        events.push(TraversalEvent::Pre(node_id));
    };

    let post_fn = |node_id: usize,
                   _draw_command: &mut DrawCommand,
                   state: &mut (Vec<usize>, Vec<u32>, Vec<TraversalEvent>)| {
        let (skipped_stack, stencil_stack, events) = state;

        if skipped_stack.last().copied() == Some(node_id) {
            skipped_stack.pop();
            stencil_stack.pop();
            events.push(TraversalEvent::Post(node_id));
            return;
        }

        if !skipped_stack.is_empty() {
            return;
        }

        stencil_stack.pop();
        events.push(TraversalEvent::Post(node_id));
    };

    match subtree_root {
        Some(root_id) => {
            draw_tree.traverse_subtree_mut(root_id, pre_fn, post_fn, &mut plan_state);
        }
        None => {
            draw_tree.traverse_mut(pre_fn, post_fn, &mut plan_state);
        }
    }

    let events = plan_state.2;
    (events, stencil_refs, parent_stencils)
}

pub(super) fn compute_node_depth(tree: &easy_tree::Tree<DrawCommand>, node_id: usize) -> usize {
    let mut depth = 0;
    let mut current = node_id;

    while let Some(parent) = tree.parent_index_unchecked(current) {
        depth += 1;
        current = parent;
    }

    depth
}

#[cfg(test)]
mod tests {
    use super::{compute_node_depth, plan_traversal, subtree_has_backdrop_effects};
    use crate::effect::EffectInstance;
    use crate::renderer::types::DrawCommand;
    use crate::shape::CachedShapeDrawData;
    use ahash::{HashMap, HashMapExt};

    #[test]
    fn compute_node_depth_returns_zero_for_root() {
        let mut tree = easy_tree::Tree::new();
        let root = tree.add_node(DrawCommand::CachedShape(CachedShapeDrawData::new(1)));

        assert_eq!(compute_node_depth(&tree, root), 0);
    }

    #[test]
    fn plan_traversal_produces_balanced_events_and_stencil_refs() {
        let mut tree = easy_tree::Tree::new();
        let root = tree.add_node(DrawCommand::CachedShape(CachedShapeDrawData::new(1)));
        let child = tree.add_child(root, DrawCommand::CachedShape(CachedShapeDrawData::new(2)));
        let grandchild =
            tree.add_child(child, DrawCommand::CachedShape(CachedShapeDrawData::new(3)));

        let effect_results: HashMap<usize, wgpu::BindGroup> = HashMap::new();
        let (events, stencil_refs, parent_stencils) =
            plan_traversal(&mut tree, &effect_results, None);

        assert_eq!(events.len(), 6);
        assert_eq!(stencil_refs.get(&root), Some(&1));
        assert_eq!(stencil_refs.get(&child), Some(&2));
        assert_eq!(stencil_refs.get(&grandchild), Some(&3));
        assert_eq!(parent_stencils.get(&root), Some(&0));
        assert_eq!(parent_stencils.get(&child), Some(&1));
        assert_eq!(parent_stencils.get(&grandchild), Some(&2));
    }

    #[test]
    fn subtree_has_backdrop_effects_detects_descendants() {
        let mut tree = easy_tree::Tree::new();
        let root = tree.add_node(DrawCommand::CachedShape(CachedShapeDrawData::new(1)));
        let child = tree.add_child(root, DrawCommand::CachedShape(CachedShapeDrawData::new(2)));
        let grandchild =
            tree.add_child(child, DrawCommand::CachedShape(CachedShapeDrawData::new(3)));

        let mut backdrop_effects = HashMap::new();
        backdrop_effects.insert(
            grandchild,
            EffectInstance {
                effect_id: 1,
                params: Vec::new(),
                params_buffer: None,
                params_bind_group: None,
            },
        );

        assert!(subtree_has_backdrop_effects(&tree, &backdrop_effects, root));
        assert!(subtree_has_backdrop_effects(
            &tree,
            &backdrop_effects,
            child
        ));
        assert!(!subtree_has_backdrop_effects(
            &tree,
            &backdrop_effects,
            grandchild
        ));
    }
}
