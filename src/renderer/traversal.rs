use super::types::{trim_hash_map_if_needed, trim_vector_if_needed, TraversalEvent};
use super::*;

const MAX_TRAVERSAL_EVENTS_CAPACITY: usize = 32_768;
const MAX_TRAVERSAL_STACK_CAPACITY: usize = 16_384;
const MAX_TRAVERSAL_MAP_CAPACITY: usize = 16_384;

#[derive(Default)]
pub(super) struct TraversalScratch {
    events: Vec<TraversalEvent>,
    stencil_refs: HashMap<usize, u32>,
    parent_stencils: HashMap<usize, u32>,
    skipped_stack: Vec<usize>,
    stencil_stack: Vec<u32>,
    excluded_depth: usize,
}

impl TraversalScratch {
    pub(super) fn new() -> Self {
        Self::default()
    }

    pub(super) fn begin(&mut self) {
        self.events.clear();
        self.stencil_refs.clear();
        self.parent_stencils.clear();
        self.skipped_stack.clear();
        self.stencil_stack.clear();
        self.excluded_depth = 0;
    }

    pub(super) fn trim_to_policy(&mut self) {
        trim_vector_if_needed(&mut self.events, MAX_TRAVERSAL_EVENTS_CAPACITY);
        trim_vector_if_needed(&mut self.skipped_stack, MAX_TRAVERSAL_STACK_CAPACITY);
        trim_vector_if_needed(&mut self.stencil_stack, MAX_TRAVERSAL_STACK_CAPACITY);
        trim_hash_map_if_needed(&mut self.stencil_refs, MAX_TRAVERSAL_MAP_CAPACITY);
        trim_hash_map_if_needed(&mut self.parent_stencils, MAX_TRAVERSAL_MAP_CAPACITY);
    }

    pub(super) fn events(&self) -> &[TraversalEvent] {
        &self.events
    }

    #[cfg(test)]
    pub(super) fn stencil_refs(&self) -> &HashMap<usize, u32> {
        &self.stencil_refs
    }

    #[cfg(test)]
    pub(super) fn parent_stencils(&self) -> &HashMap<usize, u32> {
        &self.parent_stencils
    }
}

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

pub(super) fn plan_traversal_in_place(
    draw_tree: &mut easy_tree::Tree<DrawCommand>,
    effect_results: &HashMap<usize, wgpu::BindGroup>,
    subtree_root: Option<usize>,
    exclude_subtree_id: Option<usize>,
    traversal_scratch: &mut TraversalScratch,
) {
    traversal_scratch.begin();

    let exclude_id = exclude_subtree_id;

    let pre_fn = |node_id: usize, _draw_command: &mut DrawCommand, state: &mut TraversalScratch| {
        // Handle excluded subtree: skip the node and all descendants entirely.
        if state.excluded_depth > 0 {
            state.excluded_depth += 1;
            return;
        }
        if exclude_id == Some(node_id) {
            state.excluded_depth = 1;
            return;
        }

        if effect_results.contains_key(&node_id) {
            state.skipped_stack.push(node_id);
        }

        if !state.skipped_stack.is_empty() && !effect_results.contains_key(&node_id) {
            return;
        }

        let parent_stencil = state.stencil_stack.last().copied().unwrap_or(0);
        let this_stencil = parent_stencil + 1;
        state.parent_stencils.insert(node_id, parent_stencil);
        state.stencil_refs.insert(node_id, this_stencil);
        state.stencil_stack.push(this_stencil);
        state.events.push(TraversalEvent::Pre(node_id));
    };

    let post_fn =
        |node_id: usize, _draw_command: &mut DrawCommand, state: &mut TraversalScratch| {
            // Handle excluded subtree.
            if state.excluded_depth > 0 {
                state.excluded_depth -= 1;
                return;
            }

            if state.skipped_stack.last().copied() == Some(node_id) {
                state.skipped_stack.pop();
                state.stencil_stack.pop();
                state.events.push(TraversalEvent::Post(node_id));
                return;
            }

            if !state.skipped_stack.is_empty() {
                return;
            }

            state.stencil_stack.pop();
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
    use super::{
        compute_node_depth, plan_traversal_in_place, subtree_has_backdrop_effects, TraversalScratch,
    };
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
        let mut traversal_scratch = TraversalScratch::new();
        plan_traversal_in_place(
            &mut tree,
            &effect_results,
            None,
            None,
            &mut traversal_scratch,
        );

        assert_eq!(traversal_scratch.events().len(), 6);
        assert_eq!(traversal_scratch.stencil_refs().get(&root), Some(&1));
        assert_eq!(traversal_scratch.stencil_refs().get(&child), Some(&2));
        assert_eq!(traversal_scratch.stencil_refs().get(&grandchild), Some(&3));
        assert_eq!(traversal_scratch.parent_stencils().get(&root), Some(&0));
        assert_eq!(traversal_scratch.parent_stencils().get(&child), Some(&1));
        assert_eq!(
            traversal_scratch.parent_stencils().get(&grandchild),
            Some(&2)
        );
    }

    #[test]
    fn plan_traversal_reuses_allocated_capacity() {
        let mut tree = easy_tree::Tree::new();
        let root = tree.add_node(DrawCommand::CachedShape(CachedShapeDrawData::new(1)));
        tree.add_child(root, DrawCommand::CachedShape(CachedShapeDrawData::new(2)));
        tree.add_child(root, DrawCommand::CachedShape(CachedShapeDrawData::new(3)));

        let effect_results: HashMap<usize, wgpu::BindGroup> = HashMap::new();
        let mut traversal_scratch = TraversalScratch::new();

        plan_traversal_in_place(
            &mut tree,
            &effect_results,
            None,
            None,
            &mut traversal_scratch,
        );
        let events_capacity = traversal_scratch.events.capacity();
        let refs_capacity = traversal_scratch.stencil_refs.capacity();
        let parents_capacity = traversal_scratch.parent_stencils.capacity();

        plan_traversal_in_place(
            &mut tree,
            &effect_results,
            None,
            None,
            &mut traversal_scratch,
        );
        assert!(traversal_scratch.events.capacity() >= events_capacity);
        assert!(traversal_scratch.stencil_refs.capacity() >= refs_capacity);
        assert!(traversal_scratch.parent_stencils.capacity() >= parents_capacity);
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
