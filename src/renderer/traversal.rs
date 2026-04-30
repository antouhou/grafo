use super::types::{trim_vector_if_needed, TraversalEvent};
use super::*;

const MAX_TRAVERSAL_EVENTS_CAPACITY: usize = 32_768;
const MAX_TRAVERSAL_STACK_CAPACITY: usize = 16_384;

#[derive(Default)]
pub(super) struct TraversalScratch {
    events: Vec<TraversalEvent>,
    skipped_stack: Vec<usize>,
    excluded_depth: usize,
}

impl TraversalScratch {
    pub(super) fn new() -> Self {
        Self::default()
    }

    pub(super) fn begin(&mut self) {
        self.events.clear();
        self.skipped_stack.clear();
        self.excluded_depth = 0;
    }

    pub(super) fn trim_to_policy(&mut self) {
        trim_vector_if_needed(&mut self.events, MAX_TRAVERSAL_EVENTS_CAPACITY);
        trim_vector_if_needed(&mut self.skipped_stack, MAX_TRAVERSAL_STACK_CAPACITY);
    }

    pub(super) fn events(&self) -> &[TraversalEvent] {
        &self.events
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
                state.events.push(TraversalEvent::Post(node_id));
                return;
            }

            if !state.skipped_stack.is_empty() {
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
    use crate::shape::{CachedShapeDrawData, CachedShapeHandle};
    use crate::vertex::CustomVertex;
    use crate::ShapeDrawCommandOptions;
    use ahash::{HashMap, HashMapExt};
    use lyon::tessellation::VertexBuffers;
    use std::sync::Arc;

    fn cached_draw_data() -> CachedShapeDrawData {
        CachedShapeDrawData::new(
            CachedShapeHandle {
                vertex_buffers: Arc::new(VertexBuffers::<CustomVertex, u16>::new()),
                is_rect: false,
                rect_bounds: None,
                local_bounds: [(0.0, 0.0), (1.0, 1.0)],
                texture_mapping_size: [1.0, 1.0],
                geometry_id: None,
            },
            &ShapeDrawCommandOptions::new(),
        )
    }

    #[test]
    fn compute_node_depth_returns_zero_for_root() {
        let mut tree = easy_tree::Tree::new();
        let root = tree.add_node(DrawCommand::CachedShape(cached_draw_data()));

        assert_eq!(compute_node_depth(&tree, root), 0);
    }

    #[test]
    fn plan_traversal_produces_balanced_events() {
        let mut tree = easy_tree::Tree::new();
        let root = tree.add_node(DrawCommand::CachedShape(cached_draw_data()));
        let child = tree.add_child(root, DrawCommand::CachedShape(cached_draw_data()));
        tree.add_child(child, DrawCommand::CachedShape(cached_draw_data()));

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
    }

    #[test]
    fn plan_traversal_reuses_allocated_capacity() {
        let mut tree = easy_tree::Tree::new();
        let root = tree.add_node(DrawCommand::CachedShape(cached_draw_data()));
        tree.add_child(root, DrawCommand::CachedShape(cached_draw_data()));
        tree.add_child(root, DrawCommand::CachedShape(cached_draw_data()));

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

        plan_traversal_in_place(
            &mut tree,
            &effect_results,
            None,
            None,
            &mut traversal_scratch,
        );
        assert!(traversal_scratch.events.capacity() >= events_capacity);
    }

    #[test]
    fn subtree_has_backdrop_effects_detects_descendants() {
        let mut tree = easy_tree::Tree::new();
        let root = tree.add_node(DrawCommand::CachedShape(cached_draw_data()));
        let child = tree.add_child(root, DrawCommand::CachedShape(cached_draw_data()));
        let grandchild = tree.add_child(child, DrawCommand::CachedShape(cached_draw_data()));

        let mut backdrop_effects = HashMap::new();
        backdrop_effects.insert(
            grandchild,
            EffectInstance {
                effect_id: 1,
                params: Vec::new(),
                params_buffer: None,
                params_bind_group: None,
                backdrop_config: None,
                backdrop_material_params_buffer: None,
                backdrop_texture_bind_group: None,
                backdrop_texture_id: None,
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
