use super::{
    compute_node_depth, plan_traversal_in_place, subtree_has_backdrop_effects, TraversalScratch,
};
use crate::cache::CachedTessellation;
use crate::effect::{BackdropEffectConfig, BackdropEffectInstance, EffectInstance};
use crate::renderer::types::{DrawTreeNode, TraversalEvent};
use crate::renderer::IntermediateTextureId;
use crate::shape::{CachedShapeDrawData, CachedShapeHandle};
use crate::vertex::CustomVertex;
use crate::ShapeDrawCommandOptions;
use ahash::{HashMap, HashMapExt};
use easy_tree::Tree;
use lyon::tessellation::VertexBuffers;
use std::sync::Arc;

fn cached_draw_data() -> CachedShapeDrawData {
    CachedShapeDrawData::new(
        CachedShapeHandle {
            tessellation: Arc::new(CachedTessellation {
                vertex_buffers: Arc::new(VertexBuffers::<CustomVertex, u16>::new()),
                local_bounds: [(0.0, 0.0), (1.0, 1.0)],
                texture_mapping_size: [1.0, 1.0],
            }),
            is_rect: false,
            rect_bounds: None,
            geometry_id: None,
        },
        &ShapeDrawCommandOptions::new(),
    )
}

#[test]
fn compute_node_depth_counts_ancestors_across_uneven_branches() {
    let mut tree = Tree::new();
    let root = tree.add_node(DrawTreeNode::CachedShape(cached_draw_data()));
    let child = tree.add_child(root, DrawTreeNode::CachedShape(cached_draw_data()));
    let grandchild = tree.add_child(child, DrawTreeNode::CachedShape(cached_draw_data()));
    let great_grandchild =
        tree.add_child(grandchild, DrawTreeNode::CachedShape(cached_draw_data()));
    let sibling = tree.add_child(root, DrawTreeNode::CachedShape(cached_draw_data()));
    let sibling_child = tree.add_child(sibling, DrawTreeNode::CachedShape(cached_draw_data()));

    for (node_id, expected_depth) in [
        (root, 0),
        (child, 1),
        (grandchild, 2),
        (great_grandchild, 3),
        (sibling, 1),
        (sibling_child, 2),
    ] {
        assert_eq!(compute_node_depth(&tree, node_id), expected_depth);
    }
}

#[test]
fn plan_traversal_produces_balanced_events() {
    let mut tree = Tree::new();
    let root = tree.add_node(DrawTreeNode::CachedShape(cached_draw_data()));
    let child = tree.add_child(root, DrawTreeNode::CachedShape(cached_draw_data()));
    let grandchild = tree.add_child(child, DrawTreeNode::CachedShape(cached_draw_data()));
    let sibling = tree.add_child(root, DrawTreeNode::CachedShape(cached_draw_data()));

    let effect_results: HashMap<usize, IntermediateTextureId> = HashMap::new();
    let mut traversal_scratch = TraversalScratch::new();
    plan_traversal_in_place(
        &mut tree,
        &effect_results,
        None,
        None,
        &mut traversal_scratch,
    );

    assert_eq!(
        traversal_scratch.events(),
        &[
            TraversalEvent::Pre(root),
            TraversalEvent::Pre(child),
            TraversalEvent::Pre(grandchild),
            TraversalEvent::Post(grandchild),
            TraversalEvent::Post(child),
            TraversalEvent::Pre(sibling),
            TraversalEvent::Post(sibling),
            TraversalEvent::Post(root),
        ]
    );
}

#[test]
fn plan_traversal_preserves_reserved_event_storage() {
    let mut tree = Tree::new();
    let root = tree.add_node(DrawTreeNode::CachedShape(cached_draw_data()));
    let child = tree.add_child(root, DrawTreeNode::CachedShape(cached_draw_data()));
    tree.add_child(root, DrawTreeNode::CachedShape(cached_draw_data()));

    let effect_results: HashMap<usize, IntermediateTextureId> = HashMap::new();
    let mut traversal_scratch = TraversalScratch::new();
    traversal_scratch.events.reserve(128);
    let events_pointer = traversal_scratch.events.as_ptr();
    let events_capacity = traversal_scratch.events.capacity();

    for subtree_root in [None, Some(child), None] {
        plan_traversal_in_place(
            &mut tree,
            &effect_results,
            subtree_root,
            None,
            &mut traversal_scratch,
        );
        assert!(!traversal_scratch.events.is_empty());
        assert_eq!(traversal_scratch.events.as_ptr(), events_pointer);
        assert_eq!(traversal_scratch.events.capacity(), events_capacity);
    }
}

#[test]
fn group_texture_replaces_nested_results() {
    let mut tree = Tree::new();
    let root = tree.add_node(DrawTreeNode::CachedShape(cached_draw_data()));
    let group = tree.add_child(root, DrawTreeNode::CachedShape(cached_draw_data()));
    let nested_group = tree.add_child(group, DrawTreeNode::CachedShape(cached_draw_data()));
    tree.add_child(nested_group, DrawTreeNode::CachedShape(cached_draw_data()));
    let sibling = tree.add_child(root, DrawTreeNode::CachedShape(cached_draw_data()));
    let mut results = HashMap::new();
    results.insert(group, IntermediateTextureId::Registered(1));
    results.insert(nested_group, IntermediateTextureId::Registered(2));
    let mut scratch = TraversalScratch::new();

    plan_traversal_in_place(&mut tree, &results, None, None, &mut scratch);
    assert_eq!(
        scratch.events(),
        &[
            TraversalEvent::Pre(root),
            TraversalEvent::Pre(group),
            TraversalEvent::Post(group),
            TraversalEvent::Pre(sibling),
            TraversalEvent::Post(sibling),
            TraversalEvent::Post(root),
        ]
    );

    plan_traversal_in_place(&mut tree, &results, None, Some(group), &mut scratch);
    assert_eq!(
        scratch.events(),
        &[
            TraversalEvent::Pre(root),
            TraversalEvent::Pre(sibling),
            TraversalEvent::Post(sibling),
            TraversalEvent::Post(root),
        ]
    );
}

#[test]
fn subtree_has_backdrop_effects_detects_descendants() {
    let mut tree = Tree::new();
    let root = tree.add_node(DrawTreeNode::CachedShape(cached_draw_data()));
    let child = tree.add_child(root, DrawTreeNode::CachedShape(cached_draw_data()));
    let grandchild = tree.add_child(child, DrawTreeNode::CachedShape(cached_draw_data()));

    let mut backdrop_effects = HashMap::new();
    backdrop_effects.insert(
        grandchild,
        BackdropEffectInstance::new(
            EffectInstance {
                effect_id: 1,
                params: Vec::new(),
            },
            BackdropEffectConfig::default(),
        ),
    );

    assert!(subtree_has_backdrop_effects(&tree, &backdrop_effects, root));
    assert!(subtree_has_backdrop_effects(
        &tree,
        &backdrop_effects,
        child
    ));
    assert!(subtree_has_backdrop_effects(
        &tree,
        &backdrop_effects,
        grandchild
    ));
}
