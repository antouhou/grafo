use super::{compute_node_depth, subtree_has_backdrop_effects};
use crate::cache::CachedTessellation;
use crate::effect::{BackdropEffectConfig, BackdropEffectInstance, EffectInstance};
use crate::renderer::types::DrawTreeNode;
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
