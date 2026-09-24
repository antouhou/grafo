use super::Planner;
use crate::renderer::types::CachedShapeDrawData;
use crate::renderer::types::{DrawCommandError, DrawTreeNode};

impl Planner {
    pub(in crate::renderer) fn add_draw_tree_node(
        &mut self,
        draw_tree_node: DrawTreeNode,
        parent_shape_id: Option<usize>,
    ) -> Result<usize, DrawCommandError> {
        if self.draw_tree.is_empty() {
            let node_id = self.draw_tree.add_node(draw_tree_node);
            Ok(node_id)
        } else if let Some(parent_shape_id) = parent_shape_id {
            if let Some(parent) = self.draw_tree.get_mut(parent_shape_id) {
                parent.set_not_leaf();
                let node_id = self.draw_tree.add_child(parent_shape_id, draw_tree_node);
                Ok(node_id)
            } else {
                Err(DrawCommandError::InvalidShapeId(parent_shape_id))
            }
        } else {
            if let Some(root) = self.draw_tree.get_mut(0) {
                root.set_not_leaf();
            }
            let node_id = self.draw_tree.add_child_to_root(draw_tree_node);
            Ok(node_id)
        }
    }

    pub(in crate::renderer) fn refresh_geometry_cache(
        &mut self,
        cached_shape_data: &CachedShapeDrawData,
    ) {
        if let Some(geometry_id) = cached_shape_data.cached_shape.geometry_id {
            self.shape_resources
                .tessellation_cache
                .refresh_tessellation(geometry_id, &cached_shape_data.cached_shape.tessellation);
        }
    }
}
