use crate::commands::ShapeDrawMaterial;
use crate::core::geometry;
use crate::core::{Size, UnsignedPhysicalRect};
use crate::scene_state::effects::ContentEffectInstance;
use crate::scene_state::types::DrawTreeNode;
use ahash::HashMap;

pub(super) fn should_skip_visible_rect_draw(
    node_id: usize,
    draw_tree_node: &DrawTreeNode,
    material: Option<ShapeDrawMaterial>,
    content_effects: &HashMap<usize, ContentEffectInstance>,
) -> bool {
    if !draw_tree_node.is_axis_aligned_rect() {
        return false;
    }

    if content_effects.contains_key(&node_id) {
        return false;
    }

    if material.is_some_and(|material| {
        material.has_gradient_fill
            || material.under_fill_texture.is_some()
            || material
                .texture_bindings
                .iter()
                .any(|texture| texture.is_present())
    }) {
        return false;
    }

    if draw_tree_node
        .instance_color_override()
        .is_some_and(|color| color[3] != 0.0)
    {
        return false;
    }

    true
}

/// Returns a scissor rect when the draw tree node's transform preserves axis alignment.
pub(super) fn try_scissor_for_rect(
    draw_tree_node: &DrawTreeNode,
    scale_factor: f64,
    physical_size: Size,
) -> Option<UnsignedPhysicalRect> {
    if !draw_tree_node.is_axis_aligned_rect() {
        return None;
    }
    geometry::logical_rect_to_scissor_rect(
        draw_tree_node.logical_screen_bounds(),
        scale_factor,
        physical_size,
    )
}

#[cfg(test)]
mod tests;
