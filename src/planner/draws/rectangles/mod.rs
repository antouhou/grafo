use crate::core::geometry;
use crate::core::{Size, UnsignedPhysicalRect};
use crate::scene::effects::{BackdropEffectInstance, EffectInstance};
use crate::scene::types::DrawTreeNode;
use ahash::HashMap;

pub(super) fn should_skip_visible_rect_draw(
    node_id: usize,
    draw_tree_node: &DrawTreeNode,
    group_effects: &HashMap<usize, EffectInstance>,
    backdrop_effects: &HashMap<usize, BackdropEffectInstance>,
) -> bool {
    if !draw_tree_node.is_axis_aligned_rect() {
        return false;
    }

    if group_effects.contains_key(&node_id) || backdrop_effects.contains_key(&node_id) {
        return false;
    }

    if draw_tree_node.texture_id(0).is_some() || draw_tree_node.texture_id(1).is_some() {
        return false;
    }

    // A gradient can be visible even when there is no solid color override.
    if draw_tree_node.has_gradient_fill() {
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
