use crate::core::effect::{BackdropEffectInstance, EffectInstance};
use crate::core::geometry;
use crate::core::{MathRect, Size, UnsignedPhysicalRect};
use crate::renderer::types::DrawTreeNode;
use ahash::HashMap;

pub(super) fn should_skip_visible_rect_draw(
    node_id: usize,
    draw_tree_node: &DrawTreeNode,
    group_effects: &HashMap<usize, EffectInstance>,
    backdrop_effects: &HashMap<usize, BackdropEffectInstance>,
) -> bool {
    if !draw_tree_node.is_rect() {
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

    geometry::extract_axis_aligned_rect_transform(draw_tree_node.transform()).is_some()
}

/// Returns a scissor rect when the draw tree node's transform preserves axis alignment.
pub(super) fn try_scissor_for_rect(
    draw_tree_node: &DrawTreeNode,
    scale_factor: f64,
    physical_size: Size,
) -> Option<UnsignedPhysicalRect> {
    if !draw_tree_node.is_rect() {
        return None;
    }
    let rect_bounds = draw_tree_node.rect_bounds()?;
    let rect = MathRect::new(rect_bounds[0].into(), rect_bounds[1].into());
    geometry::compute_scissor_rect(
        rect,
        draw_tree_node.transform(),
        scale_factor,
        physical_size,
    )
}

#[cfg(test)]
mod tests;
