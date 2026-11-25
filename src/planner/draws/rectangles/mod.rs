use super::{DrawPlanner, DrawPlanningInput};
use crate::commands::ShapeDrawMaterial;
use crate::core::geometry;
use crate::core::shape::boundary;
use crate::core::{BlendMode, MathRect, Size, UnsignedPhysicalRect};
use crate::scene_state::content_groups::ContentGroup;
use crate::scene_state::types::DrawTreeNode;
use ahash::HashMap;

fn clip_contains_bounds(node: &DrawTreeNode, bounds: MathRect, scale_factor: f64) -> bool {
    if !bounds.is_finite() || !scale_factor.is_finite() || scale_factor <= 0.0 {
        return false;
    }
    if bounds.is_empty() {
        return true;
    }
    // Include every pixel touched by a child's antialiased edge.
    let margin = (1.0 / scale_factor) as f32;
    let antialiased_bounds = bounds.inflate(margin, margin);
    match node {
        DrawTreeNode::ClipRect(clip) => {
            clip.logical_screen_bounds.contains_box(&antialiased_bounds)
        }
        DrawTreeNode::CachedShape(shape) => {
            let Some(geometry) = shape.instance.tessellation() else {
                return false;
            };
            let Some(local) = geometry::logical_screen_rect_to_local_bounds(
                antialiased_bounds,
                shape.instance.transform,
            ) else {
                return false;
            };
            if !MathRect::new(
                geometry.local_bounds[0].into(),
                geometry.local_bounds[1].into(),
            )
            .contains_box(&local)
            {
                return false;
            }
            if geometry.is_rect {
                return true;
            }
            if let Some(boundary) = &geometry.rounded_rect_boundary {
                boundary.contains_rect(&geometry.boundary_edges, local)
            } else {
                boundary::path_contains_rect(&geometry.boundary_edges, local)
            }
        }
        DrawTreeNode::Group(_) => false,
    }
}

impl DrawPlanner {
    pub(super) fn child_bounds(
        &self,
        node_id: usize,
        input: &DrawPlanningInput<'_, '_>,
    ) -> Option<MathRect> {
        let mut bounds = MathRect::zero();
        for child in input.tree.children(node_id) {
            bounds = bounds.union(&self.subtree_bounds.get(child).copied().flatten()?);
        }
        Some(bounds)
    }

    pub(super) fn resolve_subtree_bounds(&mut self, input: &DrawPlanningInput<'_, '_>) {
        self.subtree_bounds.clear();
        self.destination_dependent_subtrees.clear();
        self.layered_subtrees.clear();
        for &node_id in self.node_order.iter().rev() {
            let node = input
                .tree
                .get(node_id)
                .expect("traversal contains live nodes");
            let bounds = if input.shape_mask_effects.contains_key(&node_id) {
                None
            } else if let Some(group) = input.content_groups.get(&node_id) {
                group
                    .output_bounds
                    .map(|bounds| bounds.to_f32() / input.scale_factor as f32)
            } else {
                self.child_bounds(node_id, input).map(|children| {
                    let children =
                        if node.has_child_clip() && node.logical_screen_bounds().is_finite() {
                            children
                                .intersection(&node.logical_screen_bounds())
                                .unwrap_or_else(MathRect::zero)
                        } else {
                            children
                        };
                    let has_visible_paint = match node {
                        DrawTreeNode::CachedShape(shape) => {
                            shape.instance.has_gradient_fill()
                                || shape
                                    .instance
                                    .textures
                                    .iter()
                                    .any(|texture| texture.texture_id.is_some())
                                || node
                                    .solid_fill_color()
                                    .is_some_and(|color| color.to_srgba_f32()[3] > 0.0)
                                || input.backdrop_effects.contains_key(&node_id)
                        }
                        _ => false,
                    };
                    if has_visible_paint {
                        children.union(&node.logical_screen_bounds())
                    } else {
                        children
                    }
                })
            };
            self.subtree_bounds.insert(node_id, bounds);
            if input.backdrop_effects.contains_key(&node_id)
                || matches!(node, DrawTreeNode::CachedShape(shape) if shape.blend_mode != BlendMode::SourceOver)
                || input
                    .tree
                    .children(node_id)
                    .iter()
                    .any(|child| self.destination_dependent_subtrees.contains(child))
            {
                self.destination_dependent_subtrees.insert(node_id);
            }
            if input.shape_mask_effects.contains_key(&node_id)
                || input.content_groups.contains_key(&node_id)
                || !node.logical_screen_bounds().is_finite()
                || input
                    .tree
                    .children(node_id)
                    .iter()
                    .any(|child| self.layered_subtrees.contains(child))
            {
                self.layered_subtrees.insert(node_id);
            }
        }
    }

    pub(super) fn children_fit_inside_clip(
        &self,
        node_id: usize,
        node: &DrawTreeNode,
        input: &DrawPlanningInput<'_, '_>,
    ) -> bool {
        self.child_bounds(node_id, input)
            .is_some_and(|bounds| clip_contains_bounds(node, bounds, input.scale_factor))
    }
}

pub(super) fn should_skip_visible_draw(
    node_id: usize,
    draw_tree_node: &DrawTreeNode,
    material: Option<ShapeDrawMaterial>,
    content_groups: &HashMap<usize, ContentGroup>,
) -> bool {
    if matches!(draw_tree_node, DrawTreeNode::CachedShape(shape) if shape.blend_mode != BlendMode::SourceOver)
    {
        return false;
    }
    if content_groups.contains_key(&node_id) {
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
        .solid_fill_color()
        .is_some_and(|color| color.to_srgba_f32()[3] != 0.0)
    {
        return false;
    }

    true
}

/// Returns a scissor rect when the draw tree node's transform preserves axis alignment.
pub(super) fn try_scissor_for_rect(
    draw_tree_node: &DrawTreeNode,
    child_bounds: Option<MathRect>,
    scale_factor: f64,
    physical_size: Size,
) -> Option<UnsignedPhysicalRect> {
    if !draw_tree_node.is_axis_aligned_rect() {
        return None;
    }
    let bounds = draw_tree_node.logical_screen_bounds();
    for (axis, extent) in [(0, physical_size.width), (1, physical_size.height)] {
        for (edge, is_minimum) in [
            (bounds.min.to_array()[axis], true),
            (bounds.max.to_array()[axis], false),
        ] {
            let physical = f64::from(edge) * scale_factor;
            if !physical.is_finite() || physical.fract() == 0.0 {
                continue;
            }
            if (is_minimum && physical <= 0.0) || (!is_minimum && physical >= f64::from(extent)) {
                continue;
            }
            let children = child_bounds.filter(|bounds| bounds.is_finite())?;
            let has_margin = if is_minimum {
                f64::from(children.min.to_array()[axis]) * scale_factor >= physical + 1.0
            } else {
                f64::from(children.max.to_array()[axis]) * scale_factor <= physical - 1.0
            };
            if !has_margin {
                return None;
            }
        }
    }
    geometry::logical_rect_to_scissor_rect(
        draw_tree_node.logical_screen_bounds(),
        scale_factor,
        physical_size,
    )
}

#[cfg(test)]
mod tests;
