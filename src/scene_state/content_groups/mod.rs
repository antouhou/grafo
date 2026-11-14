use super::effect_damage::{EffectDamageEntry, EffectDamageKind};
use super::effects::{ContentEffectInstance, ShapeMaskEffectInstance};
use super::types::DrawTreeNode;
use super::SceneState;
use crate::core::effect::{content_effect_output_bounds, content_input_bounds};
use crate::core::{geometry, BlendMode, MathRect, PhysicalRect, UnsignedPhysicalRect, Viewport};

pub(super) fn requires_content_group(
    node: &DrawTreeNode,
    effect: Option<ContentEffectInstance>,
) -> bool {
    effect.is_some()
        || matches!(node, DrawTreeNode::CachedShape(shape) if shape.blend_mode != BlendMode::SourceOver)
}

fn content_input_bounds_for_node(
    node: &DrawTreeNode,
    shape_mask_effect: Option<&ShapeMaskEffectInstance>,
    viewport: Viewport,
    maximum_texture_dimension: u32,
) -> MathRect {
    let shape_mask_effect_bounds = shape_mask_effect
        .filter(|_| matches!(node, DrawTreeNode::CachedShape(shape) if shape.has_geometry()))
        .map(|effect| effect.bounds);
    content_input_bounds(
        node.logical_screen_bounds(),
        shape_mask_effect_bounds,
        viewport.physical_size.into(),
        maximum_texture_dimension,
    )
}

/// A content capture's optional effect and physical input and output bounds.
#[derive(Clone, Copy)]
pub(crate) struct ContentGroup {
    pub(crate) effect: Option<ContentEffectInstance>,
    /// Viewport-clipped input pixels, or `None` when the input is offscreen or invalid.
    pub(crate) input_bounds: Option<UnsignedPhysicalRect>,
    pub(crate) output_bounds: Option<PhysicalRect>,
}

impl ContentGroup {
    pub(crate) fn new(
        effect: Option<ContentEffectInstance>,
        node: &DrawTreeNode,
        shape_mask_effect: Option<&ShapeMaskEffectInstance>,
        viewport: Viewport,
        maximum_texture_dimension: u32,
    ) -> Self {
        let config = effect.map(|effect| effect.config).unwrap_or_default();
        let logical_input_bounds = content_input_bounds_for_node(
            node,
            shape_mask_effect,
            viewport,
            maximum_texture_dimension,
        );
        let input_bounds =
            geometry::logical_rect_to_physical_rect(logical_input_bounds, viewport.scale_factor)
                .and_then(|bounds| {
                    bounds
                        .to_i64()
                        .intersection(
                            &UnsignedPhysicalRect::from_size(viewport.physical_size.into())
                                .to_i64(),
                        )
                        .map(|bounds| bounds.to_u32())
                });
        Self {
            effect,
            input_bounds,
            output_bounds: content_effect_output_bounds(
                logical_input_bounds,
                config,
                viewport,
                maximum_texture_dimension,
            ),
        }
    }
}

pub(super) fn resolve_content_group(
    node_id: usize,
    node: &DrawTreeNode,
    effect: Option<ContentEffectInstance>,
    shape_mask_effect: Option<&ShapeMaskEffectInstance>,
    viewport: Viewport,
    maximum_texture_dimension: u32,
) -> (ContentGroup, Option<EffectDamageEntry>) {
    let group = ContentGroup::new(
        effect,
        node,
        shape_mask_effect,
        viewport,
        maximum_texture_dimension,
    );
    let entry = EffectDamageEntry::content(
        node_id,
        group.input_bounds,
        group.output_bounds,
        viewport.physical_size.into(),
    );
    (group, entry)
}

impl SceneState {
    pub(super) fn replace_content_group(
        &mut self,
        node_id: usize,
        group: ContentGroup,
        entry: Option<EffectDamageEntry>,
    ) {
        self.effect_damage
            .replace(EffectDamageKind::Content, node_id, entry);
        self.content_groups.insert(node_id, group);
    }

    pub(super) fn refresh_content_group_bounds(
        &mut self,
        node_id: usize,
        maximum_texture_dimension: u32,
    ) {
        let Some(group) = self.content_groups.get(&node_id) else {
            return;
        };
        let Some(node) = self.draw_tree.get(node_id) else {
            return;
        };
        if !requires_content_group(node, group.effect) {
            self.content_groups.remove(&node_id);
            self.effect_damage
                .remove(EffectDamageKind::Content, node_id);
            return;
        }
        let (updated, entry) = resolve_content_group(
            node_id,
            node,
            group.effect,
            self.shape_mask_effects.get(&node_id),
            self.viewport,
            maximum_texture_dimension,
        );
        self.replace_content_group(node_id, updated, entry);
    }

    pub(super) fn refresh_all_content_group_bounds(&mut self, maximum_texture_dimension: u32) {
        self.content_groups.retain(|&node_id, group| {
            let Some(node) = self.draw_tree.get(node_id) else {
                self.effect_damage
                    .remove(EffectDamageKind::Content, node_id);
                return false;
            };
            if !requires_content_group(node, group.effect) {
                self.effect_damage
                    .remove(EffectDamageKind::Content, node_id);
                return false;
            }
            let (updated, entry) = resolve_content_group(
                node_id,
                node,
                group.effect,
                self.shape_mask_effects.get(&node_id),
                self.viewport,
                maximum_texture_dimension,
            );
            *group = updated;
            self.effect_damage
                .replace(EffectDamageKind::Content, node_id, entry);
            true
        });
    }
}

#[cfg(test)]
mod tests;
