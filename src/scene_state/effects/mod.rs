use super::content_groups::resolve_content_group;
use super::effect_damage::{EffectDamageEntry, EffectDamageKind};
use super::types::DrawTreeNode;
use super::{SceneError, SceneState};
use crate::core::effect::{
    BackdropEffectConfig, ContentEffectConfig, ShapeMaskEffectBounds, ShapeMaskEffectConfig,
};
use crate::core::{geometry, Viewport};
pub(crate) use attachments::{
    EffectAttachment, EffectAttachmentKind, EffectAttachments, EffectConfig, EffectInstance,
    RemovedEffectAttachment,
};
pub(crate) use backdrops::validate_backdrop_config;
pub(crate) use backdrops::BackdropEffectInstance;
pub(super) use group_index::GroupEffectIndex;
use parameters::EffectParametersHandle;

mod attachments;
mod backdrops;
mod group_index;
mod groups;
pub(crate) mod parameters;

/// A cached shape mask effect attachment. GPU parameter resources are created only on cache misses.
#[derive(Clone, Copy)]
pub(crate) struct ShapeMaskEffectInstance {
    pub effect_id: u64,
    pub parameters: EffectParametersHandle,
    pub config: ShapeMaskEffectConfig,
    pub bounds: ShapeMaskEffectBounds,
}

pub(crate) fn validate_shape_mask_effect_config(
    config: &ShapeMaskEffectConfig,
) -> Result<(), SceneError> {
    if !(config.downsample > 0.0 && config.downsample <= 1.0) {
        return Err(SceneError::InvalidParams(format!(
            "shape mask effect downsample must be in the range (0.0, 1.0], got {}",
            config.downsample
        )));
    }

    let outsets = [
        config.left_outset,
        config.top_outset,
        config.right_outset,
        config.bottom_outset,
    ];
    if outsets
        .iter()
        .any(|outset| !outset.is_finite() || *outset < 0.0)
    {
        return Err(SceneError::InvalidParams(
            "shape mask effect outsets must be finite and non-negative".to_string(),
        ));
    }

    Ok(())
}

/// A content shader attachment and its filtering configuration.
#[derive(Clone, Copy)]
pub(crate) struct ContentEffectInstance {
    pub effect_id: u64,
    pub parameters: EffectParametersHandle,
    pub config: ContentEffectConfig,
}

pub(super) fn resolve_backdrop_effect(
    node_id: usize,
    node: &DrawTreeNode,
    effect: EffectInstance,
    config: BackdropEffectConfig,
    viewport: Viewport,
    maximum_texture_dimension: u32,
) -> (BackdropEffectInstance, Option<EffectDamageEntry>) {
    let instance = BackdropEffectInstance::new(
        effect,
        config,
        node.logical_screen_bounds(),
        viewport,
        maximum_texture_dimension,
    );
    let entry = EffectDamageEntry::backdrop(node_id, node, instance.capture_bounds, viewport);
    (instance, entry)
}

impl SceneState {
    pub(crate) fn compact_effect_parameters(&mut self) {
        let parameters = self
            .content_groups
            .values_mut()
            .filter_map(|group| group.effect.as_mut().map(|effect| &mut effect.parameters))
            .chain(
                self.backdrop_effects
                    .values_mut()
                    .map(|effect| &mut effect.effect.parameters),
            )
            .chain(
                self.shape_mask_effects
                    .values_mut()
                    .map(|effect| &mut effect.parameters),
            )
            .chain(
                self.draw_tree
                    .iter_mut()
                    .filter_map(|(_, node)| match node {
                        DrawTreeNode::Group(group) => Some(&mut group.effects),
                        _ => None,
                    })
                    .flat_map(|effects| {
                        effects
                            .iter_mut()
                            .map(|effect| &mut effect.instance.parameters)
                    }),
            );
        self.effect_parameters.compact(parameters);
    }

    pub(crate) fn shape_mask_effect(
        &self,
        node_id: usize,
    ) -> Result<&ShapeMaskEffectInstance, SceneError> {
        self.shape_mask_effects
            .get(&node_id)
            .ok_or(SceneError::NodeNotFound(node_id))
    }

    pub(crate) fn set_content_effect(
        &mut self,
        node_id: usize,
        effect_id: u64,
        parameters: &[u8],
        config: ContentEffectConfig,
        maximum_texture_dimension: u32,
    ) -> Result<(), SceneError> {
        self.shape(node_id)?;
        if !config.is_valid() {
            return Err(SceneError::InvalidParams(
                "content effect outsets must be finite and non-negative, and downsample must be in the range (0.0, 1.0]".to_string(),
            ));
        }
        let parameters = self.effect_parameters.store(parameters);
        if self
            .content_groups
            .get(&node_id)
            .is_some_and(|group| group.effect.is_some())
        {
            self.effect_parameters.request_compaction();
        }
        let (group, entry) = resolve_content_group(
            node_id,
            self.draw_tree.get_unchecked(node_id),
            Some(ContentEffectInstance {
                effect_id,
                parameters,
                config,
            }),
            self.shape_mask_effects.get(&node_id),
            self.viewport,
            maximum_texture_dimension,
        );
        self.replace_content_group(node_id, group, entry);
        self.set_shape_effect_source(node_id, EffectAttachmentKind::Content, None);
        Ok(())
    }

    pub(crate) fn remove_content_effect(&mut self, node_id: usize) -> bool {
        if !self
            .content_groups
            .get(&node_id)
            .is_some_and(|group| group.effect.is_some())
        {
            return false;
        }
        self.content_groups.remove(&node_id);
        self.set_shape_effect_source(node_id, EffectAttachmentKind::Content, None);
        self.effect_parameters.request_compaction();
        self.effect_damage
            .remove(EffectDamageKind::Content, node_id);
        true
    }

    pub(super) fn replace_backdrop_effect(
        &mut self,
        node_id: usize,
        instance: BackdropEffectInstance,
        entry: Option<EffectDamageEntry>,
    ) {
        self.effect_damage
            .replace(EffectDamageKind::Backdrop, node_id, entry);
        self.backdrop_effects.insert(node_id, instance);
    }

    pub(crate) fn set_backdrop_effect(
        &mut self,
        node_id: usize,
        effect_id: u64,
        parameters: &[u8],
        config: BackdropEffectConfig,
        maximum_texture_dimension: u32,
    ) -> Result<(), SceneError> {
        let node = self
            .draw_tree
            .get(node_id)
            .ok_or(SceneError::NodeNotFound(node_id))?;
        let DrawTreeNode::CachedShape(_) = node else {
            return Err(SceneError::UnsupportedNodeOperation(node_id, "effects"));
        };
        validate_backdrop_config(&config)?;
        let parameters = self.effect_parameters.store(parameters);
        if self.backdrop_effects.contains_key(&node_id) {
            self.effect_parameters.request_compaction();
        }
        let effect = EffectInstance {
            effect_id,
            parameters,
        };
        let (instance, entry) = resolve_backdrop_effect(
            node_id,
            node,
            effect,
            config,
            self.viewport,
            maximum_texture_dimension,
        );
        self.replace_backdrop_effect(node_id, instance, entry);
        self.set_shape_effect_source(node_id, EffectAttachmentKind::Backdrop, None);
        Ok(())
    }

    /// Refreshes capture bounds, viewport overlap and allocation limits after viewport changes.
    pub(super) fn refresh_backdrop_capture_bounds(&mut self, maximum_texture_dimension: u32) {
        let entries = self
            .backdrop_effects
            .iter_mut()
            .filter_map(|(&node_id, instance)| {
                let node = self.draw_tree.get(node_id)?;
                let (updated, entry) = resolve_backdrop_effect(
                    node_id,
                    node,
                    instance.effect,
                    instance.config,
                    self.viewport,
                    maximum_texture_dimension,
                );
                *instance = updated;
                entry
            });
        self.effect_damage.rebuild(entries);
    }

    pub(crate) fn remove_backdrop_effect(&mut self, node_id: usize) -> bool {
        if self.backdrop_effects.remove(&node_id).is_none() {
            return false;
        }
        self.set_shape_effect_source(node_id, EffectAttachmentKind::Backdrop, None);
        self.effect_parameters.request_compaction();
        self.effect_damage
            .remove(EffectDamageKind::Backdrop, node_id);
        true
    }

    /// Attaches an effect using the scene's rasterization settings.
    /// Leaves the previous attachment unchanged if its bounds cannot be calculated.
    pub(crate) fn set_shape_mask_effect(
        &mut self,
        node_id: usize,
        effect_id: u64,
        parameters: &[u8],
        config: ShapeMaskEffectConfig,
        maximum_texture_dimension: u32,
    ) -> Result<(), SceneError> {
        let shape = self.shape(node_id)?;
        validate_shape_mask_effect_config(&config)?;
        let bounds = ShapeMaskEffectBounds::new(
            shape.local_bounds(),
            config,
            shape.transform,
            self.viewport.scale_factor,
        )
        .ok_or(SceneError::InvalidShapeMaskEffectBounds(node_id))?;
        let parameters = self.effect_parameters.store(parameters);
        if self.shape_mask_effects.contains_key(&node_id) {
            self.effect_parameters.request_compaction();
        }
        self.shape_mask_effects.insert(
            node_id,
            ShapeMaskEffectInstance {
                effect_id,
                parameters,
                config,
                bounds,
            },
        );
        self.set_shape_effect_source(node_id, EffectAttachmentKind::ShapeMask, None);
        self.refresh_content_group_bounds(node_id, maximum_texture_dimension);
        Ok(())
    }

    /// Refreshes cached rectangles using the supplied rasterization settings.
    pub(super) fn refresh_shape_mask_effect_bounds(
        &mut self,
        scale_factor: f64,
    ) -> Result<(), SceneError> {
        for (&node_id, effect) in &mut self.shape_mask_effects {
            let node = self
                .draw_tree
                .get(node_id)
                .ok_or(SceneError::NodeNotFound(node_id))?;
            let DrawTreeNode::CachedShape(shape) = node else {
                return Err(SceneError::UnsupportedNodeOperation(node_id, "effects"));
            };
            effect.bounds = ShapeMaskEffectBounds::new(
                shape.instance.local_bounds(),
                effect.config,
                shape.instance.transform,
                scale_factor,
            )
            .ok_or(SceneError::InvalidShapeMaskEffectBounds(node_id))?;
        }
        Ok(())
    }

    pub(crate) fn remove_shape_mask_effect(
        &mut self,
        node_id: usize,
        maximum_texture_dimension: u32,
    ) {
        if self.shape_mask_effects.remove(&node_id).is_some() {
            self.set_shape_effect_source(node_id, EffectAttachmentKind::ShapeMask, None);
            self.effect_parameters.request_compaction();
            self.refresh_content_group_bounds(node_id, maximum_texture_dimension);
        }
    }

    pub(crate) fn remove_effect_attachments(
        &mut self,
        effect_id: u64,
        maximum_texture_dimension: u32,
        mut removed: impl FnMut(RemovedEffectAttachment),
    ) {
        self.remove_group_effect_attachments(effect_id);
        self.content_groups.retain(|node_id, group| {
            if !group
                .effect
                .is_some_and(|effect| effect.effect_id == effect_id)
            {
                return true;
            }
            if let Some(DrawTreeNode::CachedShape(shape)) = self.draw_tree.get_mut(*node_id) {
                shape.inherited_effect_sources.content = None;
            }
            let damage_bounds = self
                .effect_damage
                .output_region(EffectDamageKind::Content, *node_id);
            self.effect_damage
                .remove(EffectDamageKind::Content, *node_id);
            self.effect_parameters.request_compaction();
            removed(RemovedEffectAttachment {
                node_id: *node_id,
                kind: EffectAttachmentKind::Content,
                damage_bounds,
            });
            false
        });
        self.backdrop_effects.retain(|node_id, instance| {
            if instance.effect.effect_id != effect_id {
                return true;
            }
            if let Some(DrawTreeNode::CachedShape(shape)) = self.draw_tree.get_mut(*node_id) {
                shape.inherited_effect_sources.backdrop = None;
            }
            self.effect_damage
                .remove(EffectDamageKind::Backdrop, *node_id);
            let damage_bounds = self.draw_tree.get(*node_id).and_then(|node| {
                geometry::logical_bounds_to_viewport_rect(
                    node.logical_screen_bounds(),
                    self.viewport,
                )
            });
            self.effect_parameters.request_compaction();
            removed(RemovedEffectAttachment {
                node_id: *node_id,
                kind: EffectAttachmentKind::Backdrop,
                damage_bounds,
            });
            false
        });
        self.shape_mask_effects.retain(|node_id, instance| {
            if instance.effect_id != effect_id {
                return true;
            }
            if let Some(DrawTreeNode::CachedShape(shape)) = self.draw_tree.get_mut(*node_id) {
                shape.inherited_effect_sources.shape_mask = None;
            }
            let mut damage_bounds = geometry::logical_bounds_to_viewport_rect(
                instance.bounds.logical_screen_bounds,
                self.viewport,
            );
            if let (Some(group), Some(node)) = (
                self.content_groups.get_mut(node_id),
                self.draw_tree.get(*node_id),
            ) {
                if let Some(output) = self
                    .effect_damage
                    .output_region(EffectDamageKind::Content, *node_id)
                {
                    damage_bounds =
                        Some(damage_bounds.map_or(output, |bounds| bounds.union(&output)));
                }
                let (updated, entry) = resolve_content_group(
                    *node_id,
                    node,
                    group.effect,
                    None,
                    self.viewport,
                    maximum_texture_dimension,
                );
                *group = updated;
                if let Some(output) = entry.and_then(|entry| entry.target_region) {
                    damage_bounds =
                        Some(damage_bounds.map_or(output, |bounds| bounds.union(&output)));
                }
                self.effect_damage
                    .replace(EffectDamageKind::Content, *node_id, entry);
            }
            self.effect_parameters.request_compaction();
            removed(RemovedEffectAttachment {
                node_id: *node_id,
                kind: EffectAttachmentKind::ShapeMask,
                damage_bounds,
            });
            false
        });
    }
}

#[cfg(test)]
mod tests;
