use super::effect_damage::{EffectDamageEntry, EffectDamageKind};
use super::types::DrawTreeNode;
use super::{SceneError, SceneState};
use crate::core::effect::{
    BackdropEffectConfig, ContentEffectBounds, ContentEffectConfig, ShapeMaskEffectBounds,
    ShapeMaskEffectConfig,
};
use crate::core::{geometry, UnsignedPhysicalRect, Viewport};
use backdrops::validate_backdrop_config;
pub(crate) use backdrops::BackdropEffectInstance;
use parameters::EffectParametersHandle;

mod backdrops;
pub(crate) mod parameters;

/// A cached shape mask effect attachment. GPU parameter resources are created only on cache misses.
#[derive(Clone, Copy)]
pub(crate) struct ShapeMaskEffectInstance {
    pub effect_id: u64,
    pub parameters: EffectParametersHandle,
    pub config: ShapeMaskEffectConfig,
    pub bounds: ShapeMaskEffectBounds,
}

/// Parameters shared by content and backdrop effect attachments.
#[derive(Clone, Copy)]
pub(crate) struct EffectInstance {
    /// The loaded effect's ID.
    pub effect_id: u64,
    pub parameters: EffectParametersHandle,
}

/// A content attachment with its resolved input and output bounds.
#[derive(Clone, Copy)]
pub(crate) struct ContentEffectInstance {
    pub effect_id: u64,
    pub parameters: EffectParametersHandle,
    pub config: ContentEffectConfig,
    pub bounds: Option<ContentEffectBounds>,
}

impl ContentEffectInstance {
    pub(crate) fn new(
        effect: EffectInstance,
        config: ContentEffectConfig,
        node: &DrawTreeNode,
        shape_mask_effect: Option<&ShapeMaskEffectInstance>,
        viewport: Viewport,
        maximum_texture_dimension: u32,
    ) -> Self {
        let shape_mask_effect_bounds = shape_mask_effect
            .filter(|_| matches!(node, DrawTreeNode::CachedShape(shape) if shape.has_geometry()))
            .map(|effect| effect.bounds);
        Self {
            effect_id: effect.effect_id,
            parameters: effect.parameters,
            config,
            bounds: ContentEffectBounds::new(
                node.logical_screen_bounds(),
                shape_mask_effect_bounds,
                config,
                viewport,
                maximum_texture_dimension,
            ),
        }
    }
}

fn resolve_content_effect(
    node_id: usize,
    node: &DrawTreeNode,
    effect: EffectInstance,
    config: ContentEffectConfig,
    shape_mask_effect: Option<&ShapeMaskEffectInstance>,
    viewport: Viewport,
    maximum_texture_dimension: u32,
) -> (ContentEffectInstance, Option<EffectDamageEntry>) {
    let instance = ContentEffectInstance::new(
        effect,
        config,
        node,
        shape_mask_effect,
        viewport,
        maximum_texture_dimension,
    );
    let entry = EffectDamageEntry::content(node_id, instance.bounds, viewport.physical_size.into());
    (instance, entry)
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
    let entry = EffectDamageEntry::backdrop(node_id, node, instance.capture_region, viewport);
    (instance, entry)
}

fn validate_shape_mask_effect_config(config: &ShapeMaskEffectConfig) -> Result<(), SceneError> {
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

#[derive(Clone, Copy)]
pub(crate) enum EffectAttachmentKind {
    Content,
    Backdrop,
    ShapeMask,
}

#[derive(Clone, Copy)]
pub(crate) struct RemovedEffectAttachment {
    pub node_id: usize,
    pub kind: EffectAttachmentKind,
    pub damage_bounds: Option<UnsignedPhysicalRect>,
}

impl SceneState {
    pub(crate) fn compact_effect_parameters(&mut self) {
        let parameters = self
            .content_effects
            .values_mut()
            .map(|effect| &mut effect.parameters)
            .chain(
                self.backdrop_effects
                    .values_mut()
                    .map(|effect| &mut effect.effect.parameters),
            )
            .chain(
                self.shape_mask_effects
                    .values_mut()
                    .map(|effect| &mut effect.parameters),
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
        if self.content_effects.contains_key(&node_id) {
            self.effect_parameters.request_compaction();
        }
        let (instance, entry) = resolve_content_effect(
            node_id,
            self.draw_tree.get_unchecked(node_id),
            EffectInstance {
                effect_id,
                parameters,
            },
            config,
            self.shape_mask_effects.get(&node_id),
            self.viewport,
            maximum_texture_dimension,
        );
        self.replace_content_effect(node_id, instance, entry);
        Ok(())
    }

    fn replace_content_effect(
        &mut self,
        node_id: usize,
        instance: ContentEffectInstance,
        entry: Option<EffectDamageEntry>,
    ) {
        self.effect_damage
            .replace(EffectDamageKind::Content, node_id, entry);
        self.content_effects.insert(node_id, instance);
    }

    pub(crate) fn remove_content_effect(&mut self, node_id: usize) -> bool {
        if self.content_effects.remove(&node_id).is_none() {
            return false;
        }
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
            return Err(SceneError::UnsupportedClipRectOperation(node_id, "effects"));
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
        Ok(())
    }

    /// Refreshes capture bounds, viewport overlap and allocation limits after viewport changes.
    pub(super) fn refresh_backdrop_capture_regions(&mut self, maximum_texture_dimension: u32) {
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
            shape.cached_shape.tessellation.local_bounds,
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
        self.refresh_content_effect_bounds(node_id, maximum_texture_dimension);
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
                return Err(SceneError::UnsupportedClipRectOperation(node_id, "effects"));
            };
            effect.bounds = ShapeMaskEffectBounds::new(
                shape.instance.cached_shape.tessellation.local_bounds,
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
            self.effect_parameters.request_compaction();
            self.refresh_content_effect_bounds(node_id, maximum_texture_dimension);
        }
    }

    pub(crate) fn remove_effect_attachments(
        &mut self,
        effect_id: u64,
        maximum_texture_dimension: u32,
        mut removed: impl FnMut(RemovedEffectAttachment),
    ) {
        self.content_effects.retain(|node_id, instance| {
            if instance.effect_id != effect_id {
                return true;
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
            let mut damage_bounds = geometry::logical_bounds_to_viewport_rect(
                instance.bounds.logical_screen_bounds,
                self.viewport,
            );
            if let (Some(group), Some(node)) = (
                self.content_effects.get_mut(node_id),
                self.draw_tree.get(*node_id),
            ) {
                if let Some(output) = self
                    .effect_damage
                    .output_region(EffectDamageKind::Content, *node_id)
                {
                    damage_bounds =
                        Some(damage_bounds.map_or(output, |bounds| bounds.union(&output)));
                }
                let (updated, entry) = resolve_content_effect(
                    *node_id,
                    node,
                    EffectInstance {
                        effect_id: group.effect_id,
                        parameters: group.parameters,
                    },
                    group.config,
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

    pub(super) fn refresh_content_effect_bounds(
        &mut self,
        node_id: usize,
        maximum_texture_dimension: u32,
    ) {
        let Some(instance) = self.content_effects.get(&node_id) else {
            return;
        };
        let Some(node) = self.draw_tree.get(node_id) else {
            return;
        };
        let (updated, entry) = resolve_content_effect(
            node_id,
            node,
            EffectInstance {
                effect_id: instance.effect_id,
                parameters: instance.parameters,
            },
            instance.config,
            self.shape_mask_effects.get(&node_id),
            self.viewport,
            maximum_texture_dimension,
        );
        self.replace_content_effect(node_id, updated, entry);
    }

    pub(super) fn refresh_all_content_effect_bounds(&mut self, maximum_texture_dimension: u32) {
        for (&node_id, instance) in &mut self.content_effects {
            let Some(node) = self.draw_tree.get(node_id) else {
                continue;
            };
            let (updated, entry) = resolve_content_effect(
                node_id,
                node,
                EffectInstance {
                    effect_id: instance.effect_id,
                    parameters: instance.parameters,
                },
                instance.config,
                self.shape_mask_effects.get(&node_id),
                self.viewport,
                maximum_texture_dimension,
            );
            *instance = updated;
            self.effect_damage
                .replace(EffectDamageKind::Content, node_id, entry);
        }
    }
}

#[cfg(test)]
mod tests;
