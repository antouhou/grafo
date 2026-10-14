use super::effect_damage::{EffectDamageEntry, EffectDamageKind};
use super::types::DrawTreeNode;
use super::{Scene, SceneError};
use crate::commands::EffectParameters;
use crate::core::effect::{
    BackdropEffectConfig, GroupEffectBounds, GroupEffectConfig, ShapeEffectBounds,
    ShapeEffectConfig,
};
use crate::core::{geometry, UnsignedPhysicalRect, Viewport};
use backdrops::validate_backdrop_config;
pub(crate) use backdrops::BackdropEffectInstance;
use std::mem;

mod backdrops;

/// A cached shape effect attachment. GPU parameter resources are created only on cache misses.
#[derive(Clone, Copy)]
pub(crate) struct ShapeEffectInstance {
    pub effect_id: u64,
    pub parameters: EffectParameters,
    pub config: ShapeEffectConfig,
    pub bounds: ShapeEffectBounds,
}

/// Parameters shared by group and backdrop effect attachments.
#[derive(Clone, Copy)]
pub(crate) struct EffectInstance {
    /// The loaded effect's ID.
    pub effect_id: u64,
    pub parameters: EffectParameters,
}

/// A group attachment with its resolved input and output bounds.
#[derive(Clone, Copy)]
pub(crate) struct GroupEffectInstance {
    pub effect_id: u64,
    pub parameters: EffectParameters,
    pub config: GroupEffectConfig,
    pub bounds: Option<GroupEffectBounds>,
}

impl GroupEffectInstance {
    pub(crate) fn new(
        effect: EffectInstance,
        config: GroupEffectConfig,
        node: &DrawTreeNode,
        shape_effect: Option<&ShapeEffectInstance>,
        viewport: Viewport,
        maximum_texture_dimension: u32,
    ) -> Self {
        let shape_effect_bounds = shape_effect
            .filter(|_| matches!(node, DrawTreeNode::CachedShape(shape) if shape.has_geometry()))
            .map(|effect| effect.bounds);
        Self {
            effect_id: effect.effect_id,
            parameters: effect.parameters,
            config,
            bounds: GroupEffectBounds::new(
                node.logical_screen_bounds(),
                shape_effect_bounds,
                config,
                viewport,
                maximum_texture_dimension,
            ),
        }
    }
}

fn resolve_group_effect(
    node_id: usize,
    node: &DrawTreeNode,
    effect: EffectInstance,
    config: GroupEffectConfig,
    shape_effect: Option<&ShapeEffectInstance>,
    viewport: Viewport,
    maximum_texture_dimension: u32,
) -> (GroupEffectInstance, Option<EffectDamageEntry>) {
    let instance = GroupEffectInstance::new(
        effect,
        config,
        node,
        shape_effect,
        viewport,
        maximum_texture_dimension,
    );
    let entry = EffectDamageEntry::group(node_id, instance.bounds, viewport.physical_size.into());
    (instance, entry)
}

pub(super) fn resolve_backdrop_effect(
    node_id: usize,
    node: &DrawTreeNode,
    effect: EffectInstance,
    config: BackdropEffectConfig,
    viewport: Viewport,
    fringe_width: f32,
    maximum_texture_dimension: u32,
) -> (BackdropEffectInstance, Option<EffectDamageEntry>) {
    let instance = BackdropEffectInstance::new(
        effect,
        config,
        node.logical_screen_bounds(),
        viewport,
        maximum_texture_dimension,
    );
    let entry = EffectDamageEntry::backdrop(
        node_id,
        node,
        instance.capture_region,
        viewport,
        fringe_width,
    );
    (instance, entry)
}

fn validate_shape_effect_config(config: &ShapeEffectConfig) -> Result<(), SceneError> {
    if !(config.downsample > 0.0 && config.downsample <= 1.0) {
        return Err(SceneError::InvalidParams(format!(
            "shape effect downsample must be in the range (0.0, 1.0], got {}",
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
            "shape effect outsets must be finite and non-negative".to_string(),
        ));
    }

    Ok(())
}

#[derive(Clone, Copy)]
pub(crate) enum EffectAttachmentKind {
    Group,
    Backdrop,
    Shape,
}

#[derive(Clone, Copy)]
pub(crate) struct RemovedEffectAttachment {
    pub node_id: usize,
    pub kind: EffectAttachmentKind,
    pub damage_bounds: Option<UnsignedPhysicalRect>,
}

impl Scene {
    pub(crate) fn compact_effect_parameters(
        &mut self,
        current: &mut Vec<u8>,
        compacted: &mut Vec<u8>,
    ) {
        compacted.clear();
        let parameters = self
            .group_effects
            .values_mut()
            .map(|effect| &mut effect.parameters)
            .chain(
                self.backdrop_effects
                    .values_mut()
                    .map(|effect| &mut effect.effect.parameters),
            )
            .chain(
                self.shape_effects
                    .values_mut()
                    .map(|effect| &mut effect.parameters),
            );
        for parameters in parameters {
            let start = compacted.len();
            compacted.extend_from_slice(&current[parameters.range.start..parameters.range.end]);
            parameters.range.start = start;
            parameters.range.end = compacted.len();
        }
        mem::swap(current, compacted);
        compacted.clear();
    }

    pub(crate) fn shape_effect(&self, node_id: usize) -> Result<&ShapeEffectInstance, SceneError> {
        self.shape_effects
            .get(&node_id)
            .ok_or(SceneError::NodeNotFound(node_id))
    }

    pub(crate) fn set_group_effect(
        &mut self,
        node_id: usize,
        effect_id: u64,
        parameters: EffectParameters,
        config: GroupEffectConfig,
        maximum_texture_dimension: u32,
    ) -> Result<(), SceneError> {
        self.shape(node_id)?;
        if !config.is_valid() {
            return Err(SceneError::InvalidParams(
                "group effect outsets must be finite and non-negative, and downsample must be in the range (0.0, 1.0]".to_string(),
            ));
        }
        let (instance, entry) = resolve_group_effect(
            node_id,
            self.draw_tree.get_unchecked(node_id),
            EffectInstance {
                effect_id,
                parameters,
            },
            config,
            self.shape_effects.get(&node_id),
            self.viewport,
            maximum_texture_dimension,
        );
        self.replace_group_effect(node_id, instance, entry);
        Ok(())
    }

    fn replace_group_effect(
        &mut self,
        node_id: usize,
        instance: GroupEffectInstance,
        entry: Option<EffectDamageEntry>,
    ) {
        self.effect_damage
            .replace(EffectDamageKind::Group, node_id, entry);
        self.group_effects.insert(node_id, instance);
    }

    pub(crate) fn remove_group_effect(&mut self, node_id: usize) -> bool {
        if self.group_effects.remove(&node_id).is_none() {
            return false;
        }
        self.effect_damage.remove(EffectDamageKind::Group, node_id);
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

    pub(crate) fn set_shape_backdrop_effect(
        &mut self,
        node_id: usize,
        effect: EffectInstance,
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
        let (instance, entry) = resolve_backdrop_effect(
            node_id,
            node,
            effect,
            config,
            self.viewport,
            self.fringe_width,
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
                    self.fringe_width,
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
        self.effect_damage
            .remove(EffectDamageKind::Backdrop, node_id);
        true
    }

    /// Attaches an effect using the scene's rasterization settings.
    /// Leaves the previous attachment unchanged if its bounds cannot be calculated.
    pub(crate) fn set_shape_effect(
        &mut self,
        node_id: usize,
        effect_id: u64,
        parameters: EffectParameters,
        config: ShapeEffectConfig,
        maximum_texture_dimension: u32,
    ) -> Result<(), SceneError> {
        let shape = self.shape(node_id)?;
        validate_shape_effect_config(&config)?;
        let bounds = ShapeEffectBounds::new(
            shape.cached_shape.tessellation.local_bounds,
            config,
            shape.transform,
            self.viewport.scale_factor,
            self.fringe_width,
        )
        .ok_or(SceneError::InvalidShapeEffectBounds(node_id))?;
        self.shape_effects.insert(
            node_id,
            ShapeEffectInstance {
                effect_id,
                parameters,
                config,
                bounds,
            },
        );
        self.refresh_group_effect_bounds(node_id, maximum_texture_dimension);
        Ok(())
    }

    /// Refreshes cached rectangles using the supplied rasterization settings.
    pub(super) fn refresh_shape_effect_bounds(
        &mut self,
        scale_factor: f64,
        fringe_width: f32,
    ) -> Result<(), SceneError> {
        for (&node_id, effect) in &mut self.shape_effects {
            let node = self
                .draw_tree
                .get(node_id)
                .ok_or(SceneError::NodeNotFound(node_id))?;
            let DrawTreeNode::CachedShape(shape) = node else {
                return Err(SceneError::UnsupportedClipRectOperation(node_id, "effects"));
            };
            effect.bounds = ShapeEffectBounds::new(
                shape.instance.cached_shape.tessellation.local_bounds,
                effect.config,
                shape.instance.transform,
                scale_factor,
                fringe_width,
            )
            .ok_or(SceneError::InvalidShapeEffectBounds(node_id))?;
        }
        Ok(())
    }

    pub(crate) fn remove_shape_effect(&mut self, node_id: usize, maximum_texture_dimension: u32) {
        if self.shape_effects.remove(&node_id).is_some() {
            self.refresh_group_effect_bounds(node_id, maximum_texture_dimension);
        }
    }

    pub(crate) fn remove_effect_attachments(
        &mut self,
        effect_id: u64,
        maximum_texture_dimension: u32,
        mut removed: impl FnMut(RemovedEffectAttachment),
    ) {
        self.group_effects.retain(|node_id, instance| {
            if instance.effect_id != effect_id {
                return true;
            }
            let damage_bounds = self
                .effect_damage
                .output_region(EffectDamageKind::Group, *node_id);
            self.effect_damage.remove(EffectDamageKind::Group, *node_id);
            removed(RemovedEffectAttachment {
                node_id: *node_id,
                kind: EffectAttachmentKind::Group,
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
                    self.fringe_width,
                )
            });
            removed(RemovedEffectAttachment {
                node_id: *node_id,
                kind: EffectAttachmentKind::Backdrop,
                damage_bounds,
            });
            false
        });
        self.shape_effects.retain(|node_id, instance| {
            if instance.effect_id != effect_id {
                return true;
            }
            let mut damage_bounds = geometry::logical_bounds_to_viewport_rect(
                instance.bounds.logical_screen_bounds,
                self.viewport,
                0.0,
            );
            if let (Some(group), Some(node)) = (
                self.group_effects.get_mut(node_id),
                self.draw_tree.get(*node_id),
            ) {
                if let Some(output) = self
                    .effect_damage
                    .output_region(EffectDamageKind::Group, *node_id)
                {
                    damage_bounds =
                        Some(damage_bounds.map_or(output, |bounds| bounds.union(&output)));
                }
                let (updated, entry) = resolve_group_effect(
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
                    .replace(EffectDamageKind::Group, *node_id, entry);
            }
            removed(RemovedEffectAttachment {
                node_id: *node_id,
                kind: EffectAttachmentKind::Shape,
                damage_bounds,
            });
            false
        });
    }

    pub(super) fn refresh_group_effect_bounds(
        &mut self,
        node_id: usize,
        maximum_texture_dimension: u32,
    ) {
        let Some(instance) = self.group_effects.get(&node_id) else {
            return;
        };
        let Some(node) = self.draw_tree.get(node_id) else {
            return;
        };
        let (updated, entry) = resolve_group_effect(
            node_id,
            node,
            EffectInstance {
                effect_id: instance.effect_id,
                parameters: instance.parameters,
            },
            instance.config,
            self.shape_effects.get(&node_id),
            self.viewport,
            maximum_texture_dimension,
        );
        self.replace_group_effect(node_id, updated, entry);
    }

    pub(super) fn refresh_all_group_effect_bounds(&mut self, maximum_texture_dimension: u32) {
        for (&node_id, instance) in &mut self.group_effects {
            let Some(node) = self.draw_tree.get(node_id) else {
                continue;
            };
            let (updated, entry) = resolve_group_effect(
                node_id,
                node,
                EffectInstance {
                    effect_id: instance.effect_id,
                    parameters: instance.parameters,
                },
                instance.config,
                self.shape_effects.get(&node_id),
                self.viewport,
                maximum_texture_dimension,
            );
            *instance = updated;
            self.effect_damage
                .replace(EffectDamageKind::Group, node_id, entry);
        }
    }
}

#[cfg(test)]
mod tests;
