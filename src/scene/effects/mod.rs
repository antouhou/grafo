use self::backdrops::validate_backdrop_config;
use super::backdrop_damage::BackdropDamageEntry;
use super::types::DrawTreeNode;
use super::{Scene, SceneError};
use crate::commands::EffectParameters;
use crate::core::effect::{BackdropEffectConfig, ShapeEffectBounds, ShapeEffectConfig};
use crate::core::{MathRect, Viewport};
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

fn update_effect_params(
    instance: &mut EffectInstance,
    parameters: EffectParameters,
) -> Result<(), SceneError> {
    let expected_size = instance.parameters.range.end - instance.parameters.range.start;
    let actual_size = parameters.range.end - parameters.range.start;
    if expected_size != actual_size {
        return Err(SceneError::ParameterSizeMismatch {
            effect_id: instance.effect_id,
            expected_size: expected_size as u64,
            actual_size: actual_size as u64,
        });
    }
    instance.parameters = parameters;
    Ok(())
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
pub(crate) enum EffectAttachment {
    Backdrop { shape_bounds: Option<MathRect> },
    Shape(ShapeEffectBounds),
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

    pub(crate) fn group_effect(&self, node_id: usize) -> Result<&EffectInstance, SceneError> {
        self.group_effects
            .get(&node_id)
            .ok_or(SceneError::NodeNotFound(node_id))
    }

    pub(crate) fn backdrop_effect(&self, node_id: usize) -> Result<&EffectInstance, SceneError> {
        Ok(&self
            .backdrop_effects
            .get(&node_id)
            .ok_or(SceneError::NodeNotFound(node_id))?
            .effect)
    }

    pub(crate) fn shape_effect(&self, node_id: usize) -> Result<&ShapeEffectInstance, SceneError> {
        self.shape_effects
            .get(&node_id)
            .ok_or(SceneError::NodeNotFound(node_id))
    }

    pub fn set_group_effect(
        &mut self,
        node_id: usize,
        effect_id: u64,
        parameters: EffectParameters,
    ) -> Result<(), SceneError> {
        self.shape(node_id)?;
        self.group_effects.insert(
            node_id,
            EffectInstance {
                effect_id,
                parameters,
            },
        );
        Ok(())
    }

    pub fn update_group_effect_params(
        &mut self,
        node_id: usize,
        parameters: EffectParameters,
    ) -> Result<(), SceneError> {
        update_effect_params(
            self.group_effects
                .get_mut(&node_id)
                .ok_or(SceneError::NodeNotFound(node_id))?,
            parameters,
        )
    }

    pub fn remove_group_effect(&mut self, node_id: usize) {
        self.group_effects.remove(&node_id);
    }

    pub(crate) fn set_shape_backdrop_effect(
        &mut self,
        node_id: usize,
        effect: EffectInstance,
        config: BackdropEffectConfig,
        viewport: Viewport,
        fringe_width: f32,
        maximum_texture_dimension: u32,
    ) -> Result<(), SceneError> {
        let node = self
            .draw_tree
            .get(node_id)
            .ok_or(SceneError::NodeNotFound(node_id))?;
        let DrawTreeNode::CachedShape(shape) = node else {
            return Err(SceneError::UnsupportedClipRectOperation(node_id, "effects"));
        };
        validate_backdrop_config(&config)?;
        let instance = BackdropEffectInstance::new(
            effect,
            config,
            shape.logical_screen_bounds,
            viewport,
            maximum_texture_dimension,
        );
        let entry = BackdropDamageEntry::new(
            node_id,
            node,
            instance.capture_region,
            viewport,
            fringe_width,
        );
        self.replace_backdrop_effect(node_id, instance, entry);
        Ok(())
    }

    fn replace_backdrop_effect(
        &mut self,
        node_id: usize,
        instance: BackdropEffectInstance,
        entry: Option<BackdropDamageEntry>,
    ) {
        self.backdrop_damage.replace(node_id, entry);
        self.backdrop_effects.insert(node_id, instance);
    }

    pub fn update_backdrop_effect_params(
        &mut self,
        node_id: usize,
        parameters: EffectParameters,
    ) -> Result<(), SceneError> {
        update_effect_params(
            &mut self
                .backdrop_effects
                .get_mut(&node_id)
                .ok_or(SceneError::NodeNotFound(node_id))?
                .effect,
            parameters,
        )
    }

    pub fn update_backdrop_effect_config(
        &mut self,
        node_id: usize,
        config: BackdropEffectConfig,
        viewport: Viewport,
        fringe_width: f32,
        maximum_texture_dimension: u32,
    ) -> Result<(), SceneError> {
        validate_backdrop_config(&config)?;
        let instance = self
            .backdrop_effects
            .get(&node_id)
            .ok_or(SceneError::NodeNotFound(node_id))?;
        let node = self
            .draw_tree
            .get(node_id)
            .ok_or(SceneError::NodeNotFound(node_id))?;
        let updated = BackdropEffectInstance::new(
            instance.effect,
            config,
            node.logical_screen_bounds(),
            viewport,
            maximum_texture_dimension,
        );
        let entry = BackdropDamageEntry::new(
            node_id,
            node,
            updated.capture_region,
            viewport,
            fringe_width,
        );
        self.replace_backdrop_effect(node_id, updated, entry);
        Ok(())
    }

    /// Refreshes capture bounds, viewport overlap and allocation limits after viewport changes.
    pub(crate) fn refresh_backdrop_capture_regions(
        &mut self,
        viewport: Viewport,
        fringe_width: f32,
        maximum_texture_dimension: u32,
    ) {
        let entries = self
            .backdrop_effects
            .iter_mut()
            .filter_map(|(&node_id, instance)| {
                let node = self.draw_tree.get(node_id)?;
                *instance = BackdropEffectInstance::new(
                    instance.effect,
                    instance.config,
                    node.logical_screen_bounds(),
                    viewport,
                    maximum_texture_dimension,
                );
                BackdropDamageEntry::new(
                    node_id,
                    node,
                    instance.capture_region,
                    viewport,
                    fringe_width,
                )
            });
        self.backdrop_damage.rebuild(entries);
    }

    pub fn remove_backdrop_effect(&mut self, node_id: usize) -> bool {
        if self.backdrop_effects.remove(&node_id).is_none() {
            return false;
        }
        self.backdrop_damage.remove(node_id);
        true
    }

    /// Attaches an effect and caches its bounds for the supplied rasterization settings.
    /// Leaves the previous attachment unchanged if its bounds cannot be calculated.
    pub fn set_shape_effect(
        &mut self,
        node_id: usize,
        effect_id: u64,
        parameters: EffectParameters,
        config: ShapeEffectConfig,
        viewport: Viewport,
        fringe_width: f32,
    ) -> Result<(), SceneError> {
        let shape = self.shape(node_id)?;
        validate_shape_effect_config(&config)?;
        let bounds = ShapeEffectBounds::new(
            shape.cached_shape.tessellation.local_bounds,
            config,
            shape.transform,
            viewport.scale_factor,
            fringe_width,
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
        Ok(())
    }

    pub fn update_shape_effect_params(
        &mut self,
        node_id: usize,
        parameters: EffectParameters,
    ) -> Result<(), SceneError> {
        let instance = self
            .shape_effects
            .get_mut(&node_id)
            .ok_or(SceneError::NodeNotFound(node_id))?;
        instance.parameters = parameters;
        Ok(())
    }

    pub fn update_shape_effect_config(
        &mut self,
        node_id: usize,
        config: ShapeEffectConfig,
        viewport: Viewport,
        fringe_width: f32,
    ) -> Result<(), SceneError> {
        validate_shape_effect_config(&config)?;
        let shape = self.shape(node_id)?;
        let bounds = ShapeEffectBounds::new(
            shape.cached_shape.tessellation.local_bounds,
            config,
            shape.transform,
            viewport.scale_factor,
            fringe_width,
        )
        .ok_or(SceneError::InvalidShapeEffectBounds(node_id))?;
        let instance = self
            .shape_effects
            .get_mut(&node_id)
            .ok_or(SceneError::NodeNotFound(node_id))?;
        instance.config = config;
        instance.bounds = bounds;
        Ok(())
    }

    /// Refreshes cached rectangles using the supplied rasterization settings.
    pub(crate) fn refresh_shape_effect_bounds(
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

    pub fn remove_shape_effect(&mut self, node_id: usize) {
        self.shape_effects.remove(&node_id);
    }

    pub(crate) fn remove_effect_attachments(
        &mut self,
        effect_id: u64,
        mut removed: impl FnMut(usize, EffectAttachment),
    ) {
        self.group_effects
            .retain(|_, instance| instance.effect_id != effect_id);
        self.backdrop_effects.retain(|node_id, instance| {
            if instance.effect.effect_id != effect_id {
                return true;
            }
            self.backdrop_damage.remove(*node_id);
            let shape_bounds = self
                .draw_tree
                .get(*node_id)
                .map(DrawTreeNode::logical_screen_bounds);
            removed(*node_id, EffectAttachment::Backdrop { shape_bounds });
            false
        });
        self.shape_effects.retain(|node_id, instance| {
            if instance.effect_id != effect_id {
                return true;
            }
            removed(*node_id, EffectAttachment::Shape(instance.bounds));
            false
        });
    }
}

#[cfg(test)]
mod tests;
