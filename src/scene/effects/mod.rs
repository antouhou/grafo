use super::{Scene, SceneError};
use crate::commands::EffectParameters;
use crate::core::effect::{BackdropCaptureArea, BackdropEffectConfig, ShapeEffectConfig};

/// A cached shape effect attachment. GPU parameter resources are created only on cache misses.
#[derive(Clone, Copy)]
pub(crate) struct ShapeEffectInstance {
    pub effect_id: u64,
    pub parameters: EffectParameters,
    pub config: ShapeEffectConfig,
}

/// Parameters shared by group and backdrop effect attachments.
#[derive(Clone, Copy)]
pub(crate) struct EffectInstance {
    /// The loaded effect's ID.
    pub effect_id: u64,
    pub parameters: EffectParameters,
}

/// A backdrop effect attachment and its capture configuration.
#[derive(Clone, Copy)]
pub(crate) struct BackdropEffectInstance {
    pub effect: EffectInstance,
    pub config: BackdropEffectConfig,
}

impl BackdropEffectInstance {
    pub(crate) fn new(effect: EffectInstance, config: BackdropEffectConfig) -> Self {
        Self { effect, config }
    }
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

fn validate_backdrop_config(config: &BackdropEffectConfig) -> Result<(), SceneError> {
    if !(config.downsample > 0.0 && config.downsample <= 1.0) {
        return Err(SceneError::InvalidParams(format!(
            "backdrop downsample must be in the range (0.0, 1.0], got {}",
            config.downsample
        )));
    }

    if !config.padding.is_finite() || config.padding < 0.0 {
        return Err(SceneError::InvalidParams(format!(
            "backdrop padding must be finite and non-negative, got {}",
            config.padding
        )));
    }

    if let BackdropCaptureArea::ScreenRect([(x0, y0), (x1, y1)]) = config.capture_area {
        if !(x0.is_finite() && y0.is_finite() && x1.is_finite() && y1.is_finite()) {
            return Err(SceneError::InvalidParams(
                "backdrop screen capture rectangles must use only finite coordinates".to_string(),
            ));
        }

        if !(x1 > x0 && y1 > y0) {
            return Err(SceneError::InvalidParams(
                "backdrop screen capture rectangles must have positive width and height"
                    .to_string(),
            ));
        }
    }

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
    Backdrop,
    Shape,
}

impl Scene {
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
    pub fn set_shape_backdrop_effect(
        &mut self,
        node_id: usize,
        effect_id: u64,
        parameters: EffectParameters,
        config: BackdropEffectConfig,
    ) -> Result<(), SceneError> {
        self.shape(node_id)?;
        validate_backdrop_config(&config)?;
        self.backdrop_effects.insert(
            node_id,
            BackdropEffectInstance::new(
                EffectInstance {
                    effect_id,
                    parameters,
                },
                config,
            ),
        );
        Ok(())
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
    ) -> Result<(), SceneError> {
        validate_backdrop_config(&config)?;
        self.backdrop_effects
            .get_mut(&node_id)
            .ok_or(SceneError::NodeNotFound(node_id))?
            .config = config;
        Ok(())
    }
    pub fn remove_backdrop_effect(&mut self, node_id: usize) {
        self.backdrop_effects.remove(&node_id);
    }
    pub fn set_shape_effect(
        &mut self,
        node_id: usize,
        effect_id: u64,
        parameters: EffectParameters,
        config: ShapeEffectConfig,
    ) -> Result<(), SceneError> {
        self.shape(node_id)?;
        validate_shape_effect_config(&config)?;
        self.shape_effects.insert(
            node_id,
            ShapeEffectInstance {
                effect_id,
                parameters,
                config,
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
    ) -> Result<(), SceneError> {
        validate_shape_effect_config(&config)?;
        self.shape_effects
            .get_mut(&node_id)
            .ok_or(SceneError::NodeNotFound(node_id))?
            .config = config;
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
            removed(*node_id, EffectAttachment::Backdrop);
            false
        });
        self.shape_effects.retain(|node_id, instance| {
            if instance.effect_id != effect_id {
                return true;
            }
            removed(*node_id, EffectAttachment::Shape);
            false
        });
    }
}

#[cfg(test)]
mod tests;
