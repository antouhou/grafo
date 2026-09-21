use super::execution::effects::BackdropEffectResources;
use super::state::BackdropPipelineResources;
use super::*;
use crate::effect::{
    BackdropCaptureArea, BackdropEffectConfig, BackdropEffectInstance, ShapeEffectConfig,
    ShapeEffectInstance,
};

fn overwrite_effect_params(storage: &mut Vec<u8>, params: &[u8]) {
    storage.clear();
    storage.extend_from_slice(params);
}

fn validate_backdrop_config(config: &BackdropEffectConfig) -> Result<(), EffectError> {
    if !(config.downsample > 0.0 && config.downsample <= 1.0) {
        return Err(EffectError::InvalidParams(format!(
            "backdrop downsample must be in the range (0.0, 1.0], got {}",
            config.downsample
        )));
    }

    if !config.padding.is_finite() || config.padding < 0.0 {
        return Err(EffectError::InvalidParams(format!(
            "backdrop padding must be finite and non-negative, got {}",
            config.padding
        )));
    }

    if let BackdropCaptureArea::ScreenRect([(x0, y0), (x1, y1)]) = config.capture_area {
        if !(x0.is_finite() && y0.is_finite() && x1.is_finite() && y1.is_finite()) {
            return Err(EffectError::InvalidParams(
                "backdrop screen capture rectangles must use only finite coordinates".to_string(),
            ));
        }

        if !(x1 > x0 && y1 > y0) {
            return Err(EffectError::InvalidParams(
                "backdrop screen capture rectangles must have positive width and height"
                    .to_string(),
            ));
        }
    }

    Ok(())
}

fn validate_shape_effect_config(config: &ShapeEffectConfig) -> Result<(), EffectError> {
    if !(config.downsample > 0.0 && config.downsample <= 1.0) {
        return Err(EffectError::InvalidParams(format!(
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
        return Err(EffectError::InvalidParams(
            "shape effect outsets must be finite and non-negative".to_string(),
        ));
    }

    Ok(())
}

impl<'a> Renderer<'a> {
    /// Loads or replaces an effect from WGSL passes.
    ///
    /// Naga validates every pass before GPU resources are created. Invalid WGSL,
    /// missing fragment entry points, and unsupported bindings return an error
    /// without replacing an existing effect. Device-specific WGPU errors are
    /// logged through `tracing`.
    pub fn load_effect(
        &mut self,
        effect_id: u64,
        pass_sources: &[&str],
    ) -> Result<(), EffectError> {
        if !self
            .effect_registry
            .load(&self.device, self.config.format, effect_id, pass_sources)?
        {
            return Ok(());
        }
        self.state.group_effects.retain(|node_id, instance| {
            if instance.effect_id != effect_id {
                return true;
            }
            if self
                .effect_registry
                .validate_params(effect_id, &instance.params)
                .is_err()
            {
                self.state.effect_execution.group_parameters.remove(node_id);
                return false;
            }
            if let Some(resources) = self
                .state
                .effect_execution
                .group_parameters
                .get_mut(node_id)
            {
                self.effect_registry
                    .rebind_parameters(&self.device, effect_id, resources);
            }
            true
        });
        self.state.backdrop_effects.retain(|node_id, instance| {
            if instance.effect.effect_id != effect_id {
                return true;
            }
            if self
                .effect_registry
                .validate_params(effect_id, &instance.effect.params)
                .is_err()
            {
                self.state.effect_execution.backdrops.remove(node_id);
                return false;
            }
            if let Some(resources) = self
                .state
                .effect_execution
                .backdrops
                .get_mut(node_id)
                .and_then(|resources| resources.parameters.as_mut())
            {
                self.effect_registry
                    .rebind_parameters(&self.device, effect_id, resources);
            }
            true
        });
        self.state.shape_effects.retain(|_, instance| {
            instance.effect_id != effect_id
                || self
                    .effect_registry
                    .validate_params(effect_id, &instance.params)
                    .is_ok()
        });
        self.state
            .shape_effect_cache
            .retain(|cache_key, _| cache_key.effect_id != effect_id);
        Ok(())
    }

    pub fn set_group_effect(
        &mut self,
        node_id: usize,
        effect_id: u64,
        params: &[u8],
    ) -> Result<(), EffectError> {
        let draw_tree_node = self
            .state
            .draw_tree
            .get(node_id)
            .ok_or(EffectError::NodeNotFound(node_id))?;
        if draw_tree_node.is_clip_rect() {
            return Err(EffectError::InvalidParams(
                "clip rectangles do not support group effects".to_string(),
            ));
        }

        self.effect_registry.validate_params(effect_id, params)?;
        let parameters = self
            .effect_registry
            .create_parameters(&self.device, effect_id, params);
        if let Some(parameters) = parameters {
            self.state
                .effect_execution
                .group_parameters
                .insert(node_id, parameters);
        } else {
            self.state
                .effect_execution
                .group_parameters
                .remove(&node_id);
        }
        let instance = EffectInstance {
            effect_id,
            params: params.to_vec(),
        };

        self.state.group_effects.insert(node_id, instance);
        Ok(())
    }

    pub fn update_group_effect_params(
        &mut self,
        node_id: usize,
        params: &[u8],
    ) -> Result<(), EffectError> {
        let instance = self
            .state
            .group_effects
            .get_mut(&node_id)
            .ok_or(EffectError::NodeNotFound(node_id))?;

        self.effect_registry
            .validate_params(instance.effect_id, params)?;
        if let Some(resources) = self.state.effect_execution.group_parameters.get(&node_id) {
            resources.update(&self.queue, instance.effect_id, params)?;
        }
        overwrite_effect_params(&mut instance.params, params);
        Ok(())
    }

    pub fn remove_group_effect(&mut self, node_id: usize) {
        self.state.group_effects.remove(&node_id);
        self.state
            .effect_execution
            .group_parameters
            .remove(&node_id);
    }

    pub fn set_shape_backdrop_effect(
        &mut self,
        node_id: usize,
        effect_id: u64,
        params: &[u8],
        backdrop_config: BackdropEffectConfig,
    ) -> Result<(), EffectError> {
        let draw_tree_node = self
            .state
            .draw_tree
            .get(node_id)
            .ok_or(EffectError::NodeNotFound(node_id))?;
        if draw_tree_node.is_clip_rect() {
            return Err(EffectError::InvalidParams(
                "clip rectangles do not support backdrop effects".to_string(),
            ));
        }

        self.effect_registry.validate_params(effect_id, params)?;
        validate_backdrop_config(&backdrop_config)?;
        let parameters = self
            .effect_registry
            .create_parameters(&self.device, effect_id, params);
        self.state.effect_execution.backdrops.insert(
            node_id,
            BackdropEffectResources {
                parameters,
                ..Default::default()
            },
        );
        let instance = EffectInstance {
            effect_id,
            params: params.to_vec(),
        };

        self.state.backdrop_effects.insert(
            node_id,
            BackdropEffectInstance::new(instance, backdrop_config),
        );
        Ok(())
    }

    pub fn update_backdrop_effect_config(
        &mut self,
        node_id: usize,
        backdrop_config: BackdropEffectConfig,
    ) -> Result<(), EffectError> {
        validate_backdrop_config(&backdrop_config)?;

        let instance = self
            .state
            .backdrop_effects
            .get_mut(&node_id)
            .ok_or(EffectError::NodeNotFound(node_id))?;
        instance.config = backdrop_config;
        self.state
            .effect_execution
            .backdrops
            .get_mut(&node_id)
            .expect("backdrop attachments have execution resources")
            .invalidate_capture_binding();
        Ok(())
    }

    pub fn update_backdrop_effect_params(
        &mut self,
        node_id: usize,
        params: &[u8],
    ) -> Result<(), EffectError> {
        let instance = self
            .state
            .backdrop_effects
            .get_mut(&node_id)
            .ok_or(EffectError::NodeNotFound(node_id))?;
        let instance = &mut instance.effect;

        self.effect_registry
            .validate_params(instance.effect_id, params)?;
        if let Some(resources) = self
            .state
            .effect_execution
            .backdrops
            .get(&node_id)
            .and_then(|resources| resources.parameters.as_ref())
        {
            resources.update(&self.queue, instance.effect_id, params)?;
        }
        overwrite_effect_params(&mut instance.params, params);
        Ok(())
    }

    pub fn remove_backdrop_effect(&mut self, node_id: usize) {
        self.state.backdrop_effects.remove(&node_id);
        self.state.effect_execution.backdrops.remove(&node_id);
    }

    /// Attaches a cached shader effect generated from the node's local coverage mask.
    pub fn set_shape_effect(
        &mut self,
        node_id: usize,
        effect_id: u64,
        params: &[u8],
        config: ShapeEffectConfig,
    ) -> Result<(), EffectError> {
        let draw_tree_node = self
            .state
            .draw_tree
            .get(node_id)
            .ok_or(EffectError::NodeNotFound(node_id))?;
        if draw_tree_node.is_clip_rect() {
            return Err(EffectError::InvalidParams(
                "clip rectangles do not support shape effects".to_string(),
            ));
        }

        self.effect_registry.validate_params(effect_id, params)?;
        validate_shape_effect_config(&config)?;
        self.state.shape_effects.insert(
            node_id,
            ShapeEffectInstance {
                effect_id,
                params: Arc::from(params),
                config,
            },
        );
        Ok(())
    }

    /// Replaces the exact parameter bytes used by an attached cached shape effect.
    pub fn update_shape_effect_params(
        &mut self,
        node_id: usize,
        params: &[u8],
    ) -> Result<(), EffectError> {
        let instance = self
            .state
            .shape_effects
            .get_mut(&node_id)
            .ok_or(EffectError::NodeNotFound(node_id))?;
        self.effect_registry
            .validate_params(instance.effect_id, params)?;
        instance.params = Arc::from(params);
        Ok(())
    }

    /// Replaces the local-space padding used by an attached cached shape effect.
    pub fn update_shape_effect_config(
        &mut self,
        node_id: usize,
        config: ShapeEffectConfig,
    ) -> Result<(), EffectError> {
        validate_shape_effect_config(&config)?;
        let instance = self
            .state
            .shape_effects
            .get_mut(&node_id)
            .ok_or(EffectError::NodeNotFound(node_id))?;
        instance.config = config;
        Ok(())
    }

    pub fn remove_shape_effect(&mut self, node_id: usize) {
        self.state.shape_effects.remove(&node_id);
    }

    pub fn unload_effect(&mut self, effect_id: u64) {
        self.effect_registry.unload(effect_id);
        self.state.group_effects.retain(|node_id, instance| {
            if instance.effect_id == effect_id {
                self.state.effect_execution.group_parameters.remove(node_id);
                return false;
            }
            true
        });
        self.state.backdrop_effects.retain(|node_id, instance| {
            if instance.effect.effect_id == effect_id {
                self.state.effect_execution.backdrops.remove(node_id);
                return false;
            }
            true
        });
        self.state
            .shape_effects
            .retain(|_, instance| instance.effect_id != effect_id);
        self.state
            .shape_effect_cache
            .retain(|cache_key, _| cache_key.effect_id != effect_id);
    }

    pub(super) fn ensure_composite_pipeline(&mut self) -> &CompositePipelineResources {
        self.pipeline_resources
            .composite_resources
            .get_or_insert_with(|| compile_composite_pipeline(&self.device, self.config.format))
    }

    pub(super) fn ensure_backdrop_pipelines(&mut self) {
        if self.pipeline_resources.backdrops.is_some() {
            return;
        }

        self.ensure_composite_pipeline();
        let resources = &self.pipeline_resources;
        let composite = resources
            .composite_resources
            .as_ref()
            .expect("composite resources were initialized above");
        self.pipeline_resources.backdrops = Some(BackdropPipelineResources::new(
            &self.device,
            self.config.format,
            self.msaa_sample_count,
            &resources.shapes,
            &composite.bind_group_layout,
        ));
    }

    pub(super) fn ensure_effect_sampler(&mut self) {
        if self.pipeline_resources.effect_sampler.is_none() {
            self.pipeline_resources.effect_sampler =
                Some(self.device.create_sampler(&wgpu::SamplerDescriptor {
                    address_mode_u: wgpu::AddressMode::ClampToEdge,
                    address_mode_v: wgpu::AddressMode::ClampToEdge,
                    address_mode_w: wgpu::AddressMode::ClampToEdge,
                    mag_filter: wgpu::FilterMode::Linear,
                    min_filter: wgpu::FilterMode::Linear,
                    mipmap_filter: wgpu::FilterMode::Linear,
                    ..Default::default()
                }));
        }
    }
}

#[cfg(test)]
mod tests;
