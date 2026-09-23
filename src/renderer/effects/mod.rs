use super::execution::shapes::TextureMaterialPipelines;
use super::state::BackdropPipelineResources;
use super::*;
use crate::effect::{
    BackdropCaptureArea, BackdropEffectConfig, BackdropEffectInstance, ShapeEffectConfig,
    ShapeEffectInstance,
};

fn update_effect_params(instance: &mut EffectInstance, params: &[u8]) -> Result<(), EffectError> {
    if instance.params.len() != params.len() {
        return Err(EffectError::ParameterSizeMismatch {
            effect_id: instance.effect_id,
            expected_size: instance.params.len() as u64,
            actual_size: params.len() as u64,
        });
    }
    instance.params.copy_from_slice(params);
    Ok(())
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
    /// without replacing an existing effect. Identical sources leave it unchanged.
    /// Replacing changed sources removes all existing attachments and cached results
    /// for this ID. GPU parameter storage remains reusable. Attach the new effect
    /// with fresh parameters through the `set_*_effect` methods. Device-specific
    /// WGPU errors are logged through `tracing`.
    pub fn load_effect(
        &mut self,
        effect_id: u64,
        pass_sources: &[&str],
    ) -> Result<(), EffectError> {
        if self
            .effect_registry
            .load(&self.device, self.config.format, effect_id, pass_sources)?
        {
            self.remove_effect_attachments(effect_id);
        }
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
        update_effect_params(instance, params)
    }

    pub fn remove_group_effect(&mut self, node_id: usize) {
        self.state.group_effects.remove(&node_id);
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
        update_effect_params(instance, params)
    }

    pub fn remove_backdrop_effect(&mut self, node_id: usize) {
        self.state.backdrop_effects.remove(&node_id);
        if let Some(resources) = self.state.shape_execution.draws.get_mut(&node_id) {
            resources.clear_under_fill_binding();
        }
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
        let DrawTreeNode::CachedShape(shape) = draw_tree_node else {
            return Err(EffectError::InvalidParams(
                "clip rectangles do not support shape effects".to_string(),
            ));
        };

        self.effect_registry.validate_params(effect_id, params)?;
        validate_shape_effect_config(&config)?;
        self.state
            .shape_execution
            .draws
            .get_mut(&node_id)
            .expect("shape draw resources were prepared when queued")
            .mask_tessellation
            .get_or_insert_with(|| Arc::clone(&shape.cached_shape.tessellation));
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
        if let Some(resources) = self.state.shape_execution.draws.get_mut(&node_id) {
            resources.mask_tessellation = None;
        }
    }

    pub fn unload_effect(&mut self, effect_id: u64) {
        self.effect_registry.unload(effect_id);
        self.remove_effect_attachments(effect_id);
    }

    fn remove_effect_attachments(&mut self, effect_id: u64) {
        self.state
            .group_effects
            .retain(|_, instance| instance.effect_id != effect_id);
        self.state.backdrop_effects.retain(|node_id, instance| {
            if instance.effect.effect_id == effect_id {
                if let Some(resources) = self.state.shape_execution.draws.get_mut(node_id) {
                    resources.clear_under_fill_binding();
                }
                return false;
            }
            true
        });
        self.state.shape_effects.retain(|node_id, instance| {
            if instance.effect_id == effect_id {
                if let Some(resources) = self.state.shape_execution.draws.get_mut(node_id) {
                    resources.mask_tessellation = None;
                }
                return false;
            }
            true
        });
        self.state.textures.invalidate_shape_effect(effect_id);
    }

    pub(super) fn ensure_composite_pipeline(&mut self) -> &CompositePipelineResources {
        self.pipeline_resources
            .composite_resources
            .get_or_insert_with(|| {
                compile_composite_pipeline(&self.device, self.config.format, self.msaa_sample_count)
            })
    }

    pub(super) fn ensure_backdrop_pipelines(&mut self) {
        if self.pipeline_resources.backdrops.is_some() {
            return;
        }

        self.ensure_composite_pipeline();
        if self
            .pipeline_resources
            .shapes
            .under_fill_pipelines
            .is_none()
        {
            let under_fill_pipelines = TextureMaterialPipelines::new(
                &self.device,
                self.config.format,
                self.msaa_sample_count,
                &self.pipeline_resources.shapes,
            );
            self.pipeline_resources.shapes.under_fill_pipelines = Some(under_fill_pipelines);
        }
        let resources = &self.pipeline_resources;
        let composite = resources
            .composite_resources
            .as_ref()
            .expect("composite resources were initialized above");
        self.pipeline_resources.backdrops = Some(BackdropPipelineResources::new(
            &self.device,
            self.config.format,
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
