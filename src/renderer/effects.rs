use super::state::BackdropPipelineResources;
use super::*;
use crate::effect::{BackdropEffectInstance, EffectParameterResources};
use wgpu::util::{BufferInitDescriptor, DeviceExt};

fn overwrite_effect_params(storage: &mut Vec<u8>, params: &[u8]) {
    storage.clear();
    storage.extend_from_slice(params);
}

fn validate_params_expectation(
    effect_id: u64,
    expects_params: bool,
    params: &[u8],
) -> Result<(), EffectError> {
    if expects_params && params.is_empty() {
        return Err(EffectError::InvalidParams(format!(
            "effect {} expects parameters but none were provided",
            effect_id
        )));
    }

    if !expects_params && !params.is_empty() {
        return Err(EffectError::InvalidParams(format!(
            "effect {} does not accept parameters but {} bytes were provided",
            effect_id,
            params.len()
        )));
    }

    Ok(())
}

fn find_effect_and_validate_params<'a>(
    loaded_effects: &'a HashMap<u64, LoadedEffect>,
    effect_id: u64,
    params: &[u8],
) -> Result<&'a LoadedEffect, EffectError> {
    let loaded_effect = loaded_effects
        .get(&effect_id)
        .ok_or(EffectError::EffectNotLoaded(effect_id))?;

    validate_params_expectation(
        effect_id,
        loaded_effect.params_bind_group_layout.is_some(),
        params,
    )?;
    Ok(loaded_effect)
}

fn validate_backdrop_config(config: &effect::BackdropEffectConfig) -> Result<(), EffectError> {
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

    if let effect::BackdropCaptureArea::ScreenRect([(x0, y0), (x1, y1)]) = config.capture_area {
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

fn validate_shape_effect_config(config: &effect::ShapeEffectConfig) -> Result<(), EffectError> {
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

fn create_effect_parameter_resources(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    params: &[u8],
    buffer_label: &'static str,
) -> EffectParameterResources {
    let buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some(buffer_label),
        contents: params,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
    });
    let bind_group = create_params_bind_group(device, layout, &buffer);
    EffectParameterResources { buffer, bind_group }
}

fn build_effect_instance(
    device: &wgpu::Device,
    loaded_effect: &LoadedEffect,
    effect_id: u64,
    params: &[u8],
    params_buffer_label: &'static str,
) -> EffectInstance {
    let parameter_resources = loaded_effect
        .params_bind_group_layout
        .as_ref()
        .map(|layout| {
            create_effect_parameter_resources(device, layout, params, params_buffer_label)
        });
    EffectInstance {
        effect_id,
        params: params.to_vec(),
        parameter_resources,
    }
}

fn update_effect_instance_params(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    loaded_effect: &LoadedEffect,
    instance: &mut EffectInstance,
    params: &[u8],
    params_buffer_label: &'static str,
) -> Result<(), EffectError> {
    if let Some(resources) = instance.parameter_resources.as_ref() {
        let expected_size = resources.buffer.size();
        let actual_size = params.len() as u64;
        if actual_size != expected_size {
            return Err(EffectError::ParameterSizeMismatch {
                effect_id: instance.effect_id,
                expected_size,
                actual_size,
            });
        }
    }

    overwrite_effect_params(&mut instance.params, params);

    let Some(params_bind_group_layout) = loaded_effect.params_bind_group_layout.as_ref() else {
        return Ok(());
    };

    if let Some(resources) = instance.parameter_resources.as_ref() {
        queue.write_buffer(&resources.buffer, 0, params);
        return Ok(());
    }

    instance.parameter_resources = Some(create_effect_parameter_resources(
        device,
        params_bind_group_layout,
        params,
        params_buffer_label,
    ));
    Ok(())
}

fn refresh_effect_instance_after_reload(
    device: &wgpu::Device,
    loaded_effect: &LoadedEffect,
    instance: &mut EffectInstance,
) -> bool {
    if validate_params_expectation(
        instance.effect_id,
        loaded_effect.params_bind_group_layout.is_some(),
        &instance.params,
    )
    .is_err()
    {
        return false;
    }

    instance.parameter_resources = loaded_effect
        .params_bind_group_layout
        .as_ref()
        .map(|layout| {
            create_effect_parameter_resources(
                device,
                layout,
                &instance.params,
                "reloaded_effect_params_buffer",
            )
        });
    true
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
        if self.loaded_effects.get(&effect_id).is_some_and(|effect| {
            effect.pass_sources.len() == pass_sources.len()
                && effect
                    .pass_sources
                    .iter()
                    .zip(pass_sources)
                    .all(|(stored, requested)| stored.as_ref() == *requested)
        }) {
            return Ok(());
        }

        let loaded_effect = compile_effect_pipeline(
            &self.device,
            pass_sources,
            self.config.format,
            &mut self.effect_shader_validator,
        )?;
        self.loaded_effects.insert(effect_id, loaded_effect);
        let loaded_effect = self
            .loaded_effects
            .get(&effect_id)
            .expect("the newly compiled effect must be stored");
        self.state.group_effects.retain(|_, instance| {
            instance.effect_id != effect_id
                || refresh_effect_instance_after_reload(&self.device, loaded_effect, instance)
        });
        self.state.backdrop_effects.retain(|_, instance| {
            instance.effect.effect_id != effect_id
                || refresh_effect_instance_after_reload(
                    &self.device,
                    loaded_effect,
                    &mut instance.effect,
                )
        });
        self.state.shape_effects.retain(|_, instance| {
            instance.effect_id != effect_id
                || validate_params_expectation(
                    effect_id,
                    loaded_effect.params_bind_group_layout.is_some(),
                    &instance.params,
                )
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

        let loaded_effect =
            find_effect_and_validate_params(&self.loaded_effects, effect_id, params)?;

        let instance = build_effect_instance(
            &self.device,
            loaded_effect,
            effect_id,
            params,
            "effect_params_buffer",
        );

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

        let loaded_effect =
            find_effect_and_validate_params(&self.loaded_effects, instance.effect_id, params)?;

        update_effect_instance_params(
            &self.device,
            &self.queue,
            loaded_effect,
            instance,
            params,
            "effect_params_buffer",
        )
    }

    pub fn remove_group_effect(&mut self, node_id: usize) {
        self.state.group_effects.remove(&node_id);
    }

    pub fn set_shape_backdrop_effect(
        &mut self,
        node_id: usize,
        effect_id: u64,
        params: &[u8],
        backdrop_config: effect::BackdropEffectConfig,
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

        let loaded_effect =
            find_effect_and_validate_params(&self.loaded_effects, effect_id, params)?;
        validate_backdrop_config(&backdrop_config)?;

        let instance = build_effect_instance(
            &self.device,
            loaded_effect,
            effect_id,
            params,
            "backdrop_effect_params_buffer",
        );

        self.state.backdrop_effects.insert(
            node_id,
            BackdropEffectInstance::new(instance, backdrop_config),
        );
        Ok(())
    }

    pub fn update_backdrop_effect_config(
        &mut self,
        node_id: usize,
        backdrop_config: effect::BackdropEffectConfig,
    ) -> Result<(), EffectError> {
        validate_backdrop_config(&backdrop_config)?;

        let instance = self
            .state
            .backdrop_effects
            .get_mut(&node_id)
            .ok_or(EffectError::NodeNotFound(node_id))?;
        instance.config = backdrop_config;
        instance.backdrop_texture_bind_group = None;
        instance.backdrop_texture_id = None;
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

        let loaded_effect =
            find_effect_and_validate_params(&self.loaded_effects, instance.effect_id, params)?;

        update_effect_instance_params(
            &self.device,
            &self.queue,
            loaded_effect,
            instance,
            params,
            "backdrop_effect_params_buffer",
        )
    }

    pub fn remove_backdrop_effect(&mut self, node_id: usize) {
        self.state.backdrop_effects.remove(&node_id);
    }

    /// Attaches a cached shader effect generated from the node's local coverage mask.
    pub fn set_shape_effect(
        &mut self,
        node_id: usize,
        effect_id: u64,
        params: &[u8],
        config: effect::ShapeEffectConfig,
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

        find_effect_and_validate_params(&self.loaded_effects, effect_id, params)?;
        validate_shape_effect_config(&config)?;
        self.state.shape_effects.insert(
            node_id,
            effect::ShapeEffectInstance {
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
        find_effect_and_validate_params(&self.loaded_effects, instance.effect_id, params)?;
        instance.params = Arc::from(params);
        Ok(())
    }

    /// Replaces the local-space padding used by an attached cached shape effect.
    pub fn update_shape_effect_config(
        &mut self,
        node_id: usize,
        config: effect::ShapeEffectConfig,
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
        self.loaded_effects.remove(&effect_id);
        self.state
            .group_effects
            .retain(|_, instance| instance.effect_id != effect_id);
        self.state
            .backdrop_effects
            .retain(|_, instance| instance.effect.effect_id != effect_id);
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
mod tests {
    use super::{validate_backdrop_config, validate_shape_effect_config};
    use crate::effect::{BackdropCaptureArea, BackdropEffectConfig, ShapeEffectConfig};

    #[test]
    fn validate_backdrop_config_rejects_non_positive_downsample() {
        let result = validate_backdrop_config(&BackdropEffectConfig::new().downsample(0.0));
        assert!(result.is_err());
    }

    #[test]
    fn validate_backdrop_config_rejects_negative_padding() {
        let result = validate_backdrop_config(&BackdropEffectConfig::new().padding(-1.0));
        assert!(result.is_err());
    }

    #[test]
    fn validate_backdrop_config_rejects_inverted_screen_rect() {
        let result = validate_backdrop_config(
            &BackdropEffectConfig::new()
                .capture_area(BackdropCaptureArea::ScreenRect([(10.0, 10.0), (5.0, 15.0)])),
        );
        assert!(result.is_err());
    }

    #[test]
    fn validate_backdrop_config_rejects_non_finite_screen_rect() {
        let result = validate_backdrop_config(&BackdropEffectConfig::new().capture_area(
            BackdropCaptureArea::ScreenRect([(0.0, 0.0), (f32::INFINITY, 15.0)]),
        ));
        assert!(result.is_err());
    }

    #[test]
    fn validate_shape_effect_config_rejects_negative_or_non_finite_outsets() {
        assert!(validate_shape_effect_config(&ShapeEffectConfig::new().outset(-1.0)).is_err());
        assert!(
            validate_shape_effect_config(&ShapeEffectConfig::new().outsets(
                0.0,
                f32::INFINITY,
                0.0,
                0.0
            ))
            .is_err()
        );
    }

    #[test]
    fn validate_shape_effect_config_rejects_out_of_range_downsample() {
        for downsample in [0.0, -0.5, f32::NAN, 1.5] {
            assert!(
                validate_shape_effect_config(&ShapeEffectConfig::new().downsample(downsample))
                    .is_err()
            );
        }
        for downsample in [1.0, 0.5, f32::EPSILON] {
            assert!(
                validate_shape_effect_config(&ShapeEffectConfig::new().downsample(downsample))
                    .is_ok()
            );
        }
    }
}
