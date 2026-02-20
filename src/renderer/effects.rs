use super::*;

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

fn validate_effect_params(
    loaded_effects: &HashMap<u64, LoadedEffect>,
    effect_id: u64,
    params: &[u8],
) -> Result<(), EffectError> {
    let loaded_effect = loaded_effects
        .get(&effect_id)
        .ok_or(EffectError::EffectNotLoaded(effect_id))?;

    validate_params_expectation(
        effect_id,
        loaded_effect.params_bind_group_layout.is_some(),
        params,
    )
}

fn build_effect_instance(
    device: &wgpu::Device,
    loaded_effects: &HashMap<u64, LoadedEffect>,
    effect_id: u64,
    params: &[u8],
    params_buffer_label: &'static str,
) -> EffectInstance {
    let mut instance = EffectInstance {
        effect_id,
        params: params.to_vec(),
        params_buffer: None,
        params_bind_group: None,
    };

    if params.is_empty() {
        return instance;
    }

    let buffer = crate::pipeline::create_buffer_init(
        device,
        Some(params_buffer_label),
        params,
        BufferUsages::UNIFORM | BufferUsages::COPY_DST,
    );

    if let Some(loaded_effect) = loaded_effects.get(&effect_id) {
        if let Some(params_bind_group_layout) = loaded_effect.params_bind_group_layout.as_ref() {
            let bind_group = create_params_bind_group(device, params_bind_group_layout, &buffer);
            instance.params_bind_group = Some(bind_group);
        }
    }

    instance.params_buffer = Some(buffer);
    instance
}

fn update_effect_instance_params(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    loaded_effects: &HashMap<u64, LoadedEffect>,
    instance: &mut EffectInstance,
    params: &[u8],
    params_buffer_label: &'static str,
) {
    overwrite_effect_params(&mut instance.params, params);

    if params.is_empty() {
        return;
    }

    if let Some(existing_buffer) = instance.params_buffer.as_ref() {
        if params.len() as u64 <= existing_buffer.size() {
            queue.write_buffer(existing_buffer, 0, params);
            return;
        }
    }

    let new_buffer = crate::pipeline::create_buffer_init(
        device,
        Some(params_buffer_label),
        params,
        BufferUsages::UNIFORM | BufferUsages::COPY_DST,
    );

    if let Some(loaded_effect) = loaded_effects.get(&instance.effect_id) {
        if let Some(params_bind_group_layout) = loaded_effect.params_bind_group_layout.as_ref() {
            let bind_group =
                create_params_bind_group(device, params_bind_group_layout, &new_buffer);
            instance.params_bind_group = Some(bind_group);
        }
    }

    instance.params_buffer = Some(new_buffer);
}

impl<'a> Renderer<'a> {
    pub fn load_effect(
        &mut self,
        effect_id: u64,
        pass_sources: &[&str],
    ) -> Result<(), EffectError> {
        let loaded_effect =
            compile_effect_pipeline(&self.device, pass_sources, self.config.format)?;
        self.loaded_effects.insert(effect_id, loaded_effect);
        Ok(())
    }

    pub fn set_group_effect(
        &mut self,
        node_id: usize,
        effect_id: u64,
        params: &[u8],
    ) -> Result<(), EffectError> {
        if self.draw_tree.get(node_id).is_none() {
            return Err(EffectError::NodeNotFound(node_id));
        }

        validate_effect_params(&self.loaded_effects, effect_id, params)?;

        let instance = build_effect_instance(
            &self.device,
            &self.loaded_effects,
            effect_id,
            params,
            "effect_params_buffer",
        );

        self.group_effects.insert(node_id, instance);
        Ok(())
    }

    pub fn update_group_effect_params(
        &mut self,
        node_id: usize,
        params: &[u8],
    ) -> Result<(), EffectError> {
        let instance = self
            .group_effects
            .get_mut(&node_id)
            .ok_or(EffectError::NodeNotFound(node_id))?;

        validate_effect_params(&self.loaded_effects, instance.effect_id, params)?;

        update_effect_instance_params(
            &self.device,
            &self.queue,
            &self.loaded_effects,
            instance,
            params,
            "effect_params_buffer",
        );

        Ok(())
    }

    pub fn remove_group_effect(&mut self, node_id: usize) {
        self.group_effects.remove(&node_id);
    }

    pub fn set_shape_backdrop_effect(
        &mut self,
        node_id: usize,
        effect_id: u64,
        params: &[u8],
    ) -> Result<(), EffectError> {
        if self.draw_tree.get(node_id).is_none() {
            return Err(EffectError::NodeNotFound(node_id));
        }

        validate_effect_params(&self.loaded_effects, effect_id, params)?;

        let instance = build_effect_instance(
            &self.device,
            &self.loaded_effects,
            effect_id,
            params,
            "backdrop_effect_params_buffer",
        );

        self.backdrop_effects.insert(node_id, instance);
        Ok(())
    }

    pub fn update_backdrop_effect_params(
        &mut self,
        node_id: usize,
        params: &[u8],
    ) -> Result<(), EffectError> {
        let instance = self
            .backdrop_effects
            .get_mut(&node_id)
            .ok_or(EffectError::NodeNotFound(node_id))?;

        validate_effect_params(&self.loaded_effects, instance.effect_id, params)?;

        update_effect_instance_params(
            &self.device,
            &self.queue,
            &self.loaded_effects,
            instance,
            params,
            "backdrop_effect_params_buffer",
        );

        Ok(())
    }

    pub fn remove_backdrop_effect(&mut self, node_id: usize) {
        self.backdrop_effects.remove(&node_id);
    }

    pub fn unload_effect(&mut self, effect_id: u64) {
        self.loaded_effects.remove(&effect_id);
        self.group_effects
            .retain(|_, instance| instance.effect_id != effect_id);
        self.backdrop_effects
            .retain(|_, instance| instance.effect_id != effect_id);
    }

    pub(super) fn ensure_composite_pipeline(&mut self) {
        if self.composite_pipeline.is_none() {
            let (pipeline, bind_group_layout) =
                compile_composite_pipeline(&self.device, self.config.format);
            self.composite_pipeline = Some(pipeline);
            self.composite_bgl = Some(bind_group_layout);
        }
    }

    pub(super) fn ensure_effect_sampler(&mut self) {
        if self.effect_sampler.is_none() {
            self.effect_sampler = Some(self.device.create_sampler(&wgpu::SamplerDescriptor {
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

    pub(super) fn ensure_backdrop_snapshot_texture(&mut self) {
        let (width, height) = self.physical_size;
        let needs_recreate = match &self.backdrop_snapshot_texture {
            Some(texture) => {
                let size = texture.size();
                size.width != width || size.height != height
            }
            None => true,
        };

        if needs_recreate {
            let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("backdrop_snapshot"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.config.format,
                usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            self.backdrop_snapshot_texture = Some(texture);
            self.backdrop_snapshot_view = Some(view);
        }
    }

    pub(super) fn ensure_stencil_only_pipeline(&mut self) {
        if self.stencil_only_pipeline.is_some() {
            return;
        }

        let uniform_bind_group_layout = self.and_pipeline.get_bind_group_layout(0);
        let pipeline = crate::pipeline::create_stencil_only_pipeline(
            &self.device,
            self.config.format,
            self.msaa_sample_count,
            &uniform_bind_group_layout,
            &self.shape_texture_bind_group_layout_background,
            &self.shape_texture_bind_group_layout_foreground,
        );
        self.stencil_only_pipeline = Some(pipeline);
    }

    pub(super) fn ensure_backdrop_color_pipeline(&mut self) {
        if self.backdrop_color_pipeline.is_some() {
            return;
        }

        let uniform_bind_group_layout = self.and_pipeline.get_bind_group_layout(0);
        let pipeline = crate::pipeline::create_stencil_keep_color_pipeline(
            &self.device,
            self.config.format,
            self.msaa_sample_count,
            &uniform_bind_group_layout,
            &self.shape_texture_bind_group_layout_background,
            &self.shape_texture_bind_group_layout_foreground,
        );
        self.backdrop_color_pipeline = Some(pipeline);
    }
}

#[cfg(test)]
mod tests {
    use super::validate_params_expectation;

    #[test]
    fn validate_effect_params_rejects_missing_required_params() {
        let result = validate_params_expectation(1, true, &[]);
        assert!(result.is_err());
    }

    #[test]
    fn validate_effect_params_rejects_unexpected_params() {
        let result = validate_params_expectation(1, false, &[1, 2, 3, 4]);
        assert!(result.is_err());
    }

    #[test]
    fn validate_effect_params_allows_empty_for_paramless_effect() {
        let result = validate_params_expectation(1, false, &[]);
        assert!(result.is_ok());
    }

    #[test]
    fn validate_effect_params_allows_non_empty_for_param_effect() {
        let result = validate_params_expectation(1, true, &[1, 2, 3, 4]);
        assert!(result.is_ok());
    }
}
