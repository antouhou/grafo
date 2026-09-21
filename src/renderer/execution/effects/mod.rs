pub(crate) use bindings::{
    backdrop_layer_params, create_backdrop_layer_composite_bind_group,
    create_backdrop_texture_sample_bind_group, create_texture_sample_bind_group,
    prepare_backdrop_layer_params_buffer, prepare_solid_backdrop_material_params_buffer,
};
pub(crate) use composite::{
    compile_backdrop_layer_composite_pipeline, compile_composite_pipeline,
    compile_texture_blit_pipeline, CompositePipelineResources,
};
pub(crate) use parameters::{
    BackdropEffectResources, EffectExecutionResources, EffectParameterResources,
};
pub(crate) use registry::EffectRegistry;
pub(crate) use textures::{OffscreenTexturePool, PooledTexture};
use wgpu::{
    BindGroup, BindGroupLayout, Color, CommandEncoder, Device, LoadOp, Operations,
    RenderPassColorAttachment, RenderPassDescriptor, Sampler, StoreOp, TextureFormat, TextureView,
};

mod bindings;
mod composite;
mod parameters;
mod registry;
mod shaders;
mod textures;

pub(crate) struct AppliedEffectOutput {
    pub(crate) composite_bind_group: Option<BindGroup>,
    pub(crate) final_output_texture: PooledTexture,
    pub(crate) recyclable_texture: Option<PooledTexture>,
}

impl AppliedEffectOutput {
    pub(crate) fn into_final_output(
        self,
        textures_to_recycle: &mut Vec<PooledTexture>,
    ) -> (PooledTexture, Option<BindGroup>) {
        if let Some(recyclable_texture) = self.recyclable_texture {
            textures_to_recycle.push(recyclable_texture);
        }
        (self.final_output_texture, self.composite_bind_group)
    }

    pub(crate) fn push_work_textures_into(
        self,
        output_textures: &mut Vec<PooledTexture>,
    ) -> Option<BindGroup> {
        output_textures.push(self.final_output_texture);
        if let Some(recyclable_texture) = self.recyclable_texture {
            output_textures.push(recyclable_texture);
        }
        self.composite_bind_group
    }

    pub(crate) fn final_output_view(&self) -> &TextureView {
        &self.final_output_texture.color_view
    }

    pub(crate) fn final_output_texture_id(&self) -> u64 {
        self.final_output_texture.texture_id
    }
}

pub(crate) struct EffectPassRunConfig<'a> {
    pub(crate) effect_id: u64,
    pub(crate) params: &'a [u8],
    pub(crate) parameter_resources: Option<&'a EffectParameterResources>,
    pub(crate) source_view: &'a TextureView,
    pub(crate) effect_sampler: &'a Sampler,
    pub(crate) composite_bind_group_layout: &'a BindGroupLayout,
    pub(crate) create_composite_bind_group: bool,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) texture_format: TextureFormat,
    pub(crate) label: &'static str,
}

pub(crate) fn apply_effect_passes(
    registry: &EffectRegistry,
    device: &Device,
    encoder: &mut CommandEncoder,
    texture_pool: &mut OffscreenTexturePool,
    config: EffectPassRunConfig<'_>,
) -> AppliedEffectOutput {
    let loaded_effect = registry
        .loaded
        .get(&config.effect_id)
        .expect("effect attachments reference registered effects");
    let number_of_passes = loaded_effect.passes.len();
    // Cached shape effects upload only on a cache miss; attachments reuse their buffers.
    let transient_parameters = if config.parameter_resources.is_none() && !config.params.is_empty()
    {
        loaded_effect
            .params_bind_group_layout
            .as_ref()
            .map(|layout| EffectParameterResources::new(device, layout, config.params))
    } else {
        None
    };
    let parameter_resources = config.parameter_resources.or(transient_parameters.as_ref());

    let effect_texture_a = texture_pool.acquire_color_only(
        device,
        config.width,
        config.height,
        config.texture_format,
        1,
    );

    let effect_texture_b = if number_of_passes > 1 {
        Some(texture_pool.acquire_color_only(
            device,
            config.width,
            config.height,
            config.texture_format,
            1,
        ))
    } else {
        None
    };

    let mut previous_input_view: &TextureView = config.source_view;

    for (pass_index, effect_pass) in loaded_effect.passes.iter().enumerate() {
        let output_view = if pass_index % 2 == 0 {
            &effect_texture_a.color_view
        } else {
            &effect_texture_b.as_ref().unwrap().color_view
        };

        let input_bind_group = create_texture_sample_bind_group(
            device,
            &loaded_effect.input_bind_group_layout,
            previous_input_view,
            config.effect_sampler,
            Some(config.label),
        );

        let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some(config.label),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: output_view,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(Color::TRANSPARENT),
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        pass.set_pipeline(&effect_pass.pipeline);
        pass.set_bind_group(0, &input_bind_group, &[]);

        if effect_pass.has_params {
            if let Some(resources) = parameter_resources {
                pass.set_bind_group(1, &resources.bind_group, &[]);
            }
        }

        pass.draw(0..3, 0..1);
        previous_input_view = output_view;
    }

    let composite_bind_group = config.create_composite_bind_group.then(|| {
        create_texture_sample_bind_group(
            device,
            config.composite_bind_group_layout,
            previous_input_view,
            config.effect_sampler,
            Some(config.label),
        )
    });

    let (final_output_texture, recyclable_texture) = if number_of_passes % 2 == 1 {
        (effect_texture_a, effect_texture_b)
    } else {
        (
            effect_texture_b.expect("an even number of effect passes needs a second texture"),
            Some(effect_texture_a),
        )
    };

    AppliedEffectOutput {
        composite_bind_group,
        final_output_texture,
        recyclable_texture,
    }
}
