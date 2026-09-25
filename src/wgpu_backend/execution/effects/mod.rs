pub(crate) use bindings::backdrop_layer_params;
pub(crate) use composite::{
    compile_backdrop_layer_composite_pipeline, compile_composite_pipeline,
    compile_texture_blit_pipeline, CompositePipelineResources,
};
pub(in crate::wgpu_backend) use instructions::EffectContext;
pub(crate) use parameters::{EffectExecutionResources, EffectParameterPool};
pub(crate) use registry::EffectRegistry;
pub(crate) use textures::{OffscreenTexturePool, PooledTexture};
use wgpu::{
    BindGroup, BindGroupLayout, Color, CommandEncoder, Device, LoadOp, Operations, Queue,
    RenderPassColorAttachment, RenderPassDescriptor, Sampler, StoreOp, TextureFormat,
};

mod bindings;
mod composite;
pub(in crate::wgpu_backend::execution) mod instructions;
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
}

pub(crate) struct EffectPassRunConfig<'a> {
    pub(crate) effect_id: u64,
    pub(crate) params: &'a [u8],
    pub(crate) source_bind_group: &'a BindGroup,
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
    queue: &Queue,
    parameters: &mut EffectParameterPool,
    encoder: &mut CommandEncoder,
    texture_pool: &mut OffscreenTexturePool,
    config: EffectPassRunConfig<'_>,
) -> AppliedEffectOutput {
    let loaded_effect = registry
        .loaded
        .get(&config.effect_id)
        .expect("effect attachments reference registered effects");
    let number_of_passes = loaded_effect.passes.len();
    let parameter_binding = loaded_effect
        .params_bind_group_layout
        .as_ref()
        .map(|layout| parameters.prepare(device, queue, layout, config.params));

    let mut effect_texture_a = texture_pool.acquire_color_only(
        device,
        config.width,
        config.height,
        config.texture_format,
        1,
    );

    let mut effect_texture_b = if number_of_passes > 1 {
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

    for (pass_index, effect_pass) in loaded_effect.passes.iter().enumerate() {
        let (input_bind_group, output_view) = if pass_index == 0 {
            (config.source_bind_group, &effect_texture_a.color_view)
        } else if pass_index % 2 == 0 {
            (
                effect_texture_b.as_mut().unwrap().input_bind_group(
                    device,
                    registry.input_bind_group_layout(),
                    config.effect_sampler,
                ),
                &effect_texture_a.color_view,
            )
        } else {
            (
                effect_texture_a.input_bind_group(
                    device,
                    registry.input_bind_group_layout(),
                    config.effect_sampler,
                ),
                &effect_texture_b.as_ref().unwrap().color_view,
            )
        };

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
        pass.set_bind_group(0, input_bind_group, &[]);

        if effect_pass.has_params {
            if let Some(binding) = parameter_binding {
                pass.set_bind_group(1, binding, &[]);
            }
        }

        pass.draw(0..3, 0..1);
    }

    let (mut final_output_texture, recyclable_texture) = if number_of_passes % 2 == 1 {
        (effect_texture_a, effect_texture_b)
    } else {
        (
            effect_texture_b.expect("an even number of effect passes needs a second texture"),
            Some(effect_texture_a),
        )
    };

    let composite_bind_group = config.create_composite_bind_group.then(|| {
        final_output_texture
            .composite_bind_group(
                device,
                config.composite_bind_group_layout,
                config.effect_sampler,
            )
            .clone()
    });

    AppliedEffectOutput {
        composite_bind_group,
        final_output_texture,
        recyclable_texture,
    }
}
