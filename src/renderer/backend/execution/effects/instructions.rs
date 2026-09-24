use super::{apply_effect_passes, EffectExecutionResources, EffectPassRunConfig, EffectRegistry};
use crate::commands::{EffectApplication, IntermediateTextureId};
use crate::renderer::backend::execution::textures::IntermediateTextureResources;
use wgpu::{BindGroupLayout, CommandEncoder, Device, Queue, Sampler, TextureFormat};

#[derive(Clone, Copy)]
pub(in crate::renderer) struct EffectContext<'a> {
    pub device: &'a Device,
    pub queue: &'a Queue,
    pub registry: &'a EffectRegistry,
    pub sampler: &'a Sampler,
    pub composite_layout: &'a BindGroupLayout,
    pub format: TextureFormat,
}

pub(in crate::renderer::backend::execution) fn execute_effect(
    encoder: &mut CommandEncoder,
    command: &EffectApplication,
    parameters: &[u8],
    context: &EffectContext<'_>,
    create_composite_bind_group: bool,
    resources: &mut EffectExecutionResources,
    textures: &mut IntermediateTextureResources,
) {
    let IntermediateTextureId::Planned(index) = command.input else {
        unreachable!("effect commands consume planned texture outputs");
    };
    let source = &mut textures.work_textures[textures.texture_id_to_work_textures_index[index]];
    let width = source.color_texture.width();
    let height = source.color_texture.height();
    let source_bind_group = source.input_bind_group(
        context.device,
        context.registry.input_bind_group_layout(),
        context.sampler,
    );
    let output = apply_effect_passes(
        context.registry,
        context.device,
        context.queue,
        &mut resources.parameters,
        encoder,
        &mut textures.pool,
        EffectPassRunConfig {
            effect_id: command.effect_id,
            params: command.parameters.bytes(parameters),
            source_bind_group,
            effect_sampler: context.sampler,
            composite_bind_group_layout: context.composite_layout,
            create_composite_bind_group,
            width,
            height,
            texture_format: context.format,
            label: "planned_effect",
        },
    );
    let (texture, _) = output.into_final_output(&mut textures.work_textures);
    textures.insert_planned(command.output, texture);
}
