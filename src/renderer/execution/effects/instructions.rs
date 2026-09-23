use super::{apply_effect_passes, EffectExecutionResources, EffectPassRunConfig};
use crate::renderer::commands::{EffectApplication, IntermediateTextureId};
use crate::renderer::execution::textures::IntermediateTextureResources;
use crate::renderer::types::BackdropContext;
use wgpu::CommandEncoder;

pub(in crate::renderer::execution) fn execute_effect(
    encoder: &mut CommandEncoder,
    command: &EffectApplication,
    parameters: &[u8],
    context: &BackdropContext<'_>,
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
        context.effect_registry.input_bind_group_layout(),
        context.effect_sampler,
    );
    let output = apply_effect_passes(
        context.effect_registry,
        context.device,
        context.queue,
        &mut resources.parameters,
        encoder,
        &mut textures.pool,
        EffectPassRunConfig {
            effect_id: command.effect_id,
            params: command.parameters.bytes(parameters),
            source_bind_group,
            effect_sampler: context.effect_sampler,
            composite_bind_group_layout: context.composite_bind_group_layout,
            create_composite_bind_group: false,
            width,
            height,
            texture_format: context.config_format,
            label: "planned_effect",
        },
    );
    let (texture, _) = output.into_final_output(&mut textures.work_textures);
    textures.insert_planned(command.output, texture);
}
