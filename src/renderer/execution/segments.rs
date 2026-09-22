use super::backdrops;
use super::backdrops::BackdropEffectParameters;
use super::draws::DrawPass;
use super::effects::EffectExecutionResources;
use super::instructions::execute_draw_instructions;
use super::shapes::ShapeExecutionResources;
use super::targets::{self, RenderTarget};
use super::textures::IntermediateTextureResources;
use crate::renderer::commands::{BackdropDraw, DrawPlan, DrawSegment, ShapeDrawId};
#[cfg(feature = "render_metrics")]
use crate::renderer::metrics::PipelineSwitchCounts;
use crate::renderer::state::{Buffers, RendererPipelineResources};
use crate::renderer::types::{
    BackdropContext, BackdropSource, BoundTextureState, Pipeline, PipelineTracker,
};
use crate::shape::ShapeTextureBinding;
use wgpu::CommandEncoder;

pub(in crate::renderer) struct SegmentRenderTarget<'a> {
    pub(in crate::renderer) output: RenderTarget<'a>,
    pub(in crate::renderer) backdrop_source: Option<BackdropSource<'a>>,
    pub(in crate::renderer) backdrop_context: Option<&'a BackdropContext<'a>>,
}

pub(in crate::renderer) struct SegmentExecutionResources<'a> {
    pub(in crate::renderer) pipelines: &'a RendererPipelineResources,
    pub(in crate::renderer) buffers: &'a Buffers,
    pub(in crate::renderer) shapes: &'a mut ShapeExecutionResources,
    pub(in crate::renderer) effects: &'a mut EffectExecutionResources,
    pub(in crate::renderer) textures: &'a mut IntermediateTextureResources,
}

pub(in crate::renderer) struct SegmentExecutionMetrics {
    #[cfg(feature = "render_metrics")]
    pub(in crate::renderer) pipeline_switches: PipelineSwitchCounts,
}

fn execute_backdrop(
    encoder: &mut CommandEncoder,
    command: &BackdropDraw,
    parameters: &[u8],
    target: &mut SegmentRenderTarget<'_>,
    resources: &mut SegmentExecutionResources<'_>,
    pipeline_tracker: &mut PipelineTracker,
    bound_textures: &mut BoundTextureState,
) {
    target.output.clear_if_needed(encoder);
    let context = target
        .backdrop_context
        .expect("backdrop commands require execution resources");
    let mut material = command.draw.material;
    if let Some(region) = command.capture {
        let layer = backdrops::apply_backdrop_effect(
            encoder,
            context,
            target
                .backdrop_source
                .expect("backdrop commands require source textures"),
            region,
            BackdropEffectParameters {
                effect_id: command.effect_id,
                params: parameters,
                downsample: command.downsample,
            },
            resources.effects,
            resources.textures,
        );
        let ShapeDrawId::Shape(node_id) = command.draw.id else {
            unreachable!("backdrops reference uploaded shapes");
        };
        let shape = resources
            .shapes
            .draws
            .get_mut(&node_id)
            .expect("backdrop shape was uploaded");
        material.under_fill_texture = shape.prepare_texture_material(
            material.has_gradient_fill(),
            encoder,
            layer,
            &mut resources.shapes.texture_materials,
            context.device,
            context.queue,
            &resources.pipelines.shapes,
            resources.textures,
        );
    }
    let mut render_pass = target.output.begin_pass(encoder, "backdrop_shape_pass");
    let mut draw_pass = DrawPass {
        render_pass: &mut render_pass,
        pipeline_tracker,
        bound_textures,
        pipelines: resources.pipelines,
        buffers: resources.buffers,
        textures: resources.textures,
    };
    targets::set_scissor(draw_pass.render_pass, command.parent_clip.scissor);
    let shape = resources.shapes.draw_resources(command.draw.id);
    draw_pass.increment_stencil(command.parent_clip.stencil_reference, shape);
    targets::set_scissor(draw_pass.render_pass, command.shape_clip.scissor);
    draw_pass.draw_shape(command.shape_clip.stencil_reference, material, shape);
    if command.decrements_stencil {
        draw_pass.decrement_bound_shape_stencil(command.shape_clip.stencil_reference, shape);
    }
    drop(render_pass);
    if let Some(layer) = material.under_fill_texture {
        if let ShapeTextureBinding::Intermediate(texture_id) = layer.texture {
            resources.textures.finish_transient(texture_id);
        }
    }
    pipeline_tracker.switch_to(Pipeline::None);
    bound_textures.invalidate();
}

pub(in crate::renderer) fn execute_segments(
    encoder: &mut CommandEncoder,
    commands: &DrawPlan,
    mut target: SegmentRenderTarget<'_>,
    mut resources: SegmentExecutionResources<'_>,
) -> SegmentExecutionMetrics {
    let mut pipeline_tracker = PipelineTracker::new();
    let mut bound_textures = BoundTextureState::default();
    for segment in &commands.segments {
        match segment {
            DrawSegment::Draws(range) => {
                let mut render_pass = target.output.begin_pass(encoder, "segment_pass");
                let mut draw_pass = DrawPass {
                    render_pass: &mut render_pass,
                    pipeline_tracker: &mut pipeline_tracker,
                    bound_textures: &mut bound_textures,
                    pipelines: resources.pipelines,
                    buffers: resources.buffers,
                    textures: resources.textures,
                };
                execute_draw_instructions(
                    &commands.instructions[range.clone()],
                    &mut draw_pass,
                    resources.shapes,
                );
            }
            DrawSegment::Backdrop(command) => execute_backdrop(
                encoder,
                command,
                &commands.effect_parameters[command.parameter_start..command.parameter_end],
                &mut target,
                &mut resources,
                &mut pipeline_tracker,
                &mut bound_textures,
            ),
        }
    }
    target.output.clear_if_needed(encoder);
    #[cfg(feature = "render_metrics")]
    {
        pipeline_tracker.counts.scissor_clips = commands.scissor_clip_count;
    }
    SegmentExecutionMetrics {
        #[cfg(feature = "render_metrics")]
        pipeline_switches: pipeline_tracker.counts,
    }
}
