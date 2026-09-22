use super::backdrops;
use super::draws::DrawPass;
use super::effects::instructions::execute_effect;
use super::effects::EffectExecutionResources;
use super::instructions::execute_draw_instructions;
use super::shapes::ShapeExecutionResources;
use super::targets::RenderTarget;
use super::textures::IntermediateTextureResources;
use crate::renderer::commands::{BackdropCaptureSource, DrawPlan, DrawSegment};
#[cfg(feature = "render_metrics")]
use crate::renderer::metrics::PipelineSwitchCounts;
use crate::renderer::state::{Buffers, RendererPipelineResources};
use crate::renderer::types::{
    BackdropContext, BackdropSource, BoundTextureState, Pipeline, PipelineTracker,
};
use wgpu::{CommandEncoder, Texture};

pub(in crate::renderer) struct SegmentRenderTarget<'a> {
    pub(in crate::renderer) output: RenderTarget<'a>,
    pub(in crate::renderer) capture_texture: Option<&'a Texture>,
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

pub(in crate::renderer) fn execute_segments(
    encoder: &mut CommandEncoder,
    commands: &DrawPlan,
    mut target: SegmentRenderTarget<'_>,
    resources: SegmentExecutionResources<'_>,
) -> SegmentExecutionMetrics {
    let mut pipeline_tracker = PipelineTracker::new();
    let mut bound_textures = BoundTextureState::default();
    for segment in &commands.segments {
        match segment {
            DrawSegment::Draws {
                instructions,
                texture_materials,
            } => {
                if !texture_materials.is_empty() {
                    let context = target
                        .backdrop_context
                        .expect("texture materials require execution resources");
                    resources.shapes.prepare_texture_materials(
                        encoder,
                        commands,
                        texture_materials.clone(),
                        context.device,
                        context.queue,
                        &resources.pipelines.shapes,
                        resources.textures,
                    );
                }
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
                    &commands.instructions[instructions.clone()],
                    &mut draw_pass,
                    resources.shapes,
                );
                // A new GPU pass must bind all state even when its first pipeline matches.
                pipeline_tracker.current = Pipeline::None;
                bound_textures.invalidate();
            }
            DrawSegment::CaptureBackdrop(command) => {
                target.output.clear_if_needed(encoder);
                let context = target
                    .backdrop_context
                    .expect("capture commands require execution resources");
                let base_texture;
                let source = match command.source {
                    BackdropCaptureSource::Target => BackdropSource::Flattened {
                        texture: target
                            .capture_texture
                            .expect("capture target has a resolved texture"),
                    },
                    BackdropCaptureSource::Layered { base } => {
                        base_texture = resources.textures.texture(base).clone();
                        BackdropSource::Layered {
                            base_texture: &base_texture,
                            foreground_view: target.output.resolved_view(),
                        }
                    }
                };
                backdrops::execute_capture(
                    encoder,
                    context,
                    source,
                    *command,
                    resources.effects,
                    resources.textures,
                );
            }
            DrawSegment::ApplyEffect(command) => execute_effect(
                encoder,
                command,
                &commands.effect_parameters,
                target
                    .backdrop_context
                    .expect("effect commands require execution resources"),
                resources.effects,
                resources.textures,
            ),
        }
    }
    target.output.clear_if_needed(encoder);
    resources.textures.finish_plan();
    #[cfg(feature = "render_metrics")]
    {
        pipeline_tracker.counts.scissor_clips = commands.scissor_clip_count;
    }
    SegmentExecutionMetrics {
        #[cfg(feature = "render_metrics")]
        pipeline_switches: pipeline_tracker.counts,
    }
}
