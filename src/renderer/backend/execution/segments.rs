use super::backdrops;
use super::draws::DrawPass;
use super::effects::instructions::{execute_effect, EffectContext};
use super::effects::EffectExecutionResources;
use super::instructions::execute_draw_instructions;
use super::shapes::ShapeExecutionResources;
use super::targets::RenderTarget;
use super::textures::IntermediateTextureResources;
use crate::commands::{BackdropCaptureSource, DrawPlan, DrawSegment, Target};
use crate::renderer::backend::resources::{Buffers, RendererPipelineResources};
#[cfg(feature = "render_metrics")]
use crate::renderer::metrics::PipelineSwitchCounts;
use crate::renderer::types::{
    BackdropContext, BackdropSource, BoundTextureState, Pipeline, PipelineTracker,
};
use std::slice::Iter;
use wgpu::{CommandEncoder, Device, Queue, Texture, TextureFormat};

pub(in crate::renderer) struct SegmentRenderTarget<'a> {
    pub(in crate::renderer) output: RenderTarget<'a>,
    pub(in crate::renderer) capture_texture: Option<&'a Texture>,
}

pub(in crate::renderer) struct SegmentExecutionContext<'a> {
    pub device: &'a Device,
    pub queue: &'a Queue,
    pub pipelines: &'a RendererPipelineResources,
    pub effects: Option<EffectContext<'a>>,
    pub backdrops: Option<BackdropContext<'a>>,
    pub format: TextureFormat,
    pub sample_count: u32,
}

pub(in crate::renderer) struct SegmentExecutionResources<'a> {
    pub context: &'a SegmentExecutionContext<'a>,
    pub(in crate::renderer) buffers: &'a Buffers,
    pub(in crate::renderer) shapes: &'a mut ShapeExecutionResources,
    pub(in crate::renderer) effects: &'a mut EffectExecutionResources,
    pub(in crate::renderer) textures: &'a mut IntermediateTextureResources,
}

pub(in crate::renderer) struct SegmentExecutionMetrics {
    #[cfg(feature = "render_metrics")]
    pub(in crate::renderer) pipeline_switches: PipelineSwitchCounts,
}

fn execute_target(
    encoder: &mut CommandEncoder,
    commands: &DrawPlan,
    segments: &mut Iter<'_, DrawSegment>,
    mut target: SegmentRenderTarget<'_>,
    resources: &mut SegmentExecutionResources<'_>,
    pipeline_tracker: &mut PipelineTracker,
) {
    let mut bound_textures = BoundTextureState::default();
    for segment in segments {
        match segment {
            DrawSegment::Draws {
                instructions,
                texture_materials,
                composites,
            } => {
                if !texture_materials.is_empty() {
                    resources.shapes.prepare_texture_materials(
                        encoder,
                        commands,
                        texture_materials.clone(),
                        resources.context.device,
                        resources.context.queue,
                        &resources.context.pipelines.shapes,
                        resources.textures,
                    );
                }
                let composite_instances = resources.shapes.composites.prepare(
                    resources.context.device,
                    resources.context.queue,
                    commands,
                    composites.clone(),
                );
                let mut render_pass = target.output.begin_pass(encoder, "segment_pass");
                let mut draw_pass = DrawPass {
                    render_pass: &mut render_pass,
                    pipeline_tracker,
                    bound_textures: &mut bound_textures,
                    pipelines: resources.context.pipelines,
                    buffers: resources.buffers,
                    textures: resources.textures,
                };
                execute_draw_instructions(
                    commands,
                    &commands.instructions[instructions.clone()],
                    &mut draw_pass,
                    resources.shapes,
                    composite_instances,
                );
                // A new GPU pass must bind all state even when its first pipeline matches.
                pipeline_tracker.current = Pipeline::None;
                bound_textures.invalidate();
            }
            DrawSegment::EndTarget => {
                target.output.clear_if_needed(encoder);
                return;
            }
            DrawSegment::BeginTarget(_) | DrawSegment::DrawShapeMask(_) => {
                unreachable!("scene target scopes cannot nest or draw masks")
            }
            DrawSegment::CaptureBackdrop(command) => {
                target.output.clear_if_needed(encoder);
                let context = resources
                    .context
                    .backdrops
                    .as_ref()
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
                resources
                    .context
                    .effects
                    .as_ref()
                    .expect("effect commands require execution resources"),
                false,
                resources.effects,
                resources.textures,
            ),
        }
    }
    unreachable!("scene target scope must end");
}

pub(in crate::renderer) fn execute_segments(
    encoder: &mut CommandEncoder,
    commands: &DrawPlan,
    surface: SegmentRenderTarget<'_>,
    mut resources: SegmentExecutionResources<'_>,
) -> SegmentExecutionMetrics {
    let mut pipeline_tracker = PipelineTracker::new();
    let mut surface = Some(surface);
    let mut segments = commands.segments.iter();
    while let Some(segment) = segments.next() {
        match segment {
            DrawSegment::BeginTarget(Target::Surface) => execute_target(
                encoder,
                commands,
                &mut segments,
                surface.take().expect("surface scope is submitted once"),
                &mut resources,
                &mut pipeline_tracker,
            ),
            DrawSegment::BeginTarget(Target::Texture { texture, size }) => {
                resources.textures.reserve_planned(*texture);
                let target = resources.textures.pool.acquire_with_depth(
                    resources.context.device,
                    size.width,
                    size.height,
                    resources.context.format,
                    resources.context.sample_count,
                );
                execute_target(
                    encoder,
                    commands,
                    &mut segments,
                    SegmentRenderTarget {
                        output: RenderTarget::for_texture(&target),
                        capture_texture: Some(
                            target
                                .resolve_texture
                                .as_ref()
                                .unwrap_or(&target.color_texture),
                        ),
                    },
                    &mut resources,
                    &mut pipeline_tracker,
                );
                resources.textures.insert_planned(*texture, target);
            }
            DrawSegment::ApplyEffect(command) => execute_effect(
                encoder,
                command,
                &commands.effect_parameters,
                resources
                    .context
                    .effects
                    .as_ref()
                    .expect("effects require execution resources"),
                true,
                resources.effects,
                resources.textures,
            ),
            _ => unreachable!("scene draws require a target scope"),
        }
    }
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
