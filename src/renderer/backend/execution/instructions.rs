use super::backdrops;
use super::composites::CompositeInstanceBuffer;
use super::draws::DrawPass;
use super::effects::instructions::{execute_effect, EffectContext};
use super::effects::EffectExecutionResources;
use super::shape_effects::ShapeEffectExecutionResources;
use super::shapes::ShapeExecutionResources;
use super::targets::{self, ActiveTarget, SurfaceTarget};
use super::textures::{IntermediateTextureResources, PlannedTexture};
use crate::commands::{
    BackdropCaptureSource, IntermediateTextureId, RenderOperation, RenderPlan, Target,
    TexturePlacement,
};
use crate::renderer::backend::resources::{Buffers, RendererPipelineResources};
#[cfg(feature = "render_metrics")]
use crate::renderer::metrics::{PipelineSwitchCounts, ShapeEffectCacheMetrics};
use crate::renderer::types::{
    BackdropContext, BackdropSource, BoundTextureState, Pipeline, PipelineTracker,
};
use wgpu::{CommandEncoder, Device, Queue, TextureFormat};

pub(in crate::renderer) struct ExecutionContext<'a> {
    pub device: &'a Device,
    pub queue: &'a Queue,
    pub pipelines: &'a RendererPipelineResources,
    pub effects: Option<EffectContext<'a>>,
    pub backdrops: Option<BackdropContext<'a>>,
    pub format: TextureFormat,
    pub sample_count: u32,
}

pub(in crate::renderer) struct ExecutionResources<'a> {
    pub context: &'a ExecutionContext<'a>,
    pub buffers: &'a Buffers,
    pub shapes: &'a mut ShapeExecutionResources,
    pub effects: &'a mut EffectExecutionResources,
    pub textures: &'a mut IntermediateTextureResources,
    #[cfg(feature = "render_metrics")]
    pub shape_effect_metrics: &'a mut ShapeEffectCacheMetrics,
}

pub(in crate::renderer) struct ExecutionMetrics {
    #[cfg(feature = "render_metrics")]
    pub pipeline_switches: PipelineSwitchCounts,
}

impl ExecutionResources<'_> {
    fn mask_resources(&mut self) -> ShapeEffectExecutionResources<'_> {
        let effects = self
            .context
            .effects
            .as_ref()
            .expect("mask effects have execution resources");
        ShapeEffectExecutionResources {
            device: self.context.device,
            queue: self.context.queue,
            registry: effects.registry,
            sampler: effects.sampler,
            composite_layout: &self
                .context
                .pipelines
                .shapes
                .shape_texture_bind_group_layout_background,
            format: self.context.format,
            pipelines: &self.context.pipelines.shape_effects,
            buffers: self.buffers,
            shapes: self.shapes,
            effects: self.effects,
            textures: self.textures,
            #[cfg(feature = "render_metrics")]
            metrics: self.shape_effect_metrics,
        }
    }

    fn begin_target(&mut self, target: Target) {
        let texture = match target {
            Target::Surface => {
                assert!(
                    self.textures.active_targets.is_empty(),
                    "the surface is the root target"
                );
                None
            }
            Target::Texture { texture, size } => {
                self.textures.reserve_planned(texture);
                Some(self.textures.pool.acquire_with_depth(
                    self.context.device,
                    size.width,
                    size.height,
                    self.context.format,
                    self.context.sample_count,
                ))
            }
            Target::Mask(mask) => {
                self.textures.reserve_planned(mask.texture);
                None
            }
        };
        self.textures.active_targets.push(ActiveTarget {
            target,
            texture,
            needs_clear: true,
        });
    }

    fn end_target(&mut self, encoder: &mut CommandEncoder, surface: &SurfaceTarget<'_>) {
        let target = self
            .textures
            .active_targets
            .pop()
            .expect("EndTarget requires an open target");
        if matches!(target.target, Target::Mask(_)) {
            assert!(!target.needs_clear, "mask target must produce its coverage");
            return;
        }
        target.attachments(surface).clear_if_needed(encoder);
        if let Target::Texture { texture, .. } = target.target {
            let mut output = target.texture.expect("target owns its texture");
            let context = self
                .context
                .effects
                .as_ref()
                .expect("texture resources are initialized");
            output.composite_bind_group(context.device, context.composite_layout, context.sampler);
            self.textures.insert_planned(texture, output);
        }
    }
}

/// Consumes draw commands directly until the next operation needs the command encoder.
#[allow(clippy::too_many_arguments)]
fn execute_draws(
    encoder: &mut CommandEncoder,
    commands: &RenderPlan,
    cursor: &mut usize,
    surface: &SurfaceTarget<'_>,
    resources: &mut ExecutionResources<'_>,
    pipeline_tracker: &mut PipelineTracker,
    composite_instances: Option<CompositeInstanceBuffer>,
    composite_instance: &mut u32,
) {
    // A backdrop's stencil increment and under-fill draw share one GPU pass.
    let material_command = if matches!(
        commands.instructions[*cursor].operation,
        RenderOperation::IncrementStencil(_)
    ) {
        *cursor + 1
    } else {
        *cursor
    };
    if let Some(
        RenderOperation::DrawShape(draw) | RenderOperation::DrawShapeAndIncrementStencil(draw),
    ) = commands
        .instructions
        .get(material_command)
        .map(|command| &command.operation)
    {
        resources.shapes.prepare_texture_material(
            encoder,
            *draw,
            resources.context.device,
            resources.context.queue,
            &resources.context.pipelines.shapes,
            resources.textures,
        );
    }
    let target = resources
        .textures
        .active_targets
        .last_mut()
        .expect("draw requires an open target");
    let mut attachments = target.attachments(surface);
    let mut render_pass = attachments.begin_pass(encoder, "command_draws");
    target.needs_clear = false;
    let mut bound_textures = BoundTextureState::default();
    pipeline_tracker.current = Pipeline::None;
    let mut pass = DrawPass {
        render_pass: &mut render_pass,
        pipeline_tracker,
        bound_textures: &mut bound_textures,
        pipelines: resources.context.pipelines,
        buffers: resources.buffers,
        textures: resources.textures,
    };
    while let Some(command) = commands.instructions.get(*cursor) {
        if *cursor != material_command
            && matches!(&command.operation,
            RenderOperation::DrawShape(draw) | RenderOperation::DrawShapeAndIncrementStencil(draw)
            if draw.material.under_fill_texture.is_some())
        {
            break;
        }
        match &command.operation {
            RenderOperation::DrawShape(_) => {
                *cursor +=
                    pass.execute_leaf_draws(&commands.instructions[*cursor..], resources.shapes);
                continue;
            }
            RenderOperation::IncrementStencil(shape) => {
                targets::set_scissor(pass.render_pass, command.clip.scissor);
                pass.increment_stencil(
                    command.clip.stencil_reference,
                    resources.shapes.draw_resources(*shape),
                );
            }
            RenderOperation::DrawShapeAndIncrementStencil(draw) => {
                targets::set_scissor(pass.render_pass, command.clip.scissor);
                pass.draw_shape_and_increment_stencil(
                    command.clip.stencil_reference,
                    draw.material,
                    resources.shapes.draw_resources(draw.id),
                );
            }
            RenderOperation::DecrementStencil(draw) => {
                targets::set_scissor(pass.render_pass, command.clip.scissor);
                pass.decrement_stencil(
                    command.clip.stencil_reference,
                    resources.shapes.draw_resources(draw.id),
                );
            }
            RenderOperation::CompositeTexture(composite) => match composite.placement {
                TexturePlacement::Target => {
                    targets::set_scissor(pass.render_pass, command.clip.scissor);
                    pass.composite_texture(command.clip.stencil_reference, composite.texture);
                }
                TexturePlacement::Local { .. } => {
                    let count = pass.execute_texture_composites(
                        &commands.instructions[*cursor..],
                        &resources.shapes.composites,
                        composite_instances.expect("local composite instances were uploaded"),
                        *composite_instance,
                    );
                    *composite_instance += count as u32;
                    *cursor += count;
                    continue;
                }
            },
            _ => break,
        }
        *cursor += 1;
    }
}

pub(in crate::renderer) fn execute_commands(
    encoder: &mut CommandEncoder,
    commands: &RenderPlan,
    surface: SurfaceTarget<'_>,
    mut resources: ExecutionResources<'_>,
) -> ExecutionMetrics {
    let mut pipeline_tracker = PipelineTracker::new();
    let composite_instances = resources.shapes.composites.prepare(
        resources.context.device,
        resources.context.queue,
        commands,
    );
    let mut composite_instance = 0;
    let mut cursor = 0;
    while let Some(command) = commands.instructions.get(cursor) {
        match &command.operation {
            RenderOperation::BeginTarget(target) => resources.begin_target(*target),
            RenderOperation::EndTarget => resources.end_target(encoder, &surface),
            RenderOperation::DrawShapeMask(draw) => {
                let target = resources
                    .textures
                    .active_targets
                    .last_mut()
                    .expect("mask requires a target");
                let Target::Mask(mask) = target.target else {
                    unreachable!("mask draw requires a mask target")
                };
                assert!(
                    target.needs_clear,
                    "a mask command produces the whole coverage texture"
                );
                target.needs_clear = false;
                let mask = resources.mask_resources().draw_mask(encoder, mask, *draw);
                resources.textures.insert_mask(mask);
            }
            RenderOperation::ApplyEffect(effect) => {
                let mask = if let IntermediateTextureId::Planned(index) = effect.input {
                    match resources.textures.planned[index] {
                        PlannedTexture::Mask(index) => {
                            Some(resources.textures.masks[index].clone())
                        }
                        _ => None,
                    }
                } else {
                    None
                };
                if let Some(mask) = mask {
                    resources
                        .mask_resources()
                        .apply_effect(encoder, effect, commands, mask);
                } else {
                    execute_effect(
                        encoder,
                        effect,
                        commands,
                        resources
                            .context
                            .effects
                            .as_ref()
                            .expect("effect resources are initialized"),
                        true,
                        resources.effects,
                        resources.textures,
                    );
                }
            }
            RenderOperation::CaptureBackdrop(capture) => {
                let target = resources
                    .textures
                    .active_targets
                    .last_mut()
                    .expect("capture requires an open target");
                target.attachments(&surface).clear_if_needed(encoder);
                target.needs_clear = false;
                let foreground = target.attachments(&surface).resolved_view().clone();
                let base = match capture.source {
                    BackdropCaptureSource::Target => match &target.texture {
                        Some(texture) => texture
                            .resolve_texture
                            .as_ref()
                            .unwrap_or(&texture.color_texture)
                            .clone(),
                        None => surface
                            .capture_texture
                            .expect("surface has a capture texture")
                            .clone(),
                    },
                    BackdropCaptureSource::Layered { base } => {
                        resources.textures.texture(base).clone()
                    }
                };
                let source = match capture.source {
                    BackdropCaptureSource::Target => BackdropSource::Flattened { texture: &base },
                    BackdropCaptureSource::Layered { .. } => BackdropSource::Layered {
                        base_texture: &base,
                        foreground_view: &foreground,
                    },
                };
                backdrops::execute_capture(
                    encoder,
                    resources
                        .context
                        .backdrops
                        .as_ref()
                        .expect("capture resources are initialized"),
                    source,
                    *capture,
                    resources.effects,
                    resources.textures,
                );
            }
            _ => {
                execute_draws(
                    encoder,
                    commands,
                    &mut cursor,
                    &surface,
                    &mut resources,
                    &mut pipeline_tracker,
                    composite_instances,
                    &mut composite_instance,
                );
                continue;
            }
        }
        cursor += 1;
    }
    assert!(
        resources.textures.active_targets.is_empty(),
        "command stream must close every target"
    );
    resources.textures.finish_plan();
    #[cfg(feature = "render_metrics")]
    {
        pipeline_tracker.counts.scissor_clips = commands.scissor_clip_count;
    }
    ExecutionMetrics {
        #[cfg(feature = "render_metrics")]
        pipeline_switches: pipeline_tracker.counts,
    }
}
