use super::draws::DrawPass;
use super::effects::instructions::{execute_effect, EffectContext};
use super::effects::EffectExecutionResources;
use super::leaf_batches::LeafBatch;
use super::render_pass::RenderPass;
use super::shape_mask_effects::ShapeMaskEffectExecutionResources;
use super::shapes::ShapeExecutionResources;
use super::target_uniforms;
use super::targets::{ActiveTarget, SurfaceTarget};
use super::textures::{IntermediateTextureResources, PlannedTexture};
use crate::commands::{
    IntermediateTextureId, RenderCommand, RenderOperation, Target, TexturePlacement,
};
#[cfg(feature = "render_metrics")]
use crate::wgpu_backend::metrics::{PipelineSwitchCounts, ShapeMaskEffectCacheMetrics};
use crate::wgpu_backend::resources::{Buffers, RendererPipelineResources};
use crate::wgpu_backend::types::BackdropContext;
use wgpu::{CommandEncoder, Device, Queue, TextureFormat};

pub(in crate::wgpu_backend) struct ExecutionContext<'a> {
    pub device: &'a Device,
    pub queue: &'a Queue,
    pub pipelines: &'a RendererPipelineResources,
    pub effects: EffectContext<'a>,
    pub backdrops: BackdropContext<'a>,
    pub format: TextureFormat,
    pub sample_count: u32,
}

pub(in crate::wgpu_backend) struct ExecutionResources<'a> {
    pub context: &'a ExecutionContext<'a>,
    pub buffers: &'a Buffers,
    pub shapes: &'a mut ShapeExecutionResources,
    pub effects: &'a mut EffectExecutionResources,
    pub textures: &'a mut IntermediateTextureResources,
    #[cfg(feature = "render_metrics")]
    pub shape_mask_effect_metrics: &'a mut ShapeMaskEffectCacheMetrics,
}

#[derive(Default)]
pub(in crate::wgpu_backend) struct ExecutionMetrics {
    #[cfg(feature = "render_metrics")]
    pub pipeline_switches: PipelineSwitchCounts,
}

impl ExecutionMetrics {
    fn accumulate(&mut self, _pass: &RenderPass<'_>) {
        #[cfg(feature = "render_metrics")]
        self.pipeline_switches
            .accumulate(&_pass.pipeline_switch_counts());
    }
}

impl ExecutionResources<'_> {
    fn mask_resources(&mut self) -> ShapeMaskEffectExecutionResources<'_> {
        let effects = &self.context.effects;
        ShapeMaskEffectExecutionResources {
            device: self.context.device,
            queue: self.context.queue,
            registry: effects.registry,
            sampler: effects.sampler,
            composite_layout: &self.context.pipelines.shapes.layouts.background_texture,
            format: self.context.format,
            pipelines: &self.context.pipelines.shape_mask_effects,
            shape_pipelines: &self.context.pipelines.shapes,
            buffers: self.buffers,
            shapes: self.shapes,
            effects: self.effects,
            textures: self.textures,
            #[cfg(feature = "render_metrics")]
            metrics: self.shape_mask_effect_metrics,
        }
    }

    fn begin_target(
        &mut self,
        encoder: &mut CommandEncoder,
        target: Target,
        surface: &SurfaceTarget<'_>,
    ) {
        let texture = match target {
            Target::Surface => {
                assert!(
                    self.textures.active_targets.is_empty(),
                    "the surface is the root target"
                );
                None
            }
            Target::Texture { texture, region } => {
                self.textures.reserve_planned(texture);
                let size = region.bounds.size().to_u32();
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
        let uniform_index = if matches!(target, Target::Mask(_)) {
            None
        } else {
            let pipelines = &self.context.pipelines.shapes;
            let mut uniforms = pipelines.and_uniforms;
            if let Target::Texture { region, .. } = target {
                uniforms.canvas_size = region
                    .bounds
                    .size()
                    .to_f32()
                    .to_array()
                    .map(|value| value / uniforms.scale_factor);
                uniforms.target_origin = region.bounds.min.to_f32().to_array();
            }
            let index = self.textures.target_uniforms.prepare(
                self.context.device,
                self.context.queue,
                uniforms,
            );
            // Surface uniforms are initialized on resize and restored after child targets.
            if matches!(target, Target::Texture { .. }) {
                target_uniforms::activate(
                    encoder,
                    self.textures.target_uniforms.buffer(index),
                    pipelines,
                );
            }
            Some(index)
        };
        self.textures.active_targets.push(ActiveTarget {
            target,
            texture,
            uniform_index,
            needs_clear: if matches!(target, Target::Surface) {
                surface.output.needs_clear
            } else {
                true
            },
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
            let context = &self.context.effects;
            output.composite_bind_group(context.device, context.composite_layout, context.sampler);
            self.textures.insert_planned(texture, output);
        }
        if let Some(index) = self
            .textures
            .active_targets
            .last()
            .and_then(|target| target.uniform_index)
        {
            target_uniforms::activate(
                encoder,
                self.textures.target_uniforms.buffer(index),
                &self.context.pipelines.shapes,
            );
        }
    }
}

fn should_end_draw_pass(command: &RenderCommand) -> bool {
    match command.operation {
        RenderOperation::DrawShape(draw) | RenderOperation::DrawShapeAndIncrementStencil(draw) => {
            draw.material.under_fill_texture.is_some()
        }
        RenderOperation::DecrementStencil(_) | RenderOperation::CompositeTexture(_) => false,
        _ => true,
    }
}

fn execute_draw_command(
    pass: &mut DrawPass<'_, '_>,
    command: RenderCommand,
    shapes: &mut ShapeExecutionResources,
    context: &ExecutionContext<'_>,
    pending_batch: &mut Option<LeafBatch>,
) {
    if let RenderOperation::DrawShape(draw) = command.operation {
        pass.queue_leaf_draw(draw, command.clip, shapes, pending_batch);
        return;
    }
    pass.flush_leaf_batch(pending_batch);
    match command.operation {
        RenderOperation::DrawShapeAndIncrementStencil(draw) => {
            pass.set_scissor(command.clip.scissor);
            pass.draw_shape_and_increment_stencil(
                command.clip.stencil_reference,
                draw.material,
                shapes.draw_resources(draw.id),
            );
        }
        RenderOperation::DecrementStencil(draw) => {
            pass.set_scissor(command.clip.scissor);
            pass.decrement_stencil(
                command.clip.stencil_reference,
                shapes.draw_resources(draw.id),
            );
        }
        RenderOperation::CompositeTexture(composite) => match composite.placement {
            TexturePlacement::Target => {
                pass.set_scissor(command.clip.scissor);
                pass.composite_texture(command.clip.stencil_reference, composite.texture);
            }
            TexturePlacement::Local { .. } => {
                pass.composite_local_texture(
                    composite,
                    command.clip,
                    &mut shapes.composites,
                    context.device,
                    context.queue,
                );
            }
        },
        _ => unreachable!("draw passes execute only draw commands"),
    }
}

/// Returns a consumed command that still needs execution after this pass closes.
fn execute_draw_pass<'scene>(
    encoder: &mut CommandEncoder,
    pending_command: RenderCommand<'scene>,
    commands: &mut impl Iterator<Item = RenderCommand<'scene>>,
    surface: &SurfaceTarget<'_>,
    resources: &mut ExecutionResources<'_>,
    metrics: &mut ExecutionMetrics,
) -> Option<RenderCommand<'scene>> {
    if let RenderOperation::DrawShape(draw) | RenderOperation::DrawShapeAndIncrementStencil(draw) =
        &pending_command.operation
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
    let root_scissor = matches!(target.target, Target::Surface).then_some(surface.root_scissor);
    let target_region = match target.target {
        Target::Texture { region, .. } => Some(region),
        _ => None,
    };
    let mut render_pass = RenderPass::new(attachments.begin_pass(encoder, "command_draws"));
    target.needs_clear = false;
    let mut pass = DrawPass {
        root_scissor,
        target_region,
        render_pass: &mut render_pass,
        pipelines: resources.context.pipelines,
        buffers: resources.buffers,
        textures: resources.textures,
    };
    let mut pending_batch = None;
    execute_draw_command(
        &mut pass,
        pending_command,
        resources.shapes,
        resources.context,
        &mut pending_batch,
    );
    let mut next_command = None;
    for command in commands {
        if should_end_draw_pass(&command) {
            next_command = Some(command);
            break;
        }
        execute_draw_command(
            &mut pass,
            command,
            resources.shapes,
            resources.context,
            &mut pending_batch,
        );
    }
    pass.flush_leaf_batch(&mut pending_batch);
    metrics.accumulate(pass.render_pass);
    next_command
}

pub(in crate::wgpu_backend) fn execute_commands<'scene>(
    encoder: &mut CommandEncoder,
    mut commands: impl Iterator<Item = RenderCommand<'scene>>,
    surface: SurfaceTarget<'_>,
    mut resources: ExecutionResources<'_>,
) -> ExecutionMetrics {
    let mut metrics = ExecutionMetrics::default();
    resources.shapes.composites.begin_render();
    resources.textures.target_uniforms.begin_render();
    let mut pending_command = commands.next();
    while let Some(command) = pending_command {
        match &command.operation {
            RenderOperation::BeginTarget(target) => {
                resources.begin_target(encoder, *target, &surface)
            }
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
                        .apply_effect(encoder, effect, mask);
                } else {
                    execute_effect(
                        encoder,
                        effect,
                        &resources.context.effects,
                        true,
                        resources.effects,
                        resources.textures,
                    );
                }
            }
            RenderOperation::Capture(capture) => resources.capture(encoder, *capture, &surface),
            RenderOperation::DrawShape(_)
            | RenderOperation::DrawShapeAndIncrementStencil(_)
            | RenderOperation::DecrementStencil(_)
            | RenderOperation::CompositeTexture(_) => {
                pending_command = execute_draw_pass(
                    encoder,
                    command,
                    &mut commands,
                    &surface,
                    &mut resources,
                    &mut metrics,
                );
                continue;
            }
        }
        pending_command = commands.next();
    }
    assert!(
        resources.textures.active_targets.is_empty(),
        "command stream must close every target"
    );
    resources.textures.finish_plan();
    metrics
}
