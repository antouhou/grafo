use super::clipping::{self, ClipScope};
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
    ClipDraw, ClipGeometry, DrawClip, IntermediateTextureId, RenderCommand, RenderOperation,
    Target, TexturePlacement,
};
use crate::core::{BlendMode, Viewport};
#[cfg(feature = "render_metrics")]
use crate::wgpu_backend::metrics::{PipelineSwitchCounts, ShapeMaskEffectCacheMetrics};
use crate::wgpu_backend::resources::{Buffers, RendererPipelineResources};
use crate::wgpu_backend::types::BackdropContext;
use std::{cell::Cell, mem};
use wgpu::{CommandEncoder, Device, Queue, TextureFormat};

pub(in crate::wgpu_backend) struct ExecutionContext<'a> {
    pub viewport: Viewport,
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
        let inherited_coverage = self.inherited_clip_coverage(target);
        let texture = match target {
            Target::Surface => {
                assert!(
                    self.textures.active_targets.is_empty(),
                    "the surface is the root target"
                );
                None
            }
            Target::Texture {
                texture, bounds, ..
            } => {
                self.textures.reserve_planned(texture);
                let size = bounds.size().to_u32();
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
            if let Target::Texture { bounds, .. } = target {
                uniforms.canvas_size = bounds
                    .size()
                    .to_f32()
                    .to_array()
                    .map(|value| value / uniforms.scale_factor);
                uniforms.target_origin = bounds.min.to_f32().to_array();
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
        let clips = self.textures.clip_scopes.pop().unwrap_or_default();
        self.textures.active_targets.push(ActiveTarget {
            uses_physical_effect_input: matches!(
                target,
                Target::Texture {
                    is_content_effect_input: true,
                    ..
                }
            ),
            stencil_mask: None,
            inherited_coverage,
            clip_backdrop: None,
            backdrop_needs_base: Cell::new(true),
            clips,
            active_layer: None,
            stencil_reference: 0,
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
            self.textures.clip_scopes.push(target.clips);
            return;
        }
        target.attachments(surface).clear_if_needed(encoder);
        assert!(target.clips.is_empty(), "target must close its clip scopes");
        self.textures.clip_scopes.push(target.clips);
        if let Some(preview) = target.clip_backdrop {
            self.textures.work_textures.push(preview);
        }
        if let Target::Texture { texture, .. } = target.target {
            let mut output = target.texture.expect("target owns its texture");
            let context = &self.context.effects;
            output.composite_bind_group(context.device, context.composite_layout, context.sampler);
            self.textures.insert_planned(texture, output);
        }
        let uniform_index = match self.textures.active_targets.last() {
            Some(target) if target.active_layer.is_some() => target.current_uniform_index(),
            _ => self
                .textures
                .active_targets
                .last()
                .and_then(|target| target.uniform_index),
        };
        if let Some(index) = uniform_index {
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
        RenderOperation::DrawShape(draw) => draw.material.under_fill_texture.is_some(),
        RenderOperation::CompositeTexture(_) => false,
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
    pass.prepared_coverage = None;
    if pass.paints_isolated_clip {
        if let RenderOperation::DrawShapeAndIncrementStencil(ClipDraw {
            geometry: ClipGeometry::Shape(draw),
            ..
        }) = command.operation
        {
            let resources = shapes.draw_resources(draw.id);
            if resources.path_coverage.is_some() {
                pass.flush_leaf_batch(pending_batch);
                pass.set_scissor(command.clip.scissor);
                pass.draw_isolated_clip_paint(pass.stencil_reference, draw.material, resources);
                return;
            }
        }
    }
    if let Some(parent) = &pass.clip_coverage {
        if let RenderOperation::DrawShape(draw)
        | RenderOperation::DrawShapeAndIncrementStencil(ClipDraw {
            geometry: ClipGeometry::Shape(draw),
            ..
        }) = command.operation
        {
            pass.prepared_coverage = clipping::prepare_draw(
                shapes,
                context,
                draw.id,
                parent,
                command.clip.scissor,
                pass.clips_draws_directly,
            );
        }
    }
    let effective_clip = DrawClip {
        scissor: command.clip.scissor,
        stencil_reference: pass.stencil_reference,
    };
    if let RenderOperation::DrawShape(draw) = command.operation {
        pass.queue_leaf_draw(draw, effective_clip, shapes, pending_batch);
        return;
    }
    if let RenderOperation::DrawShapeAndIncrementStencil(ClipDraw {
        geometry: ClipGeometry::Shape(draw),
        ..
    }) = command.operation
    {
        if shapes.draw_resources(draw.id).path_coverage.is_some() {
            pass.queue_leaf_draw(draw, effective_clip, shapes, pending_batch);
            return;
        }
    }
    pass.flush_leaf_batch(pending_batch);
    match command.operation {
        RenderOperation::DrawShapeAndIncrementStencil(ClipDraw {
            geometry: ClipGeometry::Shape(draw),
            ..
        }) => {
            pass.set_scissor(effective_clip.scissor);
            pass.draw_shape_and_increment_stencil(
                effective_clip.stencil_reference - 1,
                draw.material,
                shapes.draw_resources(draw.id),
            );
        }
        RenderOperation::DecrementStencil(ClipDraw {
            geometry: ClipGeometry::Shape(draw),
            ..
        }) => {
            pass.set_scissor(effective_clip.scissor);
            pass.decrement_stencil(
                effective_clip.stencil_reference + 1,
                shapes.draw_resources(draw.id),
            );
        }
        RenderOperation::CompositeTexture(composite)
            if composite.blend_mode != BlendMode::SourceOver =>
        {
            pass.composite_blended_texture(
                composite,
                effective_clip,
                &mut shapes.composites,
                context.device,
                context.queue,
            );
        }
        RenderOperation::CompositeTexture(composite) => match composite.placement {
            TexturePlacement::Target => {
                pass.set_scissor(effective_clip.scissor);
                pass.composite_texture(effective_clip.stencil_reference, composite.source);
            }
            TexturePlacement::Local { .. } => {
                pass.composite_local_texture(
                    composite,
                    effective_clip,
                    &mut shapes.composites,
                    context.device,
                    context.queue,
                );
            }
        },
        _ => unreachable!("draw passes execute only draw commands"),
    }
}

fn update_direct_clip(
    command: &RenderCommand<'_>,
    pass: &mut DrawPass<'_, '_>,
    shapes: &mut ShapeExecutionResources,
    context: &ExecutionContext<'_>,
    scopes: &mut Vec<ClipScope>,
) -> bool {
    match command.operation {
        RenderOperation::DrawShapeAndIncrementStencil(draw)
            if !draw.requires_destination && !draw.requires_layer =>
        {
            if matches!(draw.geometry, ClipGeometry::Shape(shape) if shape.material.under_fill_texture.is_some())
            {
                return false;
            }
            let Some(coverage) = clipping::prepare_clip(
                shapes,
                context,
                draw.geometry,
                command.clip.scissor,
                pass.clip_coverage.as_ref(),
            ) else {
                return false;
            };
            scopes.push(ClipScope {
                uses_coverage: true,
                coverage: Some(coverage.clone()),
                layer: None,
                previous_stencil: pass.stencil_reference,
                previous_layer: None,
            });
            pass.clip_coverage = Some(coverage);
            true
        }
        RenderOperation::DecrementStencil(_)
            if scopes.last().is_some_and(|scope| scope.uses_coverage) =>
        {
            scopes.pop();
            pass.clip_coverage = scopes.last().and_then(|scope| scope.coverage.clone());
            true
        }
        _ => false,
    }
}

fn execute_pass_commands<'scene>(
    pending_command: RenderCommand<'scene>,
    commands: &mut impl Iterator<Item = RenderCommand<'scene>>,
    pass: &mut DrawPass<'_, '_>,
    shapes: &mut ShapeExecutionResources,
    context: &ExecutionContext<'_>,
    direct_scopes: &mut Vec<ClipScope>,
) -> Option<RenderCommand<'scene>> {
    let mut pending_batch = None;
    execute_draw_command(pass, pending_command, shapes, context, &mut pending_batch);
    let mut next_command = None;
    for command in commands {
        if pass.clips_draws_directly
            && matches!(
                command.operation,
                RenderOperation::DrawShapeAndIncrementStencil(_)
                    | RenderOperation::DecrementStencil(_)
            )
        {
            pass.flush_leaf_batch(&mut pending_batch);
            if update_direct_clip(&command, pass, shapes, context, direct_scopes) {
                if matches!(
                    command.operation,
                    RenderOperation::DrawShapeAndIncrementStencil(ClipDraw {
                        geometry: ClipGeometry::Shape(draw),
                        ..
                    }) if !clipping::is_empty_clip_paint(shapes, draw)
                ) {
                    execute_draw_command(pass, command, shapes, context, &mut pending_batch);
                }
                continue;
            }
        }
        if should_end_draw_pass(&command) {
            next_command = Some(command);
            break;
        }
        execute_draw_command(pass, command, shapes, context, &mut pending_batch);
    }
    pass.flush_leaf_batch(&mut pending_batch);
    next_command
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
    if let RenderOperation::DrawShape(draw)
    | RenderOperation::DrawShapeAndIncrementStencil(ClipDraw {
        geometry: ClipGeometry::Shape(draw),
        ..
    }) = &pending_command.operation
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
    let clip_coverage = target.current_coverage().cloned();
    let clips_draws_directly = target.active_layer.is_none() && target.inherited_coverage.is_none();
    let mut direct_scopes = if clips_draws_directly {
        mem::take(&mut target.clips)
    } else {
        Vec::new()
    };
    let paints_isolated_clip = matches!(
        pending_command.operation,
        RenderOperation::DrawShapeAndIncrementStencil(_)
    ) && target
        .clips
        .last()
        .and_then(|scope| scope.layer.as_ref())
        .is_some_and(|layer| layer.backdrop.is_none());
    let stencil_reference = target.stencil_reference;
    if target.active_layer.is_some() || target.clip_backdrop.is_some() {
        target.invalidate_clip_capture();
    }
    let mut attachments = target.attachments(surface);
    let root_scissor = (matches!(target.target, Target::Surface) && target.active_layer.is_none())
        .then_some(surface.root_scissor);
    let target_clip = if target.active_layer.is_some() {
        let draw_scissor = match target.target {
            Target::Texture { draw_scissor, .. } => draw_scissor,
            _ => surface.root_scissor,
        };
        Some((target.bounds(surface), draw_scissor))
    } else {
        match target.target {
            Target::Texture {
                bounds,
                draw_scissor,
                ..
            } => Some((bounds, draw_scissor)),
            _ => None,
        }
    };
    let mut render_pass = RenderPass::new(attachments.begin_pass(encoder, "command_draws"));
    target.mark_initialized();
    let mut pass = DrawPass {
        paints_isolated_clip,
        clips_draws_directly,
        clip_coverage,
        prepared_coverage: None,
        stencil_reference,
        root_scissor,
        target_clip,
        render_pass: &mut render_pass,
        pipelines: resources.context.pipelines,
        buffers: resources.buffers,
        textures: resources.textures,
    };
    let next = execute_pass_commands(
        pending_command,
        commands,
        &mut pass,
        resources.shapes,
        resources.context,
        &mut direct_scopes,
    );
    metrics.accumulate(pass.render_pass);
    drop(render_pass);
    if clips_draws_directly {
        resources
            .textures
            .active_targets
            .last_mut()
            .expect("draw requires an open target")
            .clips = direct_scopes;
    }
    next
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
    resources.textures.clip_bindings.begin_render();
    let mut pending_command = commands.next();
    while let Some(command) = pending_command {
        if matches!(
            &command.operation,
            RenderOperation::DrawShapeAndIncrementStencil(_) | RenderOperation::DecrementStencil(_)
        ) && resources.execute_clip_operation(encoder, &command, &surface)
        {
            pending_command = commands.next();
            continue;
        }
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
                target.mark_initialized();
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
                    resources.finish_clip_backdrop_filter(encoder, effect);
                }
            }
            RenderOperation::Capture(capture) => {
                let next_command = commands.next();
                resources.capture_before(encoder, *capture, &surface, next_command.as_ref());
                pending_command = next_command;
                continue;
            }
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
