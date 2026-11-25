use super::{copies, ClipCapture, ClipLayer, ClipLayerBackdrop, ClipScope};
use crate::commands::{
    ClipDraw, ClipGeometry, DrawClip, RenderCommand, RenderOperation, ShapeDraw,
};
use crate::core::{PhysicalRect, UnsignedPhysicalRect};
use crate::wgpu_backend::execution::instructions::{ExecutionContext, ExecutionResources};
use crate::wgpu_backend::execution::render_pass::RenderPass;
use crate::wgpu_backend::execution::shapes::coverage::clipping::ClipCoverage;
use crate::wgpu_backend::execution::shapes::ShapeExecutionResources;
use crate::wgpu_backend::execution::target_uniforms;
use crate::wgpu_backend::execution::targets::{RenderTarget, SurfaceTarget};
use std::{cell::Cell, sync::Arc};
use wgpu::CommandEncoder;

fn clip_texture_bounds(parent: PhysicalRect, clip: PhysicalRect) -> PhysicalRect {
    let Some(bounds) = parent
        .intersection(&clip)
        .filter(|bounds| !bounds.is_empty())
    else {
        return PhysicalRect::new(parent.min, [parent.min.x + 1, parent.min.y + 1].into());
    };
    // Animated fractional bounds share textures within a fixed size bucket.
    let width = ((bounds.width() as u32).div_ceil(64) * 64).min(parent.width() as u32) as i32;
    let height = ((bounds.height() as u32).div_ceil(64) * 64).min(parent.height() as u32) as i32;
    let origin = [
        bounds.min.x.min(parent.max.x - width),
        bounds.min.y.min(parent.max.y - height),
    ];
    PhysicalRect::new(
        origin.into(),
        [origin[0] + width, origin[1] + height].into(),
    )
}

fn clip_source(
    shapes: &mut ShapeExecutionResources,
    context: &ExecutionContext<'_>,
    draw: ClipGeometry,
) -> Option<Arc<ClipCoverage>> {
    let viewport = context.viewport;
    if let ClipGeometry::Rectangle(bounds) = draw {
        return Some(shapes.clip_coverage().rectangle(
            [
                f64::from(bounds.min.x) * viewport.scale_factor,
                f64::from(bounds.min.y) * viewport.scale_factor,
                f64::from(bounds.max.x) * viewport.scale_factor,
                f64::from(bounds.max.y) * viewport.scale_factor,
            ],
            UnsignedPhysicalRect::from_size(viewport.physical_size.into()),
        ));
    }
    let ClipGeometry::Shape(draw) = draw else {
        unreachable!("rectangle clips were prepared above")
    };
    shapes.clip_coverage();
    let coverage = shapes.draws[&draw.id.0].path_coverage.as_ref()?;
    Some(
        shapes
            .clip_coverage
            .as_deref_mut()
            .expect("clip cache was initialized")
            .shape(coverage, viewport),
    )
}

pub(in crate::wgpu_backend::execution) fn prepare_clip(
    shapes: &mut ShapeExecutionResources,
    context: &ExecutionContext<'_>,
    geometry: ClipGeometry,
    scissor: UnsignedPhysicalRect,
    parent: Option<&Arc<ClipCoverage>>,
) -> Option<Arc<ClipCoverage>> {
    let source = clip_source(shapes, context, geometry)?;
    let scissor = shapes.clip_coverage().rectangle(
        [
            f64::from(scissor.min.x),
            f64::from(scissor.min.y),
            f64::from(scissor.max.x),
            f64::from(scissor.max.y),
        ],
        UnsignedPhysicalRect::from_size(context.viewport.physical_size.into()),
    );
    let source = shapes.clip_coverage().intersection(&source, &scissor);
    Some(match parent {
        Some(parent) => shapes.clip_coverage().intersection(parent, &source),
        None => source,
    })
}

pub(in crate::wgpu_backend::execution) fn is_empty_clip_paint(
    shapes: &ShapeExecutionResources,
    draw: ShapeDraw,
) -> bool {
    let material = draw.material;
    if material.has_gradient_fill || material.under_fill_texture.is_some() {
        return false;
    }
    for binding in &material.texture_bindings {
        if binding.is_present() {
            return false;
        }
    }
    let Some(location) = shapes.draws[&draw.id.0].geometry_buffer_location else {
        return false;
    };
    shapes.instance_colors[location.instance_index].color[3] == 0.0
}

impl ExecutionResources<'_> {
    pub(in crate::wgpu_backend::execution) fn execute_clip_operation(
        &mut self,
        encoder: &mut CommandEncoder,
        command: &RenderCommand<'_>,
        surface: &SurfaceTarget<'_>,
    ) -> bool {
        match &command.operation {
            RenderOperation::DrawShapeAndIncrementStencil(draw) => {
                if self.begin_clip(encoder, *draw, command.clip, surface) {
                    if match draw.geometry {
                        ClipGeometry::Rectangle(_) => true,
                        ClipGeometry::Shape(draw) => is_empty_clip_paint(self.shapes, draw),
                    } {
                        return true;
                    }
                } else {
                    let target = self
                        .textures
                        .active_targets
                        .last_mut()
                        .expect("clip requires a target");
                    *target.stencil_mask_mut() = None;
                    target.stencil_reference += 1;
                }
                false
            }
            RenderOperation::DecrementStencil(_) => self.end_clip(encoder, surface),
            _ => false,
        }
    }

    pub(in crate::wgpu_backend::execution) fn begin_clip(
        &mut self,
        encoder: &mut CommandEncoder,
        draw: ClipDraw,
        clip: DrawClip,
        surface: &SurfaceTarget<'_>,
    ) -> bool {
        let target = self
            .textures
            .active_targets
            .last()
            .expect("clip requires a target");
        let previous_stencil = target.stencil_reference;
        let previous_layer = target.active_layer;
        let base_bounds = target.base_bounds(surface);
        let parent = target.current_coverage().cloned();
        let inherited_coverage = target.inherited_coverage.clone();
        let combined = prepare_clip(
            self.shapes,
            self.context,
            draw.geometry,
            clip.scissor,
            parent.as_ref(),
        );
        let Some(combined) = combined else {
            let target = self
                .textures
                .active_targets
                .last_mut()
                .expect("clip requires a target");
            target.clips.push(ClipScope {
                uses_coverage: false,
                coverage: parent,
                layer: None,
                previous_stencil,
                previous_layer,
            });
            return false;
        };
        let can_clip_draws_directly = !draw.requires_destination
            && !draw.requires_layer
            && target.active_layer.is_none()
            && target.inherited_coverage.is_none();
        if can_clip_draws_directly
            || parent
                .as_ref()
                .is_some_and(|parent| Arc::ptr_eq(parent, &combined))
        {
            let target = self
                .textures
                .active_targets
                .last_mut()
                .expect("clip requires a target");
            target.clips.push(ClipScope {
                uses_coverage: true,
                coverage: Some(combined),
                layer: None,
                previous_stencil,
                previous_layer,
            });
            return true;
        }
        let closing = self.shapes.clip_coverage().prepare(
            &combined,
            parent.as_ref(),
            None,
            self.context.device,
            self.context.queue,
            &self.context.pipelines.shapes,
        );
        let parent_bounds = target.bounds(surface);
        let visible_prefix = if draw.requires_destination {
            self.capture_clips(encoder, surface)
        } else {
            None
        };
        let bounds = clip_texture_bounds(parent_bounds, combined.bounds);
        let size = bounds.size().to_u32();
        let working = self.textures.pool.acquire_with_depth(
            self.context.device,
            size.width,
            size.height,
            self.context.format,
            self.context.sample_count,
        );
        let parameters = [
            [
                parent_bounds.min.x as f32,
                parent_bounds.min.y as f32,
                parent_bounds.width() as f32,
                parent_bounds.height() as f32,
            ],
            [bounds.min.x as f32, bounds.min.y as f32, 0.0, 0.0],
        ];
        let closing_binding = self.textures.clip_bindings.coverage(
            self.context.device,
            self.context.queue,
            &self.context.pipelines.shapes,
            &closing,
            parameters,
        );
        let backdrop = if draw.requires_destination {
            let background = self.textures.pool.acquire_color_only(
                self.context.device,
                size.width,
                size.height,
                self.context.format,
                self.context.sample_count,
            );
            let backdrop_background = if let Some(source) = visible_prefix {
                let texture = self.textures.pool.acquire_color_only(
                    self.context.device,
                    size.width,
                    size.height,
                    self.context.format,
                    self.context.sample_count,
                );
                copies::copy_colors(
                    encoder,
                    self.context,
                    &mut self.textures.clip_bindings,
                    &source,
                    &texture,
                    bounds,
                    &closing,
                );
                Some(texture)
            } else {
                None
            };
            let target = self
                .textures
                .active_targets
                .last_mut()
                .expect("clip requires a target");
            target.attachments(surface).clear_if_needed(encoder);
            target.mark_initialized();
            let attachments = target.attachments(surface);
            let source = ClipCapture {
                texture: target.resolved_texture(surface),
                view: attachments.resolved_view().clone(),
                samples: attachments.color_view().clone(),
                bounds: parent_bounds,
            };
            copies::copy_colors(
                encoder,
                self.context,
                &mut self.textures.clip_bindings,
                &source,
                &background,
                bounds,
                &closing,
            );
            let initial_images = self.textures.clip_bindings.images(
                self.context.device,
                self.context.clip_pipelines(),
                [
                    &background.color_view,
                    &background.color_view,
                    &background.color_view,
                ],
            );
            {
                let mut target = RenderTarget::for_texture(&working);
                let mut pass = target.begin_pass(encoder, "clip_initialize");
                pass.set_pipeline(&self.context.clip_pipelines().initialize);
                pass.set_bind_group(0, &closing_binding, &[]);
                pass.set_bind_group(1, &initial_images, &[]);
                pass.draw(0..3, 0..1);
            }
            let capture_coverage = self.shapes.clip_coverage().prepare(
                &combined,
                inherited_coverage.as_ref(),
                None,
                self.context.device,
                self.context.queue,
                &self.context.pipelines.shapes,
            );
            let capture_binding = self.textures.clip_bindings.coverage(
                self.context.device,
                self.context.queue,
                &self.context.pipelines.shapes,
                &capture_coverage,
                [
                    [
                        base_bounds.min.x as f32,
                        base_bounds.min.y as f32,
                        base_bounds.width() as f32,
                        base_bounds.height() as f32,
                    ],
                    parameters[1],
                ],
            );
            let capture_images = self.textures.clip_bindings.images(
                self.context.device,
                self.context.clip_pipelines(),
                [
                    &background.color_view,
                    &working.color_view,
                    backdrop_background
                        .as_ref()
                        .map_or(&background.color_view, |texture| &texture.color_view),
                ],
            );
            Some(ClipLayerBackdrop {
                backdrop_background,
                capture_coverage,
                capture_binding,
                capture_images,
                is_capture_dirty: Cell::new(false),
                background,
            })
        } else {
            None
        };
        let mut needs_clear = backdrop.is_none();
        let stencil_mask = self.clip_stencil_mask(encoder, surface, &closing);
        if let Some(mask) = &stencil_mask {
            self.initialize_clip_stencil(encoder, mask, &working, bounds, &closing, needs_clear);
            needs_clear = false;
        }
        let background_view = backdrop.as_ref().map_or(&working.color_view, |backdrop| {
            &backdrop.background.color_view
        });
        let images = self.textures.clip_bindings.images(
            self.context.device,
            self.context.clip_pipelines(),
            [background_view, &working.color_view, background_view],
        );
        let mut uniforms = self.context.pipelines.shapes.and_uniforms;
        uniforms.target_origin = bounds.min.to_f32().to_array();
        uniforms.canvas_size = [
            size.width as f32 / uniforms.scale_factor,
            size.height as f32 / uniforms.scale_factor,
        ];
        let uniform_index = self.textures.target_uniforms.prepare(
            self.context.device,
            self.context.queue,
            uniforms,
        );
        target_uniforms::activate(
            encoder,
            self.textures.target_uniforms.buffer(uniform_index),
            &self.context.pipelines.shapes,
        );
        let target = self
            .textures
            .active_targets
            .last_mut()
            .expect("clip requires a target");
        target.active_layer = Some(target.clips.len());
        target.stencil_reference = u32::from(stencil_mask.is_some());
        target.clips.push(ClipScope {
            uses_coverage: true,
            coverage: Some(combined),
            layer: Some(ClipLayer {
                stencil_mask,
                backdrop,
                needs_clear,
                working,
                bounds,
                uniform_index,
                closing_coverage: closing,
                closing_binding,
                images,
            }),
            previous_stencil,
            previous_layer,
        });
        true
    }

    pub(in crate::wgpu_backend::execution) fn end_clip(
        &mut self,
        encoder: &mut CommandEncoder,
        surface: &SurfaceTarget<'_>,
    ) -> bool {
        let target = self
            .textures
            .active_targets
            .last_mut()
            .expect("clip requires a target");
        let scope = target.clips.pop().expect("clip scopes are balanced");
        target.active_layer = scope.previous_layer;
        target.stencil_reference = scope.previous_stencil;
        target.invalidate_clip_capture();
        let Some(layer) = scope.layer else {
            if !scope.uses_coverage {
                *target.stencil_mask_mut() = None;
            }
            return scope.uses_coverage;
        };
        if layer.needs_clear {
            RenderTarget::for_texture(&layer.working).clear_if_needed(encoder);
        }
        {
            let mut attachments = target.attachments(surface);
            let mut pass = RenderPass::new(attachments.begin_pass(encoder, "clip_close"));
            target.mark_initialized();
            pass.set_bind_group(0, &layer.closing_binding, &[]);
            pass.set_bind_group(1, &layer.images, &[]);
            pass.set_stencil_reference(scope.previous_stencil);
            layer.closing_coverage.buffers().bind(&mut pass);
            layer.closing_coverage.draw_with_pipeline(
                &mut pass,
                if layer.backdrop.is_some() {
                    &self.context.clip_pipelines().close
                } else {
                    &self.context.clip_pipelines().close_isolated
                },
                0..1,
            );
        }
        self.textures.work_textures.push(layer.working);
        if let Some(backdrop) = layer.backdrop {
            self.textures.work_textures.push(backdrop.background);
            if let Some(texture) = backdrop.backdrop_background {
                self.textures.work_textures.push(texture);
            }
        }
        if let Some(index) = target.current_uniform_index() {
            target_uniforms::activate(
                encoder,
                self.textures.target_uniforms.buffer(index),
                &self.context.pipelines.shapes,
            );
        }
        true
    }
}
