use super::ClipStencilMask;
use crate::core::PhysicalRect;
use crate::wgpu_backend::execution::effects::PooledTexture;
use crate::wgpu_backend::execution::instructions::ExecutionResources;
use crate::wgpu_backend::execution::shapes::coverage::CoverageAllocation;
use crate::wgpu_backend::execution::targets::{RenderTarget, SurfaceTarget};
use std::sync::Arc;
use wgpu::{
    Color, CommandEncoder, LoadOp, Operations, RenderPassColorAttachment,
    RenderPassDepthStencilAttachment, RenderPassDescriptor, StoreOp,
};

impl ExecutionResources<'_> {
    /// Saves a fallback clip's sample mask once for all nested coverage clips.
    pub(super) fn clip_stencil_mask(
        &mut self,
        encoder: &mut CommandEncoder,
        surface: &SurfaceTarget<'_>,
        coverage: &Arc<CoverageAllocation>,
    ) -> Option<ClipStencilMask> {
        let target = self
            .textures
            .active_targets
            .last_mut()
            .expect("clip requires a target");
        if target.stencil_reference == 0 {
            return None;
        }
        if let Some(mask) = target.stencil_mask_mut() {
            return Some(mask.clone());
        }
        let bounds = target.bounds(surface);
        let size = bounds.size().to_u32();
        let mask = self.textures.pool.acquire_color_only(
            self.context.device,
            size.width,
            size.height,
            self.context.format,
            self.context.sample_count,
        );
        let parameters = self.textures.clip_bindings.coverage(
            self.context.device,
            self.context.queue,
            &self.context.pipelines.shapes,
            coverage,
            [[0.0; 4]; 2],
        );
        let source = target.attachments(surface).color_view();
        let images = self.textures.clip_bindings.images(
            self.context.device,
            self.context.clip_pipelines(),
            [source, source, source],
        );
        let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("clip_export_stencil"),
            color_attachments: &[Some(RenderPassColorAttachment {
                depth_slice: None,
                view: &mask.color_view,
                resolve_target: mask.resolve_view.as_ref(),
                ops: Operations {
                    load: LoadOp::Clear(Color::TRANSPARENT),
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                view: target.attachments(surface).depth_stencil_view(),
                depth_ops: None,
                stencil_ops: Some(Operations {
                    load: LoadOp::Load,
                    store: StoreOp::Store,
                }),
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.context.clip_pipelines().export_stencil);
        pass.set_bind_group(0, &parameters, &[]);
        pass.set_bind_group(1, &images, &[]);
        pass.set_stencil_reference(target.stencil_reference);
        pass.draw(0..3, 0..1);
        drop(pass);
        let saved = ClipStencilMask {
            view: mask.color_view.clone(),
            bounds,
        };
        *target.stencil_mask_mut() = Some(saved.clone());
        self.textures.work_textures.push(mask);
        Some(saved)
    }

    pub(super) fn initialize_clip_stencil(
        &mut self,
        encoder: &mut CommandEncoder,
        mask: &ClipStencilMask,
        working: &PooledTexture,
        bounds: PhysicalRect,
        coverage: &Arc<CoverageAllocation>,
        needs_clear: bool,
    ) {
        let offset = bounds.min - mask.bounds.min;
        let parameters = self.textures.clip_bindings.coverage(
            self.context.device,
            self.context.queue,
            &self.context.pipelines.shapes,
            coverage,
            [[0.0; 4], [0.0, 0.0, offset.x as f32, offset.y as f32]],
        );
        let images = self.textures.clip_bindings.images(
            self.context.device,
            self.context.clip_pipelines(),
            [&mask.view, &mask.view, &mask.view],
        );
        let mut attachments = RenderTarget::for_texture(working);
        attachments.needs_clear = needs_clear;
        let mut pass = attachments.begin_pass(encoder, "clip_import_stencil");
        pass.set_pipeline(&self.context.clip_pipelines().import_stencil);
        pass.set_bind_group(0, &parameters, &[]);
        pass.set_bind_group(1, &images, &[]);
        pass.set_stencil_reference(1);
        pass.draw(0..3, 0..1);
    }
}
