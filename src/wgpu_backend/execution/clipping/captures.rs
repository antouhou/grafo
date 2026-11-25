use super::ClipCapture;
use crate::wgpu_backend::execution::instructions::ExecutionResources;
use crate::wgpu_backend::execution::render_pass::RenderPass;
use crate::wgpu_backend::execution::targets::{RenderTarget, SurfaceTarget};
use wgpu::CommandEncoder;

impl ExecutionResources<'_> {
    /// Updates the backdrop from the current clip without walking its ancestors.
    pub(in crate::wgpu_backend::execution) fn capture_clips(
        &mut self,
        encoder: &mut CommandEncoder,
        surface: &SurfaceTarget<'_>,
    ) -> Option<ClipCapture> {
        let target = self
            .textures
            .active_targets
            .last_mut()
            .expect("capture requires a target");
        let index = target.active_layer?;
        let layer = target.clips[index]
            .layer
            .as_ref()
            .expect("active clip owns a layer");
        let layer = layer.backdrop.as_ref()?;
        let bounds = target.base_bounds(surface);
        if target.clip_backdrop.is_none() {
            let size = bounds.size().to_u32();
            target.clip_backdrop = Some(self.textures.pool.acquire_with_depth(
                self.context.device,
                size.width,
                size.height,
                self.context.format,
                self.context.sample_count,
            ));
            target.backdrop_needs_base.set(true);
        }
        let preview = target
            .clip_backdrop
            .as_ref()
            .expect("clip preview was acquired");
        if target.backdrop_needs_base.replace(false) {
            let source = target.base_attachments(surface).color_view();
            let images = self.textures.clip_bindings.images(
                self.context.device,
                self.context.clip_pipelines(),
                [source, source, source],
            );
            let mut attachments = RenderTarget::for_texture(preview);
            let mut pass = attachments.begin_pass(encoder, "clip_backdrop_initialize");
            pass.set_bind_group(0, &layer.capture_binding, &[]);
            pass.set_bind_group(1, &images, &[]);
            pass.set_pipeline(&self.context.clip_pipelines().initialize);
            pass.draw(0..3, 0..1);
            layer.is_capture_dirty.set(true);
        }
        if layer.is_capture_dirty.replace(false) {
            let mut attachments = RenderTarget::for_texture(preview);
            attachments.needs_clear = false;
            let mut pass = RenderPass::new(attachments.begin_pass(encoder, "clip_backdrop"));
            pass.set_bind_group(0, &layer.capture_binding, &[]);
            pass.set_bind_group(1, &layer.capture_images, &[]);
            layer.capture_coverage.buffers().bind(&mut pass);
            layer.capture_coverage.draw_with_pipeline(
                &mut pass,
                &self.context.clip_pipelines().capture,
                0..1,
            );
        }
        Some(ClipCapture {
            texture: preview
                .resolve_texture
                .as_ref()
                .unwrap_or(&preview.color_texture)
                .clone(),
            view: preview
                .resolve_view
                .as_ref()
                .unwrap_or(&preview.color_view)
                .clone(),
            samples: preview.color_view.clone(),
            bounds,
        })
    }
}
