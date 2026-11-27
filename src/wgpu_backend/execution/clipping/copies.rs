use super::{bindings::ClipBindings, ClipCapture};
use crate::core::PhysicalRect;
use crate::wgpu_backend::execution::effects::PooledTexture;
use crate::wgpu_backend::execution::instructions::ExecutionContext;
use crate::wgpu_backend::execution::shapes::coverage::CoverageAllocation;
use std::sync::Arc;
use wgpu::{
    Color, CommandEncoder, Extent3d, LoadOp, Operations, Origin3d, RenderPassColorAttachment,
    RenderPassDescriptor, StoreOp, TexelCopyTextureInfo, TextureAspect,
};

/// Preserves individual MSAA samples when saving a clip's background.
pub(super) fn copy_colors(
    encoder: &mut CommandEncoder,
    context: &ExecutionContext<'_>,
    bindings: &mut ClipBindings,
    source: &ClipCapture,
    destination: &PooledTexture,
    bounds: PhysicalRect,
    coverage: &Arc<CoverageAllocation>,
) {
    let copy = source.bounds.intersection(&bounds);
    if context.sample_count == 1 && copy == Some(bounds) {
        let origin = bounds.min - source.bounds.min;
        encoder.copy_texture_to_texture(
            TexelCopyTextureInfo {
                texture: &source.texture,
                mip_level: 0,
                origin: Origin3d {
                    x: origin.x as u32,
                    y: origin.y as u32,
                    z: 0,
                },
                aspect: TextureAspect::All,
            },
            TexelCopyTextureInfo {
                texture: &destination.color_texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            Extent3d {
                width: bounds.width() as u32,
                height: bounds.height() as u32,
                depth_or_array_layers: 1,
            },
        );
        return;
    }
    let offset = bounds.min - source.bounds.min;
    let parameters = bindings.coverage(
        context.device,
        context.queue,
        &context.pipelines.shapes,
        coverage,
        [[0.0; 4], [0.0, 0.0, offset.x as f32, offset.y as f32]],
    );
    let images = bindings.images(
        context.device,
        context.clip_pipelines(),
        [&source.samples, &source.samples, &source.samples],
    );
    let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
        label: Some("clip_copy_colors"),
        color_attachments: &[Some(RenderPassColorAttachment {
            depth_slice: None,
            view: &destination.color_view,
            resolve_target: destination.resolve_view.as_ref(),
            ops: Operations {
                load: LoadOp::Clear(Color::TRANSPARENT),
                store: StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    pass.set_pipeline(&context.clip_pipelines().copy);
    pass.set_bind_group(0, &parameters, &[]);
    pass.set_bind_group(1, &images, &[]);
    pass.draw(0..3, 0..1);
}
