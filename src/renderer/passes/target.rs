use crate::pipeline::{begin_render_pass_with_load_ops, RenderPassLoadOperations};
use crate::renderer::types::{BackdropContext, BackdropSource};
use wgpu::{Color, CommandEncoder, LoadOp, RenderPass, TextureView};

/// Backdrop sources and resources are always supplied together.
pub(in crate::renderer) struct BackdropInputs<'a> {
    pub(in crate::renderer) source: BackdropSource<'a>,
    pub(in crate::renderer) context: &'a BackdropContext<'a>,
}

/// Attachments and backdrop inputs for one traversal's output.
pub(in crate::renderer) struct SegmentRenderTarget<'a> {
    pub(in crate::renderer) color_view: &'a TextureView,
    pub(in crate::renderer) color_resolve_target: Option<&'a TextureView>,
    pub(in crate::renderer) depth_stencil_view: &'a TextureView,
    pub(in crate::renderer) backdrop: Option<BackdropInputs<'a>>,
}

impl<'target> SegmentRenderTarget<'target> {
    pub(super) fn begin_pass<'encoder>(
        &self,
        encoder: &'encoder mut CommandEncoder,
        label: &'static str,
        should_clear: bool,
    ) -> RenderPass<'encoder>
    where
        'target: 'encoder,
    {
        begin_render_pass_with_load_ops(
            encoder,
            Some(label),
            self.color_view,
            self.color_resolve_target,
            self.depth_stencil_view,
            RenderPassLoadOperations {
                color_load_op: if should_clear {
                    LoadOp::Clear(Color::TRANSPARENT)
                } else {
                    LoadOp::Load
                },
                depth_load_op: if should_clear {
                    LoadOp::Clear(1.0)
                } else {
                    LoadOp::Load
                },
                stencil_load_op: if should_clear {
                    LoadOp::Clear(0)
                } else {
                    LoadOp::Load
                },
            },
        )
    }
}
