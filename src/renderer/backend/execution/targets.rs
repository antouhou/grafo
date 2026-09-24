use super::effects::PooledTexture;
use crate::UnsignedPhysicalRect;
use wgpu::{
    Color, CommandEncoder, LoadOp, Operations, RenderPass, RenderPassColorAttachment,
    RenderPassDepthStencilAttachment, RenderPassDescriptor, StoreOp, TextureView,
};

pub(in crate::renderer) fn set_scissor(
    render_pass: &mut RenderPass<'_>,
    scissor: UnsignedPhysicalRect,
) {
    render_pass.set_scissor_rect(
        scissor.min.x,
        scissor.min.y,
        scissor.width(),
        scissor.height(),
    );
}

/// Borrows one target's attachments across passes split by backdrop captures.
pub(in crate::renderer) struct RenderTarget<'a> {
    color_view: &'a TextureView,
    resolve_target: Option<&'a TextureView>,
    depth_stencil_view: &'a TextureView,
    needs_clear: bool,
}

impl<'a> RenderTarget<'a> {
    pub(in crate::renderer) fn resolved_view(&self) -> &'a TextureView {
        self.resolve_target.unwrap_or(self.color_view)
    }

    pub(in crate::renderer) fn for_output(
        output_view: &'a TextureView,
        multisample_view: Option<&'a TextureView>,
        depth_stencil_view: &'a TextureView,
    ) -> Self {
        Self {
            color_view: multisample_view.unwrap_or(output_view),
            resolve_target: multisample_view.map(|_| output_view),
            depth_stencil_view,
            needs_clear: true,
        }
    }

    pub(in crate::renderer) fn for_texture(texture: &'a PooledTexture) -> Self {
        Self {
            color_view: &texture.color_view,
            resolve_target: texture.resolve_view.as_ref(),
            depth_stencil_view: texture
                .depth_stencil_view
                .as_ref()
                .expect("scene targets include depth/stencil"),
            needs_clear: true,
        }
    }

    /// Initializes a capture source even when no draws precede the first capture.
    pub(in crate::renderer) fn clear_if_needed(&mut self, encoder: &mut CommandEncoder) {
        if self.needs_clear {
            self.begin_pass(encoder, "backdrop_precapture_pass");
        }
    }

    /// Clears once, then preserves attachments. Each draw supplies its own scissor.
    pub(in crate::renderer) fn begin_pass<'pass>(
        &mut self,
        encoder: &'pass mut CommandEncoder,
        label: &'pass str,
    ) -> RenderPass<'pass> {
        let render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some(label),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: self.color_view,
                resolve_target: self.resolve_target,
                ops: Operations {
                    load: if self.needs_clear {
                        LoadOp::Clear(Color::TRANSPARENT)
                    } else {
                        LoadOp::Load
                    },
                    // MSAA color must survive the resolve for subsequent load passes.
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                view: self.depth_stencil_view,
                depth_ops: Some(Operations {
                    load: if self.needs_clear {
                        LoadOp::Clear(1.0)
                    } else {
                        LoadOp::Load
                    },
                    store: StoreOp::Store,
                }),
                stencil_ops: Some(Operations {
                    load: if self.needs_clear {
                        LoadOp::Clear(0)
                    } else {
                        LoadOp::Load
                    },
                    store: StoreOp::Store,
                }),
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        self.needs_clear = false;
        render_pass
    }
}
