use super::clipping::{ClipScope, ClipStencilMask};
use super::effects::PooledTexture;
use super::shapes::coverage::clipping::ClipCoverage;
use crate::commands::Target;
use crate::{PhysicalRect, UnsignedPhysicalRect};
use std::{cell::Cell, sync::Arc};
use wgpu::{
    Color, CommandEncoder, LoadOp, Operations, RenderPass, RenderPassColorAttachment,
    RenderPassDepthStencilAttachment, RenderPassDescriptor, RenderPipeline, StoreOp, Texture,
    TextureView,
};

pub(in crate::wgpu_backend) fn set_scissor(
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

#[derive(Clone, Copy)]
pub(in crate::wgpu_backend) struct RegionClear<'a> {
    pub pipeline: &'a RenderPipeline,
    pub scissor: UnsignedPhysicalRect,
}

/// Borrows one target's attachments across passes split by backdrop captures.
pub(in crate::wgpu_backend) struct RenderTarget<'a> {
    color_view: &'a TextureView,
    resolve_target: Option<&'a TextureView>,
    depth_stencil_view: &'a TextureView,
    region_clear: Option<RegionClear<'a>>,
    pub(super) needs_clear: bool,
}

impl<'a> RenderTarget<'a> {
    pub(super) fn color_view(&self) -> &'a TextureView {
        self.color_view
    }

    pub(super) fn depth_stencil_view(&self) -> &'a TextureView {
        self.depth_stencil_view
    }

    pub(in crate::wgpu_backend) fn resolved_view(&self) -> &'a TextureView {
        self.resolve_target.unwrap_or(self.color_view)
    }

    pub(in crate::wgpu_backend) fn for_output(
        output_view: &'a TextureView,
        multisample_view: Option<&'a TextureView>,
        depth_stencil_view: &'a TextureView,
        region_clear: Option<RegionClear<'a>>,
    ) -> Self {
        Self {
            color_view: multisample_view.unwrap_or(output_view),
            resolve_target: multisample_view.map(|_| output_view),
            depth_stencil_view,
            region_clear,
            needs_clear: true,
        }
    }

    pub(in crate::wgpu_backend) fn for_texture(texture: &'a PooledTexture) -> Self {
        Self {
            color_view: &texture.color_view,
            resolve_target: texture.resolve_view.as_ref(),
            depth_stencil_view: texture
                .depth_stencil_view
                .as_ref()
                .expect("scene targets include depth/stencil"),
            region_clear: None,
            needs_clear: true,
        }
    }

    /// Initializes a capture source even when no draws precede the first capture.
    pub(in crate::wgpu_backend) fn clear_if_needed(&mut self, encoder: &mut CommandEncoder) {
        if self.needs_clear {
            self.begin_pass(encoder, "backdrop_precapture_pass");
        }
    }

    /// Clears once, then preserves attachments. Each draw supplies its own scissor.
    pub(in crate::wgpu_backend) fn begin_pass<'pass>(
        &mut self,
        encoder: &'pass mut CommandEncoder,
        label: &'pass str,
    ) -> RenderPass<'pass> {
        let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some(label),
            color_attachments: &[Some(RenderPassColorAttachment {
                depth_slice: None,
                view: self.color_view,
                resolve_target: self.resolve_target,
                ops: Operations {
                    load: if self.needs_clear && self.region_clear.is_none() {
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
            multiview_mask: None,
        });
        // Keep the damage clear and the first scene draws in the same pass.
        if let Some(clear) = self.region_clear.filter(|_| self.needs_clear) {
            set_scissor(&mut render_pass, clear.scissor);
            render_pass.set_pipeline(clear.pipeline);
            render_pass.draw(0..3, 0..1);
        }
        self.needs_clear = false;
        render_pass
    }
}

/// A suspended target owns its attachments until its matching EndTarget.
pub(super) struct ActiveTarget {
    pub(super) uses_physical_effect_input: bool,
    pub(super) stencil_mask: Option<ClipStencilMask>,
    pub(super) inherited_coverage: Option<Arc<ClipCoverage>>,
    pub(super) clip_backdrop: Option<PooledTexture>,
    pub(super) backdrop_needs_base: Cell<bool>,
    pub(super) clips: Vec<ClipScope>,
    pub(super) active_layer: Option<usize>,
    pub(super) stencil_reference: u32,
    pub(super) target: Target,
    pub(super) texture: Option<PooledTexture>,
    pub(super) needs_clear: bool,
    pub(super) uniform_index: Option<usize>,
}

pub(in crate::wgpu_backend) struct SurfaceTarget<'a> {
    pub(in crate::wgpu_backend) root_scissor: UnsignedPhysicalRect,
    pub(in crate::wgpu_backend) output: RenderTarget<'a>,
    pub(in crate::wgpu_backend) capture_texture: Option<&'a Texture>,
}

impl ActiveTarget {
    pub(super) fn mark_initialized(&mut self) {
        if let Some(index) = self.active_layer {
            self.clips[index]
                .layer
                .as_mut()
                .expect("active clip owns a layer")
                .needs_clear = false;
        } else {
            self.needs_clear = false;
        }
    }

    pub(super) fn stencil_mask_mut(&mut self) -> &mut Option<ClipStencilMask> {
        if let Some(index) = self.active_layer {
            &mut self.clips[index]
                .layer
                .as_mut()
                .expect("active clip owns a layer")
                .stencil_mask
        } else {
            &mut self.stencil_mask
        }
    }

    pub(super) fn current_coverage(&self) -> Option<&Arc<ClipCoverage>> {
        if let Some(scope) = self.clips.last() {
            if let Some(coverage) = &scope.coverage {
                return Some(coverage);
            }
        }
        self.inherited_coverage.as_ref()
    }
    pub(super) fn attachments<'a>(&'a self, surface: &'a SurfaceTarget<'_>) -> RenderTarget<'a> {
        if let Some(layer) = self.active_layer {
            let layer = self.clips[layer]
                .layer
                .as_ref()
                .expect("active clip owns a layer");
            let mut target = RenderTarget::for_texture(&layer.working);
            target.needs_clear = layer.needs_clear;
            return target;
        }
        self.base_attachments(surface)
    }

    pub(super) fn base_attachments<'a>(
        &'a self,
        surface: &'a SurfaceTarget<'_>,
    ) -> RenderTarget<'a> {
        let mut target = match self.target {
            Target::Surface => RenderTarget {
                color_view: surface.output.color_view,
                resolve_target: surface.output.resolve_target,
                depth_stencil_view: surface.output.depth_stencil_view,
                region_clear: surface.output.region_clear,
                needs_clear: self.needs_clear,
            },
            Target::Texture { .. } => {
                RenderTarget::for_texture(self.texture.as_ref().expect("target owns its texture"))
            }
            Target::Mask(_) => unreachable!("mask draws use the mask pipeline"),
        };
        target.needs_clear = self.needs_clear;
        target
    }

    pub(super) fn bounds(&self, surface: &SurfaceTarget<'_>) -> PhysicalRect {
        if let Some(layer) = self.active_layer {
            return self.clips[layer]
                .layer
                .as_ref()
                .expect("active clip owns a layer")
                .bounds;
        }
        self.base_bounds(surface)
    }

    pub(super) fn base_bounds(&self, surface: &SurfaceTarget<'_>) -> PhysicalRect {
        match self.target {
            Target::Texture { bounds, .. } => bounds,
            _ => {
                let texture = surface
                    .capture_texture
                    .expect("scene target has a resolved texture");
                PhysicalRect::from_size((texture.width() as i32, texture.height() as i32).into())
            }
        }
    }

    pub(super) fn invalidate_clip_capture(&self) {
        if let Some(index) = self.active_layer {
            if let Some(backdrop) = &self.clips[index]
                .layer
                .as_ref()
                .expect("active clip owns a layer")
                .backdrop
            {
                backdrop.is_capture_dirty.set(true);
            }
        } else {
            self.backdrop_needs_base.set(true);
        }
    }

    pub(super) fn current_uniform_index(&self) -> Option<usize> {
        if let Some(index) = self.active_layer {
            Some(
                self.clips[index]
                    .layer
                    .as_ref()
                    .expect("active clip owns a layer")
                    .uniform_index,
            )
        } else {
            self.uniform_index
        }
    }

    pub(super) fn resolved_texture(&self, surface: &SurfaceTarget<'_>) -> Texture {
        let texture = if let Some(index) = self.active_layer {
            Some(
                &self.clips[index]
                    .layer
                    .as_ref()
                    .expect("active clip owns a layer")
                    .working,
            )
        } else {
            self.texture.as_ref()
        };
        match texture {
            Some(texture) => texture
                .resolve_texture
                .as_ref()
                .unwrap_or(&texture.color_texture)
                .clone(),
            None => surface
                .capture_texture
                .expect("scene target has a resolved texture")
                .clone(),
        }
    }
}
