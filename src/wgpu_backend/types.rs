use super::execution::effects::EffectContext;
use crate::core::PhysicalRect;
use thiserror::Error;

/// Geometry cannot be addressed by an indexed draw command.
#[derive(Error, Debug)]
pub enum GeometryBufferError {
    #[error("Aggregated vertex offset exceeds the draw command limit")]
    VertexOffsetOverflow,
    #[error("Aggregated index range exceeds the draw command limit")]
    IndexRangeOverflow,
}

#[derive(Clone, Copy)]
pub(crate) enum BackdropSource<'a> {
    /// The source already contains every layer painted before the backdrop node.
    Flattened {
        texture: &'a wgpu::Texture,
        origin: [i32; 2],
    },
    /// Group rendering keeps the outside scene separate from its transparent subtree output.
    Layered {
        base_texture: &'a wgpu::Texture,
        base_origin: [i32; 2],
        foreground_view: &'a wgpu::TextureView,
        foreground_bounds: PhysicalRect,
    },
}

impl<'a> BackdropSource<'a> {
    pub(crate) fn base(self) -> (&'a wgpu::Texture, PhysicalRect) {
        let (texture, origin) = match self {
            Self::Flattened { texture, origin } => (texture, origin),
            Self::Layered {
                base_texture,
                base_origin,
                ..
            } => (base_texture, base_origin),
        };
        let bounds = PhysicalRect::new(
            origin.into(),
            (
                origin[0] + texture.width() as i32,
                origin[1] + texture.height() as i32,
            )
                .into(),
        );
        (texture, bounds)
    }

    pub(crate) fn foreground(self) -> Option<(&'a wgpu::TextureView, PhysicalRect)> {
        match self {
            Self::Flattened { .. } => None,
            Self::Layered {
                foreground_view,
                foreground_bounds,
                ..
            } => Some((foreground_view, foreground_bounds)),
        }
    }
}

/// Backdrop-specific rendering resources. Only needed when backdrop effects exist.
/// Callers pass shared pipelines, buffers, and textures separately.
pub(crate) struct BackdropContext<'a> {
    pub(crate) effects: EffectContext<'a>,
    pub(crate) texture_blit_pipeline: &'a wgpu::RenderPipeline,
    pub(crate) backdrop_layer_composite_pipeline: &'a wgpu::RenderPipeline,
    pub(crate) backdrop_layer_composite_bind_group_layout: &'a wgpu::BindGroupLayout,
}
