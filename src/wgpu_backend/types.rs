use super::execution::effects::EffectContext;
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
    Flattened { texture: &'a wgpu::Texture },
    /// Group rendering keeps the outside scene separate from its transparent subtree output.
    Layered {
        base_texture: &'a wgpu::Texture,
        foreground_view: &'a wgpu::TextureView,
    },
}

impl<'a> BackdropSource<'a> {
    pub(crate) fn base_texture(self) -> &'a wgpu::Texture {
        match self {
            Self::Flattened { texture } => texture,
            Self::Layered { base_texture, .. } => base_texture,
        }
    }

    pub(crate) fn foreground_view(self) -> Option<&'a wgpu::TextureView> {
        match self {
            Self::Flattened { .. } => None,
            Self::Layered {
                foreground_view, ..
            } => Some(foreground_view),
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
