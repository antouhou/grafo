use super::execution::effects::EffectContext;
#[cfg(feature = "render_metrics")]
use super::metrics::PipelineSwitchCounts;
use crate::commands::ShapeTextureBinding;
use thiserror::Error;

/// Geometry cannot be addressed by an indexed draw command.
#[derive(Error, Debug)]
pub enum GeometryBufferError {
    #[error("Aggregated vertex offset exceeds the draw command limit")]
    VertexOffsetOverflow,
    #[error("Aggregated index range exceeds the draw command limit")]
    IndexRangeOverflow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Pipeline {
    None,
    StencilIncrement,
    StencilIncrementGradient,
    StencilIncrementOnly,
    StencilIncrementTexture,
    StencilIncrementGradientTexture,
    StencilDecrement,
    LeafDraw,
    LeafDrawGradient,
    LeafDrawTexture,
    LeafDrawGradientTexture,
}

/// Wraps [`Pipeline`] tracking with optional per-frame switch counters.
///
/// When the `render_metrics` feature is enabled the tracker also counts
/// every GPU `set_pipeline` call and every scissor-clip substitution so
/// the numbers can be queried after the frame.
pub(crate) struct PipelineTracker {
    pub(crate) current: Pipeline,
    #[cfg(feature = "render_metrics")]
    pub(crate) counts: PipelineSwitchCounts,
}

impl PipelineTracker {
    pub(crate) fn new() -> Self {
        Self {
            current: Pipeline::None,
            #[cfg(feature = "render_metrics")]
            counts: PipelineSwitchCounts::default(),
        }
    }

    /// Records a GPU pipeline switch when the pipeline changes.
    pub(crate) fn switch_to(&mut self, pipeline: Pipeline) {
        if self.current == pipeline {
            return;
        }
        self.current = pipeline;
        #[cfg(feature = "render_metrics")]
        {
            self.counts.total_switches += 1;
            match pipeline {
                Pipeline::StencilIncrement
                | Pipeline::StencilIncrementGradient
                | Pipeline::StencilIncrementOnly
                | Pipeline::StencilIncrementTexture
                | Pipeline::StencilIncrementGradientTexture => {
                    self.counts.to_stencil_increment += 1
                }
                Pipeline::StencilDecrement => self.counts.to_stencil_decrement += 1,
                Pipeline::LeafDraw
                | Pipeline::LeafDrawGradient
                | Pipeline::LeafDrawTexture
                | Pipeline::LeafDrawGradientTexture => self.counts.to_leaf_draw += 1,
                Pipeline::None => self.counts.to_composite += 1,
            }
        }
    }

    /// Record one draw pass that modifies the stencil buffer.
    #[cfg(feature = "render_metrics")]
    pub(crate) fn record_stencil_pass(&mut self) {
        self.counts.stencil_passes += 1;
    }
}

/// Tracks the currently-bound texture sources to skip redundant `set_bind_group` calls.
#[derive(Debug, Clone, Default)]
pub(crate) struct BoundTextureState {
    layers: [Option<ShapeTextureBinding>; 2],
}

impl BoundTextureState {
    /// Forget both bindings after a pipeline switch resets the bind group state.
    pub(crate) fn invalidate(&mut self) {
        self.layers = [None, None];
    }

    /// Returns `true` when the given texture source is not already bound on `layer`.
    pub(crate) fn needs_rebind(&self, layer: usize, texture_binding: &ShapeTextureBinding) -> bool {
        self.layers[layer].as_ref() != Some(texture_binding)
    }

    /// Record the texture source after setting its bind group.
    pub(crate) fn mark_bound(&mut self, layer: usize, texture_binding: ShapeTextureBinding) {
        self.layers[layer] = Some(texture_binding);
    }
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
