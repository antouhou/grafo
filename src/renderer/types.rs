use super::commands::DrawPlan;
use super::execution::effects::EffectRegistry;
#[cfg(feature = "render_metrics")]
use super::metrics::PipelineSwitchCounts;
use super::plan::draws::DrawPlanner;
use super::plan::shape_effects::ShapeEffectPlan;
use super::traversal::TraversalScratch;
use super::IntermediateTextureId;
use crate::shape::{CachedShapeDrawData, ShapeTextureBinding};
use crate::vertex::InstanceTransform;
use ahash::{HashMap, HashMapExt};
use thiserror::Error;
use wgpu::SurfaceError;

// TODO: probably some parts of it also can be cached, so we don't need to copy it all the time.
#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
pub(super) enum DrawTreeNode {
    CachedShape(CachedShapeDrawData),
    ClipRect(ClipRectDrawData),
}

#[derive(Debug)]
pub(super) struct ClipRectDrawData {
    pub(super) rect_bounds: [(f32, f32); 2],
    pub(super) transform: Option<InstanceTransform>,
    pub(super) is_leaf: bool,
    pub(super) clips_children: bool,
}

impl ClipRectDrawData {
    pub(super) fn new(
        rect_bounds: [(f32, f32); 2],
        transform: Option<InstanceTransform>,
        clips_children: bool,
    ) -> Self {
        Self {
            rect_bounds,
            transform,
            clips_children,
            is_leaf: true,
        }
    }
}

impl DrawTreeNode {
    /// Whether this node has no children in the draw tree
    /// Starts as `true`; set to `false` when a child is added.
    pub(super) fn is_leaf(&self) -> bool {
        match self {
            DrawTreeNode::CachedShape(s) => s.is_leaf,
            DrawTreeNode::ClipRect(clip_rect) => clip_rect.is_leaf,
        }
    }

    pub(super) fn set_not_leaf(&mut self) {
        match self {
            DrawTreeNode::CachedShape(s) => s.is_leaf = false,
            DrawTreeNode::ClipRect(clip_rect) => clip_rect.is_leaf = false,
        }
    }

    pub(super) fn is_clip_rect(&self) -> bool {
        matches!(self, DrawTreeNode::ClipRect(_))
    }
}

impl DrawTreeNode {
    pub(super) fn transform(&self) -> Option<InstanceTransform> {
        match self {
            DrawTreeNode::CachedShape(cached_shape) => cached_shape.transform,
            DrawTreeNode::ClipRect(clip_rect) => clip_rect.transform,
        }
    }

    pub(super) fn texture_id(&self, layer: usize) -> Option<u64> {
        match self {
            DrawTreeNode::CachedShape(cached_shape) => cached_shape
                .texture_bindings
                .get(layer)
                .and_then(ShapeTextureBinding::managed_texture_id),
            DrawTreeNode::ClipRect(_) => None,
        }
    }

    pub(super) fn local_bounds(&self) -> [(f32, f32); 2] {
        match self {
            DrawTreeNode::CachedShape(cached_shape) => cached_shape.cached_shape.local_bounds(),
            DrawTreeNode::ClipRect(clip_rect) => clip_rect.rect_bounds,
        }
    }

    pub(super) fn instance_color_override(&self) -> Option<[f32; 4]> {
        match self {
            DrawTreeNode::CachedShape(cached_shape) => cached_shape.color_override,
            DrawTreeNode::ClipRect(_) => None,
        }
    }

    pub(super) fn has_gradient_fill(&self) -> bool {
        match self {
            DrawTreeNode::CachedShape(cached_shape) => cached_shape.has_gradient_fill(),
            DrawTreeNode::ClipRect(_) => false,
        }
    }

    pub(super) fn clips_children(&self) -> bool {
        match self {
            DrawTreeNode::CachedShape(cached_shape) => cached_shape.clips_children,
            DrawTreeNode::ClipRect(clip_rect) => clip_rect.clips_children,
        }
    }

    pub(super) fn is_rect(&self) -> bool {
        match self {
            DrawTreeNode::CachedShape(cached_shape) => cached_shape.cached_shape.is_rect,
            DrawTreeNode::ClipRect(_) => true,
        }
    }

    pub(super) fn rect_bounds(&self) -> Option<[(f32, f32); 2]> {
        match self {
            DrawTreeNode::CachedShape(cached_shape) => cached_shape.cached_shape.rect_bounds,
            DrawTreeNode::ClipRect(clip_rect) => Some(clip_rect.rect_bounds),
        }
    }
}

/// Geometry cannot be addressed by an indexed draw command.
#[derive(Error, Debug)]
pub enum GeometryBufferError {
    #[error("Aggregated vertex offset exceeds the draw command limit")]
    VertexOffsetOverflow,
    #[error("Aggregated index range exceeds the draw command limit")]
    IndexRangeOverflow,
}

/// The surface texture could not be acquired.
#[derive(Error, Debug)]
pub enum RenderError {
    #[error(transparent)]
    Surface(#[from] SurfaceError),
}

#[derive(Error, Debug)]
#[non_exhaustive]
pub enum DrawCommandError {
    #[error(transparent)]
    GeometryBuffer(#[from] GeometryBufferError),
    #[error("Shape with id {0} doesn't exist in the draw tree.")]
    InvalidShapeId(usize),
    #[error("Shape with id {0} has not been loaded yet")]
    ShapeNotLoaded(u64),
    #[error("Clip rect node only supports axis-aligned transforms.")]
    UnsupportedClipRectTransform,
    #[error("Clip rect node {0} does not support {1}.")]
    UnsupportedClipRectOperation(usize, &'static str),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TraversalEvent {
    Pre(usize),
    Post(usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Pipeline {
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
pub(super) struct PipelineTracker {
    pub(super) current: Pipeline,
    #[cfg(feature = "render_metrics")]
    pub(super) counts: PipelineSwitchCounts,
}

impl PipelineTracker {
    pub(super) fn new() -> Self {
        Self {
            current: Pipeline::None,
            #[cfg(feature = "render_metrics")]
            counts: PipelineSwitchCounts::default(),
        }
    }

    /// Records a GPU pipeline switch when the pipeline changes.
    pub(super) fn switch_to(&mut self, pipeline: Pipeline) {
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
    pub(super) fn record_stencil_pass(&mut self) {
        self.counts.stencil_passes += 1;
    }
}

/// Tracks the currently-bound texture sources to skip redundant `set_bind_group` calls.
#[derive(Debug, Clone, Default)]
pub(super) struct BoundTextureState {
    layers: [Option<ShapeTextureBinding>; 2],
}

impl BoundTextureState {
    /// Forget both bindings after a pipeline switch resets the bind group state.
    pub(super) fn invalidate(&mut self) {
        self.layers = [None, None];
    }

    /// Returns `true` when the given texture source is not already bound on `layer`.
    pub(super) fn needs_rebind(&self, layer: usize, texture_binding: &ShapeTextureBinding) -> bool {
        self.layers[layer].as_ref() != Some(texture_binding)
    }

    /// Record the texture source after setting its bind group.
    pub(super) fn mark_bound(&mut self, layer: usize, texture_binding: ShapeTextureBinding) {
        self.layers[layer] = Some(texture_binding);
    }
}

#[derive(Clone, Copy)]
pub(super) enum BackdropSource<'a> {
    /// The source already contains every layer painted before the backdrop node.
    Flattened { texture: &'a wgpu::Texture },
    /// Group rendering keeps the outside scene separate from its transparent subtree output.
    Layered {
        base_texture: &'a wgpu::Texture,
        foreground_view: &'a wgpu::TextureView,
    },
}

impl<'a> BackdropSource<'a> {
    pub(super) fn base_texture(self) -> &'a wgpu::Texture {
        match self {
            Self::Flattened { texture } => texture,
            Self::Layered { base_texture, .. } => base_texture,
        }
    }

    pub(super) fn foreground_view(self) -> Option<&'a wgpu::TextureView> {
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
pub(super) struct BackdropContext<'a> {
    pub(super) effect_registry: &'a EffectRegistry,
    pub(super) effect_sampler: &'a wgpu::Sampler,
    pub(super) texture_blit_pipeline: &'a wgpu::RenderPipeline,
    pub(super) composite_bind_group_layout: &'a wgpu::BindGroupLayout,
    pub(super) backdrop_layer_composite_pipeline: &'a wgpu::RenderPipeline,
    pub(super) backdrop_layer_composite_bind_group_layout: &'a wgpu::BindGroupLayout,
    pub(super) device: &'a wgpu::Device,
    pub(super) queue: &'a wgpu::Queue,
    pub(super) config_format: wgpu::TextureFormat,
    pub(super) max_texture_dimension_2d: u32,
}

pub(super) struct RendererScratch {
    pub(super) effect_results: HashMap<usize, IntermediateTextureId>,
    pub(super) shape_effect_plan: ShapeEffectPlan,
    pub(super) effect_node_ids: Vec<(usize, usize)>,
    pub(super) draw_planner: DrawPlanner,
    pub(super) draw_plan: DrawPlan,
    /// CPU storage reused for mapped readback data.
    pub(super) readback_bytes: Vec<u8>,
    pub(super) traversal_scratch: TraversalScratch,
}

impl RendererScratch {
    pub(super) fn new() -> Self {
        Self {
            effect_results: HashMap::new(),
            shape_effect_plan: ShapeEffectPlan::new(),
            effect_node_ids: Vec::new(),
            draw_planner: DrawPlanner::default(),
            draw_plan: DrawPlan::default(),
            readback_bytes: Vec::new(),
            traversal_scratch: TraversalScratch::new(),
        }
    }

    pub(super) fn begin_frame(&mut self) {
        self.draw_plan.clear();
        self.effect_results.clear();
        self.shape_effect_plan.clear();
        self.effect_node_ids.clear();
        self.readback_bytes.clear();
        self.traversal_scratch.begin();
    }
}
