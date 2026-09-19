//! Renderer for the Grafo library.
#[cfg(feature = "render_metrics")]
use self::metrics::RenderLoopMetricsTracker;
use self::state::{RendererPipelineResources, RendererState};
use self::types::{DrawCommand, RendererScratch};
use crate::effect::{
    self, compile_composite_pipeline, compile_effect_pipeline, create_params_bind_group,
    CompositePipelineResources, EffectError, EffectInstance, LoadedEffect, OffscreenTexturePool,
};
use crate::pipeline::{
    compute_padded_bytes_per_row, create_and_depth_texture, create_argb_swizzle_bind_group,
    create_argb_swizzle_pipeline, create_msaa_color_texture, create_offscreen_color_texture,
    create_pipeline, create_readback_buffer, encode_copy_texture_to_buffer, ArgbParams,
    PipelineType,
};
use crate::shape::{CachedShapeDrawData, Shape};
use crate::texture_manager::TextureManager;
use crate::util::{to_logical, ShapeResources};
use crate::vertex::{
    CustomVertex, GeometryBufferRange, InstanceColor, InstanceMetadata, InstanceTransform,
    TextureUvTransform,
};
use crate::CachedShapeHandle;
use ahash::{HashMap, HashMapExt};
pub use construction::RendererCreationError;
use lyon::tessellation::FillTessellator;
use naga::valid::Validator;
pub use readback::ReadbackError;
use std::sync::{Arc, RwLock};
use std::time::Duration;
use tracing::warn;
use wgpu::{BufferUsages, CompositeAlphaMode, SurfaceTarget};

mod construction;
mod draw_queue;
mod effects;
#[cfg(feature = "render_metrics")]
pub mod metrics;
mod passes;
mod preparation;
mod readback;
mod rect_utils;
mod rendering;
mod shape_effects;
mod state;
mod surface;
mod traversal;
pub(crate) mod types;

pub type MathRect = lyon::math::Box2D;

/// Semantic texture layers for a shape. Background is layer 0, Foreground is layer 1.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum TextureLayer {
    Background,
    Foreground,
}

impl From<TextureLayer> for usize {
    fn from(value: TextureLayer) -> Self {
        match value {
            TextureLayer::Background => 0,
            TextureLayer::Foreground => 1,
        }
    }
}

/// Controls whether a shape clips descendants attached to it in the draw tree.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash)]
pub enum ShapeOverflow {
    /// Descendants are clipped to this shape. This is the default.
    #[default]
    Hidden,
    /// Descendants can render outside this shape, while still inheriting ancestor clips.
    Visible,
}

/// GPU resources shared by renderers.
///
/// Create a context once and pass clones to renderers to share the GPU device, queue,
/// texture storage, and loaded shapes. Shape cache keys belong to the context. Use the
/// same content-derived key to reuse a shape across renderers. Loading a different shape
/// under that key, or removing it, affects every renderer using the context.
#[derive(Clone)]
pub struct RendererContext {
    pub(crate) inner: Arc<RendererContextInner>,
}

pub(crate) struct RendererContextInner {
    pub(crate) instance: Arc<wgpu::Instance>,
    pub(crate) adapter: Arc<wgpu::Adapter>,
    pub(crate) supports_base_vertex: bool,
    pub(crate) device: Arc<wgpu::Device>,
    pub(crate) queue: Arc<wgpu::Queue>,
    pub(crate) texture_manager: TextureManager,
    pub(crate) shape_cache: RwLock<HashMap<u64, CachedShapeHandle>>,
}

/// Renders shapes and images with its own draw queue and an optional window surface.
///
/// Multiple renderers can share GPU resources through a [`RendererContext`].
pub struct Renderer<'a> {
    /// Outward AA fringe width in physical pixels.
    fringe_width: f32,

    // WGPU components
    context: RendererContext,
    instance: Arc<wgpu::Instance>,
    surface: Option<wgpu::Surface<'a>>,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    config: wgpu::SurfaceConfiguration,

    tessellator: FillTessellator,

    pipeline_resources: RendererPipelineResources,

    temp_vertices: Vec<CustomVertex>,
    temp_indices: Vec<u16>,

    /// Shared buffer locations for each uploaded geometry ID.
    geometry_dedup_map: HashMap<u64, GeometryBufferRange>,

    /// Per-frame instance transforms for shapes.
    temp_instance_transforms: Vec<InstanceTransform>,
    /// Per-frame instance colors for shapes.
    temp_instance_colors: Vec<InstanceColor>,
    /// Per-frame instance metadata (draw order) for shapes.
    temp_instance_metadata: Vec<InstanceMetadata>,

    // Cached resources for render_to_argb32 compute swizzle path
    argb_cs_bgl: Option<wgpu::BindGroupLayout>,
    argb_cs_pipeline: Option<wgpu::ComputePipeline>,
    argb_swizzle_bind_group: Option<wgpu::BindGroup>,
    argb_params_buffer: Option<wgpu::Buffer>,
    argb_input_buffer: Option<wgpu::Buffer>,
    argb_output_storage_buffer: Option<wgpu::Buffer>,
    argb_readback_buffer: Option<wgpu::Buffer>,
    argb_input_buffer_size: u64,
    argb_output_buffer_size: u64,
    argb_cached_width: u32,
    argb_cached_height: u32,
    argb_offscreen_texture: Option<wgpu::Texture>,

    // Cached resources for render_to_buffer (BGRA bytes) path
    rtb_offscreen_texture: Option<wgpu::Texture>,
    rtb_readback_buffer: Option<wgpu::Buffer>,
    rtb_cached_width: u32,
    rtb_cached_height: u32,

    /// Current MSAA sample count (1 = off, 4 = 4x, etc.)
    msaa_sample_count: u32,

    /// The multisampled color texture (None when sample_count == 1).
    msaa_color_texture: Option<wgpu::Texture>,
    /// View of the MSAA color texture.
    msaa_color_texture_view: Option<wgpu::TextureView>,

    /// Cached depth/stencil texture, reused across frames.
    /// Recreated on resize or MSAA sample count change.
    depth_stencil_texture: Option<wgpu::Texture>,
    /// View of the cached depth/stencil texture.
    depth_stencil_view: Option<wgpu::TextureView>,

    /// Reuses validation scratch storage across effect loads.
    effect_shader_validator: Validator,
    /// Loaded (compiled) effects, keyed by user-provided effect_id.
    loaded_effects: HashMap<u64, LoadedEffect>,
    #[cfg(feature = "render_metrics")]
    /// Tracking for cumulative render-loop timing metrics.
    render_loop_metrics_tracker: RenderLoopMetricsTracker,

    #[cfg(feature = "render_metrics")]
    /// Per-phase timing breakdown for the most recently rendered frame.
    last_phase_timings: self::metrics::PhaseTimings,

    /// Wall-clock CPU time spent inside the most recent `render_to_texture_view()` call.
    ///
    /// This measures CPU-side traversal planning, render/effect pass encoding,
    /// and `queue.submit`, but excludes presentation, readback mapping, and any
    /// forced GPU waits after submission.
    last_render_to_texture_view_cpu_time: Duration,

    state: RendererState,
}

/// Default AA fringe width in physical pixels.
const DEFAULT_FRINGE_WIDTH: f32 = 0.75;

impl<'a> Renderer<'a> {
    const DEFAULT_FRINGE_WIDTH: f32 = DEFAULT_FRINGE_WIDTH;

    pub(super) fn begin_frame_scratch(&mut self) {
        self.state.scratch.begin_frame();
    }

    pub(super) fn trim_scratch_storage(&mut self) {
        self.state.shape_resources.aa_fringe_scratch.trim();
        self.state.scratch.trim_to_policy();
    }

    /// Returns the wall-clock CPU time spent in the most recent `render_to_texture_view()` call.
    pub fn last_render_to_texture_view_cpu_time(&self) -> Duration {
        self.last_render_to_texture_view_cpu_time
    }
}
