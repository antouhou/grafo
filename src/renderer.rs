//! Renderer for the Grafo library.
#[cfg(feature = "render_metrics")]
use self::metrics::RenderLoopMetricsTracker;
use self::readback::{ArgbReadbackResources, BgraReadbackResources};
use self::state::{RendererPipelineResources, RendererState};
use self::types::{DrawCommand, RendererScratch};
use crate::effect::{
    self, compile_composite_pipeline, compile_effect_pipeline, create_params_bind_group,
    CompositePipelineResources, EffectError, EffectInstance, LoadedEffect, OffscreenTexturePool,
};
use crate::pipeline::{
    create_and_depth_texture, create_msaa_color_texture, create_pipeline, PipelineType,
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
mod execution;
#[cfg(feature = "render_metrics")]
pub mod metrics;
mod passes;
mod plan;
mod preparation;
mod readback;
mod rect_utils;
mod rendering;
mod shape_effects;
mod state;
mod surface;
mod traversal;
pub(crate) mod types;

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
    /// Per-frame instance metadata, including draw order.
    temp_instance_metadata: Vec<InstanceMetadata>,

    argb_readback: Option<ArgbReadbackResources>,
    bgra_readback: Option<BgraReadbackResources>,

    /// MSAA sample count. A value of 1 disables MSAA.
    msaa_sample_count: u32,

    /// The multisampled color texture. `None` when MSAA is disabled.
    msaa_color_texture: Option<wgpu::Texture>,
    msaa_color_texture_view: Option<wgpu::TextureView>,

    /// Cached depth/stencil texture, reused across frames.
    /// Recreated on resize or MSAA sample count change.
    depth_stencil_texture: Option<wgpu::Texture>,
    depth_stencil_view: Option<wgpu::TextureView>,

    /// Reuses validation scratch storage across effect loads.
    effect_shader_validator: Validator,
    /// Compiled effects keyed by the user-provided `effect_id`.
    loaded_effects: HashMap<u64, LoadedEffect>,
    #[cfg(feature = "render_metrics")]
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
