use self::execution::effects::EffectRegistry;
use self::readback::{ArgbReadbackResources, BgraReadbackResources};
use self::resources::{BackendResources, RendererPipelineResources};
#[cfg(feature = "render_metrics")]
use super::metrics::PhaseTimings;
use super::Viewport;
use crate::TextureManager;
use std::sync::Arc;
use std::time::Duration;

mod construction;
mod effects;
pub(in crate::renderer) mod execution;
pub(crate) mod gradient;
mod pipelines;
pub(in crate::renderer) mod readback;
mod rendering;
pub(in crate::renderer) mod resources;
mod surface;
pub(crate) mod vertex;

pub(crate) struct WgpuContext {
    pub(crate) instance: Arc<wgpu::Instance>,
    pub(crate) adapter: Arc<wgpu::Adapter>,
    pub(crate) supports_base_vertex: bool,
    pub(crate) device: Arc<wgpu::Device>,
    pub(crate) queue: Arc<wgpu::Queue>,
    pub(crate) texture_manager: TextureManager,
}

/// WGPU resources, uploads, execution caches and submission state.
/// No scene tree or planner is reachable through this type.
pub struct WgpuBackend {
    pub(in crate::renderer) context: Arc<WgpuContext>,
    pub(in crate::renderer) instance: Arc<wgpu::Instance>,
    pub(in crate::renderer) device: Arc<wgpu::Device>,
    pub(in crate::renderer) queue: Arc<wgpu::Queue>,
    pub(in crate::renderer) config: wgpu::SurfaceConfiguration,

    pub(in crate::renderer) readback_bytes: Vec<u8>,
    pub(in crate::renderer) viewport: Viewport,
    pub(in crate::renderer) fringe_width: f32,

    pub(in crate::renderer) pipeline_resources: RendererPipelineResources,

    pub(in crate::renderer) argb_readback: Option<ArgbReadbackResources>,
    pub(in crate::renderer) bgra_readback: Option<BgraReadbackResources>,

    /// MSAA sample count. A value of 1 disables MSAA.
    pub(in crate::renderer) msaa_sample_count: u32,

    /// The multisampled color texture. `None` when MSAA is disabled.
    pub(in crate::renderer) msaa_color_texture: Option<wgpu::Texture>,
    pub(in crate::renderer) msaa_color_texture_view: Option<wgpu::TextureView>,

    /// Cached depth/stencil texture, reused across frames.
    /// Recreated on resize or MSAA sample count change.
    pub(in crate::renderer) depth_stencil_texture: Option<wgpu::Texture>,
    pub(in crate::renderer) depth_stencil_view: Option<wgpu::TextureView>,

    pub(in crate::renderer) effect_registry: EffectRegistry,

    #[cfg(feature = "render_metrics")]
    /// Per-phase timing breakdown for the most recently rendered frame.
    pub(in crate::renderer) last_phase_timings: PhaseTimings,

    /// Wall-clock CPU time spent inside the most recent `render_to_texture_view()` call.
    ///
    /// This measures render/effect pass encoding and `queue.submit`. Planning
    /// and uploads run during preparation. Presentation, readback mapping, and
    /// forced GPU waits after submission are also excluded.
    pub(in crate::renderer) last_render_to_texture_view_cpu_time: Duration,

    pub(in crate::renderer) resources: BackendResources,
}

impl WgpuBackend {
    /// Shared source textures used by this backend.
    pub fn texture_manager(&self) -> &TextureManager {
        &self.pipeline_resources.shapes.texture_manager
    }
}
