//! GPU uploads, resource caches and command execution.
pub use self::context::BackendCreationError;
pub use self::contract::RenderBackend;
pub use self::errors::{EffectResourceError, EffectShaderError, WgpuBackendError};
use self::execution::effects::EffectRegistry;
#[cfg(feature = "render_metrics")]
use self::metrics::PhaseTimings;
pub use self::readback::ReadbackError;
use self::readback::{ArgbReadbackResources, BgraReadbackResources};
use self::resources::{BackendResources, RendererPipelineResources};
use self::texture_manager::TextureManager;
pub use self::types::GeometryBufferError;
use crate::core::Viewport;
use std::sync::Arc;
#[cfg(feature = "render_metrics")]
use std::time::Duration;

mod construction;
mod context;
mod contract;
#[cfg(feature = "render_metrics")]
mod diagnostics;
mod effects;
mod errors;
pub(in crate::backend) mod execution;
mod gradient;
mod interface;
#[cfg(feature = "render_metrics")]
pub mod metrics;
mod pipeline;
mod pipelines;
pub(in crate::backend) mod readback;
mod rendering;
pub(in crate::backend) mod resources;
mod surface;
pub mod texture_manager;
mod types;
mod vertex;

pub struct WgpuContext {
    instance: Arc<wgpu::Instance>,
    adapter: Arc<wgpu::Adapter>,
    supports_base_vertex: bool,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    texture_manager: TextureManager,
}

/// WGPU resources, uploads, execution caches and submission state.
/// No scene tree or planner is reachable through this type.
pub struct WgpuBackend {
    pub(in crate::backend) context: Arc<WgpuContext>,
    pub(in crate::backend) device: Arc<wgpu::Device>,
    pub(in crate::backend) queue: Arc<wgpu::Queue>,
    pub(in crate::backend) config: wgpu::SurfaceConfiguration,

    pub(in crate::backend) readback_bytes: Vec<u8>,
    pub(in crate::backend) viewport: Viewport,
    pub(in crate::backend) fringe_width: f32,

    pub(in crate::backend) pipeline_resources: RendererPipelineResources,

    pub(in crate::backend) argb_readback: Option<ArgbReadbackResources>,
    pub(in crate::backend) bgra_readback: Option<BgraReadbackResources>,

    /// MSAA sample count. A value of 1 disables MSAA.
    pub(in crate::backend) msaa_sample_count: u32,

    /// The multisampled color texture. `None` when MSAA is disabled.
    pub(in crate::backend) msaa_color_texture: Option<wgpu::Texture>,
    pub(in crate::backend) msaa_color_texture_view: Option<wgpu::TextureView>,

    /// Cached depth/stencil texture, reused across frames.
    /// Recreated on resize or MSAA sample count change.
    pub(in crate::backend) depth_stencil_texture: Option<wgpu::Texture>,
    pub(in crate::backend) depth_stencil_view: Option<wgpu::TextureView>,

    pub(in crate::backend) effect_registry: EffectRegistry,

    #[cfg(feature = "render_metrics")]
    /// Per-phase timing breakdown for the most recently rendered frame.
    pub(in crate::backend) last_phase_timings: PhaseTimings,

    #[cfg(feature = "render_metrics")]
    /// Last CPU duration reported by [`Self::last_render_to_texture_view_cpu_time`].
    pub(in crate::backend) last_render_to_texture_view_cpu_time: Duration,

    pub(in crate::backend) resources: BackendResources,
}

impl WgpuBackend {
    /// Shared source textures used by this backend.
    pub fn texture_manager(&self) -> &TextureManager {
        &self.pipeline_resources.shapes.texture_manager
    }
}
