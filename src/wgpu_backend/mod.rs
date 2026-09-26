//! WGPU uploads, resource caches and command execution.
pub use self::context::BackendCreationError;
pub use self::errors::{EffectResourceError, EffectShaderError, WgpuBackendError};
use self::execution::effects::EffectRegistry;
#[cfg(feature = "render_metrics")]
use self::metrics::PhaseTimings;
pub use self::readback::ReadbackError;
use self::readback::{ArgbReadbackResources, BgraReadbackResources};
use self::resources::{BackendResources, RendererPipelineResources};
use self::texture_manager::WgpuTextureManager;
pub use self::types::GeometryBufferError;
use crate::core::Viewport;
use std::sync::Arc;
#[cfg(feature = "render_metrics")]
use std::time::Duration;

mod construction;
mod context;
#[cfg(feature = "render_metrics")]
mod diagnostics;
mod effects;
mod errors;
pub(in crate::wgpu_backend) mod execution;
mod gradient;
mod interface;
#[cfg(feature = "render_metrics")]
pub mod metrics;
mod pipeline;
mod pipelines;
pub(in crate::wgpu_backend) mod readback;
mod rendering;
pub(in crate::wgpu_backend) mod resources;
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
    texture_manager: WgpuTextureManager,
}

/// WGPU resources, uploads, execution caches and submission state.
/// No scene tree or planner is reachable through this type.
pub struct WgpuBackend {
    pub(in crate::wgpu_backend) context: Arc<WgpuContext>,
    pub(in crate::wgpu_backend) device: Arc<wgpu::Device>,
    pub(in crate::wgpu_backend) queue: Arc<wgpu::Queue>,
    pub(in crate::wgpu_backend) config: wgpu::SurfaceConfiguration,

    pub(in crate::wgpu_backend) readback_bytes: Vec<u8>,
    pub(in crate::wgpu_backend) viewport: Viewport,
    pub(in crate::wgpu_backend) fringe_width: f32,

    pub(in crate::wgpu_backend) pipeline_resources: RendererPipelineResources,

    pub(in crate::wgpu_backend) argb_readback: Option<ArgbReadbackResources>,
    pub(in crate::wgpu_backend) bgra_readback: Option<BgraReadbackResources>,

    /// MSAA sample count. A value of 1 disables MSAA.
    pub(in crate::wgpu_backend) msaa_sample_count: u32,

    /// The multisampled color texture. `None` when MSAA is disabled.
    pub(in crate::wgpu_backend) msaa_color_texture: Option<wgpu::Texture>,
    pub(in crate::wgpu_backend) msaa_color_texture_view: Option<wgpu::TextureView>,

    /// Cached depth/stencil texture, reused across frames.
    /// Recreated on resize or MSAA sample count change.
    pub(in crate::wgpu_backend) depth_stencil_texture: Option<wgpu::Texture>,
    pub(in crate::wgpu_backend) depth_stencil_view: Option<wgpu::TextureView>,

    pub(in crate::wgpu_backend) effect_registry: EffectRegistry,

    #[cfg(feature = "render_metrics")]
    /// Per-phase timing breakdown for the most recently rendered frame.
    pub(in crate::wgpu_backend) last_phase_timings: PhaseTimings,

    #[cfg(feature = "render_metrics")]
    /// Last CPU duration reported by [`Self::last_render_to_texture_view_cpu_time`].
    pub(in crate::wgpu_backend) last_render_to_texture_view_cpu_time: Duration,

    pub(in crate::wgpu_backend) resources: BackendResources,
}

impl WgpuBackend {
    /// Shared source textures used by this backend.
    pub fn texture_manager(&self) -> &WgpuTextureManager {
        &self.pipeline_resources.shapes.texture_manager
    }
}
