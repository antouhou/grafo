//! Renderer for the Grafo library.
pub(crate) use self::backend::execution::shapes::TextureSamplingUniform;
pub use self::backend::WgpuBackend;
use self::backend::WgpuContext;
pub use self::contract::RenderBackend;
#[cfg(feature = "render_metrics")]
use self::metrics::RenderLoopMetricsTracker;
use self::plan::Planner;
pub use self::plan::Viewport;
use crate::CachedShapeHandle;
use ahash::HashMap;
pub use backend::readback::ReadbackError;
pub use construction::RendererCreationError;
use std::sync::{Arc, RwLock};
use std::time::Duration;
#[cfg(feature = "render_metrics")]
use std::time::Instant;

pub(crate) mod backend;
mod construction;
mod contract;
mod diagnostics;
mod draw_queue;
mod effects;
pub use effects::{EffectError, EffectShaderError};
#[cfg(feature = "render_metrics")]
pub mod metrics;
mod plan;
mod preparation;
mod readback;
mod rect_utils;
mod surface;
pub(crate) mod types;

/// GPU resources shared by renderers.
///
/// Create a context once and pass clones to renderers to share the GPU device, queue,
/// texture storage, and loaded shapes. Shape cache keys belong to the context. Use the
/// same content-derived key to reuse a shape across renderers. Loading a different shape
/// under that key, or removing it, affects every renderer using the context.
#[derive(Clone)]
pub struct RendererContext {
    pub(crate) gpu: Arc<WgpuContext>,
    pub(crate) loaded_shapes: Arc<RwLock<HashMap<u64, CachedShapeHandle>>>,
}

/// Renders a planned scene onto the surface owned by this renderer.
///
/// The backend receives completed commands and cannot access the planner or tree.
pub struct Renderer<'surface, B: RenderBackend<'surface> = WgpuBackend> {
    planner: Planner,
    surface: B::Surface,
    backend: B,
    viewport: Viewport,
    #[cfg(feature = "render_metrics")]
    last_planning_time: Duration,
    #[cfg(feature = "render_metrics")]
    render_loop_metrics_tracker: RenderLoopMetricsTracker,
}

/// Default AA fringe width in physical pixels.
const DEFAULT_FRINGE_WIDTH: f32 = 0.75;

impl Renderer<'_> {
    /// Returns CPU encoding and submission time, excluding planning, uploads and readback.
    pub fn last_render_to_texture_view_cpu_time(&self) -> Duration {
        self.backend.last_render_to_texture_view_cpu_time
    }
}

impl<'surface, B: RenderBackend<'surface>> Renderer<'surface, B> {
    /// Compiles the scene before passing its commands and this renderer's surface to execution.
    pub fn render(&mut self) -> Result<(), B::Error> {
        #[cfg(feature = "render_metrics")]
        let started_at = Instant::now();
        let commands = self.planner.plan(self.viewport);
        #[cfg(feature = "render_metrics")]
        {
            self.last_planning_time = started_at.elapsed();
        }
        self.backend.render(commands, &mut self.surface)?;
        #[cfg(feature = "render_metrics")]
        self.render_loop_metrics_tracker
            .record_presented_frame(started_at, Instant::now());
        Ok(())
    }
}

#[cfg(test)]
mod tests;
