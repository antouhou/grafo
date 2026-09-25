//! Coordinates scene mutation, planning and backend execution.
#[cfg(feature = "render_metrics")]
use self::metrics::RenderLoopMetricsTracker;
pub use self::types::EffectError;
pub use crate::backend::{
    EffectShaderError, ReadbackError, RenderBackend, WgpuBackend, WgpuContext,
};
use crate::core::Viewport;
use crate::planner::Planner;
use crate::scene::{Scene, SceneContext};
pub use construction::RendererCreationError;
use std::sync::Arc;
use std::time::Duration;
#[cfg(feature = "render_metrics")]
use std::time::Instant;
mod construction;
mod diagnostics;
mod draw_queue;
mod effects;
#[cfg(feature = "render_metrics")]
pub mod metrics;
mod readback;
mod surface;
pub(crate) mod types;

/// Shared CPU shape storage and backend context. Each renderer owns its own queue and output.
#[derive(Clone)]
pub struct RendererContext<C = Arc<WgpuContext>> {
    backend: C,
    scene: SceneContext,
}
impl<C> RendererContext<C> {
    pub fn from_parts(backend: C, scene: SceneContext) -> Self {
        Self { backend, scene }
    }
    pub fn backend(&self) -> &C {
        &self.backend
    }
    pub fn scene(&self) -> &SceneContext {
        &self.scene
    }
}

/// Coordinates CPU scene construction and planning, then submits the flat command stream.
pub struct Renderer<'surface, B: RenderBackend<'surface> = WgpuBackend> {
    scene: Scene,
    planner: Planner,
    surface: B::Surface,
    backend: B,
    viewport: Viewport,
    fringe_width: f32,
    #[cfg(feature = "render_metrics")]
    last_planning_time: Duration,
    #[cfg(feature = "render_metrics")]
    render_loop_metrics_tracker: RenderLoopMetricsTracker,
}

impl<'surface, B: RenderBackend<'surface>> Renderer<'surface, B> {
    /// Creates an empty scene for an initialized backend and its output.
    /// Loaded CPU shapes can be shared through `context` without copying geometry.
    pub fn from_backend(backend: B, surface: B::Surface, context: SceneContext) -> Self {
        Self {
            scene: Scene::new(context),
            planner: Planner::default(),
            surface,
            viewport: backend.viewport(),
            fringe_width: backend.fringe_width(),
            backend,
            #[cfg(feature = "render_metrics")]
            last_planning_time: Duration::ZERO,
            #[cfg(feature = "render_metrics")]
            render_loop_metrics_tracker: RenderLoopMetricsTracker::default(),
        }
    }

    /// Returns CPU encoding and submission time, excluding planning, uploads and readback.
    pub fn last_render_to_texture_view_cpu_time(&self) -> Duration {
        self.backend.last_render_to_texture_view_cpu_time()
    }

    /// Plans the scene before passing completed commands to the backend.
    pub fn render(&mut self) -> Result<(), B::Error> {
        #[cfg(feature = "render_metrics")]
        let started_at = Instant::now();
        let commands = self.planner.plan(
            &self.scene,
            self.viewport,
            self.fringe_width,
            self.backend.maximum_texture_dimension(),
        );
        self.scene.finish_preparation();
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
