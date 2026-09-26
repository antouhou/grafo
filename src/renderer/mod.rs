//! Coordinates scene mutation, planning and backend execution.
#[cfg(feature = "render_metrics")]
use self::metrics::RenderLoopMetricsTracker;
pub use self::types::{DrawCommandError, EffectError};
use crate::core::Viewport;
use crate::planner::Planner;
use crate::render_backend::RenderBackend;
use crate::scene::{Scene, SceneContext};
#[cfg(feature = "render_metrics")]
use std::time::Instant;
mod draw_queue;
mod effects;
#[cfg(feature = "render_metrics")]
pub mod metrics;
mod readback;
mod surface;
mod types;

/// Shared CPU shape storage and backend context. Each renderer owns its own queue and output.
#[derive(Clone)]
pub struct RendererContext<B> {
    backend: B,
    scene: SceneContext,
}

impl<B> RendererContext<B> {
    pub fn from_parts(backend: B, scene: SceneContext) -> Self {
        Self { backend, scene }
    }

    pub fn backend(&self) -> &B {
        &self.backend
    }

    pub fn scene(&self) -> &SceneContext {
        &self.scene
    }
    /// Consumes the context without cloning either shared resource owner.
    pub fn into_parts(self) -> (B, SceneContext) {
        (self.backend, self.scene)
    }
}

/// Coordinates CPU scene construction and planning, then submits the flat command stream.
pub struct Renderer<'surface, B: RenderBackend<'surface>> {
    scene: Scene,
    planner: Planner,
    surface: B::Surface,
    backend: B,
    viewport: Viewport,
    fringe_width: f32,
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
            render_loop_metrics_tracker: RenderLoopMetricsTracker::default(),
        }
    }

    /// Provides read-only access to backend-specific resources and diagnostics.
    pub fn backend(&self) -> &B {
        &self.backend
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
        self.backend.render(commands, &mut self.surface)?;
        #[cfg(feature = "render_metrics")]
        self.render_loop_metrics_tracker
            .record_presented_frame(started_at, Instant::now());
        Ok(())
    }
}
#[cfg(test)]
mod tests;
