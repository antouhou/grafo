//! Coordinates scene mutation, planning and backend execution.
#[cfg(feature = "render_metrics")]
use self::metrics::RenderLoopMetricsTracker;
pub use self::types::{DrawCommandError, EffectError};
use crate::commands::ShapeDrawId;
use crate::core::{UnsignedPhysicalRect, Viewport};
use crate::planner::Planner;
use crate::render_backend::render_target::RenderTarget;
use crate::render_backend::RenderBackend;
use crate::scene::{Scene, SceneContext};
#[cfg(feature = "render_metrics")]
use std::time::Instant;
mod draw_queue;
mod effects;
#[cfg(feature = "render_metrics")]
pub mod metrics;
mod types;
mod viewport;

/// CPU shape storage and backend context shared between renderers.
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
    pub fn into_parts(self) -> (B, SceneContext) {
        (self.backend, self.scene)
    }
}

/// Coordinates CPU scene construction and planning, then submits the flat command stream.
pub struct Renderer<B: RenderBackend> {
    scene: Scene,
    planner: Planner,
    backend: B,
    viewport: Viewport,
    fringe_width: f32,
    removed_shape_ids: Vec<ShapeDrawId>,
    dirty_bounds: Option<UnsignedPhysicalRect>,
    #[cfg(feature = "render_metrics")]
    render_loop_metrics_tracker: RenderLoopMetricsTracker,
}

impl<B: RenderBackend> Renderer<B> {
    /// Creates a renderer with an empty draw queue using the supplied backend.
    /// Loaded shapes are shared through `context`.
    pub fn from_backend(backend: B, context: SceneContext) -> Self {
        Self {
            scene: Scene::new(context),
            planner: Planner::default(),
            viewport: backend.viewport(),
            fringe_width: backend.fringe_width(),
            removed_shape_ids: Vec::new(),
            dirty_bounds: Some(UnsignedPhysicalRect::from_size(
                backend.viewport().physical_size.into(),
            )),
            backend,
            #[cfg(feature = "render_metrics")]
            render_loop_metrics_tracker: RenderLoopMetricsTracker::default(),
        }
    }

    /// Provides read-only access to backend-specific resources and diagnostics.
    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// Renders the draw queue to `target`, resizing the renderer to match its physical size.
    ///
    /// Pixmap pixels are ready on success and unchanged on error. Surface rendering
    /// submits and presents the frame, though GPU work may still be pending.
    /// The draw queue and output image are retained for another render. Additions and
    /// removals redraw their combined bounds; an unchanged queue preserves the image.
    /// This prototype does not track texture uploads or effect changes as damage.
    pub fn render<'target>(
        &mut self,
        target: impl Into<RenderTarget<'target, B::Surface>>,
    ) -> Result<(), B::Error>
    where
        B::Surface: 'target,
    {
        let target = target.into();
        let maximum = self.backend.maximum_texture_dimension();
        let size = target.validate_size(maximum)?;
        if self.viewport.physical_size != size {
            self.resize(size);
        }
        #[cfg(feature = "render_metrics")]
        let started_at = Instant::now();
        let commands = self.planner.plan(
            &self.scene,
            self.viewport,
            self.fringe_width,
            self.backend.maximum_texture_dimension(),
            self.dirty_bounds,
        );
        self.scene.finish_preparation();
        self.backend.render(commands, target)?;
        self.dirty_bounds = None;
        #[cfg(feature = "render_metrics")]
        self.render_loop_metrics_tracker
            .record_presented_frame(started_at, Instant::now());
        Ok(())
    }
}
#[cfg(test)]
mod tests;
