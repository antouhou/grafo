//! Coordinates scene mutation, planning and backend execution.
use crate::commands::ShapeDrawId;
use crate::core::UnsignedPhysicalRect;
use crate::planner::Planner;
use crate::render_backend::render_target::RenderTarget;
use crate::render_backend::RenderBackend;
use crate::scene_state::{SceneState, SceneStateContext};
use damage::{mark_dirty, PendingDamage};
#[cfg(feature = "render_metrics")]
use metrics::RenderLoopMetricsTracker;
#[cfg(feature = "render_metrics")]
use std::time::Instant;
pub use types::{DrawCommandError, EffectError};
mod damage;
mod draw_queue;
mod effects;
#[cfg(feature = "render_metrics")]
pub mod metrics;
mod types;
mod viewport;

/// CPU shape storage and backend context shared between scenes.
#[derive(Clone)]
pub struct SceneContext<B> {
    pub(crate) backend: B,
    pub(crate) state: SceneStateContext,
}

/// A retained scene of shapes, clipping rectangles, and effects that can render to a target.
pub struct Scene<B: RenderBackend> {
    state: SceneState,
    planner: Planner,
    backend: B,
    removed_shape_ids: Vec<ShapeDrawId>,
    dirty_bounds: Option<UnsignedPhysicalRect>,
    pending_damage: PendingDamage,
    #[cfg(feature = "render_metrics")]
    render_loop_metrics_tracker: RenderLoopMetricsTracker,
}

impl<B: RenderBackend> Scene<B> {
    /// Creates an empty scene using the supplied backend.
    /// Loaded shapes are shared through `context`.
    pub(crate) fn from_parts(backend: B, context: SceneStateContext) -> Self {
        let viewport = backend.viewport();
        Self {
            state: SceneState::new(context, viewport, backend.fringe_width()),
            planner: Planner::default(),
            removed_shape_ids: Vec::new(),
            pending_damage: PendingDamage::default(),
            dirty_bounds: Some(UnsignedPhysicalRect::from_size(
                viewport.physical_size.into(),
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

    /// Enables a red overlay at 25% opacity over the current surface redraw bounds.
    /// Disabled by default. Pixmap output and the retained scene stay unchanged.
    /// Toggling the overlay does not request a scene redraw.
    pub fn set_dirty_region_overlay_enabled(&mut self, enabled: bool) {
        self.backend.set_dirty_region_overlay_enabled(enabled);
    }

    /// Returns whether surface redraw bounds are highlighted.
    pub fn is_dirty_region_overlay_enabled(&self) -> bool {
        self.backend.is_dirty_region_overlay_enabled()
    }

    /// Renders the scene to `target`, resizing it to match the target's physical size.
    ///
    /// Pixmap pixels are ready on success and unchanged on error. Surface rendering
    /// submits and presents the frame, though GPU work may still be pending.
    /// Retains the scene and output image for later renders. Changes to shapes, effects, or
    /// textures redraw only the affected areas. An unchanged scene preserves the image.
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
        if self.size() != size {
            self.resize(size);
        }
        #[cfg(feature = "render_metrics")]
        let started_at = Instant::now();
        self.state.compact_effect_parameters();
        self.pending_damage
            .apply(&self.state, &mut self.dirty_bounds);
        let viewport = self.state.viewport();
        let fringe_width = self.state.fringe_width();
        for (id, shape) in self.state.shapes_with_updated_textures() {
            self.backend.refresh_uv(ShapeDrawId(id), &shape.instance);
            mark_dirty(
                &mut self.dirty_bounds,
                shape.logical_screen_bounds,
                viewport,
                fringe_width,
            );
        }
        self.dirty_bounds = self.state.expand_effect_damage(self.dirty_bounds);
        self.state.finish_preparation();
        let commands = self
            .planner
            .plan(&self.state, self.backend.maximum_texture_dimension());
        self.backend.render(self.dirty_bounds, commands, target)?;
        self.dirty_bounds = None;
        #[cfg(feature = "render_metrics")]
        self.render_loop_metrics_tracker
            .record_presented_frame(started_at, Instant::now());
        Ok(())
    }
}
#[cfg(test)]
mod tests;
