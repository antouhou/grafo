//! Coordinates scene mutation, planning and backend execution.
use self::damage::PendingClipDamage;
#[cfg(feature = "render_metrics")]
use self::metrics::RenderLoopMetricsTracker;
pub use self::types::{DrawCommandError, EffectError};
use crate::commands::ShapeDrawId;
use crate::core::UnsignedPhysicalRect;
use crate::planner::Planner;
use crate::render_backend::render_target::RenderTarget;
use crate::render_backend::RenderBackend;
use crate::scene::{Scene, SceneContext};
#[cfg(feature = "render_metrics")]
use std::time::Instant;
mod damage;
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
    removed_shape_ids: Vec<ShapeDrawId>,
    should_compact_effect_parameters: bool,
    dirty_bounds: Option<UnsignedPhysicalRect>,
    pending_clip_damage: PendingClipDamage,
    #[cfg(feature = "render_metrics")]
    render_loop_metrics_tracker: RenderLoopMetricsTracker,
}

impl<B: RenderBackend> Renderer<B> {
    /// Creates a renderer with an empty draw queue using the supplied backend.
    /// Loaded shapes are shared through `context`.
    pub fn from_backend(backend: B, context: SceneContext) -> Self {
        let viewport = backend.viewport();
        Self {
            scene: Scene::new(context, viewport, backend.fringe_width()),
            planner: Planner::default(),
            removed_shape_ids: Vec::new(),
            should_compact_effect_parameters: false,
            pending_clip_damage: PendingClipDamage::default(),
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

    /// Renders the draw queue to `target`, resizing the renderer to match its physical size.
    ///
    /// Pixmap pixels are ready on success and unchanged on error. Surface rendering
    /// submits and presents the frame, though GPU work may still be pending.
    /// The draw queue and output image are retained for another render. Additions and
    /// removals redraw their combined bounds; an unchanged queue preserves the image.
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
        if self.should_compact_effect_parameters {
            self.planner.compact_effect_parameters(&mut self.scene);
            self.should_compact_effect_parameters = false;
        }
        self.pending_clip_damage
            .apply(&self.scene, &mut self.dirty_bounds);
        self.dirty_bounds = self.scene.expand_backdrop_damage(self.dirty_bounds);
        let commands = self.planner.plan(
            &self.scene,
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
