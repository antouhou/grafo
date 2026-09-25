use super::{RenderBackend, Renderer};
#[cfg(feature = "render_metrics")]
use std::time::Instant;

impl<'surface, B: RenderBackend<'surface>> Renderer<'surface, B> {
    /// Reads tightly packed BGRA pixels, leaving `buffer` unchanged on readback failure.
    pub fn render_to_buffer(&mut self, buffer: &mut Vec<u8>) -> Result<(), B::Error> {
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
        self.backend.render_to_buffer(commands, buffer)?;
        #[cfg(feature = "render_metrics")]
        self.render_loop_metrics_tracker
            .record_presented_frame(started_at, Instant::now());
        Ok(())
    }

    /// Reads ARGB pixels into the viewport-sized prefix, leaving it unchanged on failure.
    pub fn render_to_argb32(&mut self, out_pixels: &mut [u32]) -> Result<(), B::Error> {
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
        self.backend.render_to_argb32(commands, out_pixels)?;
        #[cfg(feature = "render_metrics")]
        self.render_loop_metrics_tracker
            .record_presented_frame(started_at, Instant::now());
        Ok(())
    }
}
