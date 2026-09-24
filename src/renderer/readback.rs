use super::{ReadbackError, Renderer};
#[cfg(feature = "render_metrics")]
use std::time::Instant;

impl Renderer<'_> {
    /// Reads tightly packed BGRA pixels, leaving `buffer` unchanged on readback failure.
    pub fn render_to_buffer(&mut self, buffer: &mut Vec<u8>) -> Result<(), ReadbackError> {
        #[cfg(feature = "render_metrics")]
        let started_at = Instant::now();
        let commands = self.planner.plan(self.viewport);
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
    pub fn render_to_argb32(&mut self, out_pixels: &mut [u32]) -> Result<(), ReadbackError> {
        let (width, height) = self.viewport.physical_size;
        let required_pixels = width as usize * height as usize;
        if out_pixels.len() < required_pixels {
            return Err(ReadbackError::OutputTooSmall {
                required_pixels,
                provided_pixels: out_pixels.len(),
            });
        }
        #[cfg(feature = "render_metrics")]
        let started_at = Instant::now();
        let commands = self.planner.plan(self.viewport);
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
