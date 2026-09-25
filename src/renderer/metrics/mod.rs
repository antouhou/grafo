use super::{RenderBackend, Renderer};
use std::collections::VecDeque;
use std::time::{Duration, Instant};

const ROLLING_WINDOW_DURATION: Duration = Duration::from_secs(1);
const MAX_ROLLING_WINDOW_SAMPLE_COUNT: usize = 16_384;

#[derive(Debug, Clone, Copy)]
struct FrameTimingSample {
    frame_presented_at: Instant,
    render_loop_duration: Duration,
}

#[derive(Debug)]
pub(super) struct RenderLoopMetricsTracker {
    total_presented_frame_count: u64,
    total_render_duration: Duration,
    first_render_started_at: Option<Instant>,
    last_frame_presented_at: Option<Instant>,
    rolling_samples: VecDeque<FrameTimingSample>,
    rolling_render_duration: Duration,
}

impl Default for RenderLoopMetricsTracker {
    fn default() -> Self {
        Self {
            total_presented_frame_count: 0,
            total_render_duration: Duration::ZERO,
            first_render_started_at: None,
            last_frame_presented_at: None,
            rolling_samples: VecDeque::with_capacity(MAX_ROLLING_WINDOW_SAMPLE_COUNT),
            rolling_render_duration: Duration::ZERO,
        }
    }
}

impl RenderLoopMetricsTracker {
    fn remove_oldest_rolling_sample(&mut self) {
        if let Some(oldest_sample) = self.rolling_samples.pop_front() {
            self.rolling_render_duration = self
                .rolling_render_duration
                .saturating_sub(oldest_sample.render_loop_duration);
        }
    }

    fn push_rolling_sample(&mut self, frame_presented_at: Instant, render_loop_duration: Duration) {
        if self.rolling_samples.len() == MAX_ROLLING_WINDOW_SAMPLE_COUNT {
            self.remove_oldest_rolling_sample();
        }

        self.rolling_samples.push_back(FrameTimingSample {
            frame_presented_at,
            render_loop_duration,
        });
        self.rolling_render_duration += render_loop_duration;
    }

    fn prune_rolling_window(&mut self, now: Instant) {
        while let Some(oldest_sample) = self.rolling_samples.front() {
            let sample_age = now.saturating_duration_since(oldest_sample.frame_presented_at);
            if sample_age <= ROLLING_WINDOW_DURATION {
                break;
            }

            self.remove_oldest_rolling_sample();
        }
    }

    pub(super) fn record_presented_frame(
        &mut self,
        render_loop_started_at: Instant,
        frame_presented_at: Instant,
    ) {
        let render_loop_duration =
            frame_presented_at.saturating_duration_since(render_loop_started_at);

        if self.first_render_started_at.is_none() {
            self.first_render_started_at = Some(render_loop_started_at);
        }

        self.last_frame_presented_at = Some(frame_presented_at);
        self.total_presented_frame_count += 1;
        self.total_render_duration += render_loop_duration;

        self.push_rolling_sample(frame_presented_at, render_loop_duration);
        self.prune_rolling_window(frame_presented_at);
    }

    pub(super) fn average_frames_per_second(&self) -> f64 {
        if self.total_presented_frame_count == 0 {
            return 0.0;
        }

        let Some(first_frame_started_at) = self.first_render_started_at else {
            return 0.0;
        };
        let Some(last_frame_presented_at) = self.last_frame_presented_at else {
            return 0.0;
        };

        let total_elapsed_duration =
            last_frame_presented_at.saturating_duration_since(first_frame_started_at);
        let total_elapsed_seconds = total_elapsed_duration.as_secs_f64();

        if total_elapsed_seconds == 0.0 {
            return 0.0;
        }

        self.total_presented_frame_count as f64 / total_elapsed_seconds
    }

    pub(super) fn average_render_loop_duration(&self) -> Duration {
        if self.total_presented_frame_count == 0 {
            return Duration::ZERO;
        }

        Duration::from_secs_f64(
            self.total_render_duration.as_secs_f64() / self.total_presented_frame_count as f64,
        )
    }

    pub(super) fn rolling_frames_per_second(&mut self) -> f64 {
        self.prune_rolling_window(Instant::now());
        self.rolling_samples.len() as f64
    }

    pub(super) fn rolling_average_render_loop_duration(&mut self) -> Duration {
        self.prune_rolling_window(Instant::now());
        if self.rolling_samples.is_empty() {
            return Duration::ZERO;
        }

        Duration::from_secs_f64(
            self.rolling_render_duration.as_secs_f64() / self.rolling_samples.len() as f64,
        )
    }

    pub(super) fn total_presented_frame_count(&self) -> u64 {
        self.total_presented_frame_count
    }

    pub(super) fn reset(&mut self) {
        *self = Self::default();
    }
}

impl<'surface, B: RenderBackend<'surface>> Renderer<'surface, B> {
    /// Returns the average frames-per-second since metrics tracking started.
    ///
    /// Divides the completed frame count by the time from the first render's start
    /// through the latest successful render's completion.
    pub fn average_frames_per_second(&self) -> f64 {
        self.render_loop_metrics_tracker.average_frames_per_second()
    }

    /// Returns the average time spent in successful render calls, including readback calls.
    ///
    /// Includes scene planning and any waiting performed by the backend.
    pub fn average_render_loop_duration(&self) -> Duration {
        self.render_loop_metrics_tracker
            .average_render_loop_duration()
    }

    /// Returns the rolling 1-second FPS based on successful render calls.
    pub fn rolling_frames_per_second(&mut self) -> f64 {
        self.render_loop_metrics_tracker.rolling_frames_per_second()
    }

    /// Returns the rolling 1-second average render-loop duration.
    ///
    /// Includes scene planning and any waiting performed by the backend.
    pub fn rolling_average_render_loop_duration(&mut self) -> Duration {
        self.render_loop_metrics_tracker
            .rolling_average_render_loop_duration()
    }

    /// Returns the number of successful render calls included in the metrics.
    pub fn total_presented_frame_count(&self) -> u64 {
        self.render_loop_metrics_tracker
            .total_presented_frame_count()
    }

    /// Resets all render-loop metrics to start a new measurement window.
    pub fn reset_render_loop_metrics(&mut self) {
        self.render_loop_metrics_tracker.reset();
    }
}

#[cfg(test)]
mod tests;
