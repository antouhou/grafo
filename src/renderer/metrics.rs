use std::collections::VecDeque;
use std::time::{Duration, Instant};

use super::Renderer;

/// Per-frame pipeline switch counts for diagnosing GPU state-change overhead.
///
/// Each field counts how many times the corresponding `set_pipeline` call was issued
/// during a single frame. `scissor_clips` counts how many times a scissor rect was
/// used *instead* of a stencil increment/decrement pair.
#[derive(Debug, Clone, Copy, Default)]
pub struct PipelineSwitchCounts {
    /// Number of switches to the stencil-increment pipeline.
    pub to_stencil_increment: u32,
    /// Number of switches to the stencil-decrement pipeline.
    pub to_stencil_decrement: u32,
    /// Number of switches to the leaf-draw pipeline.
    pub to_leaf_draw: u32,
    /// Number of switches to the composite (effect) pipeline, which resets tracking.
    pub to_composite: u32,
    /// Total GPU pipeline switches (`set_pipeline` calls).
    pub total_switches: u32,
    /// Number of parent shapes clipped via scissor rect instead of stencil.
    pub scissor_clips: u32,
}

impl PipelineSwitchCounts {
    /// Merge another frame's counts into this accumulator.
    pub fn accumulate(&mut self, other: &Self) {
        self.to_stencil_increment += other.to_stencil_increment;
        self.to_stencil_decrement += other.to_stencil_decrement;
        self.to_leaf_draw += other.to_leaf_draw;
        self.to_composite += other.to_composite;
        self.total_switches += other.total_switches;
        self.scissor_clips += other.scissor_clips;
    }
}

/// Per-phase timing breakdown for a single frame.
///
/// Provides wall-clock durations for each phase of the render loop.
/// Available when the `render_metrics` feature is enabled.
#[derive(Debug, Clone, Copy, Default)]
pub struct PhaseTimings {
    /// Time spent in `prepare_render()` — CPU-side buffer aggregation and GPU upload.
    pub prepare: Duration,
    /// Time spent encoding GPU commands and submitting them (`render_to_texture_view` + `queue.submit`).
    pub encode_and_submit: Duration,
    /// Time spent on presentation or readback (present, or map + poll + copy for offscreen).
    pub present_or_readback: Duration,
    /// Time spent waiting for the GPU to finish all submitted work (`device.poll(Wait)`).
    /// This reveals actual GPU execution time that is otherwise hidden by async submit.
    pub gpu_wait: Duration,
    /// Total frame time (sum of all phases including GPU wait).
    pub total: Duration,
}

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
    total_render_loop_duration: Duration,
    first_presented_frame_render_loop_start: Option<Instant>,
    last_presented_frame_time: Option<Instant>,
    rolling_window_samples: VecDeque<FrameTimingSample>,
    rolling_window_total_render_loop_duration: Duration,
}

impl Default for RenderLoopMetricsTracker {
    fn default() -> Self {
        Self {
            total_presented_frame_count: 0,
            total_render_loop_duration: Duration::ZERO,
            first_presented_frame_render_loop_start: None,
            last_presented_frame_time: None,
            rolling_window_samples: VecDeque::with_capacity(MAX_ROLLING_WINDOW_SAMPLE_COUNT),
            rolling_window_total_render_loop_duration: Duration::ZERO,
        }
    }
}

impl RenderLoopMetricsTracker {
    fn remove_oldest_rolling_sample(&mut self) {
        if let Some(oldest_sample) = self.rolling_window_samples.pop_front() {
            self.rolling_window_total_render_loop_duration = self
                .rolling_window_total_render_loop_duration
                .saturating_sub(oldest_sample.render_loop_duration);
        }
    }

    fn push_rolling_sample(&mut self, frame_presented_at: Instant, render_loop_duration: Duration) {
        if self.rolling_window_samples.len() == MAX_ROLLING_WINDOW_SAMPLE_COUNT {
            self.remove_oldest_rolling_sample();
        }

        self.rolling_window_samples.push_back(FrameTimingSample {
            frame_presented_at,
            render_loop_duration,
        });
        self.rolling_window_total_render_loop_duration += render_loop_duration;
    }

    fn prune_rolling_window(&mut self, now: Instant) {
        while let Some(oldest_sample) = self.rolling_window_samples.front() {
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

        if self.first_presented_frame_render_loop_start.is_none() {
            self.first_presented_frame_render_loop_start = Some(render_loop_started_at);
        }

        self.last_presented_frame_time = Some(frame_presented_at);
        self.total_presented_frame_count += 1;
        self.total_render_loop_duration += render_loop_duration;

        self.push_rolling_sample(frame_presented_at, render_loop_duration);
        self.prune_rolling_window(frame_presented_at);
    }

    pub(super) fn cumulative_average_frames_per_second(&self) -> f64 {
        if self.total_presented_frame_count == 0 {
            return 0.0;
        }

        let Some(first_frame_started_at) = self.first_presented_frame_render_loop_start else {
            return 0.0;
        };
        let Some(last_frame_presented_at) = self.last_presented_frame_time else {
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

    pub(super) fn cumulative_average_render_loop_duration(&self) -> Duration {
        if self.total_presented_frame_count == 0 {
            return Duration::ZERO;
        }

        Duration::from_secs_f64(
            self.total_render_loop_duration.as_secs_f64() / self.total_presented_frame_count as f64,
        )
    }

    pub(super) fn rolling_one_second_frames_per_second(&self) -> f64 {
        self.rolling_window_samples.len() as f64
    }

    pub(super) fn rolling_one_second_average_render_loop_duration(&self) -> Duration {
        if self.rolling_window_samples.is_empty() {
            return Duration::ZERO;
        }

        Duration::from_secs_f64(
            self.rolling_window_total_render_loop_duration.as_secs_f64()
                / self.rolling_window_samples.len() as f64,
        )
    }

    pub(super) fn total_presented_frame_count(&self) -> u64 {
        self.total_presented_frame_count
    }

    pub(super) fn reset(&mut self) {
        *self = Self::default();
    }
}

impl<'a> Renderer<'a> {
    /// Returns the cumulative average frames-per-second since metrics tracking started.
    ///
    /// FPS is computed as:
    /// `total_presented_frames / elapsed_time_between_first_render_loop_start_and_last_present`.
    pub fn overall_average_frames_per_second(&self) -> f64 {
        self.render_loop_metrics_tracker
            .cumulative_average_frames_per_second()
    }

    /// Returns the cumulative average time spent in `render()` for successfully presented frames.
    ///
    /// This measures from the start of the render loop to `present()` completion.
    pub fn average_render_loop_duration(&self) -> Duration {
        self.render_loop_metrics_tracker
            .cumulative_average_render_loop_duration()
    }

    /// Returns the rolling 1-second FPS based on successfully presented frames.
    pub fn rolling_one_second_frames_per_second(&self) -> f64 {
        self.render_loop_metrics_tracker
            .rolling_one_second_frames_per_second()
    }

    /// Returns the rolling 1-second average render-loop duration.
    ///
    /// This measures from `render()` start to `present()` completion.
    pub fn rolling_one_second_average_render_loop_duration(&self) -> Duration {
        self.render_loop_metrics_tracker
            .rolling_one_second_average_render_loop_duration()
    }

    /// Returns the number of successfully presented frames included in the metrics.
    pub fn total_presented_frame_count(&self) -> u64 {
        self.render_loop_metrics_tracker
            .total_presented_frame_count()
    }

    /// Resets all render-loop metrics to start a new measurement window.
    pub fn reset_render_loop_metrics(&mut self) {
        self.render_loop_metrics_tracker.reset();
    }

    /// Returns the per-phase timing breakdown for the most recently rendered frame.
    pub fn last_phase_timings(&self) -> PhaseTimings {
        self.last_phase_timings
    }

    /// Returns the pipeline switch counts for the most recently rendered frame.
    ///
    /// Shows how many times each GPU pipeline was bound, and how many parent shapes
    /// used scissor clipping instead of stencil increment/decrement.
    pub fn last_pipeline_switch_counts(&self) -> PipelineSwitchCounts {
        self.last_pipeline_switch_counts
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use super::RenderLoopMetricsTracker;

    fn assert_approximately_equal(left: f64, right: f64, tolerance: f64) {
        assert!(
            (left - right).abs() <= tolerance,
            "expected {left} to be within {tolerance} of {right}"
        );
    }

    #[test]
    fn metrics_tracker_returns_zero_values_when_no_frames_presented() {
        let metrics_tracker = RenderLoopMetricsTracker::default();

        assert_eq!(metrics_tracker.total_presented_frame_count(), 0);
        assert_eq!(metrics_tracker.cumulative_average_frames_per_second(), 0.0);
        assert_eq!(
            metrics_tracker.cumulative_average_render_loop_duration(),
            Duration::ZERO
        );
        assert_eq!(metrics_tracker.rolling_one_second_frames_per_second(), 0.0);
        assert_eq!(
            metrics_tracker.rolling_one_second_average_render_loop_duration(),
            Duration::ZERO
        );
    }

    #[test]
    fn metrics_tracker_accumulates_cumulative_averages() {
        let mut metrics_tracker = RenderLoopMetricsTracker::default();
        let first_frame_started_at = Instant::now();
        let first_frame_presented_at = first_frame_started_at + Duration::from_millis(10);
        let second_frame_started_at = first_frame_started_at + Duration::from_millis(20);
        let second_frame_presented_at = first_frame_started_at + Duration::from_millis(35);

        metrics_tracker.record_presented_frame(first_frame_started_at, first_frame_presented_at);
        metrics_tracker.record_presented_frame(second_frame_started_at, second_frame_presented_at);

        assert_eq!(metrics_tracker.total_presented_frame_count(), 2);
        assert_eq!(
            metrics_tracker.cumulative_average_render_loop_duration(),
            Duration::from_secs_f64(0.0125)
        );
        assert_approximately_equal(
            metrics_tracker.cumulative_average_frames_per_second(),
            2.0 / 0.035,
            1e-9,
        );
    }

    #[test]
    fn metrics_tracker_keeps_only_last_second_for_rolling_metrics() {
        let mut metrics_tracker = RenderLoopMetricsTracker::default();
        let first_frame_started_at = Instant::now();
        let first_frame_presented_at = first_frame_started_at + Duration::from_millis(10);
        let second_frame_started_at = first_frame_started_at + Duration::from_millis(500);
        let second_frame_presented_at = second_frame_started_at + Duration::from_millis(20);
        let third_frame_started_at = first_frame_started_at + Duration::from_millis(1_300);
        let third_frame_presented_at = third_frame_started_at + Duration::from_millis(30);

        metrics_tracker.record_presented_frame(first_frame_started_at, first_frame_presented_at);
        metrics_tracker.record_presented_frame(second_frame_started_at, second_frame_presented_at);
        metrics_tracker.record_presented_frame(third_frame_started_at, third_frame_presented_at);

        assert_eq!(metrics_tracker.rolling_one_second_frames_per_second(), 2.0);
        assert_eq!(
            metrics_tracker.rolling_one_second_average_render_loop_duration(),
            Duration::from_millis(25)
        );
    }

    #[test]
    fn metrics_tracker_reset_clears_all_accumulated_values() {
        let mut metrics_tracker = RenderLoopMetricsTracker::default();
        let frame_started_at = Instant::now();
        let frame_presented_at = frame_started_at + Duration::from_millis(16);
        metrics_tracker.record_presented_frame(frame_started_at, frame_presented_at);

        metrics_tracker.reset();

        assert_eq!(metrics_tracker.total_presented_frame_count(), 0);
        assert_eq!(metrics_tracker.cumulative_average_frames_per_second(), 0.0);
        assert_eq!(
            metrics_tracker.cumulative_average_render_loop_duration(),
            Duration::ZERO
        );
        assert_eq!(metrics_tracker.rolling_one_second_frames_per_second(), 0.0);
        assert_eq!(
            metrics_tracker.rolling_one_second_average_render_loop_duration(),
            Duration::ZERO
        );
    }
}
