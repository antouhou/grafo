use super::RenderLoopMetricsTracker;
use std::time::{Duration, Instant};

fn assert_approximately_equal(left: f64, right: f64, tolerance: f64) {
    assert!(
        (left - right).abs() <= tolerance,
        "expected {left} to be within {tolerance} of {right}"
    );
}

#[test]
fn metrics_tracker_returns_zero_values_when_no_frames_presented() {
    let mut metrics_tracker = RenderLoopMetricsTracker::default();

    assert_eq!(metrics_tracker.total_presented_frame_count(), 0);
    assert_eq!(metrics_tracker.average_frames_per_second(), 0.0);
    assert_eq!(
        metrics_tracker.average_render_loop_duration(),
        Duration::ZERO
    );
    assert_eq!(metrics_tracker.rolling_frames_per_second(), 0.0);
    assert_eq!(
        metrics_tracker.rolling_average_render_loop_duration(),
        Duration::ZERO
    );
}

#[test]
fn metrics_tracker_accumulates_averages() {
    let mut metrics_tracker = RenderLoopMetricsTracker::default();
    let first_frame_started_at = Instant::now();
    let first_frame_presented_at = first_frame_started_at + Duration::from_millis(10);
    let second_frame_started_at = first_frame_started_at + Duration::from_millis(20);
    let second_frame_presented_at = first_frame_started_at + Duration::from_millis(35);

    metrics_tracker.record_presented_frame(first_frame_started_at, first_frame_presented_at);
    metrics_tracker.record_presented_frame(second_frame_started_at, second_frame_presented_at);

    assert_eq!(metrics_tracker.total_presented_frame_count(), 2);
    assert_eq!(
        metrics_tracker.average_render_loop_duration(),
        Duration::from_secs_f64(0.0125)
    );
    assert_approximately_equal(
        metrics_tracker.average_frames_per_second(),
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

    assert_eq!(metrics_tracker.rolling_frames_per_second(), 2.0);
    assert_eq!(
        metrics_tracker.rolling_average_render_loop_duration(),
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
    assert_eq!(metrics_tracker.average_frames_per_second(), 0.0);
    assert_eq!(
        metrics_tracker.average_render_loop_duration(),
        Duration::ZERO
    );
    assert_eq!(metrics_tracker.rolling_frames_per_second(), 0.0);
    assert_eq!(
        metrics_tracker.rolling_average_render_loop_duration(),
        Duration::ZERO
    );
}
