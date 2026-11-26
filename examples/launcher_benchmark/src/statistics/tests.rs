use super::{Statistics, WorkloadStatistics};
use crate::animation::Workload;
use std::time::{Duration, Instant};

#[test]
fn warmup_idle_and_workload_changes_do_not_contaminate_active_fps() {
    let start = Instant::now();
    let mut statistics = Statistics::default();
    let at = |milliseconds| start + Duration::from_millis(milliseconds);
    statistics.record(Some(Workload::Pages), true, at(0), at(200), true);
    statistics.record(Some(Workload::Pages), false, at(300), at(310), true);
    statistics.record(Some(Workload::Pages), false, at(315), at(325), true);
    statistics.record(None, false, at(500), at(501), true);
    statistics.record(Some(Workload::Pages), false, at(1000), at(1010), true);
    let pages = &statistics.workloads[0];
    assert_eq!(pages.samples, 3);
    assert_eq!(pages.mean_ms(), 10.0);
    assert!((pages.fps() - 3000.0 / 35.0).abs() < 0.001);
    statistics.record(Some(Workload::Opening), false, at(2000), at(2020), true);
    assert_eq!(statistics.workloads[1].fps(), 50.0);
}

#[test]
fn failed_acquisitions_count_active_time_without_counting_frames() {
    let start = Instant::now();
    let mut statistics = Statistics::default();
    let at = |milliseconds| start + Duration::from_millis(milliseconds);
    statistics.record(Some(Workload::Pages), true, at(0), at(200), false);
    statistics.record(Some(Workload::Pages), false, at(1000), at(1100), false);
    statistics.record(Some(Workload::Pages), false, at(1110), at(1120), true);
    statistics.record(Some(Workload::Pages), false, at(1130), at(1230), false);
    statistics.record(Some(Workload::Pages), false, at(1240), at(1340), false);
    statistics.record(None, false, at(1400), at(1401), true);
    let pages = &statistics.workloads[0];
    assert_eq!(pages.samples, 1);
    assert_eq!(pages.mean_ms(), 10.0);
    assert_eq!(pages.p95_ms(), 10.5);
    assert!((pages.fps() - 1000.0 / 340.0).abs() < 0.001);
    statistics.record(Some(Workload::Pages), false, at(2000), at(2010), true);
    assert!((statistics.workloads[0].fps() - 2000.0 / 350.0).abs() < 0.001);
}

#[test]
fn percentile_includes_stalls_and_handles_histogram_overflow() {
    let mut statistics = WorkloadStatistics::default();
    for _ in 0..90 {
        statistics.record(Some(Duration::from_millis(10)), Duration::from_millis(12));
    }
    for _ in 0..10 {
        statistics.record(Some(Duration::from_millis(40)), Duration::from_millis(45));
    }
    assert_eq!(statistics.p95_ms(), 40.5);
    let mut stalled = WorkloadStatistics::default();
    stalled.record(Some(Duration::from_secs(1)), Duration::from_secs(1));
    assert_eq!(stalled.p95_ms(), 1000.0);
}

#[test]
fn final_pose_and_view_teardown_are_measured_once_and_pause_breaks_continuity() {
    let start = Instant::now();
    let at = |milliseconds| start + Duration::from_millis(milliseconds);
    let mut statistics = Statistics::default();
    for (workload, warming, begin, end) in [
        (Some(Workload::PowerOpening), true, 0, 10),
        (None, true, 20, 50),
        (Some(Workload::PowerOpening), false, 1000, 1010),
        (None, false, 1020, 1270),
        (None, false, 2000, 2010),
    ] {
        let measured = statistics.measure_motion(workload, warming);
        let (workload, warming) = measured.map_or((None, warming), |(workload, warming)| {
            (Some(workload), warming)
        });
        statistics.record(workload, warming, at(begin), at(end), true);
    }
    let power = &statistics.workloads[Workload::PowerOpening.index()];
    assert_eq!(power.samples, 2);
    assert_eq!(power.max_ms(), 250.0);
    assert!((power.fps() - 2000.0 / 270.0).abs() < 0.001);
    statistics.measure_motion(Some(Workload::Scroll), false);
    statistics.break_sequence();
    assert_eq!(statistics.measure_motion(None, false), None);
}

#[test]
fn failed_final_draw_keeps_settling_pending_until_success() {
    let start = Instant::now();
    let at = |milliseconds| start + Duration::from_millis(milliseconds);
    let mut statistics = Statistics::default();
    let motion = Some((Workload::PowerClosing, false));
    assert_eq!(
        statistics.measure_motion(Some(Workload::PowerClosing), false),
        motion
    );
    statistics.record(Some(Workload::PowerClosing), false, at(0), at(10), true);
    for (begin, end, success) in [(20, 120, false), (130, 230, false), (240, 260, true)] {
        assert_eq!(statistics.measure_motion(None, false), motion);
        statistics.record(
            Some(Workload::PowerClosing),
            false,
            at(begin),
            at(end),
            success,
        );
    }
    assert_eq!(statistics.measure_motion(None, false), None);
    let power = &statistics.workloads[Workload::PowerClosing.index()];
    assert_eq!(power.samples, 2);
    assert_eq!(power.max_ms(), 20.0);
    assert!((power.fps() - 2000.0 / 260.0).abs() < 0.001);
}
