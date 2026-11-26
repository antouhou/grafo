use super::{sequence, Animation, Scenario, Workload};
use std::time::{Duration, Instant};

#[test]
fn scenarios_warm_up_then_measure_each_transition_and_wait_during_holds() {
    let start = Instant::now();
    for scenario in [
        Scenario::Mixed,
        Scenario::Pages,
        Scenario::Settings,
        Scenario::Scroll,
        Scenario::Power,
    ] {
        let mut animation = Animation::new(start);
        animation.select(scenario, start);
        let cycle: f64 = sequence(scenario).map(|segment| segment.duration).sum();
        let mut offset = 0.0;
        let first = sequence(scenario).next().unwrap();
        let last = sequence(scenario).last().unwrap();
        assert_eq!(last.to, first.from, "cycle boundary for {scenario:?}");
        for segment in sequence(scenario) {
            let middle = offset + segment.duration * 0.5;
            let warmup = animation.sample(start + Duration::from_secs_f64(middle));
            let measured = animation.sample(start + Duration::from_secs_f64(cycle + middle));
            assert!(warmup.is_warming_up);
            assert!(!measured.is_warming_up);
            assert_eq!(measured.workload, segment.workload);
            if segment.workload.is_some() {
                assert_ne!(measured.pose, segment.from);
                assert_ne!(measured.pose, segment.to);
                assert!(measured.next_transition.is_none());
            } else {
                assert!(measured.next_transition.is_some());
            }
            offset += segment.duration;
        }
    }
}

#[test]
fn pausing_preserves_motion_and_manual_actions_finish_without_a_redraw_loop() {
    let start = Instant::now();
    let mut animation = Animation::new(start);
    let moment = start + Duration::from_millis(80);
    let before = animation.sample(moment).pose;
    animation.toggle_pause(moment);
    let paused = animation.sample(start + Duration::from_secs(60));
    assert_eq!(paused.pose, before);
    assert!(paused.workload.is_none() && paused.next_transition.is_none());
    animation.toggle_pause(start + Duration::from_secs(60));
    assert_eq!(
        animation.sample(start + Duration::from_secs(60)).pose,
        before
    );
    animation.toggle_settings(start + Duration::from_secs(61));
    let done = animation.sample(start + Duration::from_secs(62));
    assert_eq!(done.pose.settings, 1.0);
    assert!(done.workload.is_none() && done.next_transition.is_none());
    animation.select(Scenario::Pages, start + Duration::from_secs(63));
    assert_eq!(
        animation.sample(start + Duration::from_secs(63)).mode_label,
        "PAGES"
    );
    assert!(
        animation
            .sample(start + Duration::from_secs(63))
            .is_warming_up
    );
}

#[test]
fn closing_settings_reverses_even_the_start_of_an_opening_transition() {
    let start = Instant::now();
    let mut animation = Animation::new(start);
    animation.toggle_settings(start);
    let closing = start + Duration::from_millis(20);
    let before = animation.sample(closing).pose.settings;
    assert!(before > 0.0 && before < 0.5);
    animation.close_settings(closing);
    let during = animation.sample(closing + Duration::from_millis(100));
    assert_eq!(during.workload, Some(Workload::Closing));
    assert!(during.pose.settings < before);
    let done = animation.sample(closing + Duration::from_millis(300));
    assert_eq!(done.pose.settings, 0.0);
    assert!(done.workload.is_none() && done.next_transition.is_none());
}

#[test]
fn scrolling_and_power_reversal_preserve_the_current_pose() {
    let start = Instant::now();
    let mut animation = Animation::new(start);
    animation.scroll_by(1.0, start);
    let moment = start + Duration::from_millis(200);
    let scrolled = animation.sample(moment).pose;
    assert!(scrolled.scroll > 0.0 && scrolled.scroll < 1.0);
    animation.toggle_power(moment);
    assert_eq!(animation.sample(moment).pose, scrolled);
    let reverse = moment + Duration::from_millis(60);
    let opening = animation.sample(reverse).pose;
    animation.close_power(reverse);
    assert_eq!(animation.sample(reverse).pose, opening);
    let done = animation.sample(reverse + Duration::from_millis(300));
    assert_eq!(done.pose.power, 0.0);
    assert_eq!(done.pose.scroll, scrolled.scroll);
    assert_eq!(done.pose.settings, 1.0);
    assert!(done.workload.is_none() && done.next_transition.is_none());
}
