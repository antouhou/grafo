use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Workload {
    Pages,
    Opening,
    Closing,
    Scroll,
    PowerOpening,
    PowerClosing,
}

impl Workload {
    pub const ALL: [Self; 6] = [
        Self::Pages,
        Self::Opening,
        Self::Closing,
        Self::Scroll,
        Self::PowerOpening,
        Self::PowerClosing,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Pages => "Pages",
            Self::Opening => "Settings open",
            Self::Closing => "Settings close",
            Self::Scroll => "Settings scroll",
            Self::PowerOpening => "Power open",
            Self::PowerClosing => "Power close",
        }
    }

    pub fn duration(self) -> f32 {
        match self {
            Self::Pages => 0.18,
            Self::Opening | Self::Closing => 0.28,
            Self::Scroll => 0.65,
            Self::PowerOpening | Self::PowerClosing => 0.24,
        }
    }

    pub fn index(self) -> usize {
        match self {
            Self::Pages => 0,
            Self::Opening => 1,
            Self::Closing => 2,
            Self::Scroll => 3,
            Self::PowerOpening => 4,
            Self::PowerClosing => 5,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scenario {
    Mixed,
    Pages,
    Settings,
    Scroll,
    Power,
}

impl Scenario {
    pub fn name(self) -> &'static str {
        match self {
            Self::Mixed => "AUTO",
            Self::Pages => "PAGES",
            Self::Settings => "SETTINGS",
            Self::Scroll => "SCROLL",
            Self::Power => "POWER",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Pose {
    pub page: f32,
    pub settings: f32,
    pub scroll: f32,
    pub power: f32,
}

impl Pose {
    const CLOSED: Self = Self {
        page: 0.0,
        settings: 0.0,
        scroll: 0.0,
        power: 0.0,
    };
    const OPEN: Self = Self {
        settings: 1.0,
        ..Self::CLOSED
    };
}

#[derive(Clone, Copy)]
struct Segment {
    from: Pose,
    to: Pose,
    duration: f64,
    workload: Option<Workload>,
}

impl Segment {
    const fn page(from: f32, to: f32) -> Self {
        Self {
            from: Pose {
                page: from,
                ..Pose::CLOSED
            },
            to: Pose {
                page: to,
                ..Pose::CLOSED
            },
            duration: 0.18,
            workload: Some(Workload::Pages),
        }
    }

    const fn settings(from: f32, to: f32, workload: Workload) -> Self {
        Self {
            from: Pose {
                page: 0.0,
                settings: from,
                ..Pose::CLOSED
            },
            to: Pose {
                page: 0.0,
                settings: to,
                ..Pose::CLOSED
            },
            duration: 0.28,
            workload: Some(workload),
        }
    }

    const fn hold(page: f32, settings: f32, duration: f64) -> Self {
        let pose = Pose {
            page,
            settings,
            ..Pose::CLOSED
        };
        Self {
            from: pose,
            to: pose,
            duration,
            workload: None,
        }
    }
}

const PAGE_SEQUENCE: [Segment; 8] = [
    Segment::page(0.0, 1.0),
    Segment::hold(1.0, 0.0, 0.32),
    Segment::page(1.0, 2.0),
    Segment::hold(2.0, 0.0, 0.32),
    Segment::page(2.0, 1.0),
    Segment::hold(1.0, 0.0, 0.32),
    Segment::page(1.0, 0.0),
    Segment::hold(0.0, 0.0, 0.32),
];

const SETTINGS_SEQUENCE: [Segment; 4] = [
    Segment::settings(0.0, 1.0, Workload::Opening),
    Segment::hold(0.0, 1.0, 0.62),
    Segment::settings(1.0, 0.0, Workload::Closing),
    Segment::hold(0.0, 0.0, 0.62),
];

const SCROLL_SEQUENCE: [Segment; 5] = [
    Segment::hold(0.0, 1.0, 0.5),
    Segment {
        from: Pose::OPEN,
        to: Pose {
            scroll: 1.0,
            ..Pose::OPEN
        },
        duration: 0.65,
        workload: Some(Workload::Scroll),
    },
    Segment {
        from: Pose {
            scroll: 1.0,
            ..Pose::OPEN
        },
        to: Pose {
            scroll: 1.0,
            ..Pose::OPEN
        },
        duration: 0.45,
        workload: None,
    },
    Segment {
        from: Pose {
            scroll: 1.0,
            ..Pose::OPEN
        },
        to: Pose::OPEN,
        duration: 0.65,
        workload: Some(Workload::Scroll),
    },
    Segment::hold(0.0, 1.0, 0.45),
];

const POWER_SEQUENCE: [Segment; 5] = [
    Segment::hold(0.0, 1.0, 0.5),
    Segment {
        from: Pose::OPEN,
        to: Pose {
            power: 1.0,
            ..Pose::OPEN
        },
        duration: 0.24,
        workload: Some(Workload::PowerOpening),
    },
    Segment {
        from: Pose {
            power: 1.0,
            ..Pose::OPEN
        },
        to: Pose {
            power: 1.0,
            ..Pose::OPEN
        },
        duration: 0.5,
        workload: None,
    },
    Segment {
        from: Pose {
            power: 1.0,
            ..Pose::OPEN
        },
        to: Pose::OPEN,
        duration: 0.24,
        workload: Some(Workload::PowerClosing),
    },
    Segment::hold(0.0, 1.0, 0.5),
];

const MIXED_FINISH: [Segment; 2] = [
    Segment::settings(1.0, 0.0, Workload::Closing),
    Segment::hold(0.0, 0.0, 0.62),
];

fn sequence(scenario: Scenario) -> impl Iterator<Item = &'static Segment> + Clone {
    let stages: [&[Segment]; 5] = match scenario {
        Scenario::Mixed => [
            &PAGE_SEQUENCE,
            &SETTINGS_SEQUENCE,
            &SCROLL_SEQUENCE,
            &POWER_SEQUENCE,
            &MIXED_FINISH,
        ],
        Scenario::Pages => [&PAGE_SEQUENCE, &[], &[], &[], &[]],
        Scenario::Settings => [&[], &SETTINGS_SEQUENCE, &[], &[], &[]],
        Scenario::Scroll => [&[], &[], &SCROLL_SEQUENCE, &[], &[]],
        Scenario::Power => [&[], &[], &[], &POWER_SEQUENCE, &[]],
    };
    stages.into_iter().flatten()
}

fn interpolate(from: Pose, to: Pose, progress: f32, workload: Option<Workload>) -> Pose {
    let progress = progress.clamp(0.0, 1.0);
    let eased = if matches!(
        workload,
        Some(Workload::PowerOpening | Workload::PowerClosing)
    ) {
        if progress < 0.5 {
            4.0 * progress.powi(3)
        } else {
            1.0 - (-2.0 * progress + 2.0).powi(3) / 2.0
        }
    } else {
        1.0 - (1.0 - progress).powi(3)
    };
    Pose {
        page: from.page + (to.page - from.page) * eased,
        settings: from.settings + (to.settings - from.settings) * eased,
        scroll: from.scroll + (to.scroll - from.scroll) * eased,
        power: from.power + (to.power - from.power) * eased,
    }
}

pub struct AnimationSample {
    pub mode_label: &'static str,
    pub pose: Pose,
    pub workload: Option<Workload>,
    pub is_warming_up: bool,
    pub next_transition: Option<Duration>,
}

struct ManualTransition {
    from: Pose,
    to: Pose,
    started: Instant,
    workload: Workload,
}

pub struct Animation {
    scenario: Scenario,
    started: Instant,
    paused_elapsed: Option<Duration>,
    manual: Option<ManualTransition>,
}

impl Animation {
    pub fn new(now: Instant) -> Self {
        Self {
            scenario: Scenario::Mixed,
            started: now,
            paused_elapsed: None,
            manual: None,
        }
    }

    pub fn scenario(&self) -> Scenario {
        self.scenario
    }

    pub fn is_paused(&self) -> bool {
        self.paused_elapsed.is_some()
    }

    pub fn select(&mut self, scenario: Scenario, now: Instant) {
        *self = Self {
            scenario,
            ..Self::new(now)
        };
    }

    pub fn toggle_pause(&mut self, now: Instant) {
        if self.manual.is_some() {
            self.select(self.scenario, now);
        } else if let Some(elapsed) = self.paused_elapsed.take() {
            self.started = now - elapsed;
        } else {
            self.paused_elapsed = Some(now.duration_since(self.started));
        }
    }

    pub fn change_page(&mut self, direction: f32, now: Instant) {
        let from = self.sample(now).pose;
        let to = Pose {
            page: (from.page.round() + direction).clamp(0.0, 2.0),
            ..Pose::CLOSED
        };
        self.manual = Some(ManualTransition {
            from,
            to,
            started: now,
            workload: Workload::Pages,
        });
        self.paused_elapsed = None;
    }

    fn set_settings(&mut self, opening: bool, now: Instant) {
        let from = self.sample(now).pose;
        self.manual = Some(ManualTransition {
            from,
            to: Pose {
                page: from.page,
                settings: if opening { 1.0 } else { 0.0 },
                ..from
            },
            started: now,
            workload: if opening {
                Workload::Opening
            } else {
                Workload::Closing
            },
        });
        self.paused_elapsed = None;
    }

    pub fn toggle_settings(&mut self, now: Instant) {
        let opening = self.sample(now).pose.settings < 0.5;
        self.set_settings(opening, now);
    }

    pub fn close_settings(&mut self, now: Instant) {
        self.set_settings(false, now);
    }

    pub fn scroll_by(&mut self, distance: f32, now: Instant) {
        let from = self.sample(now).pose;
        self.manual = Some(ManualTransition {
            from: Pose {
                settings: 1.0,
                power: 0.0,
                ..from
            },
            to: Pose {
                settings: 1.0,
                power: 0.0,
                scroll: (from.scroll + distance).clamp(0.0, 1.0),
                ..from
            },
            started: now,
            workload: Workload::Scroll,
        });
        self.paused_elapsed = None;
    }

    fn set_power(&mut self, opening: bool, now: Instant) {
        let from = self.sample(now).pose;
        self.manual = Some(ManualTransition {
            from: Pose {
                settings: 1.0,
                ..from
            },
            to: Pose {
                settings: 1.0,
                power: if opening { 1.0 } else { 0.0 },
                ..from
            },
            started: now,
            workload: if opening {
                Workload::PowerOpening
            } else {
                Workload::PowerClosing
            },
        });
        self.paused_elapsed = None;
    }

    pub fn toggle_power(&mut self, now: Instant) {
        self.set_power(self.sample(now).pose.power < 0.5, now);
    }

    pub fn close_power(&mut self, now: Instant) {
        self.set_power(false, now);
    }

    pub fn sample(&self, now: Instant) -> AnimationSample {
        if let Some(manual) = &self.manual {
            let duration = manual.workload.duration();
            let progress = now.duration_since(manual.started).as_secs_f32() / duration;
            return AnimationSample {
                mode_label: "MANUAL",
                pose: interpolate(manual.from, manual.to, progress, Some(manual.workload)),
                workload: (progress < 1.0).then_some(manual.workload),
                is_warming_up: false,
                next_transition: None,
            };
        }
        let elapsed = self
            .paused_elapsed
            .unwrap_or_else(|| now.duration_since(self.started))
            .as_secs_f64();
        let segments = sequence(self.scenario);
        let total: f64 = segments.clone().map(|segment| segment.duration).sum();
        let mut position = elapsed % total;
        for segment in segments {
            if position < segment.duration {
                let is_paused = self.is_paused();
                return AnimationSample {
                    mode_label: if is_paused {
                        "PAUSED"
                    } else {
                        self.scenario.name()
                    },
                    pose: interpolate(
                        segment.from,
                        segment.to,
                        (position / segment.duration) as f32,
                        segment.workload,
                    ),
                    workload: segment.workload.filter(|_| !is_paused),
                    is_warming_up: elapsed < total,
                    next_transition: if segment.workload.is_none() && !is_paused {
                        Some(Duration::from_secs_f64(segment.duration - position))
                    } else {
                        None
                    },
                };
            }
            position -= segment.duration;
        }
        unreachable!("the cycle position belongs to a segment")
    }
}

#[cfg(test)]
mod tests;
