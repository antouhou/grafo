use std::time::{Duration, Instant};

pub const RUN_CYCLES: u32 = 10;

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
pub enum RunStatus {
    Running,
    Stopped,
    Complete,
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

    const fn hold(pose: Pose, duration: f64) -> Self {
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
    Segment::hold(
        Pose {
            page: 1.0,
            ..Pose::CLOSED
        },
        0.32,
    ),
    Segment::page(1.0, 2.0),
    Segment::hold(
        Pose {
            page: 2.0,
            ..Pose::CLOSED
        },
        0.32,
    ),
    Segment::page(2.0, 1.0),
    Segment::hold(
        Pose {
            page: 1.0,
            ..Pose::CLOSED
        },
        0.32,
    ),
    Segment::page(1.0, 0.0),
    Segment::hold(Pose::CLOSED, 0.32),
];

const SETTINGS_SEQUENCE: [Segment; 4] = [
    Segment::settings(0.0, 1.0, Workload::Opening),
    Segment::hold(Pose::OPEN, 0.62),
    Segment::settings(1.0, 0.0, Workload::Closing),
    Segment::hold(Pose::CLOSED, 0.62),
];

const SCROLL_SEQUENCE: [Segment; 5] = [
    Segment::hold(Pose::OPEN, 0.5),
    Segment {
        from: Pose::OPEN,
        to: Pose {
            scroll: 1.0,
            ..Pose::OPEN
        },
        duration: 0.65,
        workload: Some(Workload::Scroll),
    },
    Segment::hold(
        Pose {
            scroll: 1.0,
            ..Pose::OPEN
        },
        0.45,
    ),
    Segment {
        from: Pose {
            scroll: 1.0,
            ..Pose::OPEN
        },
        to: Pose::OPEN,
        duration: 0.65,
        workload: Some(Workload::Scroll),
    },
    Segment::hold(Pose::OPEN, 0.45),
];

const POWER_SEQUENCE: [Segment; 5] = [
    Segment::hold(Pose::OPEN, 0.5),
    Segment {
        from: Pose::OPEN,
        to: Pose {
            power: 1.0,
            ..Pose::OPEN
        },
        duration: 0.24,
        workload: Some(Workload::PowerOpening),
    },
    Segment::hold(
        Pose {
            power: 1.0,
            ..Pose::OPEN
        },
        0.5,
    ),
    Segment {
        from: Pose {
            power: 1.0,
            ..Pose::OPEN
        },
        to: Pose::OPEN,
        duration: 0.24,
        workload: Some(Workload::PowerClosing),
    },
    Segment::hold(Pose::OPEN, 0.5),
];

const MIXED_FINISH: [Segment; 2] = [
    Segment::settings(1.0, 0.0, Workload::Closing),
    Segment::hold(Pose::CLOSED, 0.62),
];

fn sequence() -> impl Iterator<Item = &'static Segment> + Clone {
    let stages: [&[Segment]; 5] = [
        &PAGE_SEQUENCE,
        &SETTINGS_SEQUENCE,
        &SCROLL_SEQUENCE,
        &POWER_SEQUENCE,
        &MIXED_FINISH,
    ];
    stages.into_iter().flatten()
}

fn cycle_duration() -> Duration {
    sequence()
        .map(|segment| Duration::from_secs_f64(segment.duration))
        .sum()
}

fn interpolate(from: Pose, to: Pose, progress: f32, workload: Option<Workload>) -> Pose {
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
    pub status: RunStatus,
    pub completed_cycles: u32,
    pub pose: Pose,
    pub workload: Option<Workload>,
    pub is_warming_up: bool,
    pub next_transition: Option<Duration>,
}

pub struct Animation {
    started: Instant,
    stopped_elapsed: Option<Duration>,
}

impl Animation {
    pub fn new(now: Instant) -> Self {
        Self {
            started: now,
            stopped_elapsed: None,
        }
    }

    pub fn is_running(&self) -> bool {
        self.stopped_elapsed.is_none()
    }

    pub fn stop(&mut self, now: Instant) {
        self.stopped_elapsed
            .get_or_insert(now.duration_since(self.started));
    }

    pub fn sample(&mut self, now: Instant) -> AnimationSample {
        let elapsed = self
            .stopped_elapsed
            .unwrap_or_else(|| now.duration_since(self.started));
        let cycle_duration = cycle_duration();
        let run_duration = cycle_duration * RUN_CYCLES;
        if elapsed >= run_duration {
            self.stopped_elapsed = Some(run_duration);
            return AnimationSample {
                status: RunStatus::Complete,
                completed_cycles: RUN_CYCLES,
                pose: Pose::CLOSED,
                workload: None,
                is_warming_up: false,
                next_transition: None,
            };
        }
        let completed_cycles = (elapsed.as_nanos() / cycle_duration.as_nanos()) as u32;
        let mut position = elapsed - cycle_duration * completed_cycles;
        for segment in sequence() {
            let duration = Duration::from_secs_f64(segment.duration);
            if position < duration {
                let is_running = self.is_running();
                return AnimationSample {
                    status: if is_running {
                        RunStatus::Running
                    } else {
                        RunStatus::Stopped
                    },
                    completed_cycles,
                    pose: interpolate(
                        segment.from,
                        segment.to,
                        position.as_secs_f32() / duration.as_secs_f32(),
                        segment.workload,
                    ),
                    workload: segment.workload.filter(|_| is_running),
                    is_warming_up: is_running && elapsed < cycle_duration,
                    next_transition: if segment.workload.is_none() && is_running {
                        Some(duration - position)
                    } else {
                        None
                    },
                };
            }
            position -= duration;
        }
        unreachable!("the cycle position belongs to a segment")
    }
}
