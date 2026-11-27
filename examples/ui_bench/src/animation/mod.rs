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
    const SECOND_PAGE: Self = Self {
        page: 1.0,
        ..Self::CLOSED
    };
    const THIRD_PAGE: Self = Self {
        page: 2.0,
        ..Self::CLOSED
    };
    const SCROLLED: Self = Self {
        scroll: 1.0,
        ..Self::OPEN
    };
    const POWER_MENU: Self = Self {
        power: 1.0,
        ..Self::OPEN
    };
}

#[derive(Clone, Copy)]
struct Segment {
    from: Pose,
    to: Pose,
    duration: f64,
    workload: Workload,
}

impl Segment {
    const fn motion(from: Pose, to: Pose, duration: f64, workload: Workload) -> Self {
        Self {
            from,
            to,
            duration,
            workload,
        }
    }
}

const SEQUENCE: [Segment; 10] = [
    Segment::motion(Pose::CLOSED, Pose::SECOND_PAGE, 0.18, Workload::Pages),
    Segment::motion(Pose::SECOND_PAGE, Pose::THIRD_PAGE, 0.18, Workload::Pages),
    Segment::motion(Pose::THIRD_PAGE, Pose::SECOND_PAGE, 0.18, Workload::Pages),
    Segment::motion(Pose::SECOND_PAGE, Pose::CLOSED, 0.18, Workload::Pages),
    Segment::motion(Pose::CLOSED, Pose::OPEN, 0.28, Workload::Opening),
    Segment::motion(Pose::OPEN, Pose::SCROLLED, 0.65, Workload::Scroll),
    Segment::motion(Pose::SCROLLED, Pose::OPEN, 0.65, Workload::Scroll),
    Segment::motion(Pose::OPEN, Pose::POWER_MENU, 0.24, Workload::PowerOpening),
    Segment::motion(Pose::POWER_MENU, Pose::OPEN, 0.24, Workload::PowerClosing),
    Segment::motion(Pose::OPEN, Pose::CLOSED, 0.28, Workload::Closing),
];

fn interpolate(from: Pose, to: Pose, progress: f32, workload: Workload) -> Pose {
    let eased = if matches!(workload, Workload::PowerOpening | Workload::PowerClosing) {
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
}

pub struct Animation {
    motion_started: Instant,
    motion_index: usize,
    completed_cycles: u32,
    stopped_elapsed: Option<Duration>,
}

impl Animation {
    pub fn new(now: Instant) -> Self {
        Self {
            motion_started: now,
            motion_index: 0,
            completed_cycles: 0,
            stopped_elapsed: None,
        }
    }

    pub fn is_running(&self) -> bool {
        self.stopped_elapsed.is_none() && self.completed_cycles < RUN_CYCLES
    }

    pub fn stop(&mut self, now: Instant) {
        self.stopped_elapsed
            .get_or_insert(now.duration_since(self.motion_started));
    }

    /// Advance after a successful endpoint render and report whether a cycle completed.
    pub fn rendered(&mut self, sampled_at: Instant) -> bool {
        if !self.is_running()
            || sampled_at.duration_since(self.motion_started)
                < Duration::from_secs_f64(SEQUENCE[self.motion_index].duration)
        {
            return false;
        }
        self.motion_started = sampled_at;
        self.motion_index += 1;
        if self.motion_index == SEQUENCE.len() {
            self.motion_index = 0;
            self.completed_cycles += 1;
            return true;
        }
        false
    }

    pub fn sample(&self, now: Instant) -> AnimationSample {
        if self.completed_cycles == RUN_CYCLES {
            return AnimationSample {
                status: RunStatus::Complete,
                completed_cycles: RUN_CYCLES,
                pose: Pose::CLOSED,
                workload: None,
                is_warming_up: false,
            };
        }
        let segment = &SEQUENCE[self.motion_index];
        let elapsed = self
            .stopped_elapsed
            .unwrap_or_else(|| now.duration_since(self.motion_started));
        let is_running = self.is_running();
        let has_completed = elapsed >= Duration::from_secs_f64(segment.duration);
        AnimationSample {
            status: if is_running {
                RunStatus::Running
            } else {
                RunStatus::Stopped
            },
            completed_cycles: self.completed_cycles,
            pose: if has_completed {
                segment.to
            } else {
                interpolate(
                    segment.from,
                    segment.to,
                    elapsed.as_secs_f32() / segment.duration as f32,
                    segment.workload,
                )
            },
            workload: is_running.then_some(segment.workload),
            is_warming_up: is_running && self.completed_cycles == 0,
        }
    }
}
