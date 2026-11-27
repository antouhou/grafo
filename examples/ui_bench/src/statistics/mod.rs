use crate::animation::Workload;
use std::{
    fmt,
    time::{Duration, Instant},
};

pub mod report;

const HISTOGRAM_BINS: usize = 512;
const BIN_MS: f64 = 0.5;

pub enum HistogramPercentile {
    Estimate(f64),
    AtLeast(f64),
}

impl fmt::Display for HistogramPercentile {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Estimate(milliseconds) => write!(formatter, "{milliseconds:.1}"),
            Self::AtLeast(milliseconds) => write!(formatter, ">={milliseconds:.1}"),
        }
    }
}

pub struct WorkloadStatistics {
    pub samples: u64,
    pub total_work_ms: f64,
    active_ms: f64,
    maximum_work_ms: f64,
    histogram: [u64; HISTOGRAM_BINS],
}

impl Default for WorkloadStatistics {
    fn default() -> Self {
        Self {
            samples: 0,
            total_work_ms: 0.0,
            active_ms: 0.0,
            maximum_work_ms: 0.0,
            histogram: [0; HISTOGRAM_BINS],
        }
    }
}

impl WorkloadStatistics {
    fn record(&mut self, work: Option<Duration>, active: Duration) {
        self.active_ms += active.as_secs_f64() * 1000.0;
        let Some(work) = work else {
            return;
        };
        let milliseconds = work.as_secs_f64() * 1000.0;
        self.samples += 1;
        self.total_work_ms += milliseconds;
        self.maximum_work_ms = self.maximum_work_ms.max(milliseconds);
        let bucket = ((milliseconds / BIN_MS) as usize).min(HISTOGRAM_BINS - 1);
        self.histogram[bucket] += 1;
    }

    pub fn fps(&self) -> f64 {
        if self.active_ms == 0.0 {
            0.0
        } else {
            self.samples as f64 * 1000.0 / self.active_ms
        }
    }

    pub fn mean_ms(&self) -> f64 {
        if self.samples == 0 {
            0.0
        } else {
            self.total_work_ms / self.samples as f64
        }
    }

    pub fn max_ms(&self) -> f64 {
        self.maximum_work_ms
    }

    pub fn p95_ms(&self) -> HistogramPercentile {
        if self.samples == 0 {
            return HistogramPercentile::Estimate(0.0);
        }
        let threshold = (self.samples as f64 * 0.95).ceil() as u64;
        let mut accumulated = 0;
        for (index, count) in self.histogram[..HISTOGRAM_BINS - 1].iter().enumerate() {
            accumulated += count;
            if accumulated >= threshold {
                return HistogramPercentile::Estimate((index + 1) as f64 * BIN_MS);
            }
        }
        HistogramPercentile::AtLeast((HISTOGRAM_BINS - 1) as f64 * BIN_MS)
    }
}

#[derive(Default)]
pub struct Statistics {
    pub workloads: [WorkloadStatistics; Workload::ALL.len()],
    pub redraw_overlay_used: bool,
    previous_completion: Option<Instant>,
}

impl Statistics {
    pub fn record(
        &mut self,
        workload: Option<Workload>,
        is_warming_up: bool,
        started: Instant,
        completed: Instant,
        was_rendered: bool,
        is_overlay_enabled: bool,
    ) {
        let Some(workload) = workload.filter(|_| !is_warming_up) else {
            self.previous_completion = None;
            return;
        };
        let active_start = self.previous_completion.unwrap_or(started);
        self.redraw_overlay_used |= was_rendered && is_overlay_enabled;
        self.workloads[workload.index()].record(
            was_rendered.then(|| completed.duration_since(started)),
            completed.duration_since(active_start),
        );
        self.previous_completion = Some(completed);
    }

    pub fn break_sequence(&mut self) {
        self.previous_completion = None;
    }
}
