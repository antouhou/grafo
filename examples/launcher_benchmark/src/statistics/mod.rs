use crate::animation::Workload;
use std::time::{Duration, Instant};

const HISTOGRAM_BINS: usize = 512;
const BIN_MS: f64 = 0.5;

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

    pub fn p95_ms(&self) -> f64 {
        if self.samples == 0 {
            return 0.0;
        }
        let threshold = (self.samples as f64 * 0.95).ceil() as u64;
        let mut accumulated = 0;
        for (index, count) in self.histogram.iter().enumerate() {
            accumulated += count;
            if accumulated >= threshold {
                return if index == HISTOGRAM_BINS - 1 {
                    self.maximum_work_ms
                } else {
                    (index + 1) as f64 * BIN_MS
                };
            }
        }
        self.maximum_work_ms
    }
}

#[derive(Default)]
pub struct Statistics {
    pub workloads: [WorkloadStatistics; Workload::ALL.len()],
    previous_completion: Option<(Workload, Instant)>,
    last_motion: Option<(Workload, bool)>,
    is_settling: bool,
}

impl Statistics {
    /// Include the first stationary draw that applies the final pose and removes the outgoing view.
    pub fn measure_motion(
        &mut self,
        workload: Option<Workload>,
        is_warming_up: bool,
    ) -> Option<(Workload, bool)> {
        let current = workload.map(|workload| (workload, is_warming_up));
        self.is_settling = current.is_none() && self.last_motion.is_some();
        if current.is_some() {
            self.last_motion = current;
        }
        self.last_motion
    }

    pub fn record(
        &mut self,
        workload: Option<Workload>,
        is_warming_up: bool,
        started: Instant,
        completed: Instant,
        was_rendered: bool,
    ) {
        if self.is_settling && was_rendered {
            self.last_motion = None;
        }
        let Some(workload) = workload.filter(|_| !is_warming_up) else {
            self.previous_completion = None;
            return;
        };
        let active_start = self
            .previous_completion
            .filter(|(previous, _)| *previous == workload)
            .map_or(started, |(_, completion)| completion);
        self.workloads[workload.index()].record(
            was_rendered.then(|| completed.duration_since(started)),
            completed.duration_since(active_start),
        );
        self.previous_completion = Some((workload, completed));
    }

    pub fn break_sequence(&mut self) {
        self.previous_completion = None;
        self.last_motion = None;
        self.is_settling = false;
    }
}
