use super::{HistogramPercentile, Statistics, BIN_MS};
use crate::animation::{Workload, RUN_CYCLES};
use crate::interface::APPS_PER_PAGE;
use crate::BenchmarkError;
use chrono::{DateTime, Local, Utc};
use serde::Serialize;
use serde_json::ser;
use std::env;
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Serialize)]
struct WorkloadReport {
    workload: &'static str,
    samples: u64,
    fps: f64,
    mean_ms: f64,
    p95_ms: f64,
    p95_is_lower_bound: bool,
    max_ms: f64,
    total_work_ms: f64,
    active_ms: f64,
}

#[derive(Serialize)]
struct RunReport {
    started_at: DateTime<Utc>,
    started_at_local: DateTime<Local>,
    completed_at: DateTime<Utc>,
    elapsed_seconds: f64,
    platform: &'static str,
    architecture: &'static str,
    width: u32,
    height: u32,
    vsync_requested: bool,
    redraw_overlay_used: bool,
    apps_per_page: usize,
    completed_cycles: u32,
    warmup_cycles: u32,
    histogram_bin_ms: f64,
    workloads: [WorkloadReport; Workload::ALL.len()],
}

fn write_report(path: &Path, contents: &[u8]) -> io::Result<()> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    let result = file.write_all(contents).and_then(|()| file.sync_all());
    drop(file);
    if result.is_err() {
        if let Err(error) = fs::remove_file(path) {
            tracing::warn!(path = %path.display(), %error, "Could not remove incomplete result");
        }
    }
    result
}

pub struct RunReportWriter {
    directory: PathBuf,
    started_at: DateTime<Local>,
    started: Instant,
}

impl RunReportWriter {
    pub fn new(directory: PathBuf, started: Instant) -> Self {
        Self {
            directory,
            started_at: Local::now(),
            started,
        }
    }

    pub fn restart(&mut self, started: Instant) {
        self.started_at = Local::now();
        self.started = started;
    }

    pub fn save(
        &self,
        statistics: &Statistics,
        vsync_requested: bool,
        size: (u32, u32),
    ) -> Result<(), BenchmarkError> {
        let report = RunReport {
            started_at: self.started_at.with_timezone(&Utc),
            started_at_local: self.started_at,
            completed_at: Utc::now(),
            elapsed_seconds: self.started.elapsed().as_secs_f64(),
            platform: env::consts::OS,
            architecture: env::consts::ARCH,
            width: size.0,
            height: size.1,
            vsync_requested,
            redraw_overlay_used: statistics.redraw_overlay_used,
            apps_per_page: APPS_PER_PAGE,
            completed_cycles: RUN_CYCLES,
            warmup_cycles: 1,
            histogram_bin_ms: BIN_MS,
            workloads: Workload::ALL.map(|workload| {
                let sample = &statistics.workloads[workload.index()];
                let (p95_ms, p95_is_lower_bound) = match sample.p95_ms() {
                    HistogramPercentile::Estimate(value) => (value, false),
                    HistogramPercentile::AtLeast(value) => (value, true),
                };
                WorkloadReport {
                    workload: workload.label(),
                    samples: sample.samples,
                    fps: sample.fps(),
                    mean_ms: sample.mean_ms(),
                    p95_ms,
                    p95_is_lower_bound,
                    max_ms: sample.max_ms(),
                    total_work_ms: sample.total_work_ms,
                    active_ms: sample.active_ms,
                }
            }),
        };
        let mut contents = ser::to_vec_pretty(&report)?;
        contents.push(b'\n');
        let filename = format!(
            "run_{}.json",
            self.started_at.format("%Y-%m-%d_%H-%M-%S_%9f")
        );
        let path = self.directory.join(filename);
        write_report(&path, &contents)?;
        tracing::info!("UI_BENCH_RESULT {}", path.display());
        Ok(())
    }
}
