use super::WgpuBackend;
use std::time::Duration;

/// Shape mask effect cache activity during one frame.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ShapeMaskEffectCacheMetrics {
    pub hits: u64,
    pub misses: u64,
    pub generated_masks: u64,
    pub mask_hits: u64,
    pub executed_passes: u64,
    pub collected_results: u64,
    pub collected_masks: u64,
}

/// Per-frame pipeline switch counts for diagnosing GPU state-change overhead.
#[derive(Debug, Clone, Copy, Default)]
pub struct PipelineSwitchCounts {
    /// Total `set_pipeline` calls.
    pub total_switches: u32,
}

impl PipelineSwitchCounts {
    /// Merge another frame's counts into this accumulator.
    pub fn accumulate(&mut self, other: &Self) {
        self.total_switches += other.total_switches;
    }
}

/// WGPU execution timings for a single render, excluding scene planning.
///
/// Provides wall-clock durations for each backend phase.
/// Available when the `render_metrics` feature is enabled.
#[derive(Debug, Clone, Copy, Default)]
pub struct PhaseTimings {
    /// Time spent preparing and uploading geometry buffers to the GPU.
    pub prepare: Duration,
    /// Time spent acquiring the output and encoding and submitting GPU commands.
    pub encode_and_submit: Duration,
    /// Time spent presenting, or mapping, waiting for, and copying offscreen pixels.
    pub present_or_readback: Duration,
    /// Time spent waiting for outstanding GPU work after presentation.
    /// GPU work can also run during earlier phases, so this is only the remaining wait.
    pub gpu_wait: Duration,
    /// Sum of all phases, including the GPU wait.
    pub total: Duration,
}

#[doc(hidden)]
impl WgpuBackend {
    /// Returns CPU time spent updating the retained scene image.
    /// Includes image allocation, render setup, command execution, submission,
    /// and composite and effect uploads performed during execution.
    /// Excludes scene planning, geometry uploads before execution, presentation,
    /// readback and explicit GPU waits after submission.
    pub fn last_retained_output_update_cpu_time(&self) -> Duration {
        self.last_retained_output_update_cpu_time
    }

    /// Returns the WGPU phase timings for the most recently completed render.
    pub fn last_phase_timings(&self) -> PhaseTimings {
        self.last_phase_timings
    }

    /// Returns pipeline bindings from the last render.
    pub fn last_pipeline_switch_counts(&self) -> PipelineSwitchCounts {
        self.resources.pipeline_switch_counts
    }

    /// Returns cached shape mask effect activity for the most recently rendered frame.
    pub fn last_shape_mask_effect_cache_metrics(&self) -> ShapeMaskEffectCacheMetrics {
        self.resources.shape_mask_effect_cache_metrics
    }
}
