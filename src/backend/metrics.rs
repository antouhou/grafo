use std::time::Duration;

/// Shape-effect cache activity during one frame.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ShapeEffectCacheMetrics {
    pub hits: u64,
    pub misses: u64,
    pub generated_masks: u64,
    pub mask_hits: u64,
    pub executed_passes: u64,
    pub collected_results: u64,
    pub collected_masks: u64,
}

/// Per-frame pipeline switch counts for diagnosing GPU state-change overhead.
///
/// Each field counts how many times the corresponding `set_pipeline` call was issued
/// during a single frame. `scissor_clips` counts how many times a scissor rect was
/// used *instead* of a stencil increment/decrement pair. `stencil_passes` counts
/// actual indexed draws that modify the stencil buffer.
#[derive(Debug, Clone, Copy, Default)]
pub struct PipelineSwitchCounts {
    /// Number of switches to the stencil-increment pipeline.
    pub to_stencil_increment: u32,
    /// Number of switches to the stencil-decrement pipeline.
    pub to_stencil_decrement: u32,
    /// Number of switches to the leaf-draw pipeline.
    pub to_leaf_draw: u32,
    /// Number of switches to the effect composite pipeline, which resets tracking.
    pub to_composite: u32,
    /// Total `set_pipeline` calls.
    pub total_switches: u32,
    /// Number of parent shapes clipped via scissor rect instead of stencil.
    pub scissor_clips: u32,
    /// Number of stencil-modifying draw passes.
    pub stencil_passes: u32,
}

impl PipelineSwitchCounts {
    /// Merge another frame's counts into this accumulator.
    pub fn accumulate(&mut self, other: &Self) {
        self.to_stencil_increment += other.to_stencil_increment;
        self.to_stencil_decrement += other.to_stencil_decrement;
        self.to_leaf_draw += other.to_leaf_draw;
        self.to_composite += other.to_composite;
        self.total_switches += other.total_switches;
        self.scissor_clips += other.scissor_clips;
        self.stencil_passes += other.stencil_passes;
    }
}

/// Per-phase timing breakdown for a single frame.
///
/// Provides wall-clock durations for each phase of the render loop.
/// Available when the `render_metrics` feature is enabled.
#[derive(Debug, Clone, Copy, Default)]
pub struct PhaseTimings {
    /// Time spent compiling commands and uploading geometry buffers to the GPU.
    pub prepare: Duration,
    /// Time spent in `render_to_texture_view` and `queue.submit`.
    pub encode_and_submit: Duration,
    /// Time spent presenting, or mapping, waiting for, and copying offscreen pixels.
    pub present_or_readback: Duration,
    /// Time spent waiting for outstanding GPU work after presentation.
    /// GPU work can also run during earlier phases, so this is only the remaining wait.
    pub gpu_wait: Duration,
    /// Sum of all phases, including the GPU wait.
    pub total: Duration,
}
