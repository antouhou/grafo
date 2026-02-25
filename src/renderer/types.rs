use std::sync::Arc;

use ahash::{HashMap, HashMapExt};

use crate::effect::{self, EffectInstance, LoadedEffect};
use crate::shape::{CachedShapeDrawData, DrawShapeCommand, ShapeDrawData};
use crate::texture_manager::TextureManager;
use crate::vertex::InstanceTransform;

use super::traversal::TraversalScratch;

#[derive(Debug)]
pub(super) enum DrawCommand {
    Shape(ShapeDrawData),
    CachedShape(CachedShapeDrawData),
}

impl DrawCommand {
    /// Whether this node is a leaf (has no children in the draw tree).
    /// Starts as `true`; set to `false` when a child is added.
    pub(super) fn is_leaf(&self) -> bool {
        match self {
            DrawCommand::Shape(s) => s.is_leaf,
            DrawCommand::CachedShape(s) => s.is_leaf,
        }
    }

    pub(super) fn set_not_leaf(&mut self) {
        match self {
            DrawCommand::Shape(s) => s.is_leaf = false,
            DrawCommand::CachedShape(s) => s.is_leaf = false,
        }
    }
}

impl DrawCommand {
    pub(super) fn set_transform(&mut self, transform: InstanceTransform) {
        match self {
            DrawCommand::Shape(shape) => shape.set_transform(transform),
            DrawCommand::CachedShape(cached_shape) => cached_shape.set_transform(transform),
        }
    }

    pub(super) fn set_texture_id(&mut self, layer: usize, texture_id: Option<u64>) {
        match self {
            DrawCommand::Shape(shape) => shape.set_texture_id(layer, texture_id),
            DrawCommand::CachedShape(cached_shape) => {
                cached_shape.set_texture_id(layer, texture_id)
            }
        }
    }

    pub(super) fn set_instance_color_override(&mut self, color: Option<[f32; 4]>) {
        match self {
            DrawCommand::Shape(shape) => shape.set_instance_color_override(color),
            DrawCommand::CachedShape(cached_shape) => {
                cached_shape.set_instance_color_override(color)
            }
        }
    }

    pub(super) fn set_clips_children(&mut self, clips: bool) {
        match self {
            DrawCommand::Shape(shape) => shape.set_clips_children(clips),
            DrawCommand::CachedShape(cached_shape) => cached_shape.set_clips_children(clips),
        }
    }

    pub(super) fn clips_children(&self) -> bool {
        match self {
            DrawCommand::Shape(shape) => shape.clips_children(),
            DrawCommand::CachedShape(cached_shape) => cached_shape.clips_children(),
        }
    }

    pub(super) fn is_rect(&self) -> bool {
        match self {
            DrawCommand::Shape(shape) => shape.is_rect(),
            DrawCommand::CachedShape(cached_shape) => cached_shape.is_rect(),
        }
    }

    pub(super) fn rect_bounds(&self) -> Option<[(f32, f32); 2]> {
        match self {
            DrawCommand::Shape(shape) => shape.rect_bounds(),
            DrawCommand::CachedShape(cached_shape) => cached_shape.rect_bounds(),
        }
    }

    pub(super) fn clear_frame_state(&mut self) {
        match self {
            DrawCommand::Shape(shape) => {
                shape.index_buffer_range = None;
                shape.stencil_ref = None;
            }
            DrawCommand::CachedShape(cached_shape) => {
                cached_shape.index_buffer_range = None;
                cached_shape.stencil_ref = None;
            }
        }
    }
}

#[derive(Debug)]
pub(super) enum TraversalEvent {
    Pre(usize),
    Post(usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Pipeline {
    None,
    StencilIncrement,
    StencilDecrement,
    LeafDraw,
}

/// Wraps [`Pipeline`] tracking with optional per-frame switch counters.
///
/// When the `render_metrics` feature is enabled the tracker also counts
/// every GPU `set_pipeline` call and every scissor-clip substitution so
/// the numbers can be queried after the frame.
pub(super) struct PipelineTracker {
    pub(super) current: Pipeline,
    #[cfg(feature = "render_metrics")]
    pub(super) counts: super::metrics::PipelineSwitchCounts,
}

impl PipelineTracker {
    pub(super) fn new() -> Self {
        Self {
            current: Pipeline::None,
            #[cfg(feature = "render_metrics")]
            counts: super::metrics::PipelineSwitchCounts::default(),
        }
    }

    /// Record a real GPU pipeline switch (only when the pipeline actually changes).
    pub(super) fn switch_to(&mut self, pipeline: Pipeline) {
        self.current = pipeline;
        #[cfg(feature = "render_metrics")]
        {
            self.counts.total_switches += 1;
            match pipeline {
                Pipeline::StencilIncrement => self.counts.to_stencil_increment += 1,
                Pipeline::StencilDecrement => self.counts.to_stencil_decrement += 1,
                Pipeline::LeafDraw => self.counts.to_leaf_draw += 1,
                Pipeline::None => self.counts.to_composite += 1,
            }
        }
    }

    /// Record that a scissor clip was used instead of stencil increment/decrement.
    #[cfg(feature = "render_metrics")]
    pub(super) fn record_scissor_clip(&mut self) {
        self.counts.scissor_clips += 1;
    }
}

/// Tracks the currently-bound texture bind groups to skip redundant `set_bind_group` calls.
///
/// Each layer stores the texture ID that is currently bound (`Some(id)` for a real texture,
/// `None` for the default white texture). The outer `Option` distinguishes "unknown / first
/// use" (`None`) from a known state (`Some(..)`).
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct BoundTextureState {
    layers: [Option<Option<u64>>; 2],
}

impl BoundTextureState {
    /// Reset both layers to "unknown" — forces the next bind call to actually issue
    /// `set_bind_group`. Call this whenever a pipeline switch resets bind group state.
    pub(super) fn invalidate(&mut self) {
        self.layers = [None, None];
    }

    /// Returns `true` when the given texture id (or `None` for the default) is not
    /// already bound on `layer`, and updates the tracked state.
    pub(super) fn needs_rebind(&mut self, layer: usize, texture_id: Option<u64>) -> bool {
        let current = self.layers[layer];
        if current == Some(texture_id) {
            return false;
        }
        self.layers[layer] = Some(texture_id);
        true
    }
}

pub(super) struct Buffers<'a> {
    pub(super) aggregated_vertex_buffer: &'a wgpu::Buffer,
    pub(super) aggregated_index_buffer: &'a wgpu::Buffer,
    pub(super) identity_instance_transform_buffer: &'a wgpu::Buffer,
    pub(super) identity_instance_color_buffer: &'a wgpu::Buffer,
    pub(super) identity_instance_metadata_buffer: &'a wgpu::Buffer,
    pub(super) aggregated_instance_transform_buffer: Option<&'a wgpu::Buffer>,
    pub(super) aggregated_instance_color_buffer: Option<&'a wgpu::Buffer>,
    pub(super) aggregated_instance_metadata_buffer: Option<&'a wgpu::Buffer>,
}

pub(super) struct Pipelines<'a> {
    pub(super) and_pipeline: &'a wgpu::RenderPipeline,
    pub(super) and_bind_group: &'a wgpu::BindGroup,
    pub(super) decrementing_pipeline: &'a wgpu::RenderPipeline,
    pub(super) decrementing_bind_group: &'a wgpu::BindGroup,
    pub(super) leaf_draw_pipeline: &'a wgpu::RenderPipeline,
    pub(super) shape_texture_bind_group_layout_background: &'a wgpu::BindGroupLayout,
    pub(super) shape_texture_bind_group_layout_foreground: &'a wgpu::BindGroupLayout,
    pub(super) default_shape_texture_bind_groups: &'a [Arc<wgpu::BindGroup>; 2],
    pub(super) shape_texture_layout_epoch: u64,
    pub(super) texture_manager: &'a TextureManager,
}

/// Backdrop-specific rendering resources. Only needed when backdrop effects exist.
/// General resources (pipelines, buffers, textures) are passed separately.
pub(super) struct BackdropContext<'a> {
    pub(super) backdrop_effects: &'a HashMap<usize, EffectInstance>,
    pub(super) loaded_effects: &'a HashMap<u64, LoadedEffect>,
    pub(super) composite_bgl: &'a wgpu::BindGroupLayout,
    pub(super) effect_sampler: &'a wgpu::Sampler,
    pub(super) stencil_only_pipeline: &'a wgpu::RenderPipeline,
    pub(super) backdrop_color_pipeline: &'a wgpu::RenderPipeline,
    pub(super) device: &'a wgpu::Device,
    pub(super) config_format: wgpu::TextureFormat,
    pub(super) backdrop_snapshot_texture: &'a wgpu::Texture,
    pub(super) backdrop_snapshot_view: &'a wgpu::TextureView,
}

const MAX_EFFECT_RESULTS_CAPACITY: usize = 4_096;
const MAX_EFFECT_NODE_IDS_CAPACITY: usize = 4_096;
const MAX_TEXTURE_RECYCLE_CAPACITY: usize = 1_024;
const MAX_EFFECT_OUTPUT_TEXTURES_CAPACITY: usize = 2_048;
const MAX_STENCIL_STACK_CAPACITY: usize = 16_384;
const MAX_SKIPPED_STACK_CAPACITY: usize = 16_384;
const MAX_SCISSOR_STACK_CAPACITY: usize = 16_384;
const MAX_READBACK_BYTES_CAPACITY: usize = 64 * 1024 * 1024;

pub(super) struct RendererScratch {
    pub(super) effect_results: HashMap<usize, wgpu::BindGroup>,
    pub(super) effect_node_ids: Vec<(usize, usize)>,
    pub(super) textures_to_recycle: Vec<effect::PooledTexture>,
    pub(super) effect_output_textures: Vec<wgpu::Texture>,
    pub(super) stencil_stack: Vec<u32>,
    pub(super) skipped_stack: Vec<usize>,
    /// Stack of intersected scissor rects (x, y, width, height) in physical pixels.
    /// Used to replace stencil clipping for axis-aligned rect parents.
    pub(super) scissor_stack: Vec<(u32, u32, u32, u32)>,
    pub(super) backdrop_work_textures: Vec<wgpu::Texture>,
    /// Reused across readback calls; intentionally not cleared on `begin_frame`
    /// because readback may run after render submission and reuse prior capacity.
    pub(super) readback_bytes: Vec<u8>,
    pub(super) traversal_scratch: TraversalScratch,
}

impl RendererScratch {
    pub(super) fn new() -> Self {
        Self {
            effect_results: HashMap::new(),
            effect_node_ids: Vec::new(),
            textures_to_recycle: Vec::new(),
            effect_output_textures: Vec::new(),
            stencil_stack: Vec::new(),
            skipped_stack: Vec::new(),
            scissor_stack: Vec::new(),
            backdrop_work_textures: Vec::new(),
            readback_bytes: Vec::new(),
            traversal_scratch: TraversalScratch::new(),
        }
    }

    pub(super) fn begin_frame(&mut self) {
        self.effect_results.clear();
        self.effect_node_ids.clear();
        self.textures_to_recycle.clear();
        self.effect_output_textures.clear();
        self.stencil_stack.clear();
        self.skipped_stack.clear();
        self.scissor_stack.clear();
        self.backdrop_work_textures.clear();
        self.traversal_scratch.begin();
        // Keep readback bytes length/capacity untouched to preserve reuse across
        // `render_to_buffer`/`render_to_argb32` calls that are not tied to frame start.
    }

    pub(super) fn trim_to_policy(&mut self) {
        trim_hash_map_if_needed(&mut self.effect_results, MAX_EFFECT_RESULTS_CAPACITY);
        trim_vector_if_needed(&mut self.effect_node_ids, MAX_EFFECT_NODE_IDS_CAPACITY);
        trim_vector_if_needed(&mut self.textures_to_recycle, MAX_TEXTURE_RECYCLE_CAPACITY);
        trim_vector_if_needed(
            &mut self.effect_output_textures,
            MAX_EFFECT_OUTPUT_TEXTURES_CAPACITY,
        );
        trim_vector_if_needed(&mut self.stencil_stack, MAX_STENCIL_STACK_CAPACITY);
        trim_vector_if_needed(&mut self.skipped_stack, MAX_SKIPPED_STACK_CAPACITY);
        trim_vector_if_needed(&mut self.scissor_stack, MAX_SCISSOR_STACK_CAPACITY);
        trim_vector_if_needed(
            &mut self.backdrop_work_textures,
            MAX_EFFECT_OUTPUT_TEXTURES_CAPACITY,
        );
        if self.readback_bytes.len() > MAX_READBACK_BYTES_CAPACITY {
            self.readback_bytes.truncate(MAX_READBACK_BYTES_CAPACITY);
        }
        trim_vector_if_needed(&mut self.readback_bytes, MAX_READBACK_BYTES_CAPACITY);
        self.traversal_scratch.trim_to_policy();
    }
}

pub(super) fn trim_vector_if_needed<T>(values: &mut Vec<T>, max_capacity: usize) {
    if values.capacity() > max_capacity {
        values.shrink_to(max_capacity);
    }
}

pub(super) fn trim_hash_map_if_needed<K, V>(values: &mut HashMap<K, V>, max_capacity: usize)
where
    K: Eq + std::hash::Hash,
{
    if values.capacity() > max_capacity {
        values.shrink_to(max_capacity);
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct BufferSizingDecision {
    pub(super) should_reallocate: bool,
}

pub(super) fn decide_buffer_sizing(
    existing_size: Option<u64>,
    required_size: usize,
) -> BufferSizingDecision {
    let required_size = required_size as u64;
    let should_reallocate = existing_size
        .map(|size| size < required_size)
        .unwrap_or(true);

    BufferSizingDecision { should_reallocate }
}

#[cfg(test)]
mod tests {
    use super::{decide_buffer_sizing, RendererScratch};

    #[test]
    fn decide_buffer_sizing_reallocates_when_missing() {
        let decision = decide_buffer_sizing(None, 128);
        assert!(decision.should_reallocate);
    }

    #[test]
    fn decide_buffer_sizing_reallocates_when_too_small() {
        let decision = decide_buffer_sizing(Some(64), 128);
        assert!(decision.should_reallocate);
    }

    #[test]
    fn decide_buffer_sizing_keeps_buffer_when_large_enough() {
        let decision = decide_buffer_sizing(Some(512), 128);
        assert!(!decision.should_reallocate);
    }

    #[test]
    fn renderer_scratch_begin_frame_clears_lengths() {
        let mut scratch = RendererScratch::new();
        scratch.effect_node_ids.extend([(1, 1), (2, 2)]);
        scratch.readback_bytes.extend([1, 2, 3, 4]);
        scratch.begin_frame();

        assert!(scratch.effect_node_ids.is_empty());
        assert_eq!(scratch.readback_bytes.len(), 4);
    }

    #[test]
    fn renderer_scratch_trims_large_capacities() {
        let mut scratch = RendererScratch::new();
        scratch
            .effect_node_ids
            .resize(super::MAX_EFFECT_NODE_IDS_CAPACITY + 2_048, (0, 0));
        scratch.effect_node_ids.clear();

        scratch.trim_to_policy();
        assert!(scratch.effect_node_ids.capacity() <= super::MAX_EFFECT_NODE_IDS_CAPACITY);
    }

    #[test]
    fn renderer_scratch_trims_readback_bytes_length_before_shrinking() {
        let mut scratch = RendererScratch::new();
        scratch
            .readback_bytes
            .resize(super::MAX_READBACK_BYTES_CAPACITY + 1_024, 0);

        scratch.trim_to_policy();

        assert!(scratch.readback_bytes.len() <= super::MAX_READBACK_BYTES_CAPACITY);
    }
}
