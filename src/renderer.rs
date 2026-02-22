//! Renderer for the Grafo library.

use std::num::NonZeroUsize;
use std::sync::Arc;

use ahash::{HashMap, HashMapExt};
use log::warn;
use lyon::tessellation::FillTessellator;
use wgpu::{BindGroup, BufferUsages, CompositeAlphaMode, InstanceDescriptor, SurfaceTarget};

use crate::effect::{
    self, compile_composite_pipeline, compile_effect_pipeline, create_params_bind_group,
    EffectError, EffectInstance, LoadedEffect, OffscreenTexturePool,
};
use crate::pipeline::{
    compute_padded_bytes_per_row, create_and_depth_texture, create_argb_swizzle_bind_group,
    create_argb_swizzle_pipeline, create_msaa_color_texture, create_offscreen_color_texture,
    create_pipeline, create_readback_buffer, create_render_pass, create_storage_input_buffer,
    create_storage_output_buffer, encode_copy_texture_to_buffer, render_buffer_range_to_texture,
    ArgbParams, PipelineType, Uniforms,
};
use crate::shape::{CachedShapeDrawData, DrawShapeCommand, Shape, ShapeDrawData};
use crate::texture_manager::TextureManager;
use crate::util::{to_logical, PoolManager};
use crate::vertex::{InstanceColor, InstanceMetadata, InstanceTransform};
use crate::CachedShape;
use crate::Color;

use self::types::{DrawCommand, RendererScratch};

mod construction;
mod draw_queue;
mod effects;
mod passes;
mod preparation;
mod readback;
mod rendering;
mod surface;
mod traversal;
mod types;

pub type MathRect = lyon::math::Box2D;

// TODO: move to the config/constructor
const MAX_CACHED_SHAPES: usize = 1024;

/// Semantic texture layers for a shape. Background is layer 0, Foreground is layer 1.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum TextureLayer {
    Background,
    Foreground,
}

impl From<TextureLayer> for usize {
    fn from(value: TextureLayer) -> Self {
        match value {
            TextureLayer::Background => 0,
            TextureLayer::Foreground => 1,
        }
    }
}

/// The renderer for the Grafo library. This is the main struct used to render shapes and images.
pub struct Renderer<'a> {
    // Window information
    /// Size of the window in pixels.
    pub(crate) physical_size: (u32, u32),
    /// Scale factor of the window (e.g., for high-DPI displays).
    scale_factor: f64,

    /// AA fringe offset in physical pixels. Controls how far the anti-aliasing
    /// fringe extends outward from shape edges. Default is 0.5.
    fringe_width: f32,

    // WGPU components
    instance: wgpu::Instance,
    surface: wgpu::Surface<'a>,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    config: wgpu::SurfaceConfiguration,

    tessellator: FillTessellator,
    buffers_pool_manager: PoolManager,
    texture_manager: TextureManager,

    /// Tree structure holding shapes to be rendered.
    draw_tree: easy_tree::Tree<DrawCommand>,
    /// Maps node metadata indices to their clip-parent node ids.
    metadata_to_clips: HashMap<usize, usize>,

    /// Uniforms for the stencil-increment ("and") rendering pipeline.
    and_uniforms: Uniforms,
    /// GPU buffer backing the "and" pipeline uniforms.
    and_uniform_buffer: wgpu::Buffer,
    /// Bind group for the stencil-increment rendering pipeline.
    and_bind_group: BindGroup,
    /// Render pipeline for stencil-increment operations.
    and_pipeline: Arc<wgpu::RenderPipeline>,
    /// Bind group layouts for shape texture layers (groups 1 and 2).
    shape_texture_bind_group_layout_background: Arc<wgpu::BindGroupLayout>,
    shape_texture_bind_group_layout_foreground: Arc<wgpu::BindGroupLayout>,
    /// Monotonic counter to invalidate cached shape texture bind groups when the layout changes.
    shape_texture_layout_epoch: u64,
    /// Default transparent texture bind groups for both layers.
    default_shape_texture_bind_groups: [Arc<wgpu::BindGroup>; 2], // [background, foreground]

    /// Render pipeline for decrementing stencil values.
    decrementing_pipeline: Arc<wgpu::RenderPipeline>,
    /// Uniforms for the decrementing pipeline.
    decrementing_uniforms: Uniforms,
    /// GPU buffer backing the decrementing pipeline uniforms.
    decrementing_uniform_buffer: wgpu::Buffer,
    /// Bind group for the decrementing pipeline.
    decrementing_bind_group: BindGroup,

    temp_vertices: Vec<crate::vertex::CustomVertex>,
    temp_indices: Vec<u16>,

    /// Per-frame instance transforms for shapes.
    temp_instance_transforms: Vec<InstanceTransform>,
    /// Per-frame instance colors for shapes.
    temp_instance_colors: Vec<InstanceColor>,
    /// Per-frame instance metadata (draw order) for shapes.
    temp_instance_metadata: Vec<InstanceMetadata>,

    aggregated_vertex_buffer: Option<wgpu::Buffer>,
    aggregated_index_buffer: Option<wgpu::Buffer>,
    aggregated_instance_transform_buffer: Option<wgpu::Buffer>,
    aggregated_instance_color_buffer: Option<wgpu::Buffer>,
    aggregated_instance_metadata_buffer: Option<wgpu::Buffer>,

    identity_instance_transform_buffer: Option<wgpu::Buffer>,
    identity_instance_color_buffer: Option<wgpu::Buffer>,
    identity_instance_metadata_buffer: Option<wgpu::Buffer>,

    /// Loaded shapes to reuse later during rendering without loading/tessellating again.
    /// Not an LRU cache: evicting a shape also results in it not being rendered.
    shape_cache: HashMap<u64, CachedShape>,

    // Cached resources for render_to_argb32 compute swizzle path
    argb_cs_bgl: Option<wgpu::BindGroupLayout>,
    argb_cs_pipeline: Option<wgpu::ComputePipeline>,
    argb_swizzle_bind_group: Option<wgpu::BindGroup>,
    argb_params_buffer: Option<wgpu::Buffer>,
    argb_input_buffer: Option<wgpu::Buffer>,
    argb_output_storage_buffer: Option<wgpu::Buffer>,
    argb_readback_buffer: Option<wgpu::Buffer>,
    argb_input_buffer_size: u64,
    argb_output_buffer_size: u64,
    argb_cached_width: u32,
    argb_cached_height: u32,
    argb_offscreen_texture: Option<wgpu::Texture>,

    // Cached resources for render_to_buffer (BGRA bytes) path
    rtb_offscreen_texture: Option<wgpu::Texture>,
    rtb_readback_buffer: Option<wgpu::Buffer>,
    rtb_cached_width: u32,
    rtb_cached_height: u32,

    /// Current MSAA sample count (1 = off, 4 = 4x, etc.)
    msaa_sample_count: u32,

    /// The multisampled color texture (None when sample_count == 1).
    msaa_color_texture: Option<wgpu::Texture>,
    /// View of the MSAA color texture.
    msaa_color_texture_view: Option<wgpu::TextureView>,

    // ── Effect system ──────────────────────────────────────────────────
    /// Loaded (compiled) effects, keyed by user-provided effect_id.
    loaded_effects: HashMap<u64, LoadedEffect>,
    /// Per-node group effect instances, keyed by node_id.
    group_effects: HashMap<usize, EffectInstance>,
    /// Per-node backdrop effect instances, keyed by node_id.
    /// A backdrop effect processes the pixels already rendered behind a shape.
    backdrop_effects: HashMap<usize, EffectInstance>,
    /// Pool of offscreen textures for effect compositing.
    offscreen_texture_pool: OffscreenTexturePool,
    /// Shared composite pipeline for drawing effect results into the parent target.
    /// Created lazily on first use.
    composite_pipeline: Option<wgpu::RenderPipeline>,
    /// Bind group layout for the composite pipeline's input texture.
    composite_bgl: Option<wgpu::BindGroupLayout>,
    /// Reusable sampler for effect texture sampling.
    effect_sampler: Option<wgpu::Sampler>,

    // ── Backdrop effect infrastructure ─────────────────────────────────
    /// Snapshot texture for backdrop effects (copy of current render target).
    backdrop_snapshot_texture: Option<wgpu::Texture>,
    backdrop_snapshot_view: Option<wgpu::TextureView>,
    /// Stencil-only pipeline: writes stencil but no color output.
    /// Used for Step 1 of the three-step backdrop draw.
    stencil_only_pipeline: Option<wgpu::RenderPipeline>,
    /// Shape color pipeline with stencil Keep: draws color but doesn't modify stencil.
    /// Used for Step 3 of the three-step backdrop draw.
    backdrop_color_pipeline: Option<wgpu::RenderPipeline>,

    // ── Reusable scratch state ───────────────────────────────────────────
    scratch: RendererScratch,
}

/// Default AA fringe width in physical pixels.
const DEFAULT_FRINGE_WIDTH: f32 = 0.5;

impl<'a> Renderer<'a> {
    const DEFAULT_FRINGE_WIDTH: f32 = DEFAULT_FRINGE_WIDTH;

    pub(super) fn begin_frame_scratch(&mut self) {
        self.scratch.begin_frame();
    }

    pub(super) fn trim_scratch_on_resize_or_policy(&mut self) {
        // This is safe to call frequently: `shrink_to` is effectively a no-op
        // when capacities are below thresholds, so this acts as amortized
        // memory hygiene for long-running sessions.
        self.scratch.trim_to_policy();
    }
}
