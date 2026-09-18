#[cfg(feature = "render_metrics")]
use super::metrics::{PipelineSwitchCounts, ShapeEffectCacheMetrics};
use super::types::{DrawCommand, RendererScratch};
use crate::effect::{CompositePipelineResources, EffectInstance, OffscreenTexturePool};
use crate::pipeline;
use crate::texture_manager::TextureManager;
use crate::util::ShapeResources;
use crate::vertex::GeometryBufferRange;
use ahash::HashMap;
use easy_tree::Tree;
use std::ops::Range;
use std::sync::Arc;
use wgpu::RenderPass;

/// Geometry and instance buffers, initialized and reused by render preparation.
pub(super) struct Buffers {
    pub(super) supports_base_vertex: bool,
    pub(super) aggregated_vertex_buffer: Option<wgpu::Buffer>,
    pub(super) aggregated_index_buffer: Option<wgpu::Buffer>,
    pub(super) identity_instance_transform_buffer: Option<wgpu::Buffer>,
    pub(super) identity_instance_color_buffer: Option<wgpu::Buffer>,
    pub(super) identity_instance_metadata_buffer: Option<wgpu::Buffer>,
    pub(super) aggregated_instance_transform_buffer: Option<wgpu::Buffer>,
    pub(super) aggregated_instance_color_buffer: Option<wgpu::Buffer>,
    pub(super) aggregated_instance_metadata_buffer: Option<wgpu::Buffer>,
}

impl Buffers {
    pub(super) fn vertex_buffer(&self) -> &wgpu::Buffer {
        self.aggregated_vertex_buffer
            .as_ref()
            .expect("aggregated vertex buffer is initialized during render preparation")
    }

    pub(super) fn index_buffer(&self) -> &wgpu::Buffer {
        self.aggregated_index_buffer
            .as_ref()
            .expect("aggregated index buffer is initialized during render preparation")
    }

    pub(super) fn identity_transform_buffer(&self) -> &wgpu::Buffer {
        self.identity_instance_transform_buffer
            .as_ref()
            .expect("identity instance transform buffer is initialized during render preparation")
    }

    pub(super) fn identity_color_buffer(&self) -> &wgpu::Buffer {
        self.identity_instance_color_buffer
            .as_ref()
            .expect("identity instance color buffer is initialized during render preparation")
    }

    pub(super) fn identity_metadata_buffer(&self) -> &wgpu::Buffer {
        self.identity_instance_metadata_buffer
            .as_ref()
            .expect("identity instance metadata buffer is initialized during render preparation")
    }

    pub(super) fn draw_indexed(
        &self,
        render_pass: &mut RenderPass<'_>,
        geometry_range: GeometryBufferRange,
        instances: Range<u32>,
    ) {
        pipeline::draw_indexed_geometry(
            render_pass,
            geometry_range,
            self.vertex_buffer(),
            self.supports_base_vertex,
            instances,
        );
    }
}

/// Shape pipelines and the resources used to bind their materials.
pub(super) struct Pipelines {
    pub(super) and_pipeline: Arc<wgpu::RenderPipeline>,
    pub(super) and_gradient_pipeline: Arc<wgpu::RenderPipeline>,
    pub(super) and_bind_group: wgpu::BindGroup,
    pub(super) decrementing_pipeline: Arc<wgpu::RenderPipeline>,
    pub(super) decrementing_bind_group: wgpu::BindGroup,
    pub(super) leaf_draw_pipeline: Arc<wgpu::RenderPipeline>,
    pub(super) leaf_draw_gradient_pipeline: Arc<wgpu::RenderPipeline>,
    pub(super) shape_texture_bind_group_layout_background: Arc<wgpu::BindGroupLayout>,
    pub(super) shape_texture_bind_group_layout_foreground: Arc<wgpu::BindGroupLayout>,
    pub(super) default_shape_texture_bind_groups: [Arc<wgpu::BindGroup>; 2],
    pub(super) texture_manager: TextureManager,
}

/// Owns draw data, GPU resources, and scratch storage for a renderer's lifetime.
pub(super) struct RendererState {
    pub(super) draw_tree: Tree<DrawCommand>,
    pub(super) shape_resources: ShapeResources,
    pub(super) group_effects: HashMap<usize, EffectInstance>,
    pub(super) backdrop_effects: HashMap<usize, EffectInstance>,
    pub(super) pipelines: Pipelines,
    pub(super) buffers: Buffers,
    pub(super) texture_pool: OffscreenTexturePool,
    /// Created lazily when group or backdrop effects need compositing.
    pub(super) composite_resources: Option<CompositePipelineResources>,
    pub(super) scratch: RendererScratch,
    /// Converts logical coordinates to physical pixels.
    pub(super) scale_factor: f64,
    /// Viewport size in physical pixels.
    pub(super) physical_size: (u32, u32),
    #[cfg(feature = "render_metrics")]
    /// Pipeline switch counts from the most recent render.
    pub(super) pipeline_switch_counts: PipelineSwitchCounts,
    #[cfg(feature = "render_metrics")]
    /// Cache activity from the most recent render.
    pub(super) shape_effect_cache_metrics: ShapeEffectCacheMetrics,
}
