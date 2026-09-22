use super::execution::effects::{CompositePipelineResources, EffectExecutionResources};
use super::execution::shapes::{ShapeExecutionResources, TextureMaterialPipelines};
use super::execution::textures::IntermediateTextureResources;
#[cfg(feature = "render_metrics")]
use super::metrics::{PipelineSwitchCounts, ShapeEffectCacheMetrics};
use super::shape_effects::ShapeEffectRendererResources;
use super::types::{DrawTreeNode, RendererScratch};
use crate::effect::{BackdropEffectInstance, EffectInstance, ShapeEffectInstance};
use crate::pipeline::{self, Uniforms};
use crate::texture_manager::TextureManager;
use crate::util::ShapeResources;
use crate::vertex::GeometryBufferRange;
use ahash::HashMap;
use easy_tree::Tree;
use std::ops::Range;
use std::sync::Arc;
use wgpu::{BindGroup, BindGroupLayout, Buffer, RenderPass, RenderPipeline, Sampler};

/// Geometry and instance buffers, initialized and reused by render preparation.
pub(super) struct Buffers {
    pub(super) supports_base_vertex: bool,
    pub(super) aggregated_vertex_buffer: Option<Buffer>,
    pub(super) aggregated_index_buffer: Option<Buffer>,
    pub(super) identity_instance_transform_buffer: Option<Buffer>,
    pub(super) identity_instance_color_buffer: Option<Buffer>,
    pub(super) identity_instance_metadata_buffer: Option<Buffer>,
    pub(super) aggregated_instance_transform_buffer: Option<Buffer>,
    pub(super) aggregated_instance_color_buffer: Option<Buffer>,
    pub(super) aggregated_instance_metadata_buffer: Option<Buffer>,
}

impl Buffers {
    pub(super) fn vertex_buffer(&self) -> &Buffer {
        self.aggregated_vertex_buffer
            .as_ref()
            .expect("aggregated vertex buffer is initialized during render preparation")
    }

    pub(super) fn index_buffer(&self) -> &Buffer {
        self.aggregated_index_buffer
            .as_ref()
            .expect("aggregated index buffer is initialized during render preparation")
    }

    pub(super) fn identity_transform_buffer(&self) -> &Buffer {
        self.identity_instance_transform_buffer
            .as_ref()
            .expect("identity instance transform buffer is initialized during render preparation")
    }

    pub(super) fn identity_color_buffer(&self) -> &Buffer {
        self.identity_instance_color_buffer
            .as_ref()
            .expect("identity instance color buffer is initialized during render preparation")
    }

    pub(super) fn identity_metadata_buffer(&self) -> &Buffer {
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
pub(super) struct ShapePipelines {
    pub(super) and_pipeline: Arc<RenderPipeline>,
    pub(super) and_gradient_pipeline: Arc<RenderPipeline>,
    pub(super) and_bind_group: BindGroup,
    pub(super) decrementing_pipeline: Arc<RenderPipeline>,
    pub(super) decrementing_bind_group: BindGroup,
    pub(super) leaf_draw_pipeline: Arc<RenderPipeline>,
    pub(super) leaf_draw_gradient_pipeline: Arc<RenderPipeline>,
    pub(super) shape_texture_bind_group_layout_background: Arc<BindGroupLayout>,
    pub(super) shape_texture_bind_group_layout_foreground: Arc<BindGroupLayout>,
    pub(super) default_shape_texture_bind_groups: [Arc<BindGroup>; 2],
    pub(super) texture_manager: TextureManager,
    pub(super) and_uniforms: Uniforms,
    pub(super) and_uniform_buffer: Buffer,
    pub(super) decrementing_uniforms: Uniforms,
    pub(super) decrementing_uniform_buffer: Buffer,
    pub(super) under_fill_pipelines: Option<TextureMaterialPipelines>,
    pub(super) stencil_only_pipeline: RenderPipeline,
    pub(super) gradient_bind_group_layout: BindGroupLayout,
    pub(super) linear_clamp_sampler: Sampler,
}

/// Pipelines created together when rendering first needs a backdrop effect.
pub(super) struct BackdropPipelineResources {
    /// Downsamples captured backdrop pixels before applying an effect.
    pub(super) texture_blit_pipeline: RenderPipeline,
    /// Layers a transparent group prefix over the scene behind the group.
    pub(super) layer_composite_resources: CompositePipelineResources,
}

/// Built-in shape and effect resources, borrowed separately from mutable draw state.
pub(super) struct RendererPipelineResources {
    pub(super) shapes: ShapePipelines,
    pub(super) shape_effects: ShapeEffectRendererResources,
    pub(super) effect_sampler: Option<Sampler>,
    pub(super) composite_resources: Option<CompositePipelineResources>,
    pub(super) backdrops: Option<BackdropPipelineResources>,
}

/// Draw tree nodes, effect attachments and caches, and reusable rendering storage.
pub(super) struct RendererState {
    pub(super) draw_tree: Tree<DrawTreeNode>,
    pub(super) shape_resources: ShapeResources,
    pub(super) group_effects: HashMap<usize, EffectInstance>,
    pub(super) backdrop_effects: HashMap<usize, BackdropEffectInstance>,
    pub(super) shape_effects: HashMap<usize, ShapeEffectInstance>,
    pub(super) buffers: Buffers,
    pub(super) shape_execution: ShapeExecutionResources,
    pub(super) effect_execution: EffectExecutionResources,
    pub(super) textures: IntermediateTextureResources,
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
