use super::execution::effects::{CompositePipelineResources, EffectExecutionResources};
use super::execution::shape_effects::ShapeEffectRendererResources;
use super::execution::shapes::{ShapeExecutionResources, TextureMaterialPipelines};
use super::execution::textures::IntermediateTextureResources;
use crate::pipeline::{self, Uniforms};
use crate::renderer::backend::vertex::GeometryBufferRange;
#[cfg(feature = "render_metrics")]
use crate::renderer::metrics::{PipelineSwitchCounts, ShapeEffectCacheMetrics};
use crate::texture_manager::TextureManager;
use std::ops::Range;
use std::sync::Arc;
use wgpu::{BindGroup, BindGroupLayout, Buffer, RenderPass, RenderPipeline, Sampler};

/// Geometry and instance buffers, initialized and reused by render preparation.
pub(in crate::renderer) struct Buffers {
    pub(in crate::renderer) supports_base_vertex: bool,
    pub(in crate::renderer) aggregated_vertex_buffer: Option<Buffer>,
    pub(in crate::renderer) aggregated_index_buffer: Option<Buffer>,
    pub(in crate::renderer) aggregated_instance_transform_buffer: Option<Buffer>,
    pub(in crate::renderer) aggregated_instance_color_buffer: Option<Buffer>,
    pub(in crate::renderer) aggregated_instance_metadata_buffer: Option<Buffer>,
}

impl Buffers {
    pub(in crate::renderer) fn vertex_buffer(&self) -> &Buffer {
        self.aggregated_vertex_buffer
            .as_ref()
            .expect("aggregated vertex buffer is initialized during render preparation")
    }

    pub(in crate::renderer) fn index_buffer(&self) -> &Buffer {
        self.aggregated_index_buffer
            .as_ref()
            .expect("aggregated index buffer is initialized during render preparation")
    }

    pub(in crate::renderer) fn instance_transform_buffer(&self) -> &Buffer {
        self.aggregated_instance_transform_buffer
            .as_ref()
            .expect("drawable shapes have an uploaded instance transform buffer")
    }

    pub(in crate::renderer) fn instance_color_buffer(&self) -> &Buffer {
        self.aggregated_instance_color_buffer
            .as_ref()
            .expect("drawable shapes have an uploaded instance color buffer")
    }

    pub(in crate::renderer) fn instance_metadata_buffer(&self) -> &Buffer {
        self.aggregated_instance_metadata_buffer
            .as_ref()
            .expect("drawable shapes have an uploaded instance metadata buffer")
    }

    pub(in crate::renderer) fn draw_indexed(
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
pub(in crate::renderer) struct ShapePipelines {
    pub(in crate::renderer) and_pipeline: Arc<RenderPipeline>,
    pub(in crate::renderer) and_gradient_pipeline: Arc<RenderPipeline>,
    pub(in crate::renderer) and_bind_group: BindGroup,
    pub(in crate::renderer) decrementing_pipeline: Arc<RenderPipeline>,
    pub(in crate::renderer) decrementing_bind_group: BindGroup,
    pub(in crate::renderer) leaf_draw_pipeline: Arc<RenderPipeline>,
    pub(in crate::renderer) leaf_draw_gradient_pipeline: Arc<RenderPipeline>,
    pub(in crate::renderer) shape_texture_bind_group_layout_background: Arc<BindGroupLayout>,
    pub(in crate::renderer) shape_texture_bind_group_layout_foreground: Arc<BindGroupLayout>,
    pub(in crate::renderer) default_shape_texture_bind_groups: [Arc<BindGroup>; 2],
    pub(in crate::renderer) texture_manager: TextureManager,
    pub(in crate::renderer) and_uniforms: Uniforms,
    pub(in crate::renderer) and_uniform_buffer: Buffer,
    pub(in crate::renderer) decrementing_uniforms: Uniforms,
    pub(in crate::renderer) decrementing_uniform_buffer: Buffer,
    pub(in crate::renderer) under_fill_pipelines: Option<TextureMaterialPipelines>,
    pub(in crate::renderer) stencil_only_pipeline: RenderPipeline,
    pub(in crate::renderer) gradient_bind_group_layout: BindGroupLayout,
    pub(in crate::renderer) linear_clamp_sampler: Sampler,
}

/// Pipelines created together when rendering first needs a backdrop effect.
pub(in crate::renderer) struct BackdropPipelineResources {
    /// Downsamples captured backdrop pixels before applying an effect.
    pub(in crate::renderer) texture_blit_pipeline: RenderPipeline,
    /// Layers a transparent group prefix over the scene behind the group.
    pub(in crate::renderer) layer_composite_resources: CompositePipelineResources,
}

/// Built-in shape and effect resources, borrowed separately from mutable draw state.
pub(in crate::renderer) struct RendererPipelineResources {
    pub(in crate::renderer) shapes: ShapePipelines,
    pub(in crate::renderer) shape_effects: ShapeEffectRendererResources,
    pub(in crate::renderer) effect_sampler: Option<Sampler>,
    pub(in crate::renderer) composite_resources: Option<CompositePipelineResources>,
    pub(in crate::renderer) backdrops: Option<BackdropPipelineResources>,
}

/// Uploaded resources and execution caches. No scene or planning state is retained.
pub(in crate::renderer) struct BackendResources {
    pub(in crate::renderer) buffers: Buffers,
    pub(in crate::renderer) shape_execution: ShapeExecutionResources,
    pub(in crate::renderer) effect_execution: EffectExecutionResources,
    pub(in crate::renderer) textures: IntermediateTextureResources,
    #[cfg(feature = "render_metrics")]
    /// Pipeline switch counts from the most recent render.
    pub(in crate::renderer) pipeline_switch_counts: PipelineSwitchCounts,
    #[cfg(feature = "render_metrics")]
    /// Cache activity from the most recent render.
    pub(in crate::renderer) shape_effect_cache_metrics: ShapeEffectCacheMetrics,
}
