use super::execution::effects::{CompositePipelineResources, EffectExecutionResources};
use super::execution::shape_effects::ShapeEffectRendererResources;
use super::execution::shapes::{ShapeExecutionResources, TextureMaterialPipelines};
use super::execution::textures::IntermediateTextureResources;
#[cfg(feature = "render_metrics")]
use crate::backend::metrics::{PipelineSwitchCounts, ShapeEffectCacheMetrics};
use crate::backend::pipeline::{self, Uniforms};
use crate::backend::texture_manager::WgpuTextureManager;
use crate::backend::vertex::GeometryBufferRange;
use std::ops::Range;
use std::sync::Arc;
use wgpu::{BindGroup, BindGroupLayout, Buffer, RenderPass, RenderPipeline, Sampler};

/// Geometry and instance buffers, initialized and reused by render preparation.
pub(in crate::backend) struct Buffers {
    pub(in crate::backend) supports_base_vertex: bool,
    pub(in crate::backend) aggregated_vertex_buffer: Option<Buffer>,
    pub(in crate::backend) aggregated_index_buffer: Option<Buffer>,
    pub(in crate::backend) aggregated_instance_transform_buffer: Option<Buffer>,
    pub(in crate::backend) aggregated_instance_color_buffer: Option<Buffer>,
    pub(in crate::backend) aggregated_instance_metadata_buffer: Option<Buffer>,
}

impl Buffers {
    pub(in crate::backend) fn vertex_buffer(&self) -> &Buffer {
        self.aggregated_vertex_buffer
            .as_ref()
            .expect("aggregated vertex buffer is initialized during render preparation")
    }

    pub(in crate::backend) fn index_buffer(&self) -> &Buffer {
        self.aggregated_index_buffer
            .as_ref()
            .expect("aggregated index buffer is initialized during render preparation")
    }

    pub(in crate::backend) fn instance_transform_buffer(&self) -> &Buffer {
        self.aggregated_instance_transform_buffer
            .as_ref()
            .expect("drawable shapes have an uploaded instance transform buffer")
    }

    pub(in crate::backend) fn instance_color_buffer(&self) -> &Buffer {
        self.aggregated_instance_color_buffer
            .as_ref()
            .expect("drawable shapes have an uploaded instance color buffer")
    }

    pub(in crate::backend) fn instance_metadata_buffer(&self) -> &Buffer {
        self.aggregated_instance_metadata_buffer
            .as_ref()
            .expect("drawable shapes have an uploaded instance metadata buffer")
    }

    pub(in crate::backend) fn draw_indexed(
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
pub(in crate::backend) struct ShapePipelines {
    pub(in crate::backend) and_pipeline: Arc<RenderPipeline>,
    pub(in crate::backend) and_gradient_pipeline: Arc<RenderPipeline>,
    pub(in crate::backend) and_bind_group: BindGroup,
    pub(in crate::backend) decrementing_pipeline: Arc<RenderPipeline>,
    pub(in crate::backend) decrementing_bind_group: BindGroup,
    pub(in crate::backend) leaf_draw_pipeline: Arc<RenderPipeline>,
    pub(in crate::backend) leaf_draw_gradient_pipeline: Arc<RenderPipeline>,
    pub(in crate::backend) shape_texture_bind_group_layout_background: Arc<BindGroupLayout>,
    pub(in crate::backend) shape_texture_bind_group_layout_foreground: Arc<BindGroupLayout>,
    pub(in crate::backend) default_shape_texture_bind_groups: [Arc<BindGroup>; 2],
    pub(in crate::backend) texture_manager: WgpuTextureManager,
    pub(in crate::backend) and_uniforms: Uniforms,
    pub(in crate::backend) and_uniform_buffer: Buffer,
    pub(in crate::backend) decrementing_uniforms: Uniforms,
    pub(in crate::backend) decrementing_uniform_buffer: Buffer,
    pub(in crate::backend) under_fill_pipelines: Option<TextureMaterialPipelines>,
    pub(in crate::backend) stencil_only_pipeline: RenderPipeline,
    pub(in crate::backend) gradient_bind_group_layout: BindGroupLayout,
    pub(in crate::backend) linear_clamp_sampler: Sampler,
}

/// Pipelines created together when rendering first needs a backdrop effect.
pub(in crate::backend) struct BackdropPipelineResources {
    /// Downsamples captured backdrop pixels before applying an effect.
    pub(in crate::backend) texture_blit_pipeline: RenderPipeline,
    /// Layers a transparent group prefix over the scene behind the group.
    pub(in crate::backend) layer_composite_resources: CompositePipelineResources,
}

/// Built-in shape and effect resources, borrowed separately from mutable draw state.
pub(in crate::backend) struct RendererPipelineResources {
    pub(in crate::backend) shapes: ShapePipelines,
    pub(in crate::backend) shape_effects: ShapeEffectRendererResources,
    pub(in crate::backend) effect_sampler: Option<Sampler>,
    pub(in crate::backend) composite_resources: Option<CompositePipelineResources>,
    pub(in crate::backend) backdrops: Option<BackdropPipelineResources>,
}

/// Uploaded resources and execution caches. No scene or planning state is retained.
pub(in crate::backend) struct BackendResources {
    pub(in crate::backend) buffers: Buffers,
    pub(in crate::backend) shape_execution: ShapeExecutionResources,
    pub(in crate::backend) effect_execution: EffectExecutionResources,
    pub(in crate::backend) textures: IntermediateTextureResources,
    #[cfg(feature = "render_metrics")]
    /// Pipeline switch counts from the most recent render.
    pub(in crate::backend) pipeline_switch_counts: PipelineSwitchCounts,
    #[cfg(feature = "render_metrics")]
    /// Cache activity from the most recent render.
    pub(in crate::backend) shape_effect_cache_metrics: ShapeEffectCacheMetrics,
}
