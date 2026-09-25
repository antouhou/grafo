use super::execution::effects::{CompositePipelineResources, EffectExecutionResources};
use super::execution::shape_effects::ShapeEffectRendererResources;
use super::execution::shapes::{ShapeExecutionResources, TextureMaterialPipelines};
use super::execution::textures::IntermediateTextureResources;
#[cfg(feature = "render_metrics")]
use crate::wgpu_backend::metrics::{PipelineSwitchCounts, ShapeEffectCacheMetrics};
use crate::wgpu_backend::pipeline::{self, Uniforms};
use crate::wgpu_backend::texture_manager::WgpuTextureManager;
use crate::wgpu_backend::vertex::GeometryBufferRange;
use std::ops::Range;
use std::sync::Arc;
use wgpu::{BindGroup, BindGroupLayout, Buffer, RenderPass, RenderPipeline, Sampler};

/// Geometry and instance buffers, initialized and reused by render preparation.
pub(in crate::wgpu_backend) struct Buffers {
    pub(in crate::wgpu_backend) supports_base_vertex: bool,
    pub(in crate::wgpu_backend) aggregated_vertex_buffer: Option<Buffer>,
    pub(in crate::wgpu_backend) aggregated_index_buffer: Option<Buffer>,
    pub(in crate::wgpu_backend) aggregated_instance_transform_buffer: Option<Buffer>,
    pub(in crate::wgpu_backend) aggregated_instance_color_buffer: Option<Buffer>,
    pub(in crate::wgpu_backend) aggregated_instance_metadata_buffer: Option<Buffer>,
}

impl Buffers {
    pub(in crate::wgpu_backend) fn vertex_buffer(&self) -> &Buffer {
        self.aggregated_vertex_buffer
            .as_ref()
            .expect("aggregated vertex buffer is initialized during render preparation")
    }

    pub(in crate::wgpu_backend) fn index_buffer(&self) -> &Buffer {
        self.aggregated_index_buffer
            .as_ref()
            .expect("aggregated index buffer is initialized during render preparation")
    }

    pub(in crate::wgpu_backend) fn instance_transform_buffer(&self) -> &Buffer {
        self.aggregated_instance_transform_buffer
            .as_ref()
            .expect("drawable shapes have an uploaded instance transform buffer")
    }

    pub(in crate::wgpu_backend) fn instance_color_buffer(&self) -> &Buffer {
        self.aggregated_instance_color_buffer
            .as_ref()
            .expect("drawable shapes have an uploaded instance color buffer")
    }

    pub(in crate::wgpu_backend) fn instance_metadata_buffer(&self) -> &Buffer {
        self.aggregated_instance_metadata_buffer
            .as_ref()
            .expect("drawable shapes have an uploaded instance metadata buffer")
    }

    pub(in crate::wgpu_backend) fn draw_indexed(
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
pub(in crate::wgpu_backend) struct ShapePipelines {
    pub(in crate::wgpu_backend) and_pipeline: Arc<RenderPipeline>,
    pub(in crate::wgpu_backend) and_gradient_pipeline: Arc<RenderPipeline>,
    pub(in crate::wgpu_backend) and_bind_group: BindGroup,
    pub(in crate::wgpu_backend) decrementing_pipeline: Arc<RenderPipeline>,
    pub(in crate::wgpu_backend) decrementing_bind_group: BindGroup,
    pub(in crate::wgpu_backend) leaf_draw_pipeline: Arc<RenderPipeline>,
    pub(in crate::wgpu_backend) leaf_draw_gradient_pipeline: Arc<RenderPipeline>,
    pub(in crate::wgpu_backend) shape_texture_bind_group_layout_background: Arc<BindGroupLayout>,
    pub(in crate::wgpu_backend) shape_texture_bind_group_layout_foreground: Arc<BindGroupLayout>,
    pub(in crate::wgpu_backend) default_shape_texture_bind_groups: [Arc<BindGroup>; 2],
    pub(in crate::wgpu_backend) texture_manager: WgpuTextureManager,
    pub(in crate::wgpu_backend) and_uniforms: Uniforms,
    pub(in crate::wgpu_backend) and_uniform_buffer: Buffer,
    pub(in crate::wgpu_backend) decrementing_uniforms: Uniforms,
    pub(in crate::wgpu_backend) decrementing_uniform_buffer: Buffer,
    pub(in crate::wgpu_backend) under_fill_pipelines: Option<TextureMaterialPipelines>,
    pub(in crate::wgpu_backend) stencil_only_pipeline: RenderPipeline,
    pub(in crate::wgpu_backend) gradient_bind_group_layout: BindGroupLayout,
    pub(in crate::wgpu_backend) linear_clamp_sampler: Sampler,
}

/// Pipelines created together when rendering first needs a backdrop effect.
pub(in crate::wgpu_backend) struct BackdropPipelineResources {
    /// Downsamples captured backdrop pixels before applying an effect.
    pub(in crate::wgpu_backend) texture_blit_pipeline: RenderPipeline,
    /// Layers a transparent group prefix over the scene behind the group.
    pub(in crate::wgpu_backend) layer_composite_resources: CompositePipelineResources,
}

/// Built-in shape and effect resources, borrowed separately from mutable draw state.
pub(in crate::wgpu_backend) struct RendererPipelineResources {
    pub(in crate::wgpu_backend) shapes: ShapePipelines,
    pub(in crate::wgpu_backend) shape_effects: ShapeEffectRendererResources,
    pub(in crate::wgpu_backend) effect_sampler: Option<Sampler>,
    pub(in crate::wgpu_backend) composite_resources: Option<CompositePipelineResources>,
    pub(in crate::wgpu_backend) backdrops: Option<BackdropPipelineResources>,
}

/// Uploaded resources and execution caches. No scene or planning state is retained.
pub(in crate::wgpu_backend) struct BackendResources {
    pub(in crate::wgpu_backend) buffers: Buffers,
    pub(in crate::wgpu_backend) shape_execution: ShapeExecutionResources,
    pub(in crate::wgpu_backend) effect_execution: EffectExecutionResources,
    pub(in crate::wgpu_backend) textures: IntermediateTextureResources,
    #[cfg(feature = "render_metrics")]
    /// Pipeline switch counts from the most recent render.
    pub(in crate::wgpu_backend) pipeline_switch_counts: PipelineSwitchCounts,
    #[cfg(feature = "render_metrics")]
    /// Cache activity from the most recent render.
    pub(in crate::wgpu_backend) shape_effect_cache_metrics: ShapeEffectCacheMetrics,
}
