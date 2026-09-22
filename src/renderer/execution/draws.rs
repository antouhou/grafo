use super::effects::CompositePipelineResources;
use super::shapes::ShapeDrawResources;
use super::textures::IntermediateTextureResources;
use crate::renderer::state::{Buffers, ShapePipelines};
use crate::renderer::types::{BoundTextureState, Pipeline, PipelineTracker};
use crate::renderer::IntermediateTextureId;
use crate::shape::{ShapeDrawMaterial, ShapeTextureBinding};
use crate::vertex::{GeometryBufferRange, InstanceColor, InstanceMetadata, InstanceTransform};
use wgpu::{BindGroup, IndexFormat, RenderPass, RenderPipeline};

fn bind_instance_buffers(
    render_pass: &mut RenderPass<'_>,
    instance_index: usize,
    buffers: &Buffers,
) {
    let transform_offset = instance_index as u64 * InstanceTransform::STRIDE;
    render_pass.set_vertex_buffer(
        1,
        buffers
            .instance_transform_buffer()
            .slice(transform_offset..transform_offset + InstanceTransform::STRIDE),
    );
    let color_offset = instance_index as u64 * InstanceColor::STRIDE;
    render_pass.set_vertex_buffer(
        2,
        buffers
            .instance_color_buffer()
            .slice(color_offset..color_offset + InstanceColor::STRIDE),
    );
    let metadata_offset = instance_index as u64 * InstanceMetadata::STRIDE;
    render_pass.set_vertex_buffer(
        3,
        buffers
            .instance_metadata_buffer()
            .slice(metadata_offset..metadata_offset + InstanceMetadata::STRIDE),
    );
}

fn pipeline_has_shared_geometry_bindings(pipeline: Pipeline) -> bool {
    !matches!(pipeline, Pipeline::None)
}

pub(super) fn bind_aggregated_geometry_buffers(
    render_pass: &mut RenderPass<'_>,
    buffers: &Buffers,
) {
    render_pass.set_vertex_buffer(0, buffers.vertex_buffer().slice(..));
    render_pass.set_index_buffer(buffers.index_buffer().slice(..), IndexFormat::Uint16);
}

fn bind_decrement_pipeline(render_pass: &mut RenderPass<'_>, pipelines: &ShapePipelines) {
    render_pass.set_pipeline(&pipelines.decrementing_pipeline);
    render_pass.set_bind_group(0, &pipelines.decrementing_bind_group, &[]);
    render_pass.set_bind_group(1, &*pipelines.default_shape_texture_bind_groups[0], &[]);
    render_pass.set_bind_group(2, &*pipelines.default_shape_texture_bind_groups[1], &[]);
}

fn draw_stencil_geometry(
    render_pass: &mut RenderPass<'_>,
    _pipeline_tracker: &mut PipelineTracker,
    stencil_reference: u32,
    geometry_range: GeometryBufferRange,
    buffers: &Buffers,
) {
    render_pass.set_stencil_reference(stencil_reference);
    buffers.draw_indexed(render_pass, geometry_range, 0..1);
    #[cfg(feature = "render_metrics")]
    _pipeline_tracker.record_stencil_pass();
}

#[allow(clippy::too_many_arguments)]
fn draw_material(
    render_pass: &mut RenderPass<'_>,
    currently_set_pipeline: &mut PipelineTracker,
    bound_texture_state: &mut BoundTextureState,
    stencil_reference: u32,
    material: ShapeDrawMaterial<'_>,
    resources: &ShapeDrawResources,
    pipelines: &ShapePipelines,
    buffers: &Buffers,
    textures: &IntermediateTextureResources,
    increments_stencil: bool,
) {
    let Some(location) = resources.location else {
        return;
    };
    let (target_pipeline, pipeline) = pipelines.material_pipeline(material, increments_stencil);
    if currently_set_pipeline.current != target_pipeline {
        render_pass.set_pipeline(pipeline);
        render_pass.set_bind_group(0, &pipelines.and_bind_group, &[]);
        render_pass.set_bind_group(1, &*pipelines.default_shape_texture_bind_groups[0], &[]);
        render_pass.set_bind_group(2, &*pipelines.default_shape_texture_bind_groups[1], &[]);
        bound_texture_state.mark_bound(0, ShapeTextureBinding::None);
        bound_texture_state.mark_bound(1, ShapeTextureBinding::None);
        if !pipeline_has_shared_geometry_bindings(currently_set_pipeline.current) {
            bind_aggregated_geometry_buffers(render_pass, buffers);
        }
        currently_set_pipeline.switch_to(target_pipeline);
    }
    textures.bind_shape_texture_layers(
        render_pass,
        material.texture_bindings,
        &pipelines.texture_manager,
        &pipelines.shape_texture_bind_group_layout_background,
        &pipelines.shape_texture_bind_group_layout_foreground,
        &pipelines.default_shape_texture_bind_groups,
        bound_texture_state,
    );
    if let Some(binding) = resources.material_bind_group(material) {
        render_pass.set_bind_group(3, binding, &[]);
    }
    bind_instance_buffers(render_pass, location.instance_index, buffers);
    render_pass.set_stencil_reference(stencil_reference);
    buffers.draw_indexed(render_pass, location.geometry_range, 0..1);
    #[cfg(feature = "render_metrics")]
    if increments_stencil {
        currently_set_pipeline.record_stencil_pass();
    }
}

/// Draws color and increments stencil samples matching the supplied reference.
#[allow(clippy::too_many_arguments)]
pub(in crate::renderer) fn draw_shape_and_increment_stencil(
    render_pass: &mut RenderPass<'_>,
    currently_set_pipeline: &mut PipelineTracker,
    bound_texture_state: &mut BoundTextureState,
    stencil_reference: u32,
    material: ShapeDrawMaterial<'_>,
    resources: &ShapeDrawResources,
    pipelines: &ShapePipelines,
    buffers: &Buffers,
    textures: &IntermediateTextureResources,
) {
    draw_material(
        render_pass,
        currently_set_pipeline,
        bound_texture_state,
        stencil_reference,
        material,
        resources,
        pipelines,
        buffers,
        textures,
        true,
    );
}

/// Decrements stencil samples matching the supplied reference without drawing color.
pub(in crate::renderer) fn decrement_stencil(
    render_pass: &mut RenderPass<'_>,
    currently_set_pipeline: &mut PipelineTracker,
    bound_texture_state: &mut BoundTextureState,
    stencil_reference: u32,
    resources: &ShapeDrawResources,
    pipelines: &ShapePipelines,
    buffers: &Buffers,
) {
    let Some(location) = resources.location else {
        return;
    };
    if !matches!(currently_set_pipeline.current, Pipeline::StencilDecrement) {
        bind_decrement_pipeline(render_pass, pipelines);
        bound_texture_state.mark_bound(0, ShapeTextureBinding::None);
        bound_texture_state.mark_bound(1, ShapeTextureBinding::None);

        if !pipeline_has_shared_geometry_bindings(currently_set_pipeline.current) {
            bind_aggregated_geometry_buffers(render_pass, buffers);
        }

        currently_set_pipeline.switch_to(Pipeline::StencilDecrement);
    }

    bind_instance_buffers(render_pass, location.instance_index, buffers);

    draw_stencil_geometry(
        render_pass,
        currently_set_pipeline,
        stencil_reference,
        location.geometry_range,
        buffers,
    );
}

/// Draws the material at the supplied reference without modifying stencil.
#[allow(clippy::too_many_arguments)]
pub(in crate::renderer) fn draw_shape(
    render_pass: &mut RenderPass<'_>,
    currently_set_pipeline: &mut PipelineTracker,
    bound_texture_state: &mut BoundTextureState,
    stencil_reference: u32,
    material: ShapeDrawMaterial<'_>,
    resources: &ShapeDrawResources,
    pipelines: &ShapePipelines,
    buffers: &Buffers,
    textures: &IntermediateTextureResources,
) {
    draw_material(
        render_pass,
        currently_set_pipeline,
        bound_texture_state,
        stencil_reference,
        material,
        resources,
        pipelines,
        buffers,
        textures,
        false,
    );
}

/// Increments stencil within the shape without drawing color.
pub(in crate::renderer) fn increment_stencil(
    render_pass: &mut RenderPass<'_>,
    currently_set_pipeline: &mut PipelineTracker,
    stencil_reference: u32,
    resources: &ShapeDrawResources,
    pipelines: &ShapePipelines,
    buffers: &Buffers,
) {
    let Some(location) = resources.location else {
        return;
    };
    render_pass.set_pipeline(&pipelines.stencil_only_pipeline);
    currently_set_pipeline.switch_to(Pipeline::StencilIncrementOnly);
    render_pass.set_bind_group(0, &pipelines.and_bind_group, &[]);
    render_pass.set_bind_group(1, &*pipelines.default_shape_texture_bind_groups[0], &[]);
    render_pass.set_bind_group(2, &*pipelines.default_shape_texture_bind_groups[1], &[]);
    bind_aggregated_geometry_buffers(render_pass, buffers);
    bind_instance_buffers(render_pass, location.instance_index, buffers);

    draw_stencil_geometry(
        render_pass,
        currently_set_pipeline,
        stencil_reference,
        location.geometry_range,
        buffers,
    );
}

/// Decrements stencil using the geometry still bound by the preceding shape draw.
pub(in crate::renderer) fn decrement_bound_shape_stencil(
    render_pass: &mut RenderPass<'_>,
    currently_set_pipeline: &mut PipelineTracker,
    stencil_reference: u32,
    resources: &ShapeDrawResources,
    pipelines: &ShapePipelines,
    buffers: &Buffers,
) {
    if let Some(location) = resources.location {
        bind_decrement_pipeline(render_pass, pipelines);
        currently_set_pipeline.switch_to(Pipeline::StencilDecrement);
        draw_stencil_geometry(
            render_pass,
            currently_set_pipeline,
            stencil_reference,
            location.geometry_range,
            buffers,
        );
    }
}

/// Composites an intermediate texture under the active scissor and stencil reference.
pub(in crate::renderer) fn composite_texture(
    render_pass: &mut RenderPass<'_>,
    currently_set_pipeline: &mut PipelineTracker,
    bound_texture_state: &mut BoundTextureState,
    stencil_reference: u32,
    texture_id: IntermediateTextureId,
    resources: &CompositePipelineResources,
    textures: &IntermediateTextureResources,
) {
    render_pass.set_pipeline(&resources.pipeline);
    render_pass.set_bind_group(0, textures.bind_group(texture_id), &[]);
    render_pass.set_stencil_reference(stencil_reference);
    render_pass.draw(0..3, 0..1);
    currently_set_pipeline.switch_to(Pipeline::None);
    bound_texture_state.invalidate();
}

/// Rasterizes the prepared shape-effect mask into the active target.
pub(in crate::renderer) fn draw_shape_mask(
    render_pass: &mut RenderPass<'_>,
    geometry_range: GeometryBufferRange,
    pipeline: &RenderPipeline,
    material: &BindGroup,
    buffers: &Buffers,
) {
    render_pass.set_pipeline(pipeline);
    render_pass.set_bind_group(0, material, &[]);
    bind_aggregated_geometry_buffers(render_pass, buffers);
    buffers.draw_indexed(render_pass, geometry_range, 0..1);
}
