use super::shapes::ShapeDrawResources;
use super::textures::IntermediateTextureResources;
use crate::backend::resources::{Buffers, RendererPipelineResources, ShapePipelines};
use crate::backend::types::{BoundTextureState, Pipeline, PipelineTracker};
use crate::backend::vertex::{GeometryBufferRange, InstanceColor, InstanceMetadata};
use crate::commands::{IntermediateTextureId, ShapeDrawMaterial, ShapeTextureBinding};
use crate::core::vertex::InstanceTransform;
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

pub(super) struct DrawPass<'pass, 'encoder> {
    pub(super) render_pass: &'pass mut RenderPass<'encoder>,
    pub(super) pipeline_tracker: &'pass mut PipelineTracker,
    pub(super) bound_textures: &'pass mut BoundTextureState,
    pub(super) pipelines: &'pass RendererPipelineResources,
    pub(super) buffers: &'pass Buffers,
    pub(super) textures: &'pass IntermediateTextureResources,
}

impl DrawPass<'_, '_> {
    fn draw_stencil_geometry(
        &mut self,
        stencil_reference: u32,
        geometry_range: GeometryBufferRange,
    ) {
        self.render_pass.set_stencil_reference(stencil_reference);
        self.buffers
            .draw_indexed(self.render_pass, geometry_range, 0..1);
        #[cfg(feature = "render_metrics")]
        self.pipeline_tracker.record_stencil_pass();
    }

    fn draw_material(
        &mut self,
        stencil_reference: u32,
        material: ShapeDrawMaterial,
        resources: &ShapeDrawResources,
        increments_stencil: bool,
    ) {
        let pipelines = &self.pipelines.shapes;
        let Some(location) = resources.location else {
            return;
        };
        let (target_pipeline, pipeline) = pipelines.material_pipeline(material, increments_stencil);
        if self.pipeline_tracker.current != target_pipeline {
            self.render_pass.set_pipeline(pipeline);
            self.render_pass
                .set_bind_group(0, &pipelines.and_bind_group, &[]);
            self.render_pass.set_bind_group(
                1,
                &*pipelines.default_shape_texture_bind_groups[0],
                &[],
            );
            self.render_pass.set_bind_group(
                2,
                &*pipelines.default_shape_texture_bind_groups[1],
                &[],
            );
            self.bound_textures.mark_bound(0, ShapeTextureBinding::None);
            self.bound_textures.mark_bound(1, ShapeTextureBinding::None);
            if !pipeline_has_shared_geometry_bindings(self.pipeline_tracker.current) {
                bind_aggregated_geometry_buffers(self.render_pass, self.buffers);
            }
            self.pipeline_tracker.switch_to(target_pipeline);
        }
        self.textures.bind_shape_texture_layers(
            self.render_pass,
            &material.texture_bindings,
            &pipelines.texture_manager,
            &pipelines.shape_texture_bind_group_layout_background,
            &pipelines.shape_texture_bind_group_layout_foreground,
            &pipelines.default_shape_texture_bind_groups,
            self.bound_textures,
        );
        if let Some(binding) = resources.material_bind_group(material) {
            self.render_pass.set_bind_group(3, binding, &[]);
        }
        bind_instance_buffers(self.render_pass, location.instance_index, self.buffers);
        self.render_pass.set_stencil_reference(stencil_reference);
        self.buffers
            .draw_indexed(self.render_pass, location.geometry_range, 0..1);
        #[cfg(feature = "render_metrics")]
        if increments_stencil {
            self.pipeline_tracker.record_stencil_pass();
        }
    }

    /// Draws color and increments stencil samples matching the supplied reference.
    pub(super) fn draw_shape_and_increment_stencil(
        &mut self,
        stencil_reference: u32,
        material: ShapeDrawMaterial,
        resources: &ShapeDrawResources,
    ) {
        self.draw_material(stencil_reference, material, resources, true);
    }

    /// Decrements stencil samples matching the supplied reference without drawing color.
    pub(super) fn decrement_stencil(
        &mut self,
        stencil_reference: u32,
        resources: &ShapeDrawResources,
    ) {
        let Some(location) = resources.location else {
            return;
        };
        if !matches!(self.pipeline_tracker.current, Pipeline::StencilDecrement) {
            bind_decrement_pipeline(self.render_pass, &self.pipelines.shapes);
            self.bound_textures.mark_bound(0, ShapeTextureBinding::None);
            self.bound_textures.mark_bound(1, ShapeTextureBinding::None);

            if !pipeline_has_shared_geometry_bindings(self.pipeline_tracker.current) {
                bind_aggregated_geometry_buffers(self.render_pass, self.buffers);
            }

            self.pipeline_tracker.switch_to(Pipeline::StencilDecrement);
        }

        bind_instance_buffers(self.render_pass, location.instance_index, self.buffers);

        self.draw_stencil_geometry(stencil_reference, location.geometry_range);
    }

    /// Draws the material at the supplied reference without modifying stencil.
    pub(super) fn draw_shape(
        &mut self,
        stencil_reference: u32,
        material: ShapeDrawMaterial,
        resources: &ShapeDrawResources,
    ) {
        self.draw_material(stencil_reference, material, resources, false);
    }

    /// Increments stencil within the shape without drawing color.
    pub(super) fn increment_stencil(
        &mut self,
        stencil_reference: u32,
        resources: &ShapeDrawResources,
    ) {
        let pipelines = &self.pipelines.shapes;
        let Some(location) = resources.location else {
            return;
        };
        self.render_pass
            .set_pipeline(&pipelines.stencil_only_pipeline);
        self.pipeline_tracker
            .switch_to(Pipeline::StencilIncrementOnly);
        self.render_pass
            .set_bind_group(0, &pipelines.and_bind_group, &[]);
        self.render_pass
            .set_bind_group(1, &*pipelines.default_shape_texture_bind_groups[0], &[]);
        self.render_pass
            .set_bind_group(2, &*pipelines.default_shape_texture_bind_groups[1], &[]);
        self.bound_textures.mark_bound(0, ShapeTextureBinding::None);
        self.bound_textures.mark_bound(1, ShapeTextureBinding::None);
        bind_aggregated_geometry_buffers(self.render_pass, self.buffers);
        bind_instance_buffers(self.render_pass, location.instance_index, self.buffers);

        self.draw_stencil_geometry(stencil_reference, location.geometry_range);
    }

    /// Composites an intermediate texture under the active scissor and stencil reference.
    pub(super) fn composite_texture(
        &mut self,
        stencil_reference: u32,
        texture_id: IntermediateTextureId,
    ) {
        let Some(resources) = &self.pipelines.composite_resources else {
            return;
        };
        self.render_pass.set_pipeline(&resources.pipeline);
        self.render_pass
            .set_bind_group(0, self.textures.bind_group(texture_id), &[]);
        self.render_pass.set_stencil_reference(stencil_reference);
        self.render_pass.draw(0..3, 0..1);
        self.pipeline_tracker.switch_to(Pipeline::None);
        self.bound_textures.invalidate();
    }
}

/// Rasterizes the prepared shape-effect mask into the active target.
pub(in crate::backend) fn draw_shape_mask(
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
