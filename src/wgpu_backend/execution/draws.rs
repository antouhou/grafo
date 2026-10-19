use super::shapes::{ShapeDrawResources, ShapeGeometry, ShapeShader, StencilAction};
use super::targets;
use super::textures::IntermediateTextureResources;
use crate::commands::{IntermediateTextureId, ShapeDrawMaterial, ShapeTextureBinding};
use crate::core::vertex::InstanceTransform;
use crate::core::UnsignedPhysicalRect;
use crate::wgpu_backend::resources::{Buffers, RendererPipelineResources, ShapePipelines};
use crate::wgpu_backend::types::{BoundTextureState, Pipeline, PipelineTracker};
use crate::wgpu_backend::vertex::{GeometryBufferRange, InstanceColor, InstanceMetadata};
use wgpu::{BindGroup, IndexFormat, RenderPass, RenderPipeline};

fn bind_instance_buffers(
    render_pass: &mut RenderPass<'_>,
    instance_index: usize,
    buffers: &Buffers,
    uses_coverage: bool,
    uses_texture_bindings: bool,
) {
    if !uses_coverage || uses_texture_bindings {
        let transform_offset = instance_index as u64 * InstanceTransform::STRIDE;
        render_pass.set_vertex_buffer(
            1,
            buffers
                .instance_transform_buffer()
                .slice(transform_offset..transform_offset + InstanceTransform::STRIDE),
        );
    }
    let uses_solid_coverage = uses_coverage && !uses_texture_bindings;
    let color_offset = instance_index as u64 * InstanceColor::STRIDE;
    render_pass.set_vertex_buffer(
        if uses_solid_coverage { 1 } else { 2 },
        buffers
            .instance_color_buffer()
            .slice(color_offset..color_offset + InstanceColor::STRIDE),
    );
    if uses_coverage || uses_texture_bindings {
        let metadata_offset = instance_index as u64 * InstanceMetadata::STRIDE;
        render_pass.set_vertex_buffer(
            if uses_solid_coverage { 2 } else { 3 },
            buffers
                .instance_metadata_buffer()
                .slice(metadata_offset..metadata_offset + InstanceMetadata::STRIDE),
        );
    }
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
    render_pass.set_pipeline(pipelines.get(
        ShapeGeometry::Tessellated,
        ShapeShader::Solid,
        StencilAction::Decrement,
    ));
    render_pass.set_bind_group(0, &pipelines.decrementing_bind_group, &[]);
    render_pass.set_bind_group(1, &*pipelines.default_shape_texture_bind_groups[0], &[]);
    render_pass.set_bind_group(2, &*pipelines.default_shape_texture_bind_groups[1], &[]);
}

pub(super) struct DrawPass<'pass, 'encoder> {
    pub(super) root_scissor: Option<UnsignedPhysicalRect>,
    pub(super) render_pass: &'pass mut RenderPass<'encoder>,
    pub(super) pipeline_tracker: &'pass mut PipelineTracker,
    pub(super) bound_textures: &'pass mut BoundTextureState,
    pub(super) pipelines: &'pass RendererPipelineResources,
    pub(super) buffers: &'pass Buffers,
    pub(super) textures: &'pass IntermediateTextureResources,
}

impl DrawPass<'_, '_> {
    pub(super) fn set_scissor(&mut self, scissor: UnsignedPhysicalRect) {
        let scissor = self.root_scissor.map_or(scissor, |root| {
            root.intersection(&scissor)
                .unwrap_or_else(UnsignedPhysicalRect::zero)
        });
        targets::set_scissor(self.render_pass, scissor);
    }

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

    pub(super) fn bind_material_textures(&mut self, material: ShapeDrawMaterial) {
        if !ShapeShader::from(material).uses_texture_bindings() {
            return;
        }
        let pipelines = &self.pipelines.shapes;
        self.textures.bind_shape_texture_layers(
            self.render_pass,
            &material.texture_bindings,
            &pipelines.texture_manager,
            &pipelines.layouts.background_texture,
            &pipelines.layouts.foreground_texture,
            &pipelines.default_shape_texture_bind_groups,
            self.bound_textures,
        );
    }

    fn draw_material(
        &mut self,
        stencil_reference: u32,
        material: ShapeDrawMaterial,
        resources: &ShapeDrawResources,
        stencil: StencilAction,
    ) {
        let pipelines = &self.pipelines.shapes;
        let shader = ShapeShader::from(material);
        let Some(location) = resources.geometry_buffer_location else {
            return;
        };
        if let Some(coverage) = &resources.path_coverage {
            let allocation = &coverage.allocation;
            let buffers = allocation.buffers();
            self.render_pass.set_bind_group(0, &buffers.bind_group, &[]);
            self.bound_textures.invalidate();
            self.bind_material_textures(material);
            if let Some(binding) = resources.material_bind_group(material) {
                self.render_pass.set_bind_group(3, binding, &[]);
            }
            bind_instance_buffers(
                self.render_pass,
                location.instance_index,
                self.buffers,
                true,
                shader.uses_texture_bindings(),
            );
            buffers.bind(self.render_pass);
            self.render_pass.set_stencil_reference(stencil_reference);
            let _draw_count = allocation.draw(
                self.render_pass,
                |boundary| pipelines.get(ShapeGeometry::coverage(boundary), shader, stencil),
                0..1,
            );
            self.pipeline_tracker.current = Pipeline::None;
            #[cfg(feature = "render_metrics")]
            self.pipeline_tracker.record_coverage_draws(
                shader.tracked_pipeline(stencil),
                _draw_count,
                stencil == StencilAction::Increment,
            );
            self.bound_textures.invalidate();
            bind_aggregated_geometry_buffers(self.render_pass, self.buffers);
            return;
        }
        let target_pipeline = shader.tracked_pipeline(stencil);
        if self.pipeline_tracker.current != target_pipeline {
            self.render_pass.set_pipeline(pipelines.get(
                ShapeGeometry::Tessellated,
                shader,
                stencil,
            ));
            self.render_pass
                .set_bind_group(0, &pipelines.and_bind_group, &[]);
            self.bound_textures.invalidate();
            if !pipeline_has_shared_geometry_bindings(self.pipeline_tracker.current) {
                bind_aggregated_geometry_buffers(self.render_pass, self.buffers);
            }
            self.pipeline_tracker.switch_to(target_pipeline);
        }
        self.bind_material_textures(material);
        if let Some(binding) = resources.material_bind_group(material) {
            self.render_pass.set_bind_group(3, binding, &[]);
        }
        bind_instance_buffers(
            self.render_pass,
            location.instance_index,
            self.buffers,
            false,
            shader.uses_texture_bindings(),
        );
        self.render_pass.set_stencil_reference(stencil_reference);
        self.buffers
            .draw_indexed(self.render_pass, location.geometry_range, 0..1);
        #[cfg(feature = "render_metrics")]
        if stencil == StencilAction::Increment {
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
        self.draw_material(
            stencil_reference,
            material,
            resources,
            StencilAction::Increment,
        );
    }

    /// Decrements stencil samples matching the supplied reference without drawing color.
    pub(super) fn decrement_stencil(
        &mut self,
        stencil_reference: u32,
        resources: &ShapeDrawResources,
    ) {
        let Some(location) = resources.geometry_buffer_location else {
            return;
        };
        if let Some(coverage) = &resources.path_coverage {
            let allocation = &coverage.allocation;
            let buffers = allocation.buffers();
            self.render_pass
                .set_bind_group(0, &buffers.decrement_bind_group, &[]);
            for layer in 0..2 {
                self.render_pass.set_bind_group(
                    layer as u32 + 1,
                    &*self.pipelines.shapes.default_shape_texture_bind_groups[layer],
                    &[],
                );
            }
            bind_instance_buffers(
                self.render_pass,
                location.instance_index,
                self.buffers,
                true,
                true,
            );
            buffers.bind(self.render_pass);
            self.render_pass.set_stencil_reference(stencil_reference);
            let _draw_count = allocation.draw(
                self.render_pass,
                |boundary| {
                    self.pipelines.shapes.get(
                        ShapeGeometry::coverage(boundary),
                        ShapeShader::Solid,
                        StencilAction::Decrement,
                    )
                },
                0..1,
            );
            self.pipeline_tracker.current = Pipeline::None;
            #[cfg(feature = "render_metrics")]
            self.pipeline_tracker.record_coverage_draws(
                Pipeline::StencilDecrement,
                _draw_count,
                true,
            );
            self.bound_textures.invalidate();
            bind_aggregated_geometry_buffers(self.render_pass, self.buffers);
            return;
        }
        if !matches!(self.pipeline_tracker.current, Pipeline::StencilDecrement) {
            bind_decrement_pipeline(self.render_pass, &self.pipelines.shapes);
            self.bound_textures.mark_bound(0, ShapeTextureBinding::None);
            self.bound_textures.mark_bound(1, ShapeTextureBinding::None);

            if !pipeline_has_shared_geometry_bindings(self.pipeline_tracker.current) {
                bind_aggregated_geometry_buffers(self.render_pass, self.buffers);
            }

            self.pipeline_tracker.switch_to(Pipeline::StencilDecrement);
        }

        bind_instance_buffers(
            self.render_pass,
            location.instance_index,
            self.buffers,
            false,
            true,
        );

        self.draw_stencil_geometry(stencil_reference, location.geometry_range);
    }

    /// Draws the material at the supplied reference without modifying stencil.
    pub(super) fn draw_shape(
        &mut self,
        stencil_reference: u32,
        material: ShapeDrawMaterial,
        resources: &ShapeDrawResources,
    ) {
        self.draw_material(stencil_reference, material, resources, StencilAction::Keep);
    }

    /// Composites an intermediate texture under the active scissor and stencil reference.
    pub(super) fn composite_texture(
        &mut self,
        stencil_reference: u32,
        texture_id: IntermediateTextureId,
    ) {
        let resources = &self.pipelines.composite_resources;
        self.render_pass.set_pipeline(&resources.pipeline);
        self.render_pass
            .set_bind_group(0, self.textures.bind_group(texture_id), &[]);
        self.render_pass.set_stencil_reference(stencil_reference);
        self.render_pass.draw(0..3, 0..1);
        self.pipeline_tracker.switch_to(Pipeline::None);
        self.bound_textures.invalidate();
    }
}

/// Rasterizes the prepared shape mask effect mask into the active target.
pub(in crate::wgpu_backend) fn draw_shape_mask(
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
