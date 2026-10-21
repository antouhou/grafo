use super::render_pass::RenderPass;
use super::shapes::{ShapeDrawResources, ShapeGeometryType, ShapeShaderType, StencilAction};
use super::textures::IntermediateTextureResources;
use crate::commands::{IntermediateTextureId, ShapeDrawMaterial};
use crate::core::vertex::InstanceTransform;
use crate::core::UnsignedPhysicalRect;
use crate::wgpu_backend::resources::{Buffers, RendererPipelineResources, ShapePipelines};
use crate::wgpu_backend::vertex::{GeometryBufferRange, InstanceColor, InstanceMetadata};
use wgpu::{BindGroup, IndexFormat, RenderPipeline};

fn bind_instance_buffers(
    render_pass: &mut RenderPass<'_>,
    instance_index: usize,
    buffers: &Buffers,
    uses_coverage: bool,
    uses_texture_bindings: bool,
) {
    if !uses_coverage {
        let transform_offset = instance_index as u64 * InstanceTransform::STRIDE;
        render_pass.set_vertex_buffer(
            1,
            buffers
                .instance_transform_buffer()
                .slice(transform_offset..transform_offset + InstanceTransform::STRIDE),
        );
    }
    let color_buffer_slot = if uses_coverage { 0 } else { 2 };
    let color_offset = instance_index as u64 * InstanceColor::STRIDE;
    render_pass.set_vertex_buffer(
        color_buffer_slot,
        buffers
            .instance_color_buffer()
            .slice(color_offset..color_offset + InstanceColor::STRIDE),
    );
    if uses_coverage || uses_texture_bindings {
        let metadata_buffer_slot = if uses_coverage { 1 } else { 3 };
        let metadata_offset = instance_index as u64 * InstanceMetadata::STRIDE;
        render_pass.set_vertex_buffer(
            metadata_buffer_slot,
            buffers
                .instance_metadata_buffer()
                .slice(metadata_offset..metadata_offset + InstanceMetadata::STRIDE),
        );
    }
}

pub(super) fn bind_aggregated_geometry_buffers(
    render_pass: &mut RenderPass<'_>,
    buffers: &Buffers,
) {
    if buffers.supports_base_vertex {
        render_pass.set_vertex_buffer(0, buffers.vertex_buffer().slice(..));
    }
    render_pass.set_index_buffer(buffers.index_buffer().slice(..), IndexFormat::Uint16);
}

fn bind_decrement_pipeline(render_pass: &mut RenderPass<'_>, pipelines: &ShapePipelines) {
    render_pass.set_pipeline(pipelines.get(
        ShapeGeometryType::Tessellated,
        ShapeShaderType::Solid,
        StencilAction::Decrement,
    ));
    render_pass.set_bind_group(0, &pipelines.decrementing_bind_group, &[]);
}

pub(super) struct DrawPass<'pass, 'encoder> {
    pub(super) root_scissor: Option<UnsignedPhysicalRect>,
    pub(super) render_pass: &'pass mut RenderPass<'encoder>,
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
        self.render_pass.set_scissor(scissor);
    }

    pub(super) fn bind_material_textures(&mut self, material: ShapeDrawMaterial) {
        if !ShapeShaderType::from(material).uses_texture_bindings() {
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
        let shader = ShapeShaderType::from(material);
        let Some(location) = resources.geometry_buffer_location else {
            return;
        };
        if let Some(coverage) = &resources.path_coverage {
            let allocation = &coverage.allocation;
            let buffers = allocation.buffers();
            self.render_pass.set_bind_group(0, &buffers.bind_group, &[]);
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
            allocation.draw(
                self.render_pass,
                |boundary| pipelines.get(ShapeGeometryType::coverage(boundary), shader, stencil),
                0..1,
            );
            return;
        }
        self.render_pass.set_pipeline(pipelines.get(
            ShapeGeometryType::Tessellated,
            shader,
            stencil,
        ));
        self.render_pass
            .set_bind_group(0, &pipelines.and_bind_group, &[]);
        bind_aggregated_geometry_buffers(self.render_pass, self.buffers);
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
            bind_instance_buffers(
                self.render_pass,
                location.instance_index,
                self.buffers,
                true,
                false,
            );
            buffers.bind(self.render_pass);
            self.render_pass.set_stencil_reference(stencil_reference);
            allocation.draw(
                self.render_pass,
                |boundary| {
                    self.pipelines.shapes.get(
                        ShapeGeometryType::coverage(boundary),
                        ShapeShaderType::Solid,
                        StencilAction::Decrement,
                    )
                },
                0..1,
            );
            return;
        }
        bind_decrement_pipeline(self.render_pass, &self.pipelines.shapes);
        bind_aggregated_geometry_buffers(self.render_pass, self.buffers);
        bind_instance_buffers(
            self.render_pass,
            location.instance_index,
            self.buffers,
            false,
            false,
        );
        self.render_pass.set_stencil_reference(stencil_reference);
        self.buffers
            .draw_indexed(self.render_pass, location.geometry_range, 0..1);
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
