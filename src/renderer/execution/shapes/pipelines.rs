use crate::pipeline;
use crate::renderer::state::ShapePipelines;
use crate::renderer::types::Pipeline;
use crate::shape::ShapeDrawMaterial;
use wgpu::{BindGroupLayout, Device, RenderPipeline, TextureFormat};

/// Shape-material variants created only when a draw needs an under-fill texture.
pub(in crate::renderer) struct TextureMaterialPipelines {
    pub(in crate::renderer) solid_layout: BindGroupLayout,
    pub(in crate::renderer) gradient_layout: BindGroupLayout,
    pub(in crate::renderer) solid_keep: RenderPipeline,
    pub(in crate::renderer) gradient_keep: RenderPipeline,
    pub(in crate::renderer) solid_increment: RenderPipeline,
    pub(in crate::renderer) gradient_increment: RenderPipeline,
}

impl TextureMaterialPipelines {
    pub(in crate::renderer) fn new(
        device: &Device,
        format: TextureFormat,
        sample_count: u32,
        shapes: &ShapePipelines,
    ) -> Self {
        let solid_layout = pipeline::create_texture_material_bind_group_layout(device);
        let gradient_layout = pipeline::create_gradient_texture_material_bind_group_layout(device);
        let uniforms = shapes.and_pipeline.get_bind_group_layout(0);
        let solid_layouts = [
            &uniforms,
            &shapes.shape_texture_bind_group_layout_background,
            &shapes.shape_texture_bind_group_layout_foreground,
            &solid_layout,
        ];
        let gradient_layouts = [
            solid_layouts[0],
            solid_layouts[1],
            solid_layouts[2],
            &gradient_layout,
        ];
        Self {
            solid_keep: pipeline::create_texture_material_pipeline(
                device,
                format,
                sample_count,
                &solid_layouts,
                false,
                false,
            ),
            gradient_keep: pipeline::create_texture_material_pipeline(
                device,
                format,
                sample_count,
                &gradient_layouts,
                true,
                false,
            ),
            solid_increment: pipeline::create_texture_material_pipeline(
                device,
                format,
                sample_count,
                &solid_layouts,
                false,
                true,
            ),
            gradient_increment: pipeline::create_texture_material_pipeline(
                device,
                format,
                sample_count,
                &gradient_layouts,
                true,
                true,
            ),
            solid_layout,
            gradient_layout,
        }
    }
}

impl ShapePipelines {
    pub(in crate::renderer) fn material_pipeline(
        &self,
        material: ShapeDrawMaterial<'_>,
        increments_stencil: bool,
    ) -> (Pipeline, &RenderPipeline) {
        let uses_gradient = material.has_gradient_fill();
        if material.under_fill_texture.is_some() {
            let pipelines = self
                .under_fill_pipelines
                .as_ref()
                .expect("texture material pipelines are initialized before drawing");
            return match (uses_gradient, increments_stencil) {
                (false, false) => (Pipeline::LeafDrawTexture, &pipelines.solid_keep),
                (true, false) => (Pipeline::LeafDrawGradientTexture, &pipelines.gradient_keep),
                (false, true) => (
                    Pipeline::StencilIncrementTexture,
                    &pipelines.solid_increment,
                ),
                (true, true) => (
                    Pipeline::StencilIncrementGradientTexture,
                    &pipelines.gradient_increment,
                ),
            };
        }
        match (uses_gradient, increments_stencil) {
            (false, false) => (Pipeline::LeafDraw, &self.leaf_draw_pipeline),
            (true, false) => (
                Pipeline::LeafDrawGradient,
                &self.leaf_draw_gradient_pipeline,
            ),
            (false, true) => (Pipeline::StencilIncrement, &self.and_pipeline),
            (true, true) => (
                Pipeline::StencilIncrementGradient,
                &self.and_gradient_pipeline,
            ),
        }
    }
}
