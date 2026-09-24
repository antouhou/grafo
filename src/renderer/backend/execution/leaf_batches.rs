use super::draws::{self, DrawPass};
use super::shapes::ShapeExecutionResources;
use super::targets;
use crate::commands::{DrawClip, RenderCommand, RenderOperation, ShapeTextureBinding};
use crate::renderer::backend::vertex::GeometryBufferRange;
use crate::renderer::types::Pipeline;

struct LeafBatch {
    geometry_range: GeometryBufferRange,
    texture_bindings: [ShapeTextureBinding; 2],
    first_instance_index: u32,
    instance_count: u32,
}

impl DrawPass<'_, '_> {
    fn draw_batch(&mut self, batch: &LeafBatch, clip: DrawClip) {
        let pipelines = &self.pipelines.shapes;
        targets::set_scissor(self.render_pass, clip.scissor);
        if self.pipeline_tracker.current != Pipeline::LeafDraw {
            self.render_pass.set_pipeline(&pipelines.leaf_draw_pipeline);
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
            self.pipeline_tracker.switch_to(Pipeline::LeafDraw);
        }
        draws::bind_aggregated_geometry_buffers(self.render_pass, self.buffers);
        self.textures.bind_shape_texture_layers(
            self.render_pass,
            &batch.texture_bindings,
            &pipelines.texture_manager,
            &pipelines.shape_texture_bind_group_layout_background,
            &pipelines.shape_texture_bind_group_layout_foreground,
            &pipelines.default_shape_texture_bind_groups,
            self.bound_textures,
        );
        self.render_pass
            .set_vertex_buffer(1, self.buffers.instance_transform_buffer().slice(..));
        self.render_pass
            .set_vertex_buffer(2, self.buffers.instance_color_buffer().slice(..));
        self.render_pass
            .set_vertex_buffer(3, self.buffers.instance_metadata_buffer().slice(..));
        self.render_pass
            .set_stencil_reference(clip.stencil_reference);
        self.buffers.draw_indexed(
            self.render_pass,
            batch.geometry_range,
            batch.first_instance_index..batch.first_instance_index + batch.instance_count,
        );
    }

    /// Batches consecutive draws that share geometry, textures, and clipping.
    /// Returns the number of consumed commands.
    pub(super) fn execute_leaf_draws(
        &mut self,
        instructions: &[RenderCommand],
        shapes: &ShapeExecutionResources,
    ) -> usize {
        let first = &instructions[0];
        let RenderOperation::DrawShape(draw) = &first.operation else {
            unreachable!("leaf batch starts with a shape draw");
        };
        let resources = shapes.draw_resources(draw.id);
        let Some(location) = resources.location else {
            return 1;
        };
        if draw.material.has_gradient_fill() || draw.material.under_fill_texture.is_some() {
            targets::set_scissor(self.render_pass, first.clip.scissor);
            self.draw_shape(first.clip.stencil_reference, draw.material, resources);
            return 1;
        }
        let mut batch = LeafBatch {
            geometry_range: location.geometry_range,
            texture_bindings: draw.material.texture_bindings,
            first_instance_index: location.instance_index as u32,
            instance_count: 1,
        };
        for next in &instructions[1..] {
            let RenderOperation::DrawShape(next_draw) = &next.operation else {
                break;
            };
            if next.clip != first.clip
                || next_draw.material.has_gradient_fill()
                || next_draw.material.under_fill_texture.is_some()
                || next_draw.material.texture_bindings != batch.texture_bindings
            {
                break;
            }
            let Some(location) = shapes.draw_resources(next_draw.id).location else {
                break;
            };
            if location.geometry_range != batch.geometry_range
                || location.instance_index as u32
                    != batch.first_instance_index + batch.instance_count
            {
                break;
            }
            batch.instance_count += 1;
        }
        self.draw_batch(&batch, first.clip);
        batch.instance_count as usize
    }
}
