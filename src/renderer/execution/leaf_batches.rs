use super::draws;
use super::shapes::ShapeDrawResources;
use super::textures::IntermediateTextureResources;
use crate::renderer::state::{Buffers, ShapePipelines};
use crate::renderer::types::{BoundTextureState, Pipeline, PipelineTracker};
use crate::shape::{CachedShapeDrawData, ShapeTextureBinding};
use crate::vertex::GeometryBufferRange;
use wgpu::RenderPass;

#[derive(Default)]
pub(in crate::renderer) struct PendingLeafBatch {
    geometry_range: GeometryBufferRange,
    texture_bindings: [ShapeTextureBinding; 2],
    stencil_reference: u32,
    first_instance_index: u32,
    instance_count: u32,
}

impl PendingLeafBatch {
    fn is_empty(&self) -> bool {
        self.instance_count == 0
    }

    fn matches(
        &self,
        geometry_range: GeometryBufferRange,
        texture_bindings: &[ShapeTextureBinding; 2],
        stencil_reference: u32,
        instance_index: u32,
    ) -> bool {
        self.geometry_range == geometry_range
            && self.texture_bindings == *texture_bindings
            && self.stencil_reference == stencil_reference
            && instance_index == self.first_instance_index + self.instance_count
    }
}

/// Ensure the leaf-draw pipeline and full instance buffers are bound,
/// then issue one `draw_indexed` call for the accumulated batch.
pub(in crate::renderer) fn flush_pending_leaf_batch(
    batch: &mut PendingLeafBatch,
    render_pass: &mut RenderPass<'_>,
    currently_set_pipeline: &mut PipelineTracker,
    bound_texture_state: &mut BoundTextureState,
    pipelines: &ShapePipelines,
    buffers: &Buffers,
    textures: &IntermediateTextureResources,
) {
    if batch.is_empty() {
        return;
    }

    if currently_set_pipeline.current != Pipeline::LeafDraw {
        render_pass.set_pipeline(&pipelines.leaf_draw_pipeline);
        render_pass.set_bind_group(0, &pipelines.and_bind_group, &[]);
        render_pass.set_bind_group(1, &*pipelines.default_shape_texture_bind_groups[0], &[]);
        render_pass.set_bind_group(2, &*pipelines.default_shape_texture_bind_groups[1], &[]);
        bound_texture_state.mark_bound(0, ShapeTextureBinding::None);
        bound_texture_state.mark_bound(1, ShapeTextureBinding::None);
        currently_set_pipeline.switch_to(Pipeline::LeafDraw);
    }

    draws::bind_aggregated_geometry_buffers(render_pass, buffers);
    textures.bind_shape_texture_layers(
        render_pass,
        &batch.texture_bindings,
        &pipelines.texture_manager,
        &pipelines.shape_texture_bind_group_layout_background,
        &pipelines.shape_texture_bind_group_layout_foreground,
        &pipelines.default_shape_texture_bind_groups,
        bound_texture_state,
    );
    render_pass.set_vertex_buffer(1, buffers.instance_transform_buffer().slice(..));
    render_pass.set_vertex_buffer(2, buffers.instance_color_buffer().slice(..));
    render_pass.set_vertex_buffer(3, buffers.instance_metadata_buffer().slice(..));

    render_pass.set_stencil_reference(batch.stencil_reference);
    let first_instance_index = batch.first_instance_index;
    buffers.draw_indexed(
        render_pass,
        batch.geometry_range,
        first_instance_index..first_instance_index + batch.instance_count,
    );
    batch.instance_count = 0;
}

/// Tries to add a leaf shape to the pending batch. Returns `true` when added.
/// On `false`, the caller must flush the batch and draw the shape separately.
fn try_batch_leaf(
    batch: &mut PendingLeafBatch,
    shape: &CachedShapeDrawData,
    resources: &ShapeDrawResources,
    stencil_reference: u32,
) -> bool {
    let Some(location) = resources.location else {
        return false;
    };
    // Shapes with per-shape gradient bind groups cannot be batched.
    if shape.has_gradient_fill() {
        return false;
    }
    let instance_index = location.instance_index as u32;
    let texture_bindings = &shape.texture_bindings;

    if batch.is_empty() {
        batch.geometry_range = location.geometry_range;
        batch.texture_bindings = *texture_bindings;
        batch.stencil_reference = stencil_reference;
        batch.first_instance_index = instance_index;
        batch.instance_count = 1;
        return true;
    }

    if batch.matches(
        location.geometry_range,
        texture_bindings,
        stencil_reference,
        instance_index,
    ) {
        batch.instance_count += 1;
        return true;
    }

    // The caller must flush the incompatible batch before drawing this shape.
    false
}

#[allow(clippy::too_many_arguments)]
pub(in crate::renderer) fn queue_or_draw_leaf(
    shape: &CachedShapeDrawData,
    resources: &ShapeDrawResources,
    stencil_reference: u32,
    pending_leaf_batch: &mut PendingLeafBatch,
    render_pass: &mut RenderPass<'_>,
    currently_set_pipeline: &mut PipelineTracker,
    bound_texture_state: &mut BoundTextureState,
    pipelines: &ShapePipelines,
    buffers: &Buffers,
    textures: &IntermediateTextureResources,
) {
    if try_batch_leaf(pending_leaf_batch, shape, resources, stencil_reference) {
        return;
    }

    flush_pending_leaf_batch(
        pending_leaf_batch,
        render_pass,
        currently_set_pipeline,
        bound_texture_state,
        pipelines,
        buffers,
        textures,
    );
    if try_batch_leaf(pending_leaf_batch, shape, resources, stencil_reference) {
        return;
    }

    draws::draw_shape(
        render_pass,
        currently_set_pipeline,
        bound_texture_state,
        stencil_reference,
        shape.material(),
        resources,
        pipelines,
        buffers,
        textures,
    );
}
