use super::{bind_aggregated_geometry_buffers, bind_shape_texture_layers, handle_leaf_draw_pass};
use crate::renderer::state::{Buffers, ShapePipelines};
use crate::renderer::types::{BoundTextureState, Pipeline, PipelineTracker};
use crate::shape::{CachedShapeDrawData, ShapeTextureBinding};
use crate::vertex::GeometryBufferRange;
use wgpu::RenderPass;

#[derive(Default)]
pub(in crate::renderer::passes) struct PendingLeafBatch {
    geometry_range: GeometryBufferRange,
    texture_bindings: [ShapeTextureBinding; 2],
    parent_stencil: u32,
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
        parent_stencil: u32,
        instance_index: u32,
    ) -> bool {
        self.geometry_range == geometry_range
            && self.texture_bindings == *texture_bindings
            && self.parent_stencil == parent_stencil
            && instance_index == self.first_instance_index + self.instance_count
    }
}

/// Ensure the leaf-draw pipeline and full instance buffers are bound,
/// then issue one `draw_indexed` call for the accumulated batch.
pub(super) fn flush_pending_leaf_batch(
    batch: &mut PendingLeafBatch,
    render_pass: &mut RenderPass<'_>,
    currently_set_pipeline: &mut PipelineTracker,
    bound_texture_state: &mut BoundTextureState,
    pipelines: &ShapePipelines,
    buffers: &Buffers,
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

    bind_aggregated_geometry_buffers(render_pass, buffers);
    bind_shape_texture_layers(
        render_pass,
        &batch.texture_bindings,
        &pipelines.texture_manager,
        &pipelines.shape_texture_bind_group_layout_background,
        &pipelines.shape_texture_bind_group_layout_foreground,
        &pipelines.default_shape_texture_bind_groups,
        bound_texture_state,
    );
    if let Some(instance_transform_buffer) = buffers.aggregated_instance_transform_buffer.as_ref() {
        render_pass.set_vertex_buffer(1, instance_transform_buffer.slice(..));
    } else {
        render_pass.set_vertex_buffer(1, buffers.identity_transform_buffer().slice(..));
    }
    if let Some(instance_color_buffer) = buffers.aggregated_instance_color_buffer.as_ref() {
        render_pass.set_vertex_buffer(2, instance_color_buffer.slice(..));
    } else {
        render_pass.set_vertex_buffer(2, buffers.identity_color_buffer().slice(..));
    }
    if let Some(instance_metadata_buffer) = buffers.aggregated_instance_metadata_buffer.as_ref() {
        render_pass.set_vertex_buffer(3, instance_metadata_buffer.slice(..));
    } else {
        render_pass.set_vertex_buffer(3, buffers.identity_metadata_buffer().slice(..));
    }

    render_pass.set_stencil_reference(batch.parent_stencil);
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
pub(super) fn try_batch_leaf(
    batch: &mut PendingLeafBatch,
    shape: &CachedShapeDrawData,
    parent_stencil: u32,
) -> bool {
    let geometry_range = match shape.geometry_buffer_range {
        Some(range) => range,
        None => return false,
    };
    if shape.is_empty {
        return false;
    }
    // Shapes with per-shape gradient bind groups cannot be batched.
    if shape.has_gradient_fill() {
        return false;
    }
    let instance_index = match shape.instance_index {
        Some(idx) => idx as u32,
        None => return false,
    };
    let texture_bindings = &shape.texture_bindings;

    if batch.is_empty() {
        batch.geometry_range = geometry_range;
        batch.texture_bindings = texture_bindings.clone();
        batch.parent_stencil = parent_stencil;
        batch.first_instance_index = instance_index;
        batch.instance_count = 1;
        return true;
    }

    if batch.matches(
        geometry_range,
        texture_bindings,
        parent_stencil,
        instance_index,
    ) {
        batch.instance_count += 1;
        return true;
    }

    // The caller must flush the incompatible batch before drawing this shape.
    false
}

#[allow(clippy::too_many_arguments)]
pub(super) fn queue_or_draw_leaf(
    shape: &mut CachedShapeDrawData,
    parent_stencil: u32,
    pending_leaf_batch: &mut PendingLeafBatch,
    render_pass: &mut RenderPass<'_>,
    currently_set_pipeline: &mut PipelineTracker,
    bound_texture_state: &mut BoundTextureState,
    stencil_stack: &[u32],
    pipelines: &ShapePipelines,
    buffers: &Buffers,
) {
    if try_batch_leaf(pending_leaf_batch, shape, parent_stencil) {
        shape.stencil_ref = Some(parent_stencil);
        return;
    }

    flush_pending_leaf_batch(
        pending_leaf_batch,
        render_pass,
        currently_set_pipeline,
        bound_texture_state,
        pipelines,
        buffers,
    );
    if try_batch_leaf(pending_leaf_batch, shape, parent_stencil) {
        shape.stencil_ref = Some(parent_stencil);
        return;
    }

    handle_leaf_draw_pass(
        render_pass,
        currently_set_pipeline,
        bound_texture_state,
        stencil_stack,
        shape,
        pipelines,
        buffers,
    );
}
