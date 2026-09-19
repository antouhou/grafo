pub(super) use self::batching::PendingLeafBatch;
use self::batching::{flush_pending_leaf_batch, queue_or_draw_leaf};
use super::super::state::{Buffers, ShapePipelines};
use super::super::types::{BoundTextureState, Pipeline, PipelineTracker};
use crate::shape::{CachedShapeDrawData, ShapeTextureBinding};
use crate::texture_manager::TextureManager;
use crate::vertex::{InstanceColor, InstanceMetadata, InstanceTransform};
use std::{mem, sync::Arc};
use wgpu::{BindGroup, BindGroupLayout, IndexFormat, RenderPass};

mod batching;

#[allow(clippy::too_many_arguments)]
pub(super) fn bind_shape_texture_layers(
    render_pass: &mut RenderPass<'_>,
    texture_bindings: &[ShapeTextureBinding; 2],
    texture_manager: &TextureManager,
    shape_texture_bind_group_layout_background: &BindGroupLayout,
    shape_texture_bind_group_layout_foreground: &BindGroupLayout,
    default_shape_texture_bind_groups: &[Arc<BindGroup>; 2],
    bound_texture_state: &mut BoundTextureState,
) {
    for (layer, texture_binding) in texture_bindings.iter().enumerate() {
        let effective_binding = match texture_binding {
            ShapeTextureBinding::Managed(texture_id)
                if !texture_manager.is_texture_loaded(*texture_id) =>
            {
                ShapeTextureBinding::None
            }
            texture_binding => texture_binding.clone(),
        };
        if !bound_texture_state.needs_rebind(layer, &effective_binding) {
            continue;
        }
        match &effective_binding {
            ShapeTextureBinding::Managed(texture_id) => {
                match texture_manager.get_or_create_shape_bind_group(
                    if layer == 0 {
                        shape_texture_bind_group_layout_background
                    } else {
                        shape_texture_bind_group_layout_foreground
                    },
                    *texture_id,
                ) {
                    Ok(bind_group) => {
                        render_pass.set_bind_group(1 + layer as u32, &*bind_group, &[]);
                    }
                    Err(_) => {
                        render_pass.set_bind_group(
                            1 + layer as u32,
                            &*default_shape_texture_bind_groups[layer],
                            &[],
                        );
                        bound_texture_state.mark_bound(layer, ShapeTextureBinding::None);
                        continue;
                    }
                }
            }
            ShapeTextureBinding::Direct { bind_group, .. } => {
                render_pass.set_bind_group(1 + layer as u32, bind_group.as_ref(), &[]);
            }
            ShapeTextureBinding::None => {
                render_pass.set_bind_group(
                    1 + layer as u32,
                    &*default_shape_texture_bind_groups[layer],
                    &[],
                );
            }
        }
        bound_texture_state.mark_bound(layer, effective_binding);
    }
}

pub(super) fn bind_instance_buffers(
    render_pass: &mut RenderPass<'_>,
    shape: &CachedShapeDrawData,
    buffers: &Buffers,
) {
    if let Some(instance_idx) = shape.instance_index {
        if let Some(instance_transform_buffer) =
            buffers.aggregated_instance_transform_buffer.as_ref()
        {
            let stride = mem::size_of::<InstanceTransform>() as u64;
            let offset = instance_idx as u64 * stride;
            render_pass
                .set_vertex_buffer(1, instance_transform_buffer.slice(offset..offset + stride));
        } else {
            render_pass.set_vertex_buffer(1, buffers.identity_transform_buffer().slice(..));
        }

        if let Some(instance_color_buffer) = buffers.aggregated_instance_color_buffer.as_ref() {
            let stride = mem::size_of::<InstanceColor>() as u64;
            let offset = instance_idx as u64 * stride;
            render_pass.set_vertex_buffer(2, instance_color_buffer.slice(offset..offset + stride));
        } else {
            render_pass.set_vertex_buffer(2, buffers.identity_color_buffer().slice(..));
        }

        if let Some(instance_metadata_buffer) = buffers.aggregated_instance_metadata_buffer.as_ref()
        {
            let stride = mem::size_of::<InstanceMetadata>() as u64;
            let offset = instance_idx as u64 * stride;
            render_pass
                .set_vertex_buffer(3, instance_metadata_buffer.slice(offset..offset + stride));
        } else {
            render_pass.set_vertex_buffer(3, buffers.identity_metadata_buffer().slice(..));
        }
    } else {
        render_pass.set_vertex_buffer(1, buffers.identity_transform_buffer().slice(..));
        render_pass.set_vertex_buffer(2, buffers.identity_color_buffer().slice(..));
        render_pass.set_vertex_buffer(3, buffers.identity_metadata_buffer().slice(..));
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

pub(super) fn handle_increment_pass<'rp>(
    render_pass: &mut RenderPass<'rp>,
    currently_set_pipeline: &mut PipelineTracker,
    bound_texture_state: &mut BoundTextureState,
    stencil_stack: &mut Vec<u32>,
    shape: &mut CachedShapeDrawData,
    pipelines: &ShapePipelines,
    buffers: &Buffers,
) {
    if let Some(geometry_range) = shape.geometry_buffer_range {
        if shape.is_empty {
            return;
        }

        let uses_gradient = shape.has_gradient_fill();
        let target_pipeline = if uses_gradient {
            Pipeline::StencilIncrementGradient
        } else {
            Pipeline::StencilIncrement
        };

        if currently_set_pipeline.current != target_pipeline {
            render_pass.set_pipeline(if uses_gradient {
                &pipelines.and_gradient_pipeline
            } else {
                &pipelines.and_pipeline
            });
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

        bind_shape_texture_layers(
            render_pass,
            &shape.texture_bindings,
            &pipelines.texture_manager,
            &pipelines.shape_texture_bind_group_layout_background,
            &pipelines.shape_texture_bind_group_layout_foreground,
            &pipelines.default_shape_texture_bind_groups,
            bound_texture_state,
        );

        if uses_gradient {
            let gradient_bg = shape
                .gradient_bind_group
                .as_ref()
                .expect("gradient shapes must prepare a gradient bind group");
            render_pass.set_bind_group(3, gradient_bg.as_ref(), &[]);
        }

        bind_instance_buffers(render_pass, shape, buffers);

        let parent_stencil = stencil_stack.last().copied().unwrap_or(0);
        render_pass.set_stencil_reference(parent_stencil);
        buffers.draw_indexed(render_pass, geometry_range, 0..1);
        #[cfg(feature = "render_metrics")]
        currently_set_pipeline.record_stencil_pass();

        let this_stencil = parent_stencil + 1;
        shape.stencil_ref = Some(this_stencil);
        stencil_stack.push(this_stencil);
    }
}

pub(super) fn handle_decrement_pass<'rp>(
    render_pass: &mut RenderPass<'rp>,
    currently_set_pipeline: &mut PipelineTracker,
    bound_texture_state: &mut BoundTextureState,
    stencil_stack: &mut Vec<u32>,
    shape: &mut CachedShapeDrawData,
    pipelines: &ShapePipelines,
    buffers: &Buffers,
) {
    if let Some(geometry_range) = shape.geometry_buffer_range {
        if shape.is_empty {
            return;
        }

        if !matches!(currently_set_pipeline.current, Pipeline::StencilDecrement) {
            render_pass.set_pipeline(&pipelines.decrementing_pipeline);
            render_pass.set_bind_group(0, &pipelines.decrementing_bind_group, &[]);
            render_pass.set_bind_group(1, &*pipelines.default_shape_texture_bind_groups[0], &[]);
            render_pass.set_bind_group(2, &*pipelines.default_shape_texture_bind_groups[1], &[]);
            bound_texture_state.mark_bound(0, ShapeTextureBinding::None);
            bound_texture_state.mark_bound(1, ShapeTextureBinding::None);

            if !pipeline_has_shared_geometry_bindings(currently_set_pipeline.current) {
                bind_aggregated_geometry_buffers(render_pass, buffers);
            }

            currently_set_pipeline.switch_to(Pipeline::StencilDecrement);
        }

        bind_instance_buffers(render_pass, shape, buffers);

        let this_shape_stencil = shape.stencil_ref.unwrap_or(0);
        render_pass.set_stencil_reference(this_shape_stencil);
        buffers.draw_indexed(render_pass, geometry_range, 0..1);
        #[cfg(feature = "render_metrics")]
        currently_set_pipeline.record_stencil_pass();

        if shape.stencil_ref.is_some() {
            stencil_stack.pop();
        }
    }
}

/// Draw a leaf at its parent's stencil reference without modifying the stencil buffer.
pub(super) fn handle_leaf_draw_pass<'rp>(
    render_pass: &mut RenderPass<'rp>,
    currently_set_pipeline: &mut PipelineTracker,
    bound_texture_state: &mut BoundTextureState,
    stencil_stack: &[u32],
    shape: &mut CachedShapeDrawData,
    pipelines: &ShapePipelines,
    buffers: &Buffers,
) {
    if let Some(geometry_range) = shape.geometry_buffer_range {
        if shape.is_empty {
            return;
        }

        let uses_gradient = shape.has_gradient_fill();
        let target_pipeline = if uses_gradient {
            Pipeline::LeafDrawGradient
        } else {
            Pipeline::LeafDraw
        };

        if currently_set_pipeline.current != target_pipeline {
            render_pass.set_pipeline(if uses_gradient {
                &pipelines.leaf_draw_gradient_pipeline
            } else {
                &pipelines.leaf_draw_pipeline
            });
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

        bind_shape_texture_layers(
            render_pass,
            &shape.texture_bindings,
            &pipelines.texture_manager,
            &pipelines.shape_texture_bind_group_layout_background,
            &pipelines.shape_texture_bind_group_layout_foreground,
            &pipelines.default_shape_texture_bind_groups,
            bound_texture_state,
        );

        if uses_gradient {
            let gradient_bg = shape
                .gradient_bind_group
                .as_ref()
                .expect("gradient shapes must prepare a gradient bind group");
            render_pass.set_bind_group(3, gradient_bg.as_ref(), &[]);
        }

        bind_instance_buffers(render_pass, shape, buffers);

        let parent_stencil = stencil_stack.last().copied().unwrap_or(0);
        render_pass.set_stencil_reference(parent_stencil);
        buffers.draw_indexed(render_pass, geometry_range, 0..1);

        // The leaf inherits its parent's stencil reference because it makes no stencil writes.
        shape.stencil_ref = Some(parent_stencil);
    }
}

/// Borrows drawing state for one pass without changing traversal clipping or batch boundaries.
pub(super) struct ShapePass<'state, 'encoder> {
    pub(super) render_pass: &'state mut RenderPass<'encoder>,
    pub(super) currently_set_pipeline: &'state mut PipelineTracker,
    pub(super) bound_texture_state: &'state mut BoundTextureState,
    pub(super) pending_leaf_batch: &'state mut PendingLeafBatch,
    pub(super) pipelines: &'state ShapePipelines,
    pub(super) buffers: &'state Buffers,
}

impl ShapePass<'_, '_> {
    pub(super) fn flush(&mut self) {
        flush_pending_leaf_batch(
            self.pending_leaf_batch,
            self.render_pass,
            self.currently_set_pipeline,
            self.bound_texture_state,
            self.pipelines,
            self.buffers,
        );
    }

    pub(super) fn queue_leaf(
        &mut self,
        shape: &mut CachedShapeDrawData,
        parent_stencil: u32,
        stencil_stack: &[u32],
    ) {
        queue_or_draw_leaf(
            shape,
            parent_stencil,
            self.pending_leaf_batch,
            self.render_pass,
            self.currently_set_pipeline,
            self.bound_texture_state,
            stencil_stack,
            self.pipelines,
            self.buffers,
        );
    }

    pub(super) fn draw_leaf(&mut self, shape: &mut CachedShapeDrawData, stencil_stack: &[u32]) {
        handle_leaf_draw_pass(
            self.render_pass,
            self.currently_set_pipeline,
            self.bound_texture_state,
            stencil_stack,
            shape,
            self.pipelines,
            self.buffers,
        );
    }

    pub(super) fn increment_stencil(
        &mut self,
        shape: &mut CachedShapeDrawData,
        stencil_stack: &mut Vec<u32>,
    ) {
        handle_increment_pass(
            self.render_pass,
            self.currently_set_pipeline,
            self.bound_texture_state,
            stencil_stack,
            shape,
            self.pipelines,
            self.buffers,
        );
    }

    pub(super) fn decrement_stencil(
        &mut self,
        shape: &mut CachedShapeDrawData,
        stencil_stack: &mut Vec<u32>,
    ) {
        handle_decrement_pass(
            self.render_pass,
            self.currently_set_pipeline,
            self.bound_texture_state,
            stencil_stack,
            shape,
            self.pipelines,
            self.buffers,
        );
    }
}
