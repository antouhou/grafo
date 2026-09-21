use super::execution::backdrops;
use super::execution::shapes::ShapeDrawResources;
use super::execution::textures::IntermediateTextureResources;
use super::plan::backdrops::compute_backdrop_capture_region;
use super::state::{Buffers, RendererPipelineResources, RendererState, ShapePipelines};
use super::types::{
    BackdropContext, BackdropSource, BoundTextureState, ClipKind, Pipeline, PipelineTracker,
    TraversalEvent,
};
use super::*;
use crate::pipeline::{begin_render_pass_with_load_ops, RenderPassLoadOperations};
use crate::renderer::rect_utils::{should_skip_visible_rect_draw, try_scissor_for_rect};
use crate::shape::{CachedShapeDrawData, ShapeTextureBinding};
use crate::{MathRect, Size, UnsignedPhysicalRect};

fn cached_shape(draw_tree_node: &DrawTreeNode) -> &CachedShapeDrawData {
    match draw_tree_node {
        DrawTreeNode::CachedShape(shape) => shape,
        DrawTreeNode::ClipRect(_) => unreachable!("clip rectangles do not own shape geometry"),
    }
}

pub(super) fn bind_instance_buffers(
    render_pass: &mut wgpu::RenderPass<'_>,
    resources: &ShapeDrawResources,
    buffers: &Buffers,
) {
    if let Some(instance_index) = resources.instance_index {
        if let Some(instance_transform_buffer) =
            buffers.aggregated_instance_transform_buffer.as_ref()
        {
            let stride = InstanceTransform::STRIDE;
            let offset = instance_index as u64 * stride;
            render_pass
                .set_vertex_buffer(1, instance_transform_buffer.slice(offset..offset + stride));
        } else {
            render_pass.set_vertex_buffer(1, buffers.identity_transform_buffer().slice(..));
        }

        if let Some(instance_color_buffer) = buffers.aggregated_instance_color_buffer.as_ref() {
            let stride = InstanceColor::STRIDE;
            let offset = instance_index as u64 * stride;
            render_pass.set_vertex_buffer(2, instance_color_buffer.slice(offset..offset + stride));
        } else {
            render_pass.set_vertex_buffer(2, buffers.identity_color_buffer().slice(..));
        }

        if let Some(instance_metadata_buffer) = buffers.aggregated_instance_metadata_buffer.as_ref()
        {
            let stride = InstanceMetadata::STRIDE;
            let offset = instance_index as u64 * stride;
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

fn bind_aggregated_geometry_buffers(render_pass: &mut wgpu::RenderPass<'_>, buffers: &Buffers) {
    render_pass.set_vertex_buffer(0, buffers.vertex_buffer().slice(..));
    render_pass.set_index_buffer(buffers.index_buffer().slice(..), wgpu::IndexFormat::Uint16);
}

/// Draws color and increments stencil samples matching the supplied reference.
#[allow(clippy::too_many_arguments)]
pub(super) fn handle_increment_pass<'rp>(
    render_pass: &mut wgpu::RenderPass<'rp>,
    currently_set_pipeline: &mut PipelineTracker,
    bound_texture_state: &mut BoundTextureState,
    stencil_reference: u32,
    shape: &CachedShapeDrawData,
    resources: &ShapeDrawResources,
    pipelines: &ShapePipelines,
    buffers: &Buffers,
    textures: &IntermediateTextureResources,
) {
    if let Some(geometry_range) = resources.geometry_buffer_range {
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

        textures.bind_shape_texture_layers(
            render_pass,
            &shape.texture_bindings,
            &pipelines.texture_manager,
            &pipelines.shape_texture_bind_group_layout_background,
            &pipelines.shape_texture_bind_group_layout_foreground,
            &pipelines.default_shape_texture_bind_groups,
            bound_texture_state,
        );

        if uses_gradient {
            let gradient_bind_group = resources
                .gradient_bind_group
                .as_ref()
                .expect("gradient shapes must prepare a gradient bind group");
            render_pass.set_bind_group(3, gradient_bind_group.as_ref(), &[]);
        }

        bind_instance_buffers(render_pass, resources, buffers);

        render_pass.set_stencil_reference(stencil_reference);
        buffers.draw_indexed(render_pass, geometry_range, 0..1);
        #[cfg(feature = "render_metrics")]
        currently_set_pipeline.record_stencil_pass();
    }
}

/// Decrements stencil samples matching the supplied reference without drawing color.
pub(super) fn handle_decrement_pass<'rp>(
    render_pass: &mut wgpu::RenderPass<'rp>,
    currently_set_pipeline: &mut PipelineTracker,
    bound_texture_state: &mut BoundTextureState,
    stencil_reference: u32,
    resources: &ShapeDrawResources,
    pipelines: &ShapePipelines,
    buffers: &Buffers,
) {
    if let Some(geometry_range) = resources.geometry_buffer_range {
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

        bind_instance_buffers(render_pass, resources, buffers);

        render_pass.set_stencil_reference(stencil_reference);
        buffers.draw_indexed(render_pass, geometry_range, 0..1);
        #[cfg(feature = "render_metrics")]
        currently_set_pipeline.record_stencil_pass();
    }
}

/// Draw a leaf at its parent's stencil reference without modifying the stencil buffer.
#[allow(clippy::too_many_arguments)]
pub(super) fn handle_leaf_draw_pass<'rp>(
    render_pass: &mut wgpu::RenderPass<'rp>,
    currently_set_pipeline: &mut PipelineTracker,
    bound_texture_state: &mut BoundTextureState,
    stencil_reference: u32,
    shape: &CachedShapeDrawData,
    resources: &ShapeDrawResources,
    pipelines: &ShapePipelines,
    buffers: &Buffers,
    textures: &IntermediateTextureResources,
) {
    if let Some(geometry_range) = resources.geometry_buffer_range {
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

        textures.bind_shape_texture_layers(
            render_pass,
            &shape.texture_bindings,
            &pipelines.texture_manager,
            &pipelines.shape_texture_bind_group_layout_background,
            &pipelines.shape_texture_bind_group_layout_foreground,
            &pipelines.default_shape_texture_bind_groups,
            bound_texture_state,
        );

        if uses_gradient {
            let gradient_bind_group = resources
                .gradient_bind_group
                .as_ref()
                .expect("gradient shapes must prepare a gradient bind group");
            render_pass.set_bind_group(3, gradient_bind_group.as_ref(), &[]);
        }

        bind_instance_buffers(render_pass, resources, buffers);

        render_pass.set_stencil_reference(stencil_reference);
        buffers.draw_indexed(render_pass, geometry_range, 0..1);
    }
}

#[derive(Default)]
pub(super) struct PendingLeafBatch {
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
    render_pass: &mut wgpu::RenderPass<'_>,
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

    bind_aggregated_geometry_buffers(render_pass, buffers);
    textures.bind_shape_texture_layers(
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
    resources: &ShapeDrawResources,
    parent_stencil: u32,
) -> bool {
    let geometry_range = match resources.geometry_buffer_range {
        Some(range) => range,
        None => return false,
    };
    // Shapes with per-shape gradient bind groups cannot be batched.
    if shape.has_gradient_fill() {
        return false;
    }
    let instance_index = match resources.instance_index {
        Some(index) => index as u32,
        None => return false,
    };
    let texture_bindings = &shape.texture_bindings;

    if batch.is_empty() {
        batch.geometry_range = geometry_range;
        batch.texture_bindings = *texture_bindings;
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
fn queue_or_draw_leaf(
    shape: &CachedShapeDrawData,
    resources: &ShapeDrawResources,
    parent_stencil: u32,
    pending_leaf_batch: &mut PendingLeafBatch,
    render_pass: &mut wgpu::RenderPass<'_>,
    currently_set_pipeline: &mut PipelineTracker,
    bound_texture_state: &mut BoundTextureState,
    pipelines: &ShapePipelines,
    buffers: &Buffers,
    textures: &IntermediateTextureResources,
) {
    if try_batch_leaf(pending_leaf_batch, shape, resources, parent_stencil) {
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
    if try_batch_leaf(pending_leaf_batch, shape, resources, parent_stencil) {
        return;
    }

    handle_leaf_draw_pass(
        render_pass,
        currently_set_pipeline,
        bound_texture_state,
        parent_stencil,
        shape,
        resources,
        pipelines,
        buffers,
        textures,
    );
}

/// Attachments and backdrop inputs for one traversal's output.
pub(super) struct SegmentRenderTarget<'a> {
    pub(super) color_view: &'a wgpu::TextureView,
    pub(super) color_resolve_target: Option<&'a wgpu::TextureView>,
    pub(super) depth_stencil_view: &'a wgpu::TextureView,
    pub(super) backdrop_source: Option<BackdropSource<'a>>,
    pub(super) backdrop_context: Option<&'a BackdropContext<'a>>,
}

/// Render traversal events, clearing the target before the first segment.
///
/// Backdrop captures split passes. The clipping stacks preserve inherited stencil
/// references and scissor rectangles across those passes.
pub(super) fn render_segments(
    encoder: &mut wgpu::CommandEncoder,
    events: &[TraversalEvent],
    effect_results: &HashMap<usize, IntermediateTextureId>,
    target: SegmentRenderTarget<'_>,
    pipeline_resources: &RendererPipelineResources,
    state: &mut RendererState,
) {
    let SegmentRenderTarget {
        color_view,
        color_resolve_target,
        depth_stencil_view,
        backdrop_source,
        backdrop_context,
    } = target;
    let pipelines = &pipeline_resources.shapes;
    let buffers = &state.buffers;
    let scratch = &mut state.scratch;
    let mut event_idx = 0;
    let mut is_first_segment = true;
    let mut currently_set_pipeline = PipelineTracker::new();
    let mut bound_texture_state = BoundTextureState::default();
    let (width, height) = state.physical_size;
    let viewport_scissor = UnsignedPhysicalRect::from_size(Size::new(width, height));
    let mut pending_leaf_batch = PendingLeafBatch::default();
    scratch.stencil_stack.clear();
    scratch.scissor_stack.clear();
    scratch.scissor_stack.push(viewport_scissor);
    scratch.clip_kind_stack.clear();

    while event_idx < events.len() {
        // A backdrop capture must include every draw before its node.
        let mut segment_end = events.len();
        let mut backdrop_node_id: Option<usize> = None;
        if backdrop_context.is_some() {
            for (idx, event) in events.iter().enumerate().skip(event_idx) {
                if let TraversalEvent::Pre(node_id) = event {
                    if state.backdrop_effects.contains_key(node_id)
                        && !effect_results.contains_key(node_id)
                    {
                        segment_end = idx;
                        backdrop_node_id = Some(*node_id);
                        break;
                    }
                }
            }
        }

        let segment_has_events = event_idx < segment_end;
        if segment_has_events {
            let mut render_pass = begin_render_pass_with_load_ops(
                encoder,
                Some(if is_first_segment {
                    "segment_clear_pass"
                } else {
                    "segment_load_pass"
                }),
                color_view,
                color_resolve_target,
                depth_stencil_view,
                RenderPassLoadOperations {
                    color_load_op: if is_first_segment {
                        wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT)
                    } else {
                        wgpu::LoadOp::Load
                    },
                    depth_load_op: if is_first_segment {
                        wgpu::LoadOp::Clear(1.0)
                    } else {
                        wgpu::LoadOp::Load
                    },
                    stencil_load_op: if is_first_segment {
                        wgpu::LoadOp::Clear(0)
                    } else {
                        wgpu::LoadOp::Load
                    },
                },
            );

            // Each render pass starts with the full viewport, so restore the inherited scissor.
            let current_scissor = scratch
                .scissor_stack
                .last()
                .copied()
                .unwrap_or(viewport_scissor);
            if current_scissor != viewport_scissor {
                render_pass.set_scissor_rect(
                    current_scissor.min.x,
                    current_scissor.min.y,
                    current_scissor.width(),
                    current_scissor.height(),
                );
            }

            for event in events.iter().take(segment_end).skip(event_idx) {
                match event {
                    TraversalEvent::PreparedLeaf(node_id) => {
                        let Some(prepared_leaf) = scratch.shape_effect_leaves.get(node_id) else {
                            continue;
                        };
                        let parent_stencil = scratch.stencil_stack.last().copied().unwrap_or(0);
                        queue_or_draw_leaf(
                            &prepared_leaf.draw_data,
                            &state.shape_execution.effect_leaves[node_id],
                            parent_stencil,
                            &mut pending_leaf_batch,
                            &mut render_pass,
                            &mut currently_set_pipeline,
                            &mut bound_texture_state,
                            pipelines,
                            buffers,
                            &state.textures,
                        );
                        #[cfg(feature = "render_metrics")]
                        {
                            state.shape_effect_cache_metrics.composited_results += 1;
                        }
                    }
                    TraversalEvent::Pre(node_id) => {
                        let node_id = *node_id;

                        if let Some(&texture_id) = effect_results.get(&node_id) {
                            flush_pending_leaf_batch(
                                &mut pending_leaf_batch,
                                &mut render_pass,
                                &mut currently_set_pipeline,
                                &mut bound_texture_state,
                                pipelines,
                                buffers,
                                &state.textures,
                            );
                            if let Some(resources) = &pipeline_resources.composite_resources {
                                let parent_stencil =
                                    scratch.stencil_stack.last().copied().unwrap_or(0);
                                render_pass.set_pipeline(&resources.pipeline);
                                render_pass.set_bind_group(
                                    0,
                                    state.textures.bind_group(texture_id),
                                    &[],
                                );
                                render_pass.set_stencil_reference(parent_stencil);
                                render_pass.draw(0..3, 0..1);
                                currently_set_pipeline.switch_to(types::Pipeline::None);
                                bound_texture_state.invalidate();
                            }
                            continue;
                        }

                        if let Some(draw_tree_node) = state.draw_tree.get(node_id) {
                            let should_skip_visible_draw = should_skip_visible_rect_draw(
                                node_id,
                                draw_tree_node,
                                &state.group_effects,
                                &state.backdrop_effects,
                            );

                            if draw_tree_node.is_leaf() {
                                if draw_tree_node.is_clip_rect() {
                                    continue;
                                }

                                if should_skip_visible_draw {
                                    continue;
                                }

                                let parent_stencil =
                                    scratch.stencil_stack.last().copied().unwrap_or(0);
                                queue_or_draw_leaf(
                                    cached_shape(draw_tree_node),
                                    &state.shape_execution.draws[&node_id],
                                    parent_stencil,
                                    &mut pending_leaf_batch,
                                    &mut render_pass,
                                    &mut currently_set_pipeline,
                                    &mut bound_texture_state,
                                    pipelines,
                                    buffers,
                                    &state.textures,
                                );
                                continue;
                            }

                            flush_pending_leaf_batch(
                                &mut pending_leaf_batch,
                                &mut render_pass,
                                &mut currently_set_pipeline,
                                &mut bound_texture_state,
                                pipelines,
                                buffers,
                                &state.textures,
                            );

                            if !draw_tree_node.clips_children() {
                                // Draw the parent as a leaf. Its children inherit the same stencil.
                                let parent_stencil =
                                    scratch.stencil_stack.last().copied().unwrap_or(0);
                                if let DrawTreeNode::CachedShape(shape) = draw_tree_node {
                                    if !should_skip_visible_draw {
                                        handle_leaf_draw_pass(
                                            &mut render_pass,
                                            &mut currently_set_pipeline,
                                            &mut bound_texture_state,
                                            parent_stencil,
                                            shape,
                                            &state.shape_execution.draws[&node_id],
                                            pipelines,
                                            buffers,
                                            &state.textures,
                                        );
                                    }
                                }
                                scratch.stencil_stack.push(parent_stencil);
                                scratch.clip_kind_stack.push(ClipKind::NonClipping);
                            } else if let Some(scissor_rect) = try_scissor_for_rect(
                                draw_tree_node,
                                state.scale_factor,
                                state.physical_size.into(),
                            ) {
                                // An axis-aligned rectangle can clip children with a hardware scissor.
                                let current_scissor = scratch
                                    .scissor_stack
                                    .last()
                                    .copied()
                                    .unwrap_or(viewport_scissor);
                                let clipped = current_scissor
                                    .intersection(&scissor_rect)
                                    .unwrap_or_else(UnsignedPhysicalRect::zero);
                                scratch.scissor_stack.push(clipped);
                                render_pass.set_scissor_rect(
                                    clipped.min.x,
                                    clipped.min.y,
                                    clipped.width(),
                                    clipped.height(),
                                );
                                #[cfg(feature = "render_metrics")]
                                currently_set_pipeline.record_scissor_clip();

                                // Draw the rect itself as a visible shape.
                                let parent_stencil =
                                    scratch.stencil_stack.last().copied().unwrap_or(0);
                                if let DrawTreeNode::CachedShape(shape) = draw_tree_node {
                                    if !should_skip_visible_draw {
                                        handle_leaf_draw_pass(
                                            &mut render_pass,
                                            &mut currently_set_pipeline,
                                            &mut bound_texture_state,
                                            parent_stencil,
                                            shape,
                                            &state.shape_execution.draws[&node_id],
                                            pipelines,
                                            buffers,
                                            &state.textures,
                                        );
                                    }
                                }
                                // Scissor clipping leaves the parent's stencil reference unchanged.
                                scratch.stencil_stack.push(parent_stencil);
                                scratch.clip_kind_stack.push(ClipKind::Scissor);
                            } else if draw_tree_node.is_clip_rect() {
                                let parent_stencil =
                                    scratch.stencil_stack.last().copied().unwrap_or(0);
                                scratch.stencil_stack.push(parent_stencil);
                                scratch.clip_kind_stack.push(ClipKind::NonClipping);
                            } else {
                                // Fall back to stencil increment.
                                let shape = cached_shape(draw_tree_node);
                                let resources = &state.shape_execution.draws[&node_id];
                                if resources.geometry_buffer_range.is_some() {
                                    let parent_stencil =
                                        scratch.stencil_stack.last().copied().unwrap_or(0);
                                    handle_increment_pass(
                                        &mut render_pass,
                                        &mut currently_set_pipeline,
                                        &mut bound_texture_state,
                                        parent_stencil,
                                        shape,
                                        resources,
                                        pipelines,
                                        buffers,
                                        &state.textures,
                                    );
                                    scratch.stencil_stack.push(parent_stencil + 1);
                                }
                                scratch.clip_kind_stack.push(ClipKind::Stencil);
                            }
                        }
                    }
                    TraversalEvent::Post(node_id) => {
                        let node_id = *node_id;

                        // Pre composited the effect result without pushing a stencil entry.
                        if effect_results.contains_key(&node_id) {
                            continue;
                        }

                        if let Some(draw_tree_node) = state.draw_tree.get(node_id) {
                            // Pre drew the leaf without changing the clip stacks.
                            if draw_tree_node.is_leaf() {
                                continue;
                            }

                            match scratch.clip_kind_stack.pop() {
                                Some(ClipKind::NonClipping) => {
                                    scratch.stencil_stack.pop();
                                }
                                Some(ClipKind::Scissor) => {
                                    flush_pending_leaf_batch(
                                        &mut pending_leaf_batch,
                                        &mut render_pass,
                                        &mut currently_set_pipeline,
                                        &mut bound_texture_state,
                                        pipelines,
                                        buffers,
                                        &state.textures,
                                    );
                                    scratch.scissor_stack.pop();
                                    let prev = scratch
                                        .scissor_stack
                                        .last()
                                        .copied()
                                        .unwrap_or(viewport_scissor);
                                    render_pass.set_scissor_rect(
                                        prev.min.x,
                                        prev.min.y,
                                        prev.width(),
                                        prev.height(),
                                    );
                                    scratch.stencil_stack.pop();
                                }
                                Some(ClipKind::Stencil) => {
                                    flush_pending_leaf_batch(
                                        &mut pending_leaf_batch,
                                        &mut render_pass,
                                        &mut currently_set_pipeline,
                                        &mut bound_texture_state,
                                        pipelines,
                                        buffers,
                                        &state.textures,
                                    );
                                    let resources = &state.shape_execution.draws[&node_id];
                                    if resources.geometry_buffer_range.is_some() {
                                        let stencil_reference =
                                            scratch.stencil_stack.last().copied().unwrap_or(0);
                                        handle_decrement_pass(
                                            &mut render_pass,
                                            &mut currently_set_pipeline,
                                            &mut bound_texture_state,
                                            stencil_reference,
                                            resources,
                                            pipelines,
                                            buffers,
                                        );
                                        scratch.stencil_stack.pop();
                                    }
                                }
                                None => {
                                    debug_assert!(
                                        false,
                                        "clip_kind_stack underflow in Post for node {node_id}"
                                    );
                                }
                            }
                        }
                    }
                }
            }

            flush_pending_leaf_batch(
                &mut pending_leaf_batch,
                &mut render_pass,
                &mut currently_set_pipeline,
                &mut bound_texture_state,
                pipelines,
                buffers,
                &state.textures,
            );

            is_first_segment = false;
        }

        event_idx = segment_end;

        if !segment_has_events && is_first_segment {
            let mut render_pass = begin_render_pass_with_load_ops(
                encoder,
                Some("backdrop_precapture_pass"),
                color_view,
                color_resolve_target,
                depth_stencil_view,
                RenderPassLoadOperations {
                    color_load_op: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    depth_load_op: wgpu::LoadOp::Clear(1.0),
                    stencil_load_op: wgpu::LoadOp::Clear(0),
                },
            );
            let current_scissor = scratch
                .scissor_stack
                .last()
                .copied()
                .unwrap_or(viewport_scissor);
            if current_scissor != viewport_scissor {
                render_pass.set_scissor_rect(
                    current_scissor.min.x,
                    current_scissor.min.y,
                    current_scissor.width(),
                    current_scissor.height(),
                );
            }
            is_first_segment = false;
            currently_set_pipeline.switch_to(Pipeline::None);
            bound_texture_state.invalidate();
        }

        if let Some(backdrop_node_id) = backdrop_node_id {
            let backdrop_context =
                backdrop_context.expect("backdrop rendering requires its context");
            // Ancestors clipped by scissor retain the nearest stencil-writing ancestor's value.
            let parent_stencil = scratch.stencil_stack.last().copied().unwrap_or(0);
            let this_stencil = parent_stencil + 1;

            let mut backdrop_material = None;

            if let Some(draw_tree_node) = state.draw_tree.get_mut(backdrop_node_id) {
                let effect_instance = state
                    .backdrop_effects
                    .get(&backdrop_node_id)
                    .expect("backdrop node must have an attached effect instance");
                let backdrop_config = effect_instance.config;
                let effect_resources = state
                    .effect_execution
                    .backdrops
                    .get_mut(&backdrop_node_id)
                    .expect("backdrop attachments have execution resources");
                let local_bounds = draw_tree_node.local_bounds();

                if let Some(capture_region) = compute_backdrop_capture_region(
                    MathRect::new(local_bounds[0].into(), local_bounds[1].into()),
                    draw_tree_node.transform(),
                    backdrop_config,
                    state.scale_factor,
                    state.physical_size.into(),
                    backdrop_context.max_texture_dimension_2d,
                ) {
                    let effect_output = backdrops::apply_backdrop_effect(
                        encoder,
                        backdrop_context,
                        backdrop_source.expect("backdrop source required for backdrop effects"),
                        capture_region,
                        effect_instance,
                        effect_resources,
                        &mut state.textures,
                    );
                    if let DrawTreeNode::CachedShape(cached_shape) = draw_tree_node {
                        let shape_resources = state
                            .shape_execution
                            .draws
                            .get_mut(&backdrop_node_id)
                            .expect("backdrop shapes have execution resources");
                        backdrop_material = backdrops::prepare_backdrop_material(
                            backdrop_context,
                            capture_region,
                            &effect_output,
                            &mut cached_shape.fill,
                            effect_resources,
                            shape_resources,
                            &mut state.shape_execution.gradient_cache,
                        );
                    }
                    effect_output.push_work_textures_into(&mut state.textures.work_textures);
                }
            }

            // Preserve the scene while drawing the backdrop result inside this shape.
            let mut render_pass = begin_render_pass_with_load_ops(
                encoder,
                Some("backdrop_shape_pass"),
                color_view,
                color_resolve_target,
                depth_stencil_view,
                RenderPassLoadOperations {
                    color_load_op: wgpu::LoadOp::Load,
                    depth_load_op: wgpu::LoadOp::Load,
                    stencil_load_op: wgpu::LoadOp::Load,
                },
            );

            // Restore scissor in the backdrop pass.
            let current_scissor = scratch
                .scissor_stack
                .last()
                .copied()
                .unwrap_or(viewport_scissor);
            if current_scissor != viewport_scissor {
                render_pass.set_scissor_rect(
                    current_scissor.min.x,
                    current_scissor.min.y,
                    current_scissor.width(),
                    current_scissor.height(),
                );
            }

            // Increment the stencil inside the backdrop shape before drawing its color.
            if state.draw_tree.get(backdrop_node_id).is_some() {
                render_pass.set_pipeline(backdrop_context.stencil_only_pipeline);
                render_pass.set_bind_group(0, &pipelines.and_bind_group, &[]);
                render_pass.set_bind_group(
                    1,
                    &*pipelines.default_shape_texture_bind_groups[0],
                    &[],
                );
                render_pass.set_bind_group(
                    2,
                    &*pipelines.default_shape_texture_bind_groups[1],
                    &[],
                );
                bind_aggregated_geometry_buffers(&mut render_pass, buffers);

                let resources = &state.shape_execution.draws[&backdrop_node_id];
                bind_instance_buffers(&mut render_pass, resources, buffers);
                let shape_geometry_range = resources.geometry_buffer_range;

                if let Some(geometry_range) = shape_geometry_range {
                    render_pass.set_stencil_reference(parent_stencil);
                    buffers.draw_indexed(&mut render_pass, geometry_range, 0..1);
                    #[cfg(feature = "render_metrics")]
                    currently_set_pipeline.record_stencil_pass();
                }
            }

            let backdrop_is_leaf = state
                .draw_tree
                .get(backdrop_node_id)
                .is_none_or(|cmd| cmd.is_leaf());
            let backdrop_clips_children = state
                .draw_tree
                .get(backdrop_node_id)
                .is_none_or(|cmd| cmd.clips_children());

            // Draw the color where the stencil matches, leaving its value unchanged.
            if let Some(draw_tree_node) = state.draw_tree.get(backdrop_node_id) {
                let uses_gradient = draw_tree_node.has_gradient_fill();
                let use_backdrop_gradient_pipeline = uses_gradient && backdrop_material.is_some();
                render_pass.set_pipeline(if use_backdrop_gradient_pipeline {
                    backdrop_context.backdrop_color_gradient_pipeline
                } else if uses_gradient {
                    &pipelines.leaf_draw_gradient_pipeline
                } else {
                    backdrop_context.backdrop_color_pipeline
                });
                render_pass.set_bind_group(0, &pipelines.and_bind_group, &[]);
                render_pass.set_bind_group(
                    1,
                    &*pipelines.default_shape_texture_bind_groups[0],
                    &[],
                );
                render_pass.set_bind_group(
                    2,
                    &*pipelines.default_shape_texture_bind_groups[1],
                    &[],
                );
                bound_texture_state.mark_bound(0, ShapeTextureBinding::None);
                bound_texture_state.mark_bound(1, ShapeTextureBinding::None);
                bind_aggregated_geometry_buffers(&mut render_pass, buffers);

                let shape = cached_shape(draw_tree_node);
                let resources = &state.shape_execution.draws[&backdrop_node_id];
                bind_instance_buffers(&mut render_pass, resources, buffers);
                let texture_bindings = shape.texture_bindings;
                let shape_geometry_range = resources.geometry_buffer_range;

                if uses_gradient {
                    if let Some(gradient_backdrop_bind_group) = backdrop_material.as_ref() {
                        render_pass.set_bind_group(3, gradient_backdrop_bind_group, &[]);
                    } else {
                        let gradient_bind_group = resources
                            .gradient_bind_group.as_ref()
                            .expect("gradient backdrop fallback should reuse the prepared gradient bind group");
                        render_pass.set_bind_group(3, gradient_bind_group.as_ref(), &[]);
                    }
                } else {
                    render_pass.set_bind_group(
                        3,
                        backdrop_material
                            .as_ref()
                            .unwrap_or(backdrop_context.default_backdrop_texture_bind_group),
                        &[],
                    );
                }

                state.textures.bind_shape_texture_layers(
                    &mut render_pass,
                    &texture_bindings,
                    &pipelines.texture_manager,
                    &pipelines.shape_texture_bind_group_layout_background,
                    &pipelines.shape_texture_bind_group_layout_foreground,
                    &pipelines.default_shape_texture_bind_groups,
                    &mut bound_texture_state,
                );

                if let Some(geometry_range) = shape_geometry_range {
                    render_pass.set_stencil_reference(this_stencil);
                    buffers.draw_indexed(&mut render_pass, geometry_range, 0..1);

                    // Restore the ancestor's stencil unless children still need this shape's clip.
                    // Clipping parents retain `this_stencil` until Post.
                    if backdrop_is_leaf || !backdrop_clips_children {
                        render_pass.set_pipeline(&pipelines.decrementing_pipeline);
                        render_pass.set_bind_group(0, &pipelines.decrementing_bind_group, &[]);
                        render_pass.set_bind_group(
                            1,
                            &*pipelines.default_shape_texture_bind_groups[0],
                            &[],
                        );
                        render_pass.set_bind_group(
                            2,
                            &*pipelines.default_shape_texture_bind_groups[1],
                            &[],
                        );
                        render_pass.set_stencil_reference(this_stencil);
                        buffers.draw_indexed(&mut render_pass, geometry_range, 0..1);
                        #[cfg(feature = "render_metrics")]
                        currently_set_pipeline.record_stencil_pass();
                    }
                }
            }

            currently_set_pipeline.switch_to(Pipeline::None);
            bound_texture_state.invalidate();
            is_first_segment = false;

            if backdrop_is_leaf {
                // The leaf is complete. Skip its Pre and Post events.
                event_idx += 2;
            } else if backdrop_clips_children {
                // Children inherit the backdrop shape's stencil.
                // Post decrements it after rendering the descendants.
                scratch.stencil_stack.push(this_stencil);
                scratch.clip_kind_stack.push(ClipKind::Stencil);
                event_idx += 1;
            } else {
                // The backdrop used this node's stencil.
                // Visible-overflow children inherit the nearest ancestor clip.
                scratch.stencil_stack.push(parent_stencil);
                scratch.clip_kind_stack.push(ClipKind::NonClipping);
                event_idx += 1;
            }
        }
    }

    #[cfg(feature = "render_metrics")]
    state
        .pipeline_switch_counts
        .accumulate(&currently_set_pipeline.counts);
}
