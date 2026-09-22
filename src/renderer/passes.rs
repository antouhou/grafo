use super::execution::leaf_batches::{
    flush_pending_leaf_batch, queue_or_draw_leaf, PendingLeafBatch,
};
use super::execution::{backdrops, draws};
use super::plan::backdrops::compute_backdrop_capture_region;
use super::state::{RendererPipelineResources, RendererState};
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
                                draws::composite_texture(
                                    &mut render_pass,
                                    &mut currently_set_pipeline,
                                    &mut bound_texture_state,
                                    parent_stencil,
                                    texture_id,
                                    resources,
                                    &state.textures,
                                );
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
                                        draws::draw_shape(
                                            &mut render_pass,
                                            &mut currently_set_pipeline,
                                            &mut bound_texture_state,
                                            parent_stencil,
                                            shape.material(),
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
                                        draws::draw_shape(
                                            &mut render_pass,
                                            &mut currently_set_pipeline,
                                            &mut bound_texture_state,
                                            parent_stencil,
                                            shape.material(),
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
                                    draws::draw_shape_and_increment_stencil(
                                        &mut render_pass,
                                        &mut currently_set_pipeline,
                                        &mut bound_texture_state,
                                        parent_stencil,
                                        shape.material(),
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
                                        draws::decrement_stencil(
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

            let mut under_fill_texture = None;

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
                    let layer = backdrops::apply_backdrop_effect(
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
                        under_fill_texture = shape_resources.prepare_texture_material(
                            cached_shape.fill.as_ref(),
                            layer,
                            &mut state.shape_execution.texture_materials,
                            backdrop_context.device,
                            backdrop_context.queue,
                            pipelines,
                            &state.textures,
                            #[cfg(feature = "render_metrics")]
                            &mut state.shape_execution.texture_material_metrics,
                        );
                    }
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

            let backdrop_node = state.draw_tree.get(backdrop_node_id);
            let backdrop_is_leaf = backdrop_node.is_none_or(|node| node.is_leaf());
            let backdrop_clips_children = backdrop_node.is_none_or(|node| node.clips_children());

            if let Some(draw_tree_node) = backdrop_node {
                let resources = &state.shape_execution.draws[&backdrop_node_id];
                draws::increment_stencil(
                    &mut render_pass,
                    &mut currently_set_pipeline,
                    parent_stencil,
                    resources,
                    pipelines,
                    buffers,
                );
                let mut material = cached_shape(draw_tree_node).material();
                material.under_fill_texture = under_fill_texture;
                draws::draw_shape(
                    &mut render_pass,
                    &mut currently_set_pipeline,
                    &mut bound_texture_state,
                    this_stencil,
                    material,
                    resources,
                    pipelines,
                    buffers,
                    &state.textures,
                );

                // Clipping parents retain their stencil until Post visits them.
                if backdrop_is_leaf || !backdrop_clips_children {
                    draws::decrement_bound_shape_stencil(
                        &mut render_pass,
                        &mut currently_set_pipeline,
                        this_stencil,
                        resources,
                        pipelines,
                        buffers,
                    );
                }
            }

            drop(render_pass);
            if let Some(layer) = under_fill_texture {
                if let ShapeTextureBinding::Intermediate(texture_id) = layer.texture {
                    state.textures.finish_transient(texture_id);
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
