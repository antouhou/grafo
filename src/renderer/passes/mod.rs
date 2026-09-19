use self::backdrop::{prepare_backdrop, BackdropMaterialBindings, BackdropPreparation};
pub(super) use self::effects::{
    apply_effect_passes, compute_downsampled_dimensions, EffectPassRunConfig,
};
use self::shapes::{
    bind_aggregated_geometry_buffers, bind_instance_buffers, bind_shape_texture_layers,
    PendingLeafBatch, ShapePass,
};
pub(super) use self::target::{BackdropInputs, SegmentRenderTarget};
use super::state::{RendererPipelineResources, RendererState};
use super::types::{
    BoundTextureState, ClipKind, DrawCommand, Pipeline, PipelineTracker, TraversalEvent,
};
use crate::renderer::rect_utils::{
    intersect_scissor, should_skip_visible_rect_draw, try_scissor_for_rect,
};
use crate::shape::{CachedShapeDrawData, ShapeTextureBinding};
use ahash::HashMap;
use wgpu::{BindGroup, CommandEncoder};

mod backdrop;
mod effects;
mod shapes;
mod target;

fn cached_shape_mut(draw_command: &mut DrawCommand) -> &mut CachedShapeDrawData {
    match draw_command {
        DrawCommand::CachedShape(shape) => shape,
        DrawCommand::ClipRect(_) => unreachable!("clip rectangles do not own shape geometry"),
    }
}

/// Render traversal events, clearing the target before the first segment.
/// Empty traversals encode no render pass.
///
/// Backdrop captures split passes. The clipping stacks preserve inherited stencil
/// references and scissor rectangles across those passes.
pub(super) fn render_segments(
    encoder: &mut CommandEncoder,
    events: &[TraversalEvent],
    effect_results: &HashMap<usize, BindGroup>,
    target: SegmentRenderTarget<'_>,
    pipeline_resources: &RendererPipelineResources,
    state: &mut RendererState,
) {
    let pipelines = &pipeline_resources.shapes;
    let buffers = &state.buffers;
    let scratch = &mut state.scratch;
    let mut event_index = 0;
    let mut is_first_segment = true;
    let mut currently_set_pipeline = PipelineTracker::new();
    let mut bound_texture_state = BoundTextureState::default();
    let (width, height) = state.physical_size;
    let viewport_scissor = (0u32, 0u32, width, height);
    let mut pending_leaf_batch = PendingLeafBatch::default();
    scratch.stencil_stack.clear();
    scratch.scissor_stack.clear();
    scratch.scissor_stack.push(viewport_scissor);
    scratch.backdrop_work_textures.clear();
    scratch.clip_kind_stack.clear();

    while event_index < events.len() {
        // A backdrop capture must include every draw before its node.
        let mut segment_end = events.len();
        let mut backdrop_node_id: Option<usize> = None;
        if target.backdrop.is_some() {
            for (index, event) in events.iter().enumerate().skip(event_index) {
                if let TraversalEvent::Pre(node_id) = event {
                    if state.backdrop_effects.contains_key(node_id)
                        && !effect_results.contains_key(node_id)
                    {
                        segment_end = index;
                        backdrop_node_id = Some(*node_id);
                        break;
                    }
                }
            }
        }

        let segment_has_events = event_index < segment_end;
        if segment_has_events {
            let mut render_pass = target.begin_pass(
                encoder,
                if is_first_segment {
                    "segment_clear_pass"
                } else {
                    "segment_load_pass"
                },
                is_first_segment,
            );

            // Each render pass starts with the full viewport, so restore the inherited scissor.
            let current_scissor = scratch
                .scissor_stack
                .last()
                .copied()
                .unwrap_or(viewport_scissor);
            if current_scissor != viewport_scissor {
                render_pass.set_scissor_rect(
                    current_scissor.0,
                    current_scissor.1,
                    current_scissor.2,
                    current_scissor.3,
                );
            }

            let mut shape_pass = ShapePass {
                render_pass: &mut render_pass,
                currently_set_pipeline: &mut currently_set_pipeline,
                bound_texture_state: &mut bound_texture_state,
                pending_leaf_batch: &mut pending_leaf_batch,
                pipelines,
                buffers,
            };

            for event in events.iter().take(segment_end).skip(event_index) {
                match event {
                    TraversalEvent::PreparedLeaf(node_id) => {
                        let Some(prepared_leaf) = scratch.shape_effect_leaves.get_mut(node_id)
                        else {
                            continue;
                        };
                        let parent_stencil = scratch.stencil_stack.last().copied().unwrap_or(0);
                        shape_pass.queue_leaf(
                            prepared_leaf,
                            parent_stencil,
                            &scratch.stencil_stack,
                        );
                        #[cfg(feature = "render_metrics")]
                        {
                            state.shape_effect_cache_metrics.composited_results += 1;
                        }
                    }
                    TraversalEvent::Pre(node_id) => {
                        let node_id = *node_id;

                        if let Some(result_bind_group) = effect_results.get(&node_id) {
                            shape_pass.flush();
                            if let Some(resources) = &pipeline_resources.composite_resources {
                                let parent_stencil =
                                    scratch.stencil_stack.last().copied().unwrap_or(0);
                                shape_pass.render_pass.set_pipeline(&resources.pipeline);
                                shape_pass
                                    .render_pass
                                    .set_bind_group(0, result_bind_group, &[]);
                                shape_pass.render_pass.set_stencil_reference(parent_stencil);
                                shape_pass.render_pass.draw(0..3, 0..1);
                                shape_pass.currently_set_pipeline.switch_to(Pipeline::None);
                                shape_pass.bound_texture_state.invalidate();
                            }
                            continue;
                        }

                        if let Some(draw_command) = state.draw_tree.get_mut(node_id) {
                            let should_skip_visible_draw = should_skip_visible_rect_draw(
                                node_id,
                                &*draw_command,
                                &state.group_effects,
                                &state.backdrop_effects,
                            );

                            if draw_command.is_leaf() {
                                if draw_command.is_clip_rect() {
                                    continue;
                                }

                                let shape = cached_shape_mut(draw_command);
                                if should_skip_visible_draw {
                                    let parent_stencil =
                                        scratch.stencil_stack.last().copied().unwrap_or(0);
                                    shape.stencil_ref = Some(parent_stencil);
                                    continue;
                                }

                                let parent_stencil =
                                    scratch.stencil_stack.last().copied().unwrap_or(0);
                                shape_pass.queue_leaf(
                                    shape,
                                    parent_stencil,
                                    &scratch.stencil_stack,
                                );
                                continue;
                            }

                            shape_pass.flush();

                            if !draw_command.clips_children() {
                                // Draw the parent as a leaf. Its children inherit the same stencil.
                                let parent_stencil =
                                    scratch.stencil_stack.last().copied().unwrap_or(0);
                                if let DrawCommand::CachedShape(shape) = draw_command {
                                    shape.stencil_ref = Some(parent_stencil);
                                    if !should_skip_visible_draw {
                                        shape_pass.draw_leaf(shape, &scratch.stencil_stack);
                                    }
                                }
                                scratch.stencil_stack.push(parent_stencil);
                                scratch.clip_kind_stack.push(ClipKind::NonClipping);
                            } else if let Some(scissor_rect) = try_scissor_for_rect(
                                draw_command,
                                state.scale_factor,
                                state.physical_size,
                            ) {
                                // An axis-aligned rectangle can clip children with a hardware scissor.
                                let current_scissor = scratch
                                    .scissor_stack
                                    .last()
                                    .copied()
                                    .unwrap_or(viewport_scissor);
                                let clipped = intersect_scissor(current_scissor, scissor_rect);
                                scratch.scissor_stack.push(clipped);
                                shape_pass
                                    .render_pass
                                    .set_scissor_rect(clipped.0, clipped.1, clipped.2, clipped.3);
                                #[cfg(feature = "render_metrics")]
                                shape_pass.currently_set_pipeline.record_scissor_clip();

                                // Draw the rect itself as a visible shape.
                                let parent_stencil =
                                    scratch.stencil_stack.last().copied().unwrap_or(0);
                                if let DrawCommand::CachedShape(shape) = draw_command {
                                    shape.stencil_ref = Some(parent_stencil);
                                    if !should_skip_visible_draw {
                                        shape_pass.draw_leaf(shape, &scratch.stencil_stack);
                                    }
                                }
                                // Scissor clipping leaves the parent's stencil reference unchanged.
                                scratch.stencil_stack.push(parent_stencil);
                                scratch.clip_kind_stack.push(ClipKind::Scissor);
                            } else if draw_command.is_clip_rect() {
                                let parent_stencil =
                                    scratch.stencil_stack.last().copied().unwrap_or(0);
                                scratch.stencil_stack.push(parent_stencil);
                                scratch.clip_kind_stack.push(ClipKind::NonClipping);
                            } else {
                                // Fall back to stencil increment.
                                shape_pass.increment_stencil(
                                    cached_shape_mut(draw_command),
                                    &mut scratch.stencil_stack,
                                );
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

                        if let Some(draw_command) = state.draw_tree.get_mut(node_id) {
                            // Pre drew the leaf without changing the clip stacks.
                            if draw_command.is_leaf() {
                                continue;
                            }

                            match scratch.clip_kind_stack.pop() {
                                Some(ClipKind::NonClipping) => {
                                    scratch.stencil_stack.pop();
                                }
                                Some(ClipKind::Scissor) => {
                                    shape_pass.flush();
                                    scratch.scissor_stack.pop();
                                    let previous_scissor = scratch
                                        .scissor_stack
                                        .last()
                                        .copied()
                                        .unwrap_or(viewport_scissor);
                                    shape_pass.render_pass.set_scissor_rect(
                                        previous_scissor.0,
                                        previous_scissor.1,
                                        previous_scissor.2,
                                        previous_scissor.3,
                                    );
                                    scratch.stencil_stack.pop();
                                }
                                Some(ClipKind::Stencil) => {
                                    shape_pass.flush();
                                    shape_pass.decrement_stencil(
                                        cached_shape_mut(draw_command),
                                        &mut scratch.stencil_stack,
                                    );
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

            shape_pass.flush();

            is_first_segment = false;
        }

        event_index = segment_end;

        if !segment_has_events && is_first_segment {
            let mut render_pass = target.begin_pass(encoder, "backdrop_precapture_pass", true);
            let current_scissor = scratch
                .scissor_stack
                .last()
                .copied()
                .unwrap_or(viewport_scissor);
            if current_scissor != viewport_scissor {
                render_pass.set_scissor_rect(
                    current_scissor.0,
                    current_scissor.1,
                    current_scissor.2,
                    current_scissor.3,
                );
            }
            is_first_segment = false;
            currently_set_pipeline.switch_to(Pipeline::None);
            bound_texture_state.invalidate();
        }

        if let Some(backdrop_node_id) = backdrop_node_id {
            let backdrop = target
                .backdrop
                .as_ref()
                .expect("backdrop boundary requires capture inputs");
            let backdrop_context = backdrop.context;
            let composite_bind_group_layout = &pipeline_resources
                .composite_resources
                .as_ref()
                .expect("backdrop rendering requires composite resources")
                .bind_group_layout;
            // Ancestors clipped by scissor retain the nearest stencil-writing ancestor's value.
            let parent_stencil = scratch.stencil_stack.last().copied().unwrap_or(0);
            let this_stencil = parent_stencil + 1;

            let backdrop_material =
                if let Some(draw_command) = state.draw_tree.get_mut(backdrop_node_id) {
                    let effect_instance = state
                        .backdrop_effects
                        .get_mut(&backdrop_node_id)
                        .expect("backdrop node must have an attached effect instance");
                    prepare_backdrop(
                        encoder,
                        draw_command,
                        effect_instance,
                        &mut state.texture_pool,
                        &mut state.shape_resources.gradient_cache,
                        &mut scratch.backdrop_work_textures,
                        BackdropPreparation {
                            context: backdrop_context,
                            source: backdrop.source,
                            composite_bind_group_layout,
                            scale_factor: state.scale_factor,
                            physical_size: state.physical_size,
                        },
                    )
                } else {
                    BackdropMaterialBindings::default()
                };
            let solid_backdrop_bind_group = backdrop_material.solid;
            let gradient_backdrop_bind_group = backdrop_material.gradient;

            // Preserve the scene while drawing the backdrop result inside this shape.
            let mut render_pass = target.begin_pass(encoder, "backdrop_shape_pass", false);

            // Restore scissor in the backdrop pass.
            let current_scissor = scratch
                .scissor_stack
                .last()
                .copied()
                .unwrap_or(viewport_scissor);
            if current_scissor != viewport_scissor {
                render_pass.set_scissor_rect(
                    current_scissor.0,
                    current_scissor.1,
                    current_scissor.2,
                    current_scissor.3,
                );
            }

            // Increment the stencil inside the backdrop shape before drawing its color.
            if let Some(draw_command) = state.draw_tree.get_mut(backdrop_node_id) {
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

                let shape = cached_shape_mut(draw_command);
                bind_instance_buffers(&mut render_pass, shape, buffers);
                let shape_geometry_range = shape.geometry_buffer_range;

                if let Some(geometry_range) = shape_geometry_range {
                    render_pass.set_stencil_reference(parent_stencil);
                    buffers.draw_indexed(&mut render_pass, geometry_range, 0..1);
                    #[cfg(feature = "render_metrics")]
                    currently_set_pipeline.record_stencil_pass();
                }

                shape.stencil_ref = Some(this_stencil);
            }

            let backdrop_is_leaf = state
                .draw_tree
                .get(backdrop_node_id)
                .is_none_or(|draw_command| draw_command.is_leaf());
            let backdrop_clips_children = state
                .draw_tree
                .get(backdrop_node_id)
                .is_none_or(|draw_command| draw_command.clips_children());

            // Draw the color where the stencil matches, leaving its value unchanged.
            if let Some(draw_command) = state.draw_tree.get_mut(backdrop_node_id) {
                let uses_gradient = draw_command.has_gradient_fill();
                let use_backdrop_gradient_pipeline =
                    uses_gradient && gradient_backdrop_bind_group.is_some();
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

                let shape = cached_shape_mut(draw_command);
                bind_instance_buffers(&mut render_pass, shape, buffers);
                let texture_bindings = shape.texture_bindings.clone();
                let shape_geometry_range = shape.geometry_buffer_range;

                if uses_gradient {
                    if let Some(gradient_backdrop_bind_group) =
                        gradient_backdrop_bind_group.as_ref()
                    {
                        render_pass.set_bind_group(3, gradient_backdrop_bind_group, &[]);
                    } else {
                        let gradient_bind_group = draw_command
                            .gradient_bind_group()
                            .expect("gradient backdrop fallback should reuse the prepared gradient bind group");
                        render_pass.set_bind_group(3, gradient_bind_group.as_ref(), &[]);
                    }
                } else {
                    render_pass.set_bind_group(
                        3,
                        solid_backdrop_bind_group
                            .as_ref()
                            .unwrap_or(backdrop_context.default_backdrop_texture_bind_group),
                        &[],
                    );
                }

                bind_shape_texture_layers(
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
                event_index += 2;
            } else if backdrop_clips_children {
                // Children inherit the backdrop shape's stencil.
                // Post decrements it after rendering the descendants.
                scratch.stencil_stack.push(this_stencil);
                scratch.clip_kind_stack.push(ClipKind::Stencil);
                event_index += 1;
            } else {
                // The backdrop used this node's stencil.
                // Visible-overflow children inherit the nearest ancestor clip.
                scratch.stencil_stack.push(parent_stencil);
                scratch.clip_kind_stack.push(ClipKind::NonClipping);
                event_index += 1;
            }
        }
    }

    #[cfg(feature = "render_metrics")]
    state
        .pipeline_switch_counts
        .accumulate(&currently_set_pipeline.counts);
}
