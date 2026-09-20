use super::execution::effects::{apply_effect_passes, EffectPassRunConfig};
use super::plan::backdrops::compute_backdrop_capture_region;
use super::state::{Buffers, RendererPipelineResources, RendererState, ShapePipelines};
use super::types::{
    BackdropContext, BackdropSource, BoundTextureState, ClipKind, Pipeline, PipelineTracker,
    TraversalEvent,
};
use super::*;
use crate::effect::PooledTexture;
use crate::pipeline::{begin_render_pass_with_load_ops, RenderPassLoadOperations};
use crate::renderer::rect_utils::{
    compute_downsampled_dimensions, should_skip_visible_rect_draw, try_scissor_for_rect,
};
use crate::shape::{CachedShapeDrawData, ShapeTextureBinding};
use crate::{Size, UnsignedPhysicalRect};

fn cached_shape_mut(draw_command: &mut DrawCommand) -> &mut CachedShapeDrawData {
    match draw_command {
        DrawCommand::CachedShape(shape) => shape,
        DrawCommand::ClipRect(_) => unreachable!("clip rectangles do not own shape geometry"),
    }
}

#[allow(clippy::too_many_arguments)]
fn bind_shape_texture_layers(
    render_pass: &mut wgpu::RenderPass<'_>,
    texture_bindings: &[ShapeTextureBinding; 2],
    texture_manager: &TextureManager,
    shape_texture_bind_group_layout_background: &wgpu::BindGroupLayout,
    shape_texture_bind_group_layout_foreground: &wgpu::BindGroupLayout,
    default_shape_texture_bind_groups: &[Arc<wgpu::BindGroup>; 2],
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
    render_pass: &mut wgpu::RenderPass<'_>,
    shape: &CachedShapeDrawData,
    buffers: &Buffers,
) {
    if let Some(instance_idx) = shape.instance_index {
        if let Some(instance_transform_buffer) =
            buffers.aggregated_instance_transform_buffer.as_ref()
        {
            let stride = std::mem::size_of::<InstanceTransform>() as u64;
            let offset = instance_idx as u64 * stride;
            render_pass
                .set_vertex_buffer(1, instance_transform_buffer.slice(offset..offset + stride));
        } else {
            render_pass.set_vertex_buffer(1, buffers.identity_transform_buffer().slice(..));
        }

        if let Some(instance_color_buffer) = buffers.aggregated_instance_color_buffer.as_ref() {
            let stride = std::mem::size_of::<InstanceColor>() as u64;
            let offset = instance_idx as u64 * stride;
            render_pass.set_vertex_buffer(2, instance_color_buffer.slice(offset..offset + stride));
        } else {
            render_pass.set_vertex_buffer(2, buffers.identity_color_buffer().slice(..));
        }

        if let Some(instance_metadata_buffer) = buffers.aggregated_instance_metadata_buffer.as_ref()
        {
            let stride = std::mem::size_of::<InstanceMetadata>() as u64;
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

fn bind_aggregated_geometry_buffers(render_pass: &mut wgpu::RenderPass<'_>, buffers: &Buffers) {
    render_pass.set_vertex_buffer(0, buffers.vertex_buffer().slice(..));
    render_pass.set_index_buffer(buffers.index_buffer().slice(..), wgpu::IndexFormat::Uint16);
}

pub(super) fn handle_increment_pass<'rp>(
    render_pass: &mut wgpu::RenderPass<'rp>,
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
    render_pass: &mut wgpu::RenderPass<'rp>,
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
    render_pass: &mut wgpu::RenderPass<'rp>,
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
fn queue_or_draw_leaf(
    shape: &mut CachedShapeDrawData,
    parent_stencil: u32,
    pending_leaf_batch: &mut PendingLeafBatch,
    render_pass: &mut wgpu::RenderPass<'_>,
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

fn clear_texture_to_transparent(
    encoder: &mut wgpu::CommandEncoder,
    output_view: &wgpu::TextureView,
    label: &str,
) {
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: output_view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
    });
}

#[allow(clippy::too_many_arguments)]
fn blit_texture_to_texture(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::RenderPipeline,
    bind_group_layout: &wgpu::BindGroupLayout,
    input_view: &wgpu::TextureView,
    output_view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
    label: &str,
) {
    let bind_group = effect::create_texture_sample_bind_group(
        device,
        bind_group_layout,
        input_view,
        sampler,
        Some(label),
    );

    let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: output_view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
    });
    render_pass.set_pipeline(pipeline);
    render_pass.set_bind_group(0, &bind_group, &[]);
    render_pass.draw(0..3, 0..1);
}

fn composite_backdrop_foreground_layer(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::RenderPipeline,
    bind_group_layout: &wgpu::BindGroupLayout,
    foreground_view: &wgpu::TextureView,
    output_view: &wgpu::TextureView,
    params_buffer: &wgpu::Buffer,
) {
    let bind_group = effect::create_backdrop_layer_composite_bind_group(
        device,
        bind_group_layout,
        foreground_view,
        params_buffer,
    );
    let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("backdrop_layer_composite_pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: output_view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
    });
    render_pass.set_pipeline(pipeline);
    render_pass.set_bind_group(0, &bind_group, &[]);
    render_pass.draw(0..3, 0..1);
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
    effect_results: &HashMap<usize, wgpu::BindGroup>,
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
    scratch.backdrop_work_textures.clear();
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
                        let Some(prepared_leaf) = scratch.shape_effect_leaves.get_mut(node_id)
                        else {
                            continue;
                        };
                        let parent_stencil = scratch.stencil_stack.last().copied().unwrap_or(0);
                        queue_or_draw_leaf(
                            prepared_leaf,
                            parent_stencil,
                            &mut pending_leaf_batch,
                            &mut render_pass,
                            &mut currently_set_pipeline,
                            &mut bound_texture_state,
                            &scratch.stencil_stack,
                            pipelines,
                            buffers,
                        );
                        #[cfg(feature = "render_metrics")]
                        {
                            state.shape_effect_cache_metrics.composited_results += 1;
                        }
                    }
                    TraversalEvent::Pre(node_id) => {
                        let node_id = *node_id;

                        if let Some(result_bind_group) = effect_results.get(&node_id) {
                            flush_pending_leaf_batch(
                                &mut pending_leaf_batch,
                                &mut render_pass,
                                &mut currently_set_pipeline,
                                &mut bound_texture_state,
                                pipelines,
                                buffers,
                            );
                            if let Some(resources) = &pipeline_resources.composite_resources {
                                let parent_stencil =
                                    scratch.stencil_stack.last().copied().unwrap_or(0);
                                render_pass.set_pipeline(&resources.pipeline);
                                render_pass.set_bind_group(0, result_bind_group, &[]);
                                render_pass.set_stencil_reference(parent_stencil);
                                render_pass.draw(0..3, 0..1);
                                currently_set_pipeline.switch_to(types::Pipeline::None);
                                bound_texture_state.invalidate();
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
                                queue_or_draw_leaf(
                                    shape,
                                    parent_stencil,
                                    &mut pending_leaf_batch,
                                    &mut render_pass,
                                    &mut currently_set_pipeline,
                                    &mut bound_texture_state,
                                    &scratch.stencil_stack,
                                    pipelines,
                                    buffers,
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
                            );

                            if !draw_command.clips_children() {
                                // Draw the parent as a leaf. Its children inherit the same stencil.
                                let parent_stencil =
                                    scratch.stencil_stack.last().copied().unwrap_or(0);
                                if let DrawCommand::CachedShape(shape) = draw_command {
                                    shape.stencil_ref = Some(parent_stencil);
                                    if !should_skip_visible_draw {
                                        handle_leaf_draw_pass(
                                            &mut render_pass,
                                            &mut currently_set_pipeline,
                                            &mut bound_texture_state,
                                            &scratch.stencil_stack,
                                            shape,
                                            pipelines,
                                            buffers,
                                        );
                                    }
                                }
                                scratch.stencil_stack.push(parent_stencil);
                                scratch.clip_kind_stack.push(ClipKind::NonClipping);
                            } else if let Some(scissor_rect) = try_scissor_for_rect(
                                draw_command,
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
                                if let DrawCommand::CachedShape(shape) = draw_command {
                                    shape.stencil_ref = Some(parent_stencil);
                                    if !should_skip_visible_draw {
                                        handle_leaf_draw_pass(
                                            &mut render_pass,
                                            &mut currently_set_pipeline,
                                            &mut bound_texture_state,
                                            &scratch.stencil_stack,
                                            shape,
                                            pipelines,
                                            buffers,
                                        );
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
                                handle_increment_pass(
                                    &mut render_pass,
                                    &mut currently_set_pipeline,
                                    &mut bound_texture_state,
                                    &mut scratch.stencil_stack,
                                    cached_shape_mut(draw_command),
                                    pipelines,
                                    buffers,
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
                                    flush_pending_leaf_batch(
                                        &mut pending_leaf_batch,
                                        &mut render_pass,
                                        &mut currently_set_pipeline,
                                        &mut bound_texture_state,
                                        pipelines,
                                        buffers,
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
                                    );
                                    handle_decrement_pass(
                                        &mut render_pass,
                                        &mut currently_set_pipeline,
                                        &mut bound_texture_state,
                                        &mut scratch.stencil_stack,
                                        cached_shape_mut(draw_command),
                                        pipelines,
                                        buffers,
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

            flush_pending_leaf_batch(
                &mut pending_leaf_batch,
                &mut render_pass,
                &mut currently_set_pipeline,
                &mut bound_texture_state,
                pipelines,
                buffers,
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
            let bctx = backdrop_context.unwrap();
            let composite_bind_group_layout = &pipeline_resources
                .composite_resources
                .as_ref()
                .expect("backdrop rendering requires composite resources")
                .bind_group_layout;
            // Ancestors clipped by scissor retain the nearest stencil-writing ancestor's value.
            let parent_stencil = scratch.stencil_stack.last().copied().unwrap_or(0);
            let this_stencil = parent_stencil + 1;

            let mut solid_backdrop_bind_group: Option<wgpu::BindGroup> = None;
            let mut gradient_backdrop_bind_group: Option<wgpu::BindGroup> = None;

            if let Some(draw_command) = state.draw_tree.get_mut(backdrop_node_id) {
                let effect_instance = state
                    .backdrop_effects
                    .get_mut(&backdrop_node_id)
                    .expect("backdrop node must have an attached effect instance");
                let backdrop_config = effect_instance.config;

                if let Some(capture_region) = compute_backdrop_capture_region(
                    draw_command,
                    backdrop_config,
                    state.scale_factor,
                    state.physical_size.into(),
                    bctx.max_texture_dimension_2d,
                ) {
                    let backdrop_sampling_uniform = capture_region.sample_uniform();
                    let capture_size = capture_region.bounds.size().to_u32();
                    let backdrop_source =
                        backdrop_source.expect("backdrop source required for backdrop effects");
                    let backdrop_capture_texture = state.texture_pool.acquire_color_only(
                        bctx.device,
                        capture_size.width,
                        capture_size.height,
                        bctx.config_format,
                        1,
                    );
                    if capture_region.source_rect.map(|rect| rect.size()) != Some(capture_size) {
                        clear_texture_to_transparent(
                            encoder,
                            &backdrop_capture_texture.color_view,
                            "backdrop_capture_clear",
                        );
                    }
                    if let Some(source_rect) = capture_region.source_rect {
                        encoder.copy_texture_to_texture(
                            wgpu::TexelCopyTextureInfo {
                                texture: backdrop_source.base_texture(),
                                mip_level: 0,
                                origin: wgpu::Origin3d {
                                    x: source_rect.min.x,
                                    y: source_rect.min.y,
                                    z: 0,
                                },
                                aspect: wgpu::TextureAspect::All,
                            },
                            wgpu::TexelCopyTextureInfo {
                                texture: &backdrop_capture_texture.color_texture,
                                mip_level: 0,
                                origin: wgpu::Origin3d {
                                    x: capture_region.copy_destination_origin.x,
                                    y: capture_region.copy_destination_origin.y,
                                    z: 0,
                                },
                                aspect: wgpu::TextureAspect::All,
                            },
                            wgpu::Extent3d {
                                width: source_rect.width(),
                                height: source_rect.height(),
                                depth_or_array_layers: 1,
                            },
                        );
                    }

                    if let Some(foreground_view) = backdrop_source.foreground_view() {
                        let layer_params = effect::backdrop_layer_params(
                            capture_region.bounds.min.to_tuple(),
                            state.physical_size,
                        );
                        let layer_params_buffer = effect::prepare_backdrop_layer_params_buffer(
                            bctx.device,
                            bctx.queue,
                            &mut effect_instance.backdrop_layer_params_buffer,
                            layer_params,
                        );
                        composite_backdrop_foreground_layer(
                            bctx.device,
                            encoder,
                            bctx.backdrop_layer_composite_pipeline,
                            bctx.backdrop_layer_composite_bind_group_layout,
                            foreground_view,
                            &backdrop_capture_texture.color_view,
                            &layer_params_buffer,
                        );
                    }

                    let effect_input_size =
                        compute_downsampled_dimensions(capture_size, backdrop_config.downsample);
                    let mut downsampled_capture_texture: Option<PooledTexture> = None;

                    if effect_input_size != capture_size {
                        let downsampled_capture_target = state.texture_pool.acquire_color_only(
                            bctx.device,
                            effect_input_size.width,
                            effect_input_size.height,
                            bctx.config_format,
                            1,
                        );
                        blit_texture_to_texture(
                            bctx.device,
                            encoder,
                            bctx.texture_blit_pipeline,
                            composite_bind_group_layout,
                            &backdrop_capture_texture.color_view,
                            &downsampled_capture_target.color_view,
                            bctx.effect_sampler,
                            "backdrop_capture_downsample",
                        );
                        downsampled_capture_texture = Some(downsampled_capture_target);
                    }

                    let loaded_effect = bctx
                        .loaded_effects
                        .get(&effect_instance.effect.effect_id)
                        .expect("loaded backdrop effect must exist");
                    let effect_output = apply_effect_passes(
                        bctx.device,
                        encoder,
                        &mut state.texture_pool,
                        EffectPassRunConfig {
                            loaded_effect,
                            params_bind_group: effect_instance
                                .effect
                                .parameter_resources
                                .as_ref()
                                .map(|resources| &resources.bind_group),
                            source_view: downsampled_capture_texture
                                .as_ref()
                                .map(|texture| &texture.color_view)
                                .unwrap_or(&backdrop_capture_texture.color_view),
                            effect_sampler: bctx.effect_sampler,
                            composite_bind_group_layout,
                            create_composite_bind_group: false,
                            width: effect_input_size.width,
                            height: effect_input_size.height,
                            texture_format: bctx.config_format,
                            label: "backdrop_effect",
                        },
                    );

                    let uses_gradient_backdrop = draw_command.has_gradient_fill();
                    if let DrawCommand::CachedShape(cached_shape) = draw_command {
                        if uses_gradient_backdrop {
                            let gradient_backdrop_material_params_buffer = cached_shape
                                .prepare_gradient_backdrop_material_params_buffer(
                                    bctx.device,
                                    bctx.queue,
                                    backdrop_sampling_uniform,
                                )
                                .expect(
                                    "gradient backdrop shapes must prepare a backdrop material params buffer",
                                );
                            let backdrop_view = effect_output.final_output_view();
                            gradient_backdrop_bind_group = cached_shape
                                .prepare_backdrop_gradient_bind_group(
                                    &mut state.shape_resources.gradient_cache,
                                    bctx.device,
                                    bctx.queue,
                                    bctx.backdrop_gradient_bind_group_layout,
                                    &gradient_backdrop_material_params_buffer,
                                    bctx.gradient_ramp_sampler,
                                    effect_output.final_output_texture_id(),
                                    backdrop_view,
                                    bctx.effect_sampler,
                                )
                                .cloned();
                        } else {
                            let solid_backdrop_material_params_buffer =
                                effect::prepare_solid_backdrop_material_params_buffer(
                                    bctx.device,
                                    bctx.queue,
                                    &mut effect_instance.backdrop_material_params_buffer,
                                    backdrop_sampling_uniform,
                                );

                            if effect_instance.backdrop_texture_id
                                != Some(effect_output.final_output_texture_id())
                            {
                                effect_instance.backdrop_texture_bind_group =
                                    Some(effect::create_backdrop_texture_sample_bind_group(
                                        bctx.device,
                                        bctx.backdrop_texture_bind_group_layout,
                                        &solid_backdrop_material_params_buffer,
                                        effect_output.final_output_view(),
                                        bctx.effect_sampler,
                                        Some("backdrop_shape_background_bind_group"),
                                    ));
                                effect_instance.backdrop_texture_id =
                                    Some(effect_output.final_output_texture_id());
                            }

                            solid_backdrop_bind_group =
                                effect_instance.backdrop_texture_bind_group.clone();
                        }
                    }

                    scratch
                        .backdrop_work_textures
                        .push(backdrop_capture_texture);
                    if let Some(downsampled_capture_texture) = downsampled_capture_texture {
                        scratch
                            .backdrop_work_textures
                            .push(downsampled_capture_texture);
                    }
                    effect_output.push_work_textures_into(&mut scratch.backdrop_work_textures);
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
            if let Some(draw_command) = state.draw_tree.get_mut(backdrop_node_id) {
                render_pass.set_pipeline(bctx.stencil_only_pipeline);
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
                .is_none_or(|cmd| cmd.is_leaf());
            let backdrop_clips_children = state
                .draw_tree
                .get(backdrop_node_id)
                .is_none_or(|cmd| cmd.clips_children());

            // Draw the color where the stencil matches, leaving its value unchanged.
            if let Some(draw_command) = state.draw_tree.get_mut(backdrop_node_id) {
                let uses_gradient = draw_command.has_gradient_fill();
                let use_backdrop_gradient_pipeline =
                    uses_gradient && gradient_backdrop_bind_group.is_some();
                render_pass.set_pipeline(if use_backdrop_gradient_pipeline {
                    bctx.backdrop_color_gradient_pipeline
                } else if uses_gradient {
                    &pipelines.leaf_draw_gradient_pipeline
                } else {
                    bctx.backdrop_color_pipeline
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
                            .unwrap_or(bctx.default_backdrop_texture_bind_group),
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
