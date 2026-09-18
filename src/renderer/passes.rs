use super::state::{Buffers, Pipelines, RendererState};
use super::types::{
    BackdropContext, BackdropSource, BoundTextureState, ClipKind, Pipeline, PipelineTracker,
    TraversalEvent,
};
use super::*;
use crate::effect::PooledTexture;
use crate::pipeline::{
    begin_render_pass_with_load_ops, BackdropSamplingUniform, RenderPassLoadOperations,
};
use crate::renderer::rect_utils::{
    intersect_scissor, should_skip_visible_rect_draw, try_scissor_for_rect,
};
use crate::shape::{CachedShapeDrawData, ShapeTextureBinding};

fn cached_shape_mut(draw_command: &mut DrawCommand) -> &mut CachedShapeDrawData {
    match draw_command {
        DrawCommand::CachedShape(shape) => shape,
        DrawCommand::ClipRect(_) => unreachable!("clip rectangles do not own shape geometry"),
    }
}

pub(super) struct AppliedEffectOutput {
    pub(super) composite_bind_group: Option<wgpu::BindGroup>,
    pub(super) final_output_texture: PooledTexture,
    pub(super) recyclable_texture: Option<PooledTexture>,
}

impl AppliedEffectOutput {
    pub(super) fn into_final_and_recyclable(
        self,
        recyclable: &mut Vec<PooledTexture>,
    ) -> (PooledTexture, Option<wgpu::BindGroup>) {
        if let Some(recyclable_texture) = self.recyclable_texture {
            recyclable.push(recyclable_texture);
        }
        (self.final_output_texture, self.composite_bind_group)
    }

    pub(super) fn push_work_textures_into(
        self,
        output_textures: &mut Vec<PooledTexture>,
    ) -> Option<wgpu::BindGroup> {
        output_textures.push(self.final_output_texture);
        if let Some(recyclable_texture) = self.recyclable_texture {
            output_textures.push(recyclable_texture);
        }
        self.composite_bind_group
    }

    pub(super) fn final_output_view(&self) -> &wgpu::TextureView {
        &self.final_output_texture.color_view
    }

    pub(super) fn final_output_texture_id(&self) -> u64 {
        self.final_output_texture.texture_id
    }
}

pub(super) struct EffectPassRunConfig<'a> {
    pub(super) loaded_effect: &'a LoadedEffect,
    pub(super) params_bind_group: Option<&'a wgpu::BindGroup>,
    pub(super) source_view: &'a wgpu::TextureView,
    pub(super) effect_sampler: &'a wgpu::Sampler,
    pub(super) composite_bind_group_layout: &'a wgpu::BindGroupLayout,
    pub(super) create_composite_bind_group: bool,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) texture_format: wgpu::TextureFormat,
    pub(super) label_prefix: &'a str,
}

pub(super) fn apply_effect_passes(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    texture_pool: &mut OffscreenTexturePool,
    config: EffectPassRunConfig<'_>,
) -> AppliedEffectOutput {
    let number_of_passes = config.loaded_effect.passes.len();

    let effect_texture_a = texture_pool.acquire_color_only(
        device,
        config.width,
        config.height,
        config.texture_format,
        1,
    );

    let effect_texture_b = if number_of_passes > 1 {
        Some(texture_pool.acquire_color_only(
            device,
            config.width,
            config.height,
            config.texture_format,
            1,
        ))
    } else {
        None
    };

    let mut previous_input_view: &wgpu::TextureView = config.source_view;

    for (pass_index, effect_pass) in config.loaded_effect.passes.iter().enumerate() {
        let output_view = if pass_index % 2 == 0 {
            &effect_texture_a.color_view
        } else {
            &effect_texture_b.as_ref().unwrap().color_view
        };

        let input_bind_group = effect::create_texture_sample_bind_group(
            device,
            &config.loaded_effect.input_bind_group_layout,
            previous_input_view,
            config.effect_sampler,
            Some(&format!("{}_pass_input_bg", config.label_prefix)),
        );

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(&format!(
                "{}_apply_pass_{}",
                config.label_prefix, pass_index
            )),
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

        pass.set_pipeline(&effect_pass.pipeline);
        pass.set_bind_group(0, &input_bind_group, &[]);

        if effect_pass.has_params {
            if let Some(params_bind_group) = config.params_bind_group {
                pass.set_bind_group(1, params_bind_group, &[]);
            }
        }

        pass.draw(0..3, 0..1);
        previous_input_view = output_view;
    }

    let composite_bind_group = config.create_composite_bind_group.then(|| {
        effect::create_texture_sample_bind_group(
            device,
            config.composite_bind_group_layout,
            previous_input_view,
            config.effect_sampler,
            Some(&format!("{}_composite_bg", config.label_prefix)),
        )
    });

    let (final_output_texture, recyclable_texture) = if number_of_passes % 2 == 1 {
        (effect_texture_a, effect_texture_b)
    } else {
        (
            effect_texture_b.expect("an even number of effect passes needs a second texture"),
            Some(effect_texture_a),
        )
    };

    AppliedEffectOutput {
        composite_bind_group,
        final_output_texture,
        recyclable_texture,
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
    pipelines: &Pipelines,
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
            // Inform the tracker that default textures are now bound on both layers.
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
    pipelines: &Pipelines,
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
    pipelines: &Pipelines,
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
    pipelines: &Pipelines,
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

/// Try to add a leaf shape to the pending batch. Returns `true` if the shape
/// was successfully batched (no draw call needed yet). Returns `false` if
/// the shape could not be batched (caller should use the normal single-draw path).
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
    pipelines: &Pipelines,
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

fn transform_point_to_logical_screen(
    point: (f32, f32),
    transform: Option<InstanceTransform>,
) -> (f32, f32) {
    let transform = transform.unwrap_or_else(InstanceTransform::identity);
    let homogeneous_x =
        transform.col0[0] * point.0 + transform.col1[0] * point.1 + transform.col3[0];
    let homogeneous_y =
        transform.col0[1] * point.0 + transform.col1[1] * point.1 + transform.col3[1];
    let homogeneous_w =
        transform.col0[3] * point.0 + transform.col1[3] * point.1 + transform.col3[3];
    let clamped_w = homogeneous_w.signum() * homogeneous_w.abs().max(1e-6);
    let inverse_w = 1.0 / clamped_w;
    (homogeneous_x * inverse_w, homogeneous_y * inverse_w)
}

fn transformed_bounds_to_logical_screen_rect(
    local_bounds: [(f32, f32); 2],
    transform: Option<InstanceTransform>,
) -> [(f32, f32); 2] {
    let corners = [
        (local_bounds[0].0, local_bounds[0].1),
        (local_bounds[1].0, local_bounds[0].1),
        (local_bounds[1].0, local_bounds[1].1),
        (local_bounds[0].0, local_bounds[1].1),
    ];

    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;

    for corner in corners {
        let (x, y) = transform_point_to_logical_screen(corner, transform);
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }

    [(min_x, min_y), (max_x, max_y)]
}

fn inflate_logical_rect(logical_rect: [(f32, f32); 2], padding: f32) -> [(f32, f32); 2] {
    if padding <= 0.0 {
        return logical_rect;
    }

    let min_x = logical_rect[0].0.min(logical_rect[1].0) - padding;
    let min_y = logical_rect[0].1.min(logical_rect[1].1) - padding;
    let max_x = logical_rect[0].0.max(logical_rect[1].0) + padding;
    let max_y = logical_rect[0].1.max(logical_rect[1].1) + padding;

    [(min_x, min_y), (max_x, max_y)]
}

fn logical_rect_is_finite(logical_rect: [(f32, f32); 2]) -> bool {
    logical_rect[0].0.is_finite()
        && logical_rect[0].1.is_finite()
        && logical_rect[1].0.is_finite()
        && logical_rect[1].1.is_finite()
}

fn round_capture_coordinate(value: f32, scale_factor: f32, round_outward: bool) -> Option<i32> {
    let scaled_value = value * scale_factor;
    if !scaled_value.is_finite() {
        return None;
    }

    let rounded_value = if round_outward {
        scaled_value.ceil()
    } else {
        scaled_value.floor()
    };
    if !rounded_value.is_finite()
        || rounded_value < i32::MIN as f32
        || rounded_value > i32::MAX as f32
    {
        return None;
    }

    Some(rounded_value as i32)
}

fn logical_rect_to_physical_capture_rect(
    logical_rect: [(f32, f32); 2],
    scale_factor: f64,
) -> Option<(i32, i32, u32, u32)> {
    let scale_factor = scale_factor as f32;
    if !scale_factor.is_finite() || scale_factor <= 0.0 || !logical_rect_is_finite(logical_rect) {
        return None;
    }

    let min_x = logical_rect[0].0.min(logical_rect[1].0);
    let min_y = logical_rect[0].1.min(logical_rect[1].1);
    let max_x = logical_rect[0].0.max(logical_rect[1].0);
    let max_y = logical_rect[0].1.max(logical_rect[1].1);

    let physical_min_x = round_capture_coordinate(min_x, scale_factor, false)?;
    let physical_min_y = round_capture_coordinate(min_y, scale_factor, false)?;
    let physical_max_x = round_capture_coordinate(max_x, scale_factor, true)?;
    let physical_max_y = round_capture_coordinate(max_y, scale_factor, true)?;

    let width = physical_max_x.saturating_sub(physical_min_x) as u32;
    let height = physical_max_y.saturating_sub(physical_min_y) as u32;
    if width == 0 || height == 0 {
        return None;
    }

    Some((physical_min_x, physical_min_y, width, height))
}

fn capture_size_exceeds_limits(capture_size: (u32, u32), max_capture_dimension: u32) -> bool {
    capture_size.0 > max_capture_dimension || capture_size.1 > max_capture_dimension
}

const MAX_BACKDROP_CAPTURE_VIEWPORT_TEXEL_MULTIPLIER: u64 = 4;

fn max_backdrop_capture_texels(physical_size: (u32, u32)) -> u64 {
    u64::from(physical_size.0)
        .saturating_mul(u64::from(physical_size.1))
        .saturating_mul(MAX_BACKDROP_CAPTURE_VIEWPORT_TEXEL_MULTIPLIER)
}

fn capture_size_exceeds_budget(capture_size: (u32, u32), physical_size: (u32, u32)) -> bool {
    let capture_texels = u64::from(capture_size.0).saturating_mul(u64::from(capture_size.1));
    capture_texels > max_backdrop_capture_texels(physical_size)
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct BackdropCaptureRegion {
    capture_origin: (i32, i32),
    capture_size: (u32, u32),
    copy_source_origin: Option<(u32, u32)>,
    copy_destination_origin: (u32, u32),
    copy_size: (u32, u32),
}

impl BackdropCaptureRegion {
    fn sample_uniform(self) -> BackdropSamplingUniform {
        BackdropSamplingUniform::new(self.capture_origin, self.capture_size)
    }
}

#[cfg(test)]
fn screen_point_to_capture_uv(
    sample_transform: BackdropSamplingUniform,
    screen_point: (f32, f32),
) -> (f32, f32) {
    (
        (screen_point.0 - sample_transform.capture_origin[0])
            * sample_transform.inverse_capture_size[0],
        (screen_point.1 - sample_transform.capture_origin[1])
            * sample_transform.inverse_capture_size[1],
    )
}

fn resolve_capture_region_to_viewport(
    requested_rect: (i32, i32, u32, u32),
    physical_size: (u32, u32),
) -> BackdropCaptureRegion {
    let (capture_x, capture_y, capture_width, capture_height) = requested_rect;
    let capture_right = capture_x.saturating_add(capture_width as i32);
    let capture_bottom = capture_y.saturating_add(capture_height as i32);

    let overlap_left = capture_x.max(0);
    let overlap_top = capture_y.max(0);
    let overlap_right = capture_right.min(physical_size.0 as i32);
    let overlap_bottom = capture_bottom.min(physical_size.1 as i32);

    let overlap_width = overlap_right.saturating_sub(overlap_left).max(0) as u32;
    let overlap_height = overlap_bottom.saturating_sub(overlap_top).max(0) as u32;
    let has_overlap = overlap_width > 0 && overlap_height > 0;

    BackdropCaptureRegion {
        capture_origin: (capture_x, capture_y),
        capture_size: (capture_width, capture_height),
        copy_source_origin: has_overlap.then_some((overlap_left as u32, overlap_top as u32)),
        copy_destination_origin: (
            overlap_left.saturating_sub(capture_x) as u32,
            overlap_top.saturating_sub(capture_y) as u32,
        ),
        copy_size: (overlap_width, overlap_height),
    }
}

fn compute_backdrop_capture_region(
    draw_command: &DrawCommand,
    backdrop_config: effect::BackdropEffectConfig,
    scale_factor: f64,
    physical_size: (u32, u32),
    max_capture_dimension: u32,
) -> Option<BackdropCaptureRegion> {
    let logical_rect = match backdrop_config.capture_area {
        effect::BackdropCaptureArea::NodeBounds => transformed_bounds_to_logical_screen_rect(
            draw_command.local_bounds(),
            draw_command.transform(),
        ),
        effect::BackdropCaptureArea::FullScene => {
            let logical_width = physical_size.0 as f32 / scale_factor as f32;
            let logical_height = physical_size.1 as f32 / scale_factor as f32;
            [(0.0, 0.0), (logical_width, logical_height)]
        }
        effect::BackdropCaptureArea::ScreenRect(rect) => rect,
    };

    let logical_rect = inflate_logical_rect(logical_rect, backdrop_config.padding);

    logical_rect_to_physical_capture_rect(logical_rect, scale_factor).and_then(|requested_rect| {
        let capture_size = (requested_rect.2, requested_rect.3);
        if capture_size_exceeds_limits(capture_size, max_capture_dimension) {
            warn!(
                requested_width = capture_size.0,
                requested_height = capture_size.1,
                max_capture_dimension,
                "Skipping backdrop capture that exceeds supported texture dimensions"
            );
            return None;
        }

        if capture_size_exceeds_budget(capture_size, physical_size) {
            warn!(
                requested_width = capture_size.0,
                requested_height = capture_size.1,
                requested_texels =
                    u64::from(capture_size.0).saturating_mul(u64::from(capture_size.1)),
                max_capture_texels = max_backdrop_capture_texels(physical_size),
                viewport_width = physical_size.0,
                viewport_height = physical_size.1,
                "Skipping backdrop capture that exceeds the per-viewport texel budget"
            );
            return None;
        }

        Some(resolve_capture_region_to_viewport(
            requested_rect,
            physical_size,
        ))
    })
}

pub(super) fn compute_downsampled_dimensions(
    capture_size: (u32, u32),
    downsample: f32,
) -> (u32, u32) {
    (
        ((capture_size.0 as f32) * downsample).ceil().max(1.0) as u32,
        ((capture_size.1 as f32) * downsample).ceil().max(1.0) as u32,
    )
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
    state: &mut RendererState,
) {
    let SegmentRenderTarget {
        color_view,
        color_resolve_target,
        depth_stencil_view,
        backdrop_source,
        backdrop_context,
    } = target;
    let pipelines = &state.pipelines;
    let buffers = &state.buffers;
    let scratch = &mut state.scratch;
    let mut event_idx = 0;
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
                    current_scissor.0,
                    current_scissor.1,
                    current_scissor.2,
                    current_scissor.3,
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
                            if let Some(resources) = &state.composite_resources {
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
                                // Non-clipping parent: draw as leaf, children inherit
                                // the same stencil.
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
                                state.physical_size,
                            ) {
                                // Scissor optimization: rect parent with axis-aligned
                                // transform. Use hardware scissor instead of stencil.
                                let current_scissor = scratch
                                    .scissor_stack
                                    .last()
                                    .copied()
                                    .unwrap_or(viewport_scissor);
                                let clipped = intersect_scissor(current_scissor, scissor_rect);
                                scratch.scissor_stack.push(clipped);
                                render_pass
                                    .set_scissor_rect(clipped.0, clipped.1, clipped.2, clipped.3);
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

                        // Effect result: Pre composited, no stencil was pushed.
                        if effect_results.contains_key(&node_id) {
                            continue;
                        }

                        if let Some(draw_command) = state.draw_tree.get_mut(node_id) {
                            // Leaf: already drew in Pre, nothing to undo.
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
                                    render_pass.set_scissor_rect(prev.0, prev.1, prev.2, prev.3);
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

            // Flush any remaining leaf batch at the end of the segment.
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
            let bctx = backdrop_context.unwrap();
            let composite_bind_group_layout = &state
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
                let backdrop_config = effect_instance.backdrop_config.unwrap_or_default();

                if let Some(capture_region) = compute_backdrop_capture_region(
                    draw_command,
                    backdrop_config,
                    state.scale_factor,
                    state.physical_size,
                    bctx.max_texture_dimension_2d,
                ) {
                    let backdrop_sampling_uniform = capture_region.sample_uniform();
                    let (capture_width, capture_height) = capture_region.capture_size;
                    let backdrop_source =
                        backdrop_source.expect("backdrop source required for backdrop effects");
                    let backdrop_capture_texture = state.texture_pool.acquire_color_only(
                        bctx.device,
                        capture_width,
                        capture_height,
                        bctx.config_format,
                        1,
                    );
                    if capture_region.copy_size != capture_region.capture_size {
                        clear_texture_to_transparent(
                            encoder,
                            &backdrop_capture_texture.color_view,
                            "backdrop_capture_clear",
                        );
                    }
                    if let Some((copy_source_x, copy_source_y)) = capture_region.copy_source_origin
                    {
                        encoder.copy_texture_to_texture(
                            wgpu::TexelCopyTextureInfo {
                                texture: backdrop_source.base_texture(),
                                mip_level: 0,
                                origin: wgpu::Origin3d {
                                    x: copy_source_x,
                                    y: copy_source_y,
                                    z: 0,
                                },
                                aspect: wgpu::TextureAspect::All,
                            },
                            wgpu::TexelCopyTextureInfo {
                                texture: &backdrop_capture_texture.color_texture,
                                mip_level: 0,
                                origin: wgpu::Origin3d {
                                    x: capture_region.copy_destination_origin.0,
                                    y: capture_region.copy_destination_origin.1,
                                    z: 0,
                                },
                                aspect: wgpu::TextureAspect::All,
                            },
                            wgpu::Extent3d {
                                width: capture_region.copy_size.0,
                                height: capture_region.copy_size.1,
                                depth_or_array_layers: 1,
                            },
                        );
                    }

                    if let Some(foreground_view) = backdrop_source.foreground_view() {
                        let layer_params = effect::backdrop_layer_params(
                            capture_region.capture_origin,
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

                    let effect_input_size = compute_downsampled_dimensions(
                        (capture_width, capture_height),
                        backdrop_config.downsample,
                    );
                    let mut downsampled_capture_texture: Option<PooledTexture> = None;

                    if effect_input_size != (capture_width, capture_height) {
                        let downsampled_capture_target = state.texture_pool.acquire_color_only(
                            bctx.device,
                            effect_input_size.0,
                            effect_input_size.1,
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
                        .get(&effect_instance.effect_id)
                        .expect("loaded backdrop effect must exist");
                    let effect_output = apply_effect_passes(
                        bctx.device,
                        encoder,
                        &mut state.texture_pool,
                        EffectPassRunConfig {
                            loaded_effect,
                            params_bind_group: effect_instance
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
                            width: effect_input_size.0,
                            height: effect_input_size.1,
                            texture_format: bctx.config_format,
                            label_prefix: "backdrop_effect",
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
                    current_scissor.0,
                    current_scissor.1,
                    current_scissor.2,
                    current_scissor.3,
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
                // Leaf: skip both Pre and Post events.
                event_idx += 2;
            } else if backdrop_clips_children {
                // Non-leaf clipping node: children inherit the backdrop shape's stencil. The
                // normal Post handler decrements after descendants render.
                scratch.stencil_stack.push(this_stencil);
                scratch.clip_kind_stack.push(ClipKind::Stencil);
                event_idx += 1;
            } else {
                // Non-leaf visible-overflow node: the backdrop effect itself used this node's
                // stencil, but descendants inherit the nearest ancestor clip.
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

#[cfg(test)]
mod tests {
    use super::{
        capture_size_exceeds_budget, capture_size_exceeds_limits, inflate_logical_rect,
        logical_rect_to_physical_capture_rect, resolve_capture_region_to_viewport,
        screen_point_to_capture_uv, transform_point_to_logical_screen,
    };
    use crate::vertex::InstanceTransform;

    #[test]
    fn physical_capture_rect_preserves_requested_size_outside_viewport() {
        let requested_rect =
            logical_rect_to_physical_capture_rect([(-10.0, 5.0), (30.0, 25.0)], 1.0)
                .expect("capture rect should be non-empty");

        assert_eq!(requested_rect, (-10, 5, 40, 20));
    }

    #[test]
    fn transform_point_to_logical_screen_preserves_negative_w_sign() {
        let transform = InstanceTransform {
            col0: [2.0, 0.0, 0.0, 0.0],
            col1: [0.0, 3.0, 0.0, 0.0],
            col2: [0.0, 0.0, 1.0, 0.0],
            col3: [0.0, 0.0, 0.0, -2.0],
        };

        let point = transform_point_to_logical_screen((1.0, 1.0), Some(transform));

        assert_eq!(point, (-1.0, -1.5));
    }

    #[test]
    fn physical_capture_rect_rejects_non_finite_coordinates() {
        let requested_rect =
            logical_rect_to_physical_capture_rect([(0.0, 0.0), (f32::INFINITY, 25.0)], 1.0);

        assert!(requested_rect.is_none());
    }

    #[test]
    fn capture_size_exceeds_limits_rejects_oversized_regions() {
        let max_capture_dimension = 4_096u32;

        assert!(capture_size_exceeds_limits(
            (max_capture_dimension + 1, 64),
            max_capture_dimension,
        ));
        assert!(!capture_size_exceeds_limits(
            (max_capture_dimension, max_capture_dimension),
            max_capture_dimension,
        ));
    }

    #[test]
    fn capture_size_exceeds_budget_rejects_large_dimension_valid_regions() {
        assert!(capture_size_exceeds_budget((1_500, 1_500), (480, 800)));
        assert!(!capture_size_exceeds_budget((480, 800), (480, 800)));
    }

    #[test]
    fn capture_region_offsets_visible_copy_into_transparent_texture() {
        let region = resolve_capture_region_to_viewport((-10, 5, 40, 20), (100, 100));

        assert_eq!(region.capture_origin, (-10, 5));
        assert_eq!(region.capture_size, (40, 20));
        assert_eq!(region.copy_source_origin, Some((0, 5)));
        assert_eq!(region.copy_destination_origin, (10, 0));
        assert_eq!(region.copy_size, (30, 20));
    }

    #[test]
    fn capture_region_skips_copy_when_fully_offscreen() {
        let offscreen_right = resolve_capture_region_to_viewport((110, 5, 20, 20), (100, 100));
        let offscreen_bottom = resolve_capture_region_to_viewport((5, 110, 20, 20), (100, 100));

        assert_eq!(offscreen_right.copy_source_origin, None);
        assert_eq!(offscreen_right.copy_size, (0, 20));
        assert_eq!(offscreen_bottom.copy_source_origin, None);
        assert_eq!(offscreen_bottom.copy_size, (20, 0));
    }

    #[test]
    fn inflate_logical_rect_expands_symmetrically() {
        let rect = inflate_logical_rect([(10.0, 20.0), (30.0, 40.0)], 5.0);

        assert_eq!(rect, [(5.0, 15.0), (35.0, 45.0)]);
    }

    #[test]
    fn padded_capture_preserves_node_window_inside_capture() {
        let padded_rect = inflate_logical_rect([(100.0, 100.0), (200.0, 200.0)], 20.0);
        let requested_rect = logical_rect_to_physical_capture_rect(padded_rect, 1.0)
            .expect("capture rect should be non-empty");
        let capture_region = resolve_capture_region_to_viewport(requested_rect, (1_000, 1_000));
        let sample_transform = capture_region.sample_uniform();

        let top_left_uv = screen_point_to_capture_uv(sample_transform, (100.5, 100.5));
        let bottom_right_uv = screen_point_to_capture_uv(sample_transform, (199.5, 199.5));

        assert!((top_left_uv.0 - (20.5 / 140.0)).abs() < 1e-6);
        assert!((top_left_uv.1 - (20.5 / 140.0)).abs() < 1e-6);
        assert!((bottom_right_uv.0 - (119.5 / 140.0)).abs() < 1e-6);
        assert!((bottom_right_uv.1 - (119.5 / 140.0)).abs() < 1e-6);
    }
}
