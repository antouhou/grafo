use super::types::TraversalEvent;
use super::*;
use crate::vertex::InstanceTransform;

/// Compute a screen-space scissor rect from a local-space axis-aligned rect and its transform.
///
/// Returns `Some((x, y, width, height))` in physical pixels if the transform preserves
/// axis-alignment (identity, translation, and/or scale — no rotation, skew, or perspective).
/// Returns `None` if scissor clipping cannot be used (the caller should fall back to stencil).
pub(super) fn compute_scissor_rect(
    rect: [(f32, f32); 2],
    transform: Option<InstanceTransform>,
    scale_factor: f64,
    physical_size: (u32, u32),
) -> Option<(u32, u32, u32, u32)> {
    let t = transform.unwrap_or_else(InstanceTransform::identity);

    // The InstanceTransform struct stores 4 "rows" which the WGSL shader treats as columns
    // of a mat4x4 (column-major on the GPU). The effective math for pos_out = M * pos_in is:
    //   out.x = row0[0]*x + row1[0]*y + row3[0]   (row3[0] = tx)
    //   out.y = row0[1]*x + row1[1]*y + row3[1]   (row3[1] = ty)
    //   out.w = row0[3]*x + row1[3]*y + row3[3]
    // For affine (non-perspective), we need out.w == 1 for all (x,y):
    //   row0[3] == 0, row1[3] == 0, row3[3] == 1
    if t.row0[3] != 0.0 || t.row1[3] != 0.0 || t.row3[3] != 1.0 {
        return None;
    }

    // Check axis-aligned: no rotation/skew in the 2D affine part.
    // In column-major GPU layout: row0[0]=sx, row0[1]=shy, row1[0]=shx, row1[1]=sy
    // For axis-aligned rects we need shx == 0 and shy == 0.
    if t.row0[1] != 0.0 || t.row1[0] != 0.0 {
        return None;
    }

    let sx = t.row0[0];
    let sy = t.row1[1];
    let tx = t.row3[0];
    let ty = t.row3[1];

    // Transform the two corners.
    let x0 = rect[0].0 * sx + tx;
    let y0 = rect[0].1 * sy + ty;
    let x1 = rect[1].0 * sx + tx;
    let y1 = rect[1].1 * sy + ty;

    // Ensure min/max ordering (scale could be negative).
    let min_x = x0.min(x1);
    let min_y = y0.min(y1);
    let max_x = x0.max(x1);
    let max_y = y0.max(y1);

    // Convert from logical pixels to physical pixels.
    let sf = scale_factor as f32;
    let px_min_x = ((min_x * sf).floor().max(0.0) as u32).min(physical_size.0);
    let px_min_y = ((min_y * sf).floor().max(0.0) as u32).min(physical_size.1);
    let px_max_x = (max_x * sf).ceil().min(physical_size.0 as f32) as u32;
    let px_max_y = (max_y * sf).ceil().min(physical_size.1 as f32) as u32;

    let width = px_max_x.saturating_sub(px_min_x);
    let height = px_max_y.saturating_sub(px_min_y);

    Some((px_min_x, px_min_y, width, height))
}

/// Intersect two scissor rects, returning the overlapping region.
/// If the rects don't overlap, returns a zero-size rect.
pub(super) fn intersect_scissor(
    a: (u32, u32, u32, u32),
    b: (u32, u32, u32, u32),
) -> (u32, u32, u32, u32) {
    let a_right = a.0 + a.2;
    let a_bottom = a.1 + a.3;
    let b_right = b.0 + b.2;
    let b_bottom = b.1 + b.3;

    let left = a.0.max(b.0);
    let top = a.1.max(b.1);
    let right = a_right.min(b_right);
    let bottom = a_bottom.min(b_bottom);

    let width = right.saturating_sub(left);
    let height = bottom.saturating_sub(top);

    (left, top, width, height)
}

/// Check whether a non-leaf draw command is eligible for scissor clipping,
/// and if so, compute the scissor rect. This centralizes the eligibility logic
/// so pre-visit and post-visit make the same deterministic decision.
pub(super) fn try_scissor_for_rect(
    draw_command: &DrawCommand,
    scale_factor: f64,
    physical_size: (u32, u32),
) -> Option<(u32, u32, u32, u32)> {
    if !draw_command.is_rect() {
        return None;
    }
    let rect_bounds = match draw_command.rect_bounds() {
        Some(b) => b,
        None => {
            return None;
        }
    };
    let transform = match draw_command {
        DrawCommand::Shape(s) => s.transform(),
        DrawCommand::CachedShape(s) => s.transform(),
    };
    compute_scissor_rect(rect_bounds, transform, scale_factor, physical_size)
}

pub(super) struct AppliedEffectOutput {
    pub(super) composite_bind_group: wgpu::BindGroup,
    pub(super) primary_work_texture: wgpu::Texture,
    pub(super) secondary_work_texture: Option<wgpu::Texture>,
}

impl AppliedEffectOutput {
    pub(super) fn push_work_textures_into(
        self,
        output_textures: &mut Vec<wgpu::Texture>,
    ) -> wgpu::BindGroup {
        output_textures.push(self.primary_work_texture);
        if let Some(secondary_work_texture) = self.secondary_work_texture {
            output_textures.push(secondary_work_texture);
        }
        self.composite_bind_group
    }
}

pub(super) struct EffectPassRunConfig<'a> {
    pub(super) loaded_effect: &'a LoadedEffect,
    pub(super) params_bind_group: Option<&'a wgpu::BindGroup>,
    pub(super) source_view: &'a wgpu::TextureView,
    pub(super) effect_sampler: &'a wgpu::Sampler,
    pub(super) composite_bind_group_layout: &'a wgpu::BindGroupLayout,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) texture_format: wgpu::TextureFormat,
    pub(super) label_prefix: &'a str,
}

pub(super) fn apply_effect_passes(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    config: EffectPassRunConfig<'_>,
) -> AppliedEffectOutput {
    let number_of_passes = config.loaded_effect.passes.len();

    let effect_texture_a = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(&format!("{}_work_a", config.label_prefix)),
        size: wgpu::Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: config.texture_format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let effect_view_a = effect_texture_a.create_view(&wgpu::TextureViewDescriptor::default());

    let effect_texture_b = if number_of_passes > 1 {
        Some(device.create_texture(&wgpu::TextureDescriptor {
            label: Some(&format!("{}_work_b", config.label_prefix)),
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: config.texture_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        }))
    } else {
        None
    };
    let effect_view_b = effect_texture_b
        .as_ref()
        .map(|texture| texture.create_view(&wgpu::TextureViewDescriptor::default()));

    let mut previous_input_view: &wgpu::TextureView = config.source_view;

    for (pass_index, effect_pass) in config.loaded_effect.passes.iter().enumerate() {
        let output_view = if pass_index % 2 == 0 {
            &effect_view_a
        } else {
            effect_view_b.as_ref().unwrap()
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

    let composite_bind_group = effect::create_texture_sample_bind_group(
        device,
        config.composite_bind_group_layout,
        previous_input_view,
        config.effect_sampler,
        Some(&format!("{}_composite_bg", config.label_prefix)),
    );

    AppliedEffectOutput {
        composite_bind_group,
        primary_work_texture: effect_texture_a,
        secondary_work_texture: effect_texture_b,
    }
}

#[allow(clippy::too_many_arguments)]
fn bind_shape_texture_layers(
    render_pass: &mut wgpu::RenderPass<'_>,
    texture_ids: [Option<u64>; 2],
    texture_manager: &TextureManager,
    shape_texture_bind_group_layout_background: &wgpu::BindGroupLayout,
    shape_texture_bind_group_layout_foreground: &wgpu::BindGroupLayout,
    default_shape_texture_bind_groups: &[Arc<wgpu::BindGroup>; 2],
    shape_texture_layout_epoch: u64,
    bound_texture_state: &mut crate::renderer::types::BoundTextureState,
) {
    for (layer, &texture_id) in texture_ids.iter().enumerate() {
        if !bound_texture_state.needs_rebind(layer, texture_id) {
            continue;
        }
        if let Some(texture_id) = texture_id {
            if texture_manager.is_texture_loaded(texture_id) {
                if let Ok(bind_group) = texture_manager.get_or_create_shape_bind_group(
                    if layer == 0 {
                        shape_texture_bind_group_layout_background
                    } else {
                        shape_texture_bind_group_layout_foreground
                    },
                    shape_texture_layout_epoch,
                    texture_id,
                ) {
                    render_pass.set_bind_group(1 + layer as u32, &*bind_group, &[]);
                }
            }
        } else {
            render_pass.set_bind_group(
                1 + layer as u32,
                &*default_shape_texture_bind_groups[layer],
                &[],
            );
        }
    }
}

pub(super) fn bind_instance_buffers(
    render_pass: &mut wgpu::RenderPass<'_>,
    shape: &(impl DrawShapeCommand + ?Sized),
    buffers: &crate::renderer::types::Buffers,
) {
    if let Some(instance_idx) = shape.instance_index() {
        if let Some(instance_transform_buffer) = buffers.aggregated_instance_transform_buffer {
            let stride = std::mem::size_of::<InstanceTransform>() as u64;
            let offset = instance_idx as u64 * stride;
            render_pass
                .set_vertex_buffer(1, instance_transform_buffer.slice(offset..offset + stride));
        } else {
            render_pass.set_vertex_buffer(1, buffers.identity_instance_transform_buffer.slice(..));
        }

        if let Some(instance_color_buffer) = buffers.aggregated_instance_color_buffer {
            let stride = std::mem::size_of::<InstanceColor>() as u64;
            let offset = instance_idx as u64 * stride;
            render_pass.set_vertex_buffer(2, instance_color_buffer.slice(offset..offset + stride));
        } else {
            render_pass.set_vertex_buffer(2, buffers.identity_instance_color_buffer.slice(..));
        }

        if let Some(instance_metadata_buffer) = buffers.aggregated_instance_metadata_buffer {
            let stride = std::mem::size_of::<InstanceMetadata>() as u64;
            let offset = instance_idx as u64 * stride;
            render_pass
                .set_vertex_buffer(3, instance_metadata_buffer.slice(offset..offset + stride));
        } else {
            render_pass.set_vertex_buffer(3, buffers.identity_instance_metadata_buffer.slice(..));
        }
    } else {
        render_pass.set_vertex_buffer(1, buffers.identity_instance_transform_buffer.slice(..));
        render_pass.set_vertex_buffer(2, buffers.identity_instance_color_buffer.slice(..));
        render_pass.set_vertex_buffer(3, buffers.identity_instance_metadata_buffer.slice(..));
    }
}

pub(super) fn handle_increment_pass<'rp>(
    render_pass: &mut wgpu::RenderPass<'rp>,
    currently_set_pipeline: &mut crate::renderer::types::PipelineTracker,
    bound_texture_state: &mut crate::renderer::types::BoundTextureState,
    stencil_stack: &mut Vec<u32>,
    shape: &mut (impl DrawShapeCommand + ?Sized),
    pipelines: &crate::renderer::types::Pipelines,
    buffers: &crate::renderer::types::Buffers,
) {
    if let Some(index_range) = shape.index_buffer_range() {
        if shape.is_empty() {
            return;
        }

        if !matches!(
            currently_set_pipeline.current,
            crate::renderer::types::Pipeline::StencilIncrement
        ) {
            render_pass.set_pipeline(pipelines.and_pipeline);
            render_pass.set_bind_group(0, pipelines.and_bind_group, &[]);
            render_pass.set_bind_group(1, &*pipelines.default_shape_texture_bind_groups[0], &[]);
            render_pass.set_bind_group(2, &*pipelines.default_shape_texture_bind_groups[1], &[]);
            // Inform the tracker that default textures are now bound on both layers.
            bound_texture_state.needs_rebind(0, None);
            bound_texture_state.needs_rebind(1, None);

            if !matches!(
                currently_set_pipeline.current,
                crate::renderer::types::Pipeline::StencilDecrement
            ) {
                render_pass.set_vertex_buffer(0, buffers.aggregated_vertex_buffer.slice(..));
                render_pass.set_index_buffer(
                    buffers.aggregated_index_buffer.slice(..),
                    wgpu::IndexFormat::Uint16,
                );
            }

            currently_set_pipeline.switch_to(crate::renderer::types::Pipeline::StencilIncrement);
        }

        bind_shape_texture_layers(
            render_pass,
            [shape.texture_id(0), shape.texture_id(1)],
            pipelines.texture_manager,
            pipelines.shape_texture_bind_group_layout_background,
            pipelines.shape_texture_bind_group_layout_foreground,
            pipelines.default_shape_texture_bind_groups,
            pipelines.shape_texture_layout_epoch,
            bound_texture_state,
        );

        bind_instance_buffers(render_pass, shape, buffers);

        let parent_stencil = stencil_stack.last().copied().unwrap_or(0);
        render_buffer_range_to_texture(index_range, render_pass, parent_stencil);

        let this_stencil = parent_stencil + 1;
        *shape.stencil_ref_mut() = Some(this_stencil);
        stencil_stack.push(this_stencil);
    }
}

pub(super) fn handle_decrement_pass<'rp>(
    render_pass: &mut wgpu::RenderPass<'rp>,
    currently_set_pipeline: &mut crate::renderer::types::PipelineTracker,
    bound_texture_state: &mut crate::renderer::types::BoundTextureState,
    stencil_stack: &mut Vec<u32>,
    shape: &mut (impl DrawShapeCommand + ?Sized),
    pipelines: &crate::renderer::types::Pipelines,
    buffers: &crate::renderer::types::Buffers,
) {
    if let Some(index_range) = shape.index_buffer_range() {
        if shape.is_empty() {
            return;
        }

        if !matches!(
            currently_set_pipeline.current,
            crate::renderer::types::Pipeline::StencilDecrement
        ) {
            render_pass.set_pipeline(pipelines.decrementing_pipeline);
            render_pass.set_bind_group(0, pipelines.decrementing_bind_group, &[]);
            render_pass.set_bind_group(1, &*pipelines.default_shape_texture_bind_groups[0], &[]);
            render_pass.set_bind_group(2, &*pipelines.default_shape_texture_bind_groups[1], &[]);
            bound_texture_state.needs_rebind(0, None);
            bound_texture_state.needs_rebind(1, None);

            if !matches!(
                currently_set_pipeline.current,
                crate::renderer::types::Pipeline::StencilIncrement
            ) {
                render_pass.set_vertex_buffer(0, buffers.aggregated_vertex_buffer.slice(..));
                render_pass.set_index_buffer(
                    buffers.aggregated_index_buffer.slice(..),
                    wgpu::IndexFormat::Uint16,
                );
            }

            currently_set_pipeline.switch_to(crate::renderer::types::Pipeline::StencilDecrement);
        }

        bind_instance_buffers(render_pass, shape, buffers);

        let this_shape_stencil = shape.stencil_ref_mut().unwrap_or(0);
        render_buffer_range_to_texture(index_range, render_pass, this_shape_stencil);

        if shape.stencil_ref_mut().is_some() {
            stencil_stack.pop();
        }
    }
}

/// O3: Leaf-node draw — single draw call with stencil Equal + Keep.
/// For nodes without children, the increment + decrement pair cancels out,
/// so we can skip both and just draw at the parent's stencil reference.
pub(super) fn handle_leaf_draw_pass<'rp>(
    render_pass: &mut wgpu::RenderPass<'rp>,
    currently_set_pipeline: &mut crate::renderer::types::PipelineTracker,
    bound_texture_state: &mut crate::renderer::types::BoundTextureState,
    stencil_stack: &[u32],
    shape: &mut (impl DrawShapeCommand + ?Sized),
    pipelines: &crate::renderer::types::Pipelines,
    buffers: &crate::renderer::types::Buffers,
) {
    if let Some(index_range) = shape.index_buffer_range() {
        if shape.is_empty() {
            return;
        }

        if !matches!(
            currently_set_pipeline.current,
            crate::renderer::types::Pipeline::LeafDraw
        ) {
            render_pass.set_pipeline(pipelines.leaf_draw_pipeline);
            render_pass.set_bind_group(0, pipelines.and_bind_group, &[]);
            render_pass.set_bind_group(1, &*pipelines.default_shape_texture_bind_groups[0], &[]);
            render_pass.set_bind_group(2, &*pipelines.default_shape_texture_bind_groups[1], &[]);
            bound_texture_state.needs_rebind(0, None);
            bound_texture_state.needs_rebind(1, None);

            if !matches!(
                currently_set_pipeline.current,
                crate::renderer::types::Pipeline::StencilIncrement
                    | crate::renderer::types::Pipeline::StencilDecrement
            ) {
                render_pass.set_vertex_buffer(0, buffers.aggregated_vertex_buffer.slice(..));
                render_pass.set_index_buffer(
                    buffers.aggregated_index_buffer.slice(..),
                    wgpu::IndexFormat::Uint16,
                );
            }

            currently_set_pipeline.switch_to(crate::renderer::types::Pipeline::LeafDraw);
        }

        bind_shape_texture_layers(
            render_pass,
            [shape.texture_id(0), shape.texture_id(1)],
            pipelines.texture_manager,
            pipelines.shape_texture_bind_group_layout_background,
            pipelines.shape_texture_bind_group_layout_foreground,
            pipelines.default_shape_texture_bind_groups,
            pipelines.shape_texture_layout_epoch,
            bound_texture_state,
        );

        bind_instance_buffers(render_pass, shape, buffers);

        let parent_stencil = stencil_stack.last().copied().unwrap_or(0);
        render_buffer_range_to_texture(index_range, render_pass, parent_stencil);

        // Leaf node: stencil was not modified, so record the parent stencil as this node's ref
        // (children would read parent_stencil + 1, but leaves have no children).
        *shape.stencil_ref_mut() = Some(parent_stencil);
    }
}

/// Accumulates consecutive leaf nodes that share the same geometry (index range),
/// texture, and parent stencil reference so they can be emitted as a single
/// multi-instance `draw_indexed` call.
#[derive(Default)]
pub(super) struct PendingLeafBatch {
    index_range: (usize, usize),
    texture_ids: [Option<u64>; 2],
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
        index_range: (usize, usize),
        texture_ids: [Option<u64>; 2],
        parent_stencil: u32,
        instance_index: u32,
    ) -> bool {
        self.index_range == index_range
            && self.texture_ids == texture_ids
            && self.parent_stencil == parent_stencil
            && instance_index == self.first_instance_index + self.instance_count
    }
}

/// Ensure the leaf-draw pipeline and full instance buffers are bound,
/// then issue one `draw_indexed` call for the accumulated batch.
pub(super) fn flush_pending_leaf_batch(
    batch: &mut PendingLeafBatch,
    render_pass: &mut wgpu::RenderPass<'_>,
    currently_set_pipeline: &mut crate::renderer::types::PipelineTracker,
    bound_texture_state: &mut crate::renderer::types::BoundTextureState,
    pipelines: &crate::renderer::types::Pipelines,
    buffers: &crate::renderer::types::Buffers,
) {
    if batch.is_empty() {
        return;
    }

    // Ensure leaf pipeline is active.
    if !matches!(
        currently_set_pipeline.current,
        crate::renderer::types::Pipeline::LeafDraw
    ) {
        render_pass.set_pipeline(pipelines.leaf_draw_pipeline);
        render_pass.set_bind_group(0, pipelines.and_bind_group, &[]);
        render_pass.set_bind_group(1, &*pipelines.default_shape_texture_bind_groups[0], &[]);
        render_pass.set_bind_group(2, &*pipelines.default_shape_texture_bind_groups[1], &[]);
        bound_texture_state.needs_rebind(0, None);
        bound_texture_state.needs_rebind(1, None);

        if !matches!(
            currently_set_pipeline.current,
            crate::renderer::types::Pipeline::StencilIncrement
                | crate::renderer::types::Pipeline::StencilDecrement
        ) {
            render_pass.set_vertex_buffer(0, buffers.aggregated_vertex_buffer.slice(..));
            render_pass.set_index_buffer(
                buffers.aggregated_index_buffer.slice(..),
                wgpu::IndexFormat::Uint16,
            );
        }

        currently_set_pipeline.switch_to(crate::renderer::types::Pipeline::LeafDraw);
    }

    // Bind textures for the batch.
    bind_shape_texture_layers(
        render_pass,
        batch.texture_ids,
        pipelines.texture_manager,
        pipelines.shape_texture_bind_group_layout_background,
        pipelines.shape_texture_bind_group_layout_foreground,
        pipelines.default_shape_texture_bind_groups,
        pipelines.shape_texture_layout_epoch,
        bound_texture_state,
    );

    // Bind the full aggregated instance buffers so the instances range selects
    // the correct transform/color/metadata for each batched shape.
    if let Some(instance_transform_buffer) = buffers.aggregated_instance_transform_buffer {
        render_pass.set_vertex_buffer(1, instance_transform_buffer.slice(..));
    } else {
        render_pass.set_vertex_buffer(1, buffers.identity_instance_transform_buffer.slice(..));
    }
    if let Some(instance_color_buffer) = buffers.aggregated_instance_color_buffer {
        render_pass.set_vertex_buffer(2, instance_color_buffer.slice(..));
    } else {
        render_pass.set_vertex_buffer(2, buffers.identity_instance_color_buffer.slice(..));
    }
    if let Some(instance_metadata_buffer) = buffers.aggregated_instance_metadata_buffer {
        render_pass.set_vertex_buffer(3, instance_metadata_buffer.slice(..));
    } else {
        render_pass.set_vertex_buffer(3, buffers.identity_instance_metadata_buffer.slice(..));
    }

    render_pass.set_stencil_reference(batch.parent_stencil);
    let index_start = batch.index_range.0 as u32;
    let index_end = (batch.index_range.0 + batch.index_range.1) as u32;
    let first = batch.first_instance_index;
    render_pass.draw_indexed(
        index_start..index_end,
        0,
        first..first + batch.instance_count,
    );

    batch.instance_count = 0;
}

/// Try to add a leaf shape to the pending batch. Returns `true` if the shape
/// was successfully batched (no draw call needed yet). Returns `false` if
/// the shape could not be batched (caller should use the normal single-draw path).
pub(super) fn try_batch_leaf(
    batch: &mut PendingLeafBatch,
    shape: &(impl DrawShapeCommand + ?Sized),
    parent_stencil: u32,
) -> bool {
    let index_range = match shape.index_buffer_range() {
        Some(range) => range,
        None => return false,
    };
    if shape.is_empty() {
        return false;
    }
    let instance_index = match shape.instance_index() {
        Some(idx) => idx as u32,
        None => return false,
    };
    let texture_ids = [shape.texture_id(0), shape.texture_id(1)];

    if batch.is_empty() {
        // Start a new batch.
        batch.index_range = index_range;
        batch.texture_ids = texture_ids;
        batch.parent_stencil = parent_stencil;
        batch.first_instance_index = instance_index;
        batch.instance_count = 1;
        return true;
    }

    if batch.matches(index_range, texture_ids, parent_stencil, instance_index) {
        batch.instance_count += 1;
        return true;
    }

    // Incompatible — caller must flush then handle this shape.
    false
}

/// Unified rendering function for all paths: main scene, effect subtrees,
/// and behind-group rendering. Processes a flat event list from
/// `plan_traversal_in_place`, breaking render passes at backdrop effect
/// boundaries when needed.
///
/// When `backdrop_ctx` is `None`, no backdrop breaks occur and the entire
/// event list is processed as a single segment (equivalent to the old
/// `traverse_mut`-based main path). When `Some`, backdrop nodes cause
/// segment breaks for framebuffer capture + effect application.
///
/// Maintains a **runtime stencil stack** that accurately tracks the actual
/// stencil buffer state, including scissor-optimized parents that skip
/// stencil writes. This fixes the stencil mismatch that previously broke
/// backdrop effects when ancestors used scissor clipping.
#[allow(clippy::too_many_arguments)]
pub(super) fn render_segments(
    draw_tree: &mut easy_tree::Tree<DrawCommand>,
    encoder: &mut wgpu::CommandEncoder,
    events: &[TraversalEvent],
    effect_results: &HashMap<usize, wgpu::BindGroup>,
    color_view: &wgpu::TextureView,
    color_resolve_target: Option<&wgpu::TextureView>,
    depth_stencil_view: &wgpu::TextureView,
    copy_source_texture: Option<&wgpu::Texture>,
    clear_first: bool,
    pipelines: &crate::renderer::types::Pipelines,
    buffers: &crate::renderer::types::Buffers,
    composite_pipeline: Option<&wgpu::RenderPipeline>,
    backdrop_ctx: Option<&crate::renderer::types::BackdropContext>,
    backdrop_work_textures: &mut Vec<wgpu::Texture>,
    stencil_stack: &mut Vec<u32>,
    scale_factor: f64,
    physical_size: (u32, u32),
    #[cfg(feature = "render_metrics")] pipeline_counts_out: &mut crate::renderer::metrics::PipelineSwitchCounts,
) {
    let mut event_idx = 0;
    let mut is_first_segment = clear_first;
    let mut currently_set_pipeline = crate::renderer::types::PipelineTracker::new();
    let mut bound_texture_state = crate::renderer::types::BoundTextureState::default();
    let (width, height) = physical_size;
    let viewport_scissor = (0u32, 0u32, width, height);
    let mut scissor_stack: Vec<(u32, u32, u32, u32)> = vec![viewport_scissor];
    let mut pending_leaf_batch = PendingLeafBatch::default();
    stencil_stack.clear();
    backdrop_work_textures.clear();

    while event_idx < events.len() {
        // --- Scan for the next backdrop boundary ---
        let mut segment_end = events.len();
        let mut backdrop_node_id: Option<usize> = None;
        if let Some(bctx) = backdrop_ctx {
            for (idx, event) in events.iter().enumerate().skip(event_idx) {
                if let TraversalEvent::Pre(node_id) = event {
                    if bctx.backdrop_effects.contains_key(node_id) {
                        segment_end = idx;
                        backdrop_node_id = Some(*node_id);
                        break;
                    }
                }
            }
        }

        // --- Process segment events [event_idx .. segment_end) ---
        if event_idx < segment_end {
            let mut render_pass = crate::pipeline::begin_render_pass_with_load_ops(
                encoder,
                Some(if is_first_segment {
                    "segment_clear_pass"
                } else {
                    "segment_load_pass"
                }),
                color_view,
                color_resolve_target,
                depth_stencil_view,
                crate::pipeline::RenderPassLoadOperations {
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

            // Render-pass boundaries reset GPU scissor — restore from our stack.
            let current_scissor = scissor_stack.last().copied().unwrap_or(viewport_scissor);
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
                    TraversalEvent::Pre(node_id) => {
                        let node_id = *node_id;

                        // --- Composite pre-rendered group effect result ---
                        if let Some(result_bind_group) = effect_results.get(&node_id) {
                            flush_pending_leaf_batch(
                                &mut pending_leaf_batch,
                                &mut render_pass,
                                &mut currently_set_pipeline,
                                &mut bound_texture_state,
                                pipelines,
                                buffers,
                            );
                            if let Some(cp) = composite_pipeline {
                                let parent_stencil =
                                    stencil_stack.last().copied().unwrap_or(0);
                                render_pass.set_pipeline(cp);
                                render_pass.set_bind_group(0, result_bind_group, &[]);
                                render_pass.set_stencil_reference(parent_stencil);
                                render_pass.draw(0..3, 0..1);
                                currently_set_pipeline.switch_to(types::Pipeline::None);
                                bound_texture_state.invalidate();
                            }
                            continue;
                        }

                        if let Some(draw_command) = draw_tree.get_mut(node_id) {
                            // --- Leaf node ---
                            if draw_command.is_leaf() {
                                let parent_stencil =
                                    stencil_stack.last().copied().unwrap_or(0);
                                let batched = match draw_command {
                                    DrawCommand::Shape(shape) => {
                                        let result = try_batch_leaf(
                                            &mut pending_leaf_batch,
                                            shape,
                                            parent_stencil,
                                        );
                                        if result {
                                            *shape.stencil_ref_mut() = Some(parent_stencil);
                                        }
                                        result
                                    }
                                    DrawCommand::CachedShape(shape) => {
                                        let result = try_batch_leaf(
                                            &mut pending_leaf_batch,
                                            shape,
                                            parent_stencil,
                                        );
                                        if result {
                                            *shape.stencil_ref_mut() = Some(parent_stencil);
                                        }
                                        result
                                    }
                                };
                                if !batched {
                                    flush_pending_leaf_batch(
                                        &mut pending_leaf_batch,
                                        &mut render_pass,
                                        &mut currently_set_pipeline,
                                        &mut bound_texture_state,
                                        pipelines,
                                        buffers,
                                    );
                                    match draw_command {
                                        DrawCommand::Shape(shape) => {
                                            if !try_batch_leaf(
                                                &mut pending_leaf_batch,
                                                shape,
                                                parent_stencil,
                                            ) {
                                                handle_leaf_draw_pass(
                                                    &mut render_pass,
                                                    &mut currently_set_pipeline,
                                                    &mut bound_texture_state,
                                                    stencil_stack,
                                                    shape,
                                                    pipelines,
                                                    buffers,
                                                );
                                            } else {
                                                *shape.stencil_ref_mut() =
                                                    Some(parent_stencil);
                                            }
                                        }
                                        DrawCommand::CachedShape(shape) => {
                                            if !try_batch_leaf(
                                                &mut pending_leaf_batch,
                                                shape,
                                                parent_stencil,
                                            ) {
                                                handle_leaf_draw_pass(
                                                    &mut render_pass,
                                                    &mut currently_set_pipeline,
                                                    &mut bound_texture_state,
                                                    stencil_stack,
                                                    shape,
                                                    pipelines,
                                                    buffers,
                                                );
                                            } else {
                                                *shape.stencil_ref_mut() =
                                                    Some(parent_stencil);
                                            }
                                        }
                                    }
                                }
                                continue;
                            }

                            // --- Non-leaf node ---
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
                                    stencil_stack.last().copied().unwrap_or(0);
                                match draw_command {
                                    DrawCommand::Shape(shape) => {
                                        *shape.stencil_ref_mut() = Some(parent_stencil);
                                        handle_leaf_draw_pass(
                                            &mut render_pass,
                                            &mut currently_set_pipeline,
                                            &mut bound_texture_state,
                                            stencil_stack,
                                            shape,
                                            pipelines,
                                            buffers,
                                        );
                                    }
                                    DrawCommand::CachedShape(shape) => {
                                        *shape.stencil_ref_mut() = Some(parent_stencil);
                                        handle_leaf_draw_pass(
                                            &mut render_pass,
                                            &mut currently_set_pipeline,
                                            &mut bound_texture_state,
                                            stencil_stack,
                                            shape,
                                            pipelines,
                                            buffers,
                                        );
                                    }
                                }
                                stencil_stack.push(parent_stencil);
                            } else if let Some(scissor_rect) = try_scissor_for_rect(
                                draw_command,
                                scale_factor,
                                physical_size,
                            ) {
                                // Scissor optimization: rect parent with axis-aligned
                                // transform. Use hardware scissor instead of stencil.
                                let current_scissor = scissor_stack
                                    .last()
                                    .copied()
                                    .unwrap_or(viewport_scissor);
                                let clipped =
                                    intersect_scissor(current_scissor, scissor_rect);
                                scissor_stack.push(clipped);
                                render_pass.set_scissor_rect(
                                    clipped.0, clipped.1, clipped.2, clipped.3,
                                );
                                #[cfg(feature = "render_metrics")]
                                currently_set_pipeline.record_scissor_clip();

                                // Draw the rect itself as a visible shape.
                                let parent_stencil =
                                    stencil_stack.last().copied().unwrap_or(0);
                                match draw_command {
                                    DrawCommand::Shape(shape) => {
                                        *shape.stencil_ref_mut() = Some(parent_stencil);
                                        handle_leaf_draw_pass(
                                            &mut render_pass,
                                            &mut currently_set_pipeline,
                                            &mut bound_texture_state,
                                            stencil_stack,
                                            shape,
                                            pipelines,
                                            buffers,
                                        );
                                    }
                                    DrawCommand::CachedShape(shape) => {
                                        *shape.stencil_ref_mut() = Some(parent_stencil);
                                        handle_leaf_draw_pass(
                                            &mut render_pass,
                                            &mut currently_set_pipeline,
                                            &mut bound_texture_state,
                                            stencil_stack,
                                            shape,
                                            pipelines,
                                            buffers,
                                        );
                                    }
                                }
                                // Push same stencil — children are clipped by scissor
                                // hardware, not by stencil buffer values.
                                stencil_stack.push(parent_stencil);
                            } else {
                                // Fall back to stencil increment.
                                match draw_command {
                                    DrawCommand::Shape(shape) => {
                                        handle_increment_pass(
                                            &mut render_pass,
                                            &mut currently_set_pipeline,
                                            &mut bound_texture_state,
                                            stencil_stack,
                                            shape,
                                            pipelines,
                                            buffers,
                                        );
                                    }
                                    DrawCommand::CachedShape(shape) => {
                                        handle_increment_pass(
                                            &mut render_pass,
                                            &mut currently_set_pipeline,
                                            &mut bound_texture_state,
                                            stencil_stack,
                                            shape,
                                            pipelines,
                                            buffers,
                                        );
                                    }
                                }
                            }
                        }
                    }
                    TraversalEvent::Post(node_id) => {
                        let node_id = *node_id;

                        // Effect result: Pre composited, no stencil was pushed.
                        if effect_results.contains_key(&node_id) {
                            continue;
                        }

                        if let Some(draw_command) = draw_tree.get_mut(node_id) {
                            // Leaf: already drew in Pre, nothing to undo.
                            if draw_command.is_leaf() {
                                continue;
                            }

                            // Non-clipping parent: pop dummy stencil entry.
                            if !draw_command.clips_children() {
                                stencil_stack.pop();
                                continue;
                            }

                            // Scissor parent: flush pending leaves (they use this
                            // scissor), then pop scissor and stencil.
                            if try_scissor_for_rect(
                                draw_command,
                                scale_factor,
                                physical_size,
                            )
                            .is_some()
                            {
                                flush_pending_leaf_batch(
                                    &mut pending_leaf_batch,
                                    &mut render_pass,
                                    &mut currently_set_pipeline,
                                    &mut bound_texture_state,
                                    pipelines,
                                    buffers,
                                );
                                scissor_stack.pop();
                                let prev = scissor_stack
                                    .last()
                                    .copied()
                                    .unwrap_or(viewport_scissor);
                                render_pass
                                    .set_scissor_rect(prev.0, prev.1, prev.2, prev.3);
                                stencil_stack.pop();
                                continue;
                            }

                            // Stencil decrement.
                            flush_pending_leaf_batch(
                                &mut pending_leaf_batch,
                                &mut render_pass,
                                &mut currently_set_pipeline,
                                &mut bound_texture_state,
                                pipelines,
                                buffers,
                            );
                            match draw_command {
                                DrawCommand::Shape(shape) => {
                                    handle_decrement_pass(
                                        &mut render_pass,
                                        &mut currently_set_pipeline,
                                        &mut bound_texture_state,
                                        stencil_stack,
                                        shape,
                                        pipelines,
                                        buffers,
                                    );
                                }
                                DrawCommand::CachedShape(shape) => {
                                    handle_decrement_pass(
                                        &mut render_pass,
                                        &mut currently_set_pipeline,
                                        &mut bound_texture_state,
                                        stencil_stack,
                                        shape,
                                        pipelines,
                                        buffers,
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

        // --- Handle the backdrop effect node ---
        if let Some(backdrop_node_id) = backdrop_node_id {
            let bctx = backdrop_ctx.unwrap();
            // Use the RUNTIME stencil stack — this correctly reflects scissor-
            // optimized ancestors that never wrote to the stencil buffer.
            let parent_stencil = stencil_stack.last().copied().unwrap_or(0);
            let this_stencil = parent_stencil + 1;

            if is_first_segment {
                crate::pipeline::begin_render_pass_with_load_ops(
                    encoder,
                    Some("segment_initial_clear"),
                    color_view,
                    color_resolve_target,
                    depth_stencil_view,
                    crate::pipeline::RenderPassLoadOperations {
                        color_load_op: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        depth_load_op: wgpu::LoadOp::Clear(1.0),
                        stencil_load_op: wgpu::LoadOp::Clear(0),
                    },
                );
            }

            // Copy current framebuffer to the backdrop snapshot texture.
            let copy_src = copy_source_texture
                .expect("copy_source_texture required for backdrop effects");
            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: copy_src,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: bctx.backdrop_snapshot_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );

            // Apply the backdrop effect (blur etc.) via ping-pong passes.
            let effect_instance =
                bctx.backdrop_effects.get(&backdrop_node_id).unwrap();
            let loaded_effect =
                bctx.loaded_effects.get(&effect_instance.effect_id).unwrap();
            let effect_output = apply_effect_passes(
                bctx.device,
                encoder,
                EffectPassRunConfig {
                    loaded_effect,
                    params_bind_group: effect_instance.params_bind_group.as_ref(),
                    source_view: bctx.backdrop_snapshot_view,
                    effect_sampler: bctx.effect_sampler,
                    composite_bind_group_layout: bctx.composite_bgl,
                    width,
                    height,
                    texture_format: bctx.config_format,
                    label_prefix: "backdrop_effect",
                },
            );
            let backdrop_composite_bind_group =
                effect_output.push_work_textures_into(backdrop_work_textures);

            // Begin the self-contained backdrop shape pass.
            let mut render_pass = crate::pipeline::begin_render_pass_with_load_ops(
                encoder,
                Some("backdrop_shape_pass"),
                color_view,
                color_resolve_target,
                depth_stencil_view,
                crate::pipeline::RenderPassLoadOperations {
                    color_load_op: wgpu::LoadOp::Load,
                    depth_load_op: wgpu::LoadOp::Load,
                    stencil_load_op: wgpu::LoadOp::Load,
                },
            );

            // Restore scissor in the backdrop pass.
            let current_scissor =
                scissor_stack.last().copied().unwrap_or(viewport_scissor);
            if current_scissor != viewport_scissor {
                render_pass.set_scissor_rect(
                    current_scissor.0,
                    current_scissor.1,
                    current_scissor.2,
                    current_scissor.3,
                );
            }

            // Step 1: Stencil-only draw (IncrementClamp at parent_stencil).
            if let Some(draw_command) = draw_tree.get_mut(backdrop_node_id) {
                render_pass.set_pipeline(bctx.stencil_only_pipeline);
                render_pass.set_bind_group(0, pipelines.and_bind_group, &[]);
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
                render_pass
                    .set_vertex_buffer(0, buffers.aggregated_vertex_buffer.slice(..));
                render_pass.set_index_buffer(
                    buffers.aggregated_index_buffer.slice(..),
                    wgpu::IndexFormat::Uint16,
                );

                let shape_index_range = match draw_command {
                    DrawCommand::Shape(shape) => {
                        bind_instance_buffers(&mut render_pass, shape, buffers);
                        shape.index_buffer_range()
                    }
                    DrawCommand::CachedShape(shape) => {
                        bind_instance_buffers(&mut render_pass, shape, buffers);
                        shape.index_buffer_range()
                    }
                };

                if let Some(sir) = shape_index_range {
                    render_pass.set_stencil_reference(parent_stencil);
                    let start = sir.0 as u32;
                    let end = (sir.0 + sir.1) as u32;
                    render_pass.draw_indexed(start..end, 0, 0..1);
                }

                match draw_command {
                    DrawCommand::Shape(shape) => {
                        *shape.stencil_ref_mut() = Some(this_stencil)
                    }
                    DrawCommand::CachedShape(shape) => {
                        *shape.stencil_ref_mut() = Some(this_stencil)
                    }
                }
            }

            // Step 2: Composite the blurred backdrop (stencil-masked fullscreen quad).
            let cp = composite_pipeline
                .expect("composite_pipeline required for backdrop effects");
            render_pass.set_pipeline(cp);
            render_pass.set_bind_group(0, &backdrop_composite_bind_group, &[]);
            render_pass.set_stencil_reference(this_stencil);
            render_pass.draw(0..3, 0..1);

            // Steps 3 + 4: Color draw then decrement (self-contained).
            if let Some(draw_command) = draw_tree.get_mut(backdrop_node_id) {
                // Step 3: Color draw (Equal + Keep).
                render_pass.set_pipeline(bctx.backdrop_color_pipeline);
                render_pass.set_bind_group(0, pipelines.and_bind_group, &[]);
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
                render_pass
                    .set_vertex_buffer(0, buffers.aggregated_vertex_buffer.slice(..));
                render_pass.set_index_buffer(
                    buffers.aggregated_index_buffer.slice(..),
                    wgpu::IndexFormat::Uint16,
                );

                let (texture_ids, shape_index_range) = match draw_command {
                    DrawCommand::Shape(shape) => {
                        bind_instance_buffers(&mut render_pass, shape, buffers);
                        (
                            [shape.texture_id(0), shape.texture_id(1)],
                            shape.index_buffer_range(),
                        )
                    }
                    DrawCommand::CachedShape(shape) => {
                        bind_instance_buffers(&mut render_pass, shape, buffers);
                        (
                            [shape.texture_id(0), shape.texture_id(1)],
                            shape.index_buffer_range(),
                        )
                    }
                };

                bind_shape_texture_layers(
                    &mut render_pass,
                    texture_ids,
                    pipelines.texture_manager,
                    pipelines.shape_texture_bind_group_layout_background,
                    pipelines.shape_texture_bind_group_layout_foreground,
                    pipelines.default_shape_texture_bind_groups,
                    pipelines.shape_texture_layout_epoch,
                    &mut bound_texture_state,
                );

                if let Some(sir) = shape_index_range {
                    render_pass.set_stencil_reference(this_stencil);
                    let start = sir.0 as u32;
                    let end = (sir.0 + sir.1) as u32;
                    render_pass.draw_indexed(start..end, 0, 0..1);

                    // Step 4: Decrement stencil to restore parent level.
                    render_pass.set_pipeline(pipelines.decrementing_pipeline);
                    render_pass.set_bind_group(
                        0,
                        pipelines.decrementing_bind_group,
                        &[],
                    );
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
                    // Instance buffers still bound from step 3.
                    render_pass.set_stencil_reference(this_stencil);
                    render_pass.draw_indexed(start..end, 0, 0..1);
                }
            }

            currently_set_pipeline
                .switch_to(crate::renderer::types::Pipeline::None);
            bound_texture_state.invalidate();
            is_first_segment = false;
            // Backdrop effects are on leaf nodes — skip both Pre and Post events.
            event_idx += 2;
        }
    }

    #[cfg(feature = "render_metrics")]
    pipeline_counts_out.accumulate(&currently_set_pipeline.counts);
}
