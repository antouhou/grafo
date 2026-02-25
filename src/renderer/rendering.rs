use super::*;
use crate::renderer::passes::{
    apply_effect_passes, flush_pending_leaf_batch, handle_decrement_pass, handle_increment_pass,
    handle_leaf_draw_pass, intersect_scissor, render_scene_behind_group, render_segments,
    try_batch_leaf, try_scissor_for_rect, EffectPassRunConfig, PendingLeafBatch,
};
use crate::renderer::traversal::{
    compute_node_depth, plan_traversal_in_place, subtree_has_backdrop_effects,
};

impl<'a> Renderer<'a> {
    pub(super) fn render_to_texture_view(
        &mut self,
        texture_view: &wgpu::TextureView,
        output_texture: Option<&wgpu::Texture>,
    ) {
        self.begin_frame_scratch();

        let mut traversal_scratch = std::mem::take(&mut self.scratch.traversal_scratch);
        let mut effect_results = std::mem::take(&mut self.scratch.effect_results);
        let mut effect_node_ids = std::mem::take(&mut self.scratch.effect_node_ids);
        let mut textures_to_recycle = std::mem::take(&mut self.scratch.textures_to_recycle);
        let mut effect_output_textures = std::mem::take(&mut self.scratch.effect_output_textures);
        let mut stencil_stack = std::mem::take(&mut self.scratch.stencil_stack);
        let mut skipped_stack = std::mem::take(&mut self.scratch.skipped_stack);
        let mut scissor_stack = std::mem::take(&mut self.scratch.scissor_stack);
        let mut backdrop_work_textures = std::mem::take(&mut self.scratch.backdrop_work_textures);

        let has_group_effects = !self.group_effects.is_empty();
        let has_backdrop_effects = !self.backdrop_effects.is_empty();

        if has_group_effects || has_backdrop_effects {
            self.ensure_composite_pipeline();
            self.ensure_effect_sampler();
        }
        if has_backdrop_effects {
            self.ensure_backdrop_snapshot_texture();
            self.ensure_stencil_only_pipeline();
            self.ensure_backdrop_color_pipeline();
        }

        // O1: Ensure depth/stencil texture exists (lazy init on first frame)
        if self.depth_stencil_view.is_none() {
            self.recreate_depth_stencil_texture();
        }

        #[cfg(feature = "render_metrics")]
        let mut frame_pipeline_counts = crate::renderer::metrics::PipelineSwitchCounts::default();

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Command Encoder"),
            });

        let pipelines = crate::renderer::types::Pipelines {
            and_pipeline: &self.and_pipeline,
            and_bind_group: &self.and_bind_group,
            decrementing_pipeline: &self.decrementing_pipeline,
            decrementing_bind_group: &self.decrementing_bind_group,
            leaf_draw_pipeline: &self.leaf_draw_pipeline,
            shape_texture_bind_group_layout_background: &self
                .shape_texture_bind_group_layout_background,
            shape_texture_bind_group_layout_foreground: &self
                .shape_texture_bind_group_layout_foreground,
            default_shape_texture_bind_groups: &self.default_shape_texture_bind_groups,
            shape_texture_layout_epoch: self.shape_texture_layout_epoch,
            texture_manager: &self.texture_manager,
        };

        let buffers = crate::renderer::types::Buffers {
            aggregated_vertex_buffer: self.aggregated_vertex_buffer.as_ref().unwrap(),
            aggregated_index_buffer: self.aggregated_index_buffer.as_ref().unwrap(),
            identity_instance_transform_buffer: self
                .identity_instance_transform_buffer
                .as_ref()
                .unwrap(),
            identity_instance_color_buffer: self.identity_instance_color_buffer.as_ref().unwrap(),
            identity_instance_metadata_buffer: self
                .identity_instance_metadata_buffer
                .as_ref()
                .unwrap(),
            aggregated_instance_transform_buffer: self
                .aggregated_instance_transform_buffer
                .as_ref(),
            aggregated_instance_color_buffer: self.aggregated_instance_color_buffer.as_ref(),
            aggregated_instance_metadata_buffer: self.aggregated_instance_metadata_buffer.as_ref(),
        };

        if has_group_effects {
            effect_node_ids.clear();
            for &node_id in self.group_effects.keys() {
                if self.draw_tree.get(node_id).is_some() {
                    let depth = compute_node_depth(&self.draw_tree, node_id);
                    effect_node_ids.push((node_id, depth));
                }
            }
            effect_node_ids.sort_by(|left, right| right.1.cmp(&left.1));

            let (width, height) = self.physical_size;
            let physical_size = self.physical_size;
            let scale_factor = self.scale_factor;

            for &(node_id, _depth) in &effect_node_ids {
                let effect_instance = match self.group_effects.get(&node_id) {
                    Some(instance) => instance,
                    None => continue,
                };
                let effect_id = effect_instance.effect_id;
                if !self.loaded_effects.contains_key(&effect_id) {
                    continue;
                }

                let subtree_texture = self.offscreen_texture_pool.acquire(
                    &self.device,
                    width,
                    height,
                    self.config.format,
                    self.msaa_sample_count,
                );

                let subtree_needs_backdrop_effects =
                    subtree_has_backdrop_effects(&self.draw_tree, &self.backdrop_effects, node_id);

                if subtree_needs_backdrop_effects {
                    let behind_texture = self.offscreen_texture_pool.acquire(
                        &self.device,
                        width,
                        height,
                        self.config.format,
                        self.msaa_sample_count,
                    );
                    let behind_depth = create_and_depth_texture(
                        &self.device,
                        (width, height),
                        self.msaa_sample_count,
                    );
                    let behind_depth_view =
                        behind_depth.create_view(&wgpu::TextureViewDescriptor::default());

                    let (behind_color_view, behind_resolve_target) =
                        if behind_texture.sample_count > 1 {
                            (
                                &behind_texture.color_view,
                                Some(behind_texture.resolve_view.as_ref().unwrap()
                                    as &wgpu::TextureView),
                            )
                        } else {
                            (&behind_texture.color_view as &wgpu::TextureView, None)
                        };

                    render_scene_behind_group(
                        &mut self.draw_tree,
                        &mut encoder,
                        &effect_results,
                        node_id,
                        behind_color_view,
                        behind_resolve_target,
                        &behind_depth_view,
                        self.composite_pipeline.as_ref(),
                        &pipelines,
                        &buffers,
                        &mut stencil_stack,
                        &mut skipped_stack,
                        self.scale_factor,
                        self.physical_size,
                    );

                    let behind_copy_source = if behind_texture.sample_count > 1 {
                        behind_texture.resolve_texture.as_ref().unwrap()
                    } else {
                        &behind_texture.color_texture
                    };

                    plan_traversal_in_place(
                        &mut self.draw_tree,
                        &effect_results,
                        Some(node_id),
                        &mut traversal_scratch,
                    );

                    let (subtree_color_view, subtree_resolve_target) =
                        if subtree_texture.sample_count > 1 {
                            (
                                &subtree_texture.color_view,
                                Some(subtree_texture.resolve_view.as_ref().unwrap()
                                    as &wgpu::TextureView),
                            )
                        } else {
                            (&subtree_texture.color_view, None)
                        };

                    let backdrop_ctx = crate::renderer::types::BackdropContext {
                        backdrop_effects: &self.backdrop_effects,
                        loaded_effects: &self.loaded_effects,
                        composite_pipeline: self.composite_pipeline.as_ref().unwrap(),
                        composite_bgl: self.composite_bgl.as_ref().unwrap(),
                        effect_sampler: self.effect_sampler.as_ref().unwrap(),
                        stencil_only_pipeline: self.stencil_only_pipeline.as_ref().unwrap(),
                        backdrop_color_pipeline: self.backdrop_color_pipeline.as_ref().unwrap(),
                        and_bind_group: &self.and_bind_group,
                        default_shape_texture_bind_groups: &self.default_shape_texture_bind_groups,
                        device: &self.device,
                        physical_size: self.physical_size,
                        scale_factor: self.scale_factor,
                        config_format: self.config.format,
                        backdrop_snapshot_texture: self.backdrop_snapshot_texture.as_ref().unwrap(),
                        backdrop_snapshot_view: self.backdrop_snapshot_view.as_ref().unwrap(),
                        texture_manager: &self.texture_manager,
                        shape_texture_bind_group_layout_background: &self
                            .shape_texture_bind_group_layout_background,
                        shape_texture_bind_group_layout_foreground: &self
                            .shape_texture_bind_group_layout_foreground,
                        shape_texture_layout_epoch: self.shape_texture_layout_epoch,
                    };

                    render_segments(
                        &mut self.draw_tree,
                        &mut encoder,
                        traversal_scratch.events(),
                        traversal_scratch.stencil_refs(),
                        traversal_scratch.parent_stencils(),
                        &effect_results,
                        subtree_color_view,
                        subtree_resolve_target,
                        &subtree_texture.depth_stencil_view,
                        behind_copy_source,
                        true,
                        &pipelines,
                        &buffers,
                        &backdrop_ctx,
                        &mut backdrop_work_textures,
                        &mut stencil_stack,
                    );

                    effect_output_textures.append(&mut backdrop_work_textures);
                    textures_to_recycle.push(behind_texture);
                } else {
                    let (view, resolve_target) = if subtree_texture.sample_count > 1 {
                        (
                            &subtree_texture.color_view,
                            subtree_texture.resolve_view.as_ref(),
                        )
                    } else {
                        (&subtree_texture.color_view, None)
                    };

                    let render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("effect_subtree_pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view,
                            resolve_target,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: &subtree_texture.depth_stencil_view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(1.0),
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(0),
                                store: wgpu::StoreOp::Store,
                            }),
                        }),
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });

                    stencil_stack.clear();
                    skipped_stack.clear();
                    scissor_stack.clear();
                    let subtree_viewport = (0u32, 0u32, width, height);
                    scissor_stack.push(subtree_viewport);
                    let current_pipeline = crate::renderer::types::PipelineTracker::new();
                    let bound_texture_state = crate::renderer::types::BoundTextureState::default();
                    let mut data = (
                        render_pass,
                        &mut stencil_stack,
                        current_pipeline,
                        &mut skipped_stack,
                        bound_texture_state,
                        &mut scissor_stack,
                    );

                    let effect_results_ref = &effect_results;
                    let composite_pipeline_ref = self.composite_pipeline.as_ref();

                    self.draw_tree.traverse_subtree_mut(
                        node_id,
                        |shape_id, draw_command, data| {
                            let (
                                render_pass,
                                stencil_stack,
                                currently_set_pipeline,
                                skipped_stack,
                                bound_texture_state,
                                scissor_stack,
                            ) = data;

                            if shape_id != node_id {
                                if let Some(result_bg) = effect_results_ref.get(&shape_id) {
                                    let parent_stencil = stencil_stack.last().copied().unwrap_or(0);
                                    if let Some(composite_pipeline) = composite_pipeline_ref {
                                        render_pass.set_pipeline(composite_pipeline);
                                        render_pass.set_bind_group(0, result_bg, &[]);
                                        render_pass.set_stencil_reference(parent_stencil);
                                        render_pass.draw(0..3, 0..1);
                                        currently_set_pipeline
                                            .switch_to(crate::renderer::types::Pipeline::None);
                                        bound_texture_state.invalidate();
                                    }
                                    skipped_stack.push(shape_id);
                                    return;
                                }
                            }

                            if !skipped_stack.is_empty() {
                                return;
                            }

                            // Scissor optimization for rect parents in effect subtrees.
                            if !draw_command.is_leaf() && draw_command.clips_children() {
                                if let Some(scissor_rect) =
                                    try_scissor_for_rect(draw_command, scale_factor, physical_size)
                                {
                                    let current_scissor =
                                        scissor_stack.last().copied().unwrap_or(subtree_viewport);
                                    let clipped = intersect_scissor(current_scissor, scissor_rect);
                                    scissor_stack.push(clipped);
                                    render_pass.set_scissor_rect(
                                        clipped.0, clipped.1, clipped.2, clipped.3,
                                    );
                                    let parent_stencil = stencil_stack.last().copied().unwrap_or(0);
                                    stencil_stack.push(parent_stencil);
                                    #[cfg(feature = "render_metrics")]
                                    currently_set_pipeline.record_scissor_clip();
                                    return;
                                }
                            }

                            match draw_command {
                                DrawCommand::Shape(shape) => {
                                    handle_increment_pass(
                                        render_pass,
                                        currently_set_pipeline,
                                        bound_texture_state,
                                        stencil_stack,
                                        shape,
                                        &pipelines,
                                        &buffers,
                                    );
                                }
                                DrawCommand::CachedShape(shape) => {
                                    handle_increment_pass(
                                        render_pass,
                                        currently_set_pipeline,
                                        bound_texture_state,
                                        stencil_stack,
                                        shape,
                                        &pipelines,
                                        &buffers,
                                    );
                                }
                            }
                        },
                        |shape_id, draw_command, data| {
                            let (
                                render_pass,
                                stencil_stack,
                                currently_set_pipeline,
                                skipped_stack,
                                bound_texture_state,
                                scissor_stack,
                            ) = data;

                            if skipped_stack.last().copied() == Some(shape_id) {
                                skipped_stack.pop();
                                return;
                            }

                            if !skipped_stack.is_empty() {
                                return;
                            }

                            // Scissor optimization: pop scissor for rect parents.
                            if !draw_command.is_leaf()
                                && draw_command.clips_children()
                                && try_scissor_for_rect(draw_command, scale_factor, physical_size)
                                    .is_some()
                            {
                                scissor_stack.pop();
                                let prev =
                                    scissor_stack.last().copied().unwrap_or(subtree_viewport);
                                render_pass.set_scissor_rect(prev.0, prev.1, prev.2, prev.3);
                                stencil_stack.pop();
                                return;
                            }

                            match draw_command {
                                DrawCommand::Shape(shape) => {
                                    handle_decrement_pass(
                                        render_pass,
                                        currently_set_pipeline,
                                        bound_texture_state,
                                        stencil_stack,
                                        shape,
                                        &pipelines,
                                        &buffers,
                                    );
                                }
                                DrawCommand::CachedShape(shape) => {
                                    handle_decrement_pass(
                                        render_pass,
                                        currently_set_pipeline,
                                        bound_texture_state,
                                        stencil_stack,
                                        shape,
                                        &pipelines,
                                        &buffers,
                                    );
                                }
                            }
                        },
                        &mut data,
                    );

                    #[cfg(feature = "render_metrics")]
                    {
                        let (_, _, ref pipeline_tracker, _, _, _) = data;
                        frame_pipeline_counts.accumulate(&pipeline_tracker.counts);
                    }
                }

                let source_view = if subtree_texture.sample_count > 1 {
                    subtree_texture.resolve_view.as_ref().unwrap()
                } else {
                    &subtree_texture.color_view
                };

                let loaded_effect = self.loaded_effects.get(&effect_id).unwrap();
                let effect_output = apply_effect_passes(
                    &self.device,
                    &mut encoder,
                    EffectPassRunConfig {
                        loaded_effect,
                        params_bind_group: effect_instance.params_bind_group.as_ref(),
                        source_view,
                        effect_sampler: self.effect_sampler.as_ref().unwrap(),
                        composite_bind_group_layout: self.composite_bgl.as_ref().unwrap(),
                        width,
                        height,
                        texture_format: self.config.format,
                        label_prefix: "group_effect",
                    },
                );

                let composite_bind_group =
                    effect_output.push_work_textures_into(&mut effect_output_textures);
                effect_results.insert(node_id, composite_bind_group);
                textures_to_recycle.push(subtree_texture);
            }
        }

        {
            let depth_texture_view = self.depth_stencil_view.as_ref().unwrap();

            if has_backdrop_effects {
                plan_traversal_in_place(
                    &mut self.draw_tree,
                    &effect_results,
                    None,
                    &mut traversal_scratch,
                );

                let (phase2_color_view, phase2_resolve_target) =
                    if let Some(msaa_view) = self.msaa_color_texture_view.as_ref() {
                        (
                            msaa_view as &wgpu::TextureView,
                            Some(texture_view as &wgpu::TextureView),
                        )
                    } else {
                        (texture_view as &wgpu::TextureView, None)
                    };

                let backdrop_ctx = crate::renderer::types::BackdropContext {
                    backdrop_effects: &self.backdrop_effects,
                    loaded_effects: &self.loaded_effects,
                    composite_pipeline: self.composite_pipeline.as_ref().unwrap(),
                    composite_bgl: self.composite_bgl.as_ref().unwrap(),
                    effect_sampler: self.effect_sampler.as_ref().unwrap(),
                    stencil_only_pipeline: self.stencil_only_pipeline.as_ref().unwrap(),
                    backdrop_color_pipeline: self.backdrop_color_pipeline.as_ref().unwrap(),
                    and_bind_group: &self.and_bind_group,
                    default_shape_texture_bind_groups: &self.default_shape_texture_bind_groups,
                    device: &self.device,
                    physical_size: self.physical_size,
                    scale_factor: self.scale_factor,
                    config_format: self.config.format,
                    backdrop_snapshot_texture: self.backdrop_snapshot_texture.as_ref().unwrap(),
                    backdrop_snapshot_view: self.backdrop_snapshot_view.as_ref().unwrap(),
                    texture_manager: &self.texture_manager,
                    shape_texture_bind_group_layout_background: &self
                        .shape_texture_bind_group_layout_background,
                    shape_texture_bind_group_layout_foreground: &self
                        .shape_texture_bind_group_layout_foreground,
                    shape_texture_layout_epoch: self.shape_texture_layout_epoch,
                };

                let src_texture =
                    output_texture.expect("output_texture required for backdrop effects");

                render_segments(
                    &mut self.draw_tree,
                    &mut encoder,
                    traversal_scratch.events(),
                    traversal_scratch.stencil_refs(),
                    traversal_scratch.parent_stencils(),
                    &effect_results,
                    phase2_color_view,
                    phase2_resolve_target,
                    depth_texture_view,
                    src_texture,
                    true,
                    &pipelines,
                    &buffers,
                    &backdrop_ctx,
                    &mut backdrop_work_textures,
                    &mut stencil_stack,
                );

                backdrop_work_textures.clear();
            } else {
                let render_pass = create_render_pass(
                    &mut encoder,
                    self.msaa_color_texture_view.as_ref(),
                    texture_view,
                    depth_texture_view,
                );

                stencil_stack.clear();
                skipped_stack.clear();
                // Initialize scissor stack with the full viewport.
                scissor_stack.clear();
                let viewport_scissor = (0u32, 0u32, self.physical_size.0, self.physical_size.1);
                scissor_stack.push(viewport_scissor);
                let current_pipeline = crate::renderer::types::PipelineTracker::new();
                let bound_texture_state = crate::renderer::types::BoundTextureState::default();
                let pending_leaf_batch = PendingLeafBatch::default();
                let scale_factor = self.scale_factor;
                let physical_size = self.physical_size;
                let mut data = (
                    render_pass,
                    &mut stencil_stack,
                    current_pipeline,
                    &mut skipped_stack,
                    bound_texture_state,
                    pending_leaf_batch,
                    &mut scissor_stack,
                );

                let effect_results_ref = &effect_results;
                let composite_pipeline_ref = self.composite_pipeline.as_ref();

                self.draw_tree.traverse_mut(
                    |shape_id, draw_command, data| {
                        let (
                            render_pass,
                            stencil_stack,
                            currently_set_pipeline,
                            skipped_stack,
                            bound_texture_state,
                            pending_batch,
                            scissor_stack,
                        ) = data;

                        if let Some(result_bind_group) = effect_results_ref.get(&shape_id) {
                            flush_pending_leaf_batch(
                                pending_batch,
                                render_pass,
                                currently_set_pipeline,
                                bound_texture_state,
                                &pipelines,
                                &buffers,
                            );
                            let parent_stencil = stencil_stack.last().copied().unwrap_or(0);
                            if let Some(composite_pipeline) = composite_pipeline_ref {
                                render_pass.set_pipeline(composite_pipeline);
                                render_pass.set_bind_group(0, result_bind_group, &[]);
                                render_pass.set_stencil_reference(parent_stencil);
                                render_pass.draw(0..3, 0..1);
                                currently_set_pipeline
                                    .switch_to(crate::renderer::types::Pipeline::None);
                                bound_texture_state.invalidate();
                            }
                            skipped_stack.push(shape_id);
                            return;
                        }

                        if !skipped_stack.is_empty() {
                            return;
                        }

                        // O3+O4: Leaf nodes — try to batch consecutive compatible leaves
                        // into a single multi-instance draw call.
                        if draw_command.is_leaf() {
                            let parent_stencil = stencil_stack.last().copied().unwrap_or(0);
                            let batched = match draw_command {
                                DrawCommand::Shape(shape) => {
                                    let result =
                                        try_batch_leaf(pending_batch, shape, parent_stencil);
                                    if result {
                                        *shape.stencil_ref_mut() = Some(parent_stencil);
                                    }
                                    result
                                }
                                DrawCommand::CachedShape(shape) => {
                                    let result =
                                        try_batch_leaf(pending_batch, shape, parent_stencil);
                                    if result {
                                        *shape.stencil_ref_mut() = Some(parent_stencil);
                                    }
                                    result
                                }
                            };
                            if !batched {
                                // Flush the current batch, then try again (starts new batch).
                                flush_pending_leaf_batch(
                                    pending_batch,
                                    render_pass,
                                    currently_set_pipeline,
                                    bound_texture_state,
                                    &pipelines,
                                    &buffers,
                                );
                                match draw_command {
                                    DrawCommand::Shape(shape) => {
                                        let parent_stencil =
                                            stencil_stack.last().copied().unwrap_or(0);
                                        if !try_batch_leaf(pending_batch, shape, parent_stencil) {
                                            // Unbatchable — fall back to single draw.
                                            handle_leaf_draw_pass(
                                                render_pass,
                                                currently_set_pipeline,
                                                bound_texture_state,
                                                stencil_stack,
                                                shape,
                                                &pipelines,
                                                &buffers,
                                            );
                                        } else {
                                            *shape.stencil_ref_mut() = Some(parent_stencil);
                                        }
                                    }
                                    DrawCommand::CachedShape(shape) => {
                                        let parent_stencil =
                                            stencil_stack.last().copied().unwrap_or(0);
                                        if !try_batch_leaf(pending_batch, shape, parent_stencil) {
                                            handle_leaf_draw_pass(
                                                render_pass,
                                                currently_set_pipeline,
                                                bound_texture_state,
                                                stencil_stack,
                                                shape,
                                                &pipelines,
                                                &buffers,
                                            );
                                        } else {
                                            *shape.stencil_ref_mut() = Some(parent_stencil);
                                        }
                                    }
                                }
                            }
                        } else {
                            // Non-leaf: flush any pending leaf batch before the increment pass.
                            flush_pending_leaf_batch(
                                pending_batch,
                                render_pass,
                                currently_set_pipeline,
                                bound_texture_state,
                                &pipelines,
                                &buffers,
                            );

                            // S5: If this parent doesn't clip children, draw it like
                            // a leaf and push the *same* parent stencil so children
                            // inherit the grandparent's clip region.
                            if !draw_command.clips_children() {
                                let parent_stencil = stencil_stack.last().copied().unwrap_or(0);
                                match draw_command {
                                    DrawCommand::Shape(shape) => {
                                        *shape.stencil_ref_mut() = Some(parent_stencil);
                                        handle_leaf_draw_pass(
                                            render_pass,
                                            currently_set_pipeline,
                                            bound_texture_state,
                                            stencil_stack,
                                            shape,
                                            &pipelines,
                                            &buffers,
                                        );
                                    }
                                    DrawCommand::CachedShape(shape) => {
                                        *shape.stencil_ref_mut() = Some(parent_stencil);
                                        handle_leaf_draw_pass(
                                            render_pass,
                                            currently_set_pipeline,
                                            bound_texture_state,
                                            stencil_stack,
                                            shape,
                                            &pipelines,
                                            &buffers,
                                        );
                                    }
                                }
                                // Push same stencil so children use grandparent's clip.
                                stencil_stack.push(parent_stencil);
                            } else if let Some(scissor_rect) =
                                try_scissor_for_rect(draw_command, scale_factor, physical_size)
                            {
                                // Scissor optimization: rect parent with axis-aligned transform.
                                // Use hardware scissor instead of stencil increment/decrement.
                                let current_scissor =
                                    scissor_stack.last().copied().unwrap_or(viewport_scissor);
                                let clipped = intersect_scissor(current_scissor, scissor_rect);
                                scissor_stack.push(clipped);
                                render_pass
                                    .set_scissor_rect(clipped.0, clipped.1, clipped.2, clipped.3);
                                #[cfg(feature = "render_metrics")]
                                currently_set_pipeline.record_scissor_clip();

                                // Draw the rect itself as a visible shape (like a leaf).
                                let parent_stencil = stencil_stack.last().copied().unwrap_or(0);
                                match draw_command {
                                    DrawCommand::Shape(shape) => {
                                        *shape.stencil_ref_mut() = Some(parent_stencil);
                                        handle_leaf_draw_pass(
                                            render_pass,
                                            currently_set_pipeline,
                                            bound_texture_state,
                                            stencil_stack,
                                            shape,
                                            &pipelines,
                                            &buffers,
                                        );
                                    }
                                    DrawCommand::CachedShape(shape) => {
                                        *shape.stencil_ref_mut() = Some(parent_stencil);
                                        handle_leaf_draw_pass(
                                            render_pass,
                                            currently_set_pipeline,
                                            bound_texture_state,
                                            stencil_stack,
                                            shape,
                                            &pipelines,
                                            &buffers,
                                        );
                                    }
                                }
                                // Push a dummy stencil entry so the stack stays balanced
                                // (post-visit will pop it). Children use the parent's stencil ref.
                                stencil_stack.push(parent_stencil);
                            } else {
                                match draw_command {
                                    DrawCommand::Shape(shape) => {
                                        handle_increment_pass(
                                            render_pass,
                                            currently_set_pipeline,
                                            bound_texture_state,
                                            stencil_stack,
                                            shape,
                                            &pipelines,
                                            &buffers,
                                        );
                                    }
                                    DrawCommand::CachedShape(shape) => {
                                        handle_increment_pass(
                                            render_pass,
                                            currently_set_pipeline,
                                            bound_texture_state,
                                            stencil_stack,
                                            shape,
                                            &pipelines,
                                            &buffers,
                                        );
                                    }
                                }
                            }
                        }
                    },
                    |shape_id, draw_command, data| {
                        let (
                            render_pass,
                            stencil_stack,
                            currently_set_pipeline,
                            skipped_stack,
                            bound_texture_state,
                            pending_batch,
                            scissor_stack,
                        ) = data;

                        if skipped_stack.last().copied() == Some(shape_id) {
                            skipped_stack.pop();
                            return;
                        }

                        if !skipped_stack.is_empty() {
                            return;
                        }

                        // O3: Leaf nodes already drew with Keep — skip decrement entirely
                        if draw_command.is_leaf() {
                            return;
                        }

                        // S5: Non-clipping parents pushed a stencil entry but never
                        // incremented — just pop and skip the decrement pass entirely.
                        if !draw_command.clips_children() {
                            stencil_stack.pop();
                            return;
                        }

                        // Scissor optimization: if pre-visit used scissor for this node,
                        // pop the scissor stack and restore the previous scissor rect.
                        // This is deterministic: the same conditions evaluated in pre-visit
                        // will produce the same result here.
                        if try_scissor_for_rect(draw_command, scale_factor, physical_size).is_some()
                        {
                            // Flush any pending leaf draws BEFORE changing the scissor,
                            // so they are clipped to this parent's rect, not the grandparent's.
                            flush_pending_leaf_batch(
                                pending_batch,
                                render_pass,
                                currently_set_pipeline,
                                bound_texture_state,
                                &pipelines,
                                &buffers,
                            );
                            scissor_stack.pop();
                            let prev = scissor_stack.last().copied().unwrap_or(viewport_scissor);
                            render_pass.set_scissor_rect(prev.0, prev.1, prev.2, prev.3);
                            stencil_stack.pop();
                            return;
                        }

                        // Flush any pending leaf batch before decrement pass.
                        flush_pending_leaf_batch(
                            pending_batch,
                            render_pass,
                            currently_set_pipeline,
                            bound_texture_state,
                            &pipelines,
                            &buffers,
                        );

                        match draw_command {
                            DrawCommand::Shape(shape) => {
                                handle_decrement_pass(
                                    render_pass,
                                    currently_set_pipeline,
                                    bound_texture_state,
                                    stencil_stack,
                                    shape,
                                    &pipelines,
                                    &buffers,
                                );
                            }
                            DrawCommand::CachedShape(shape) => {
                                handle_decrement_pass(
                                    render_pass,
                                    currently_set_pipeline,
                                    bound_texture_state,
                                    stencil_stack,
                                    shape,
                                    &pipelines,
                                    &buffers,
                                );
                            }
                        }
                    },
                    &mut data,
                );
                // Flush any leftover batch after the traversal completes.
                {
                    let (
                        render_pass,
                        _,
                        currently_set_pipeline,
                        _,
                        bound_texture_state,
                        pending_batch,
                        _,
                    ) = &mut data;
                    flush_pending_leaf_batch(
                        pending_batch,
                        render_pass,
                        currently_set_pipeline,
                        bound_texture_state,
                        &pipelines,
                        &buffers,
                    );
                }
                #[cfg(feature = "render_metrics")]
                {
                    let (_, _, ref pipeline_tracker, _, _, _, _) = data;
                    frame_pipeline_counts.accumulate(&pipeline_tracker.counts);
                }
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));

        self.offscreen_texture_pool
            .recycle(&mut textures_to_recycle);
        effect_output_textures.clear();

        self.draw_tree
            .iter_mut()
            .for_each(|(_, draw_command)| draw_command.clear_frame_state());

        self.scratch.traversal_scratch = traversal_scratch;
        self.scratch.effect_results = effect_results;
        self.scratch.effect_node_ids = effect_node_ids;
        self.scratch.textures_to_recycle = textures_to_recycle;
        self.scratch.effect_output_textures = effect_output_textures;
        self.scratch.stencil_stack = stencil_stack;
        self.scratch.skipped_stack = skipped_stack;
        self.scratch.scissor_stack = scissor_stack;
        self.scratch.backdrop_work_textures = backdrop_work_textures;

        #[cfg(feature = "render_metrics")]
        {
            self.last_pipeline_switch_counts = frame_pipeline_counts;
        }
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        #[cfg(feature = "render_metrics")]
        let frame_render_loop_started_at = std::time::Instant::now();
        self.prepare_render();

        #[cfg(feature = "render_metrics")]
        let after_prepare = std::time::Instant::now();

        let output = self.surface.get_current_texture()?;
        let output_texture_view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        self.render_to_texture_view(&output_texture_view, Some(&output.texture));

        #[cfg(feature = "render_metrics")]
        let after_submit = std::time::Instant::now();

        output.present();
        #[cfg(feature = "render_metrics")]
        {
            let after_present = std::time::Instant::now();
            // Force GPU completion to measure actual GPU execution time.
            let _ = self.device.poll(wgpu::MaintainBase::Wait);
            let after_gpu_wait = std::time::Instant::now();

            let prepare_dur = after_prepare.saturating_duration_since(frame_render_loop_started_at);
            let encode_submit_dur = after_submit.saturating_duration_since(after_prepare);
            let present_dur = after_present.saturating_duration_since(after_submit);
            let gpu_wait_dur = after_gpu_wait.saturating_duration_since(after_present);
            let total_dur = after_gpu_wait.saturating_duration_since(frame_render_loop_started_at);
            self.last_phase_timings = crate::renderer::metrics::PhaseTimings {
                prepare: prepare_dur,
                encode_and_submit: encode_submit_dur,
                present_or_readback: present_dur,
                gpu_wait: gpu_wait_dur,
                total: total_dur,
            };
            self.render_loop_metrics_tracker
                .record_presented_frame(frame_render_loop_started_at, after_gpu_wait);
        }
        Ok(())
    }
}
