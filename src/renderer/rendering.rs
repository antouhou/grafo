use super::*;
use crate::renderer::passes::{apply_effect_passes, render_segments, EffectPassRunConfig};
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
        let skipped_stack = std::mem::take(&mut self.scratch.skipped_stack);
        let mut scissor_stack = std::mem::take(&mut self.scratch.scissor_stack);
        let mut clip_kind_stack = std::mem::take(&mut self.scratch.clip_kind_stack);
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

                // --- Behind-group rendering (when subtree has backdrop effects) ---
                let behind_texture = if subtree_needs_backdrop_effects {
                    let behind_tex = self.offscreen_texture_pool.acquire(
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

                    let (behind_color_view, behind_resolve_target) = if behind_tex.sample_count > 1
                    {
                        (
                            &behind_tex.color_view,
                            Some(behind_tex.resolve_view.as_ref().unwrap() as &wgpu::TextureView),
                        )
                    } else {
                        (&behind_tex.color_view as &wgpu::TextureView, None)
                    };

                    // Use plan_traversal (full tree, excluding this subtree)
                    // + render_segments to render the scene behind the group.
                    plan_traversal_in_place(
                        &mut self.draw_tree,
                        &effect_results,
                        None,
                        Some(node_id),
                        &mut traversal_scratch,
                    );
                    render_segments(
                        &mut self.draw_tree,
                        &mut encoder,
                        traversal_scratch.events(),
                        &effect_results,
                        behind_color_view,
                        behind_resolve_target,
                        &behind_depth_view,
                        None,
                        true,
                        &pipelines,
                        &buffers,
                        self.composite_pipeline.as_ref(),
                        None,
                        &mut backdrop_work_textures,
                        &mut stencil_stack,
                        &mut scissor_stack,
                        &mut clip_kind_stack,
                        self.scale_factor,
                        self.physical_size,
                        #[cfg(feature = "render_metrics")]
                        &mut frame_pipeline_counts,
                    );
                    Some(behind_tex)
                } else {
                    None
                };

                // --- Subtree rendering (unified: always use plan_traversal + render_segments) ---
                plan_traversal_in_place(
                    &mut self.draw_tree,
                    &effect_results,
                    Some(node_id),
                    None,
                    &mut traversal_scratch,
                );

                let (subtree_color_view, subtree_resolve_target) = if subtree_texture.sample_count
                    > 1
                {
                    (
                        &subtree_texture.color_view,
                        Some(subtree_texture.resolve_view.as_ref().unwrap() as &wgpu::TextureView),
                    )
                } else {
                    (&subtree_texture.color_view, None)
                };

                let backdrop_ctx_opt = if subtree_needs_backdrop_effects {
                    Some(crate::renderer::types::BackdropContext {
                        backdrop_effects: &self.backdrop_effects,
                        loaded_effects: &self.loaded_effects,
                        composite_bgl: self.composite_bgl.as_ref().unwrap(),
                        effect_sampler: self.effect_sampler.as_ref().unwrap(),
                        stencil_only_pipeline: self.stencil_only_pipeline.as_ref().unwrap(),
                        backdrop_color_pipeline: self.backdrop_color_pipeline.as_ref().unwrap(),
                        device: &self.device,
                        config_format: self.config.format,
                        backdrop_snapshot_texture: self.backdrop_snapshot_texture.as_ref().unwrap(),
                        backdrop_snapshot_view: self.backdrop_snapshot_view.as_ref().unwrap(),
                    })
                } else {
                    None
                };

                let copy_source = behind_texture.as_ref().map(|tex| {
                    if tex.sample_count > 1 {
                        tex.resolve_texture.as_ref().unwrap() as &wgpu::Texture
                    } else {
                        &tex.color_texture as &wgpu::Texture
                    }
                });

                render_segments(
                    &mut self.draw_tree,
                    &mut encoder,
                    traversal_scratch.events(),
                    &effect_results,
                    subtree_color_view,
                    subtree_resolve_target,
                    &subtree_texture.depth_stencil_view,
                    copy_source,
                    true,
                    &pipelines,
                    &buffers,
                    self.composite_pipeline.as_ref(),
                    backdrop_ctx_opt.as_ref(),
                    &mut backdrop_work_textures,
                    &mut stencil_stack,
                    &mut scissor_stack,
                    &mut clip_kind_stack,
                    scale_factor,
                    physical_size,
                    #[cfg(feature = "render_metrics")]
                    &mut frame_pipeline_counts,
                );

                effect_output_textures.append(&mut backdrop_work_textures);
                if let Some(behind_tex) = behind_texture {
                    textures_to_recycle.push(behind_tex);
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

            // Unified main-scene rendering: always plan_traversal + render_segments.
            plan_traversal_in_place(
                &mut self.draw_tree,
                &effect_results,
                None,
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

            let backdrop_ctx_opt = if has_backdrop_effects {
                Some(crate::renderer::types::BackdropContext {
                    backdrop_effects: &self.backdrop_effects,
                    loaded_effects: &self.loaded_effects,
                    composite_bgl: self.composite_bgl.as_ref().unwrap(),
                    effect_sampler: self.effect_sampler.as_ref().unwrap(),
                    stencil_only_pipeline: self.stencil_only_pipeline.as_ref().unwrap(),
                    backdrop_color_pipeline: self.backdrop_color_pipeline.as_ref().unwrap(),
                    device: &self.device,
                    config_format: self.config.format,
                    backdrop_snapshot_texture: self.backdrop_snapshot_texture.as_ref().unwrap(),
                    backdrop_snapshot_view: self.backdrop_snapshot_view.as_ref().unwrap(),
                })
            } else {
                None
            };

            let copy_src = if has_backdrop_effects {
                Some(output_texture.expect("output_texture required for backdrop effects"))
            } else {
                None
            };

            render_segments(
                &mut self.draw_tree,
                &mut encoder,
                traversal_scratch.events(),
                &effect_results,
                phase2_color_view,
                phase2_resolve_target,
                depth_texture_view,
                copy_src,
                true,
                &pipelines,
                &buffers,
                self.composite_pipeline.as_ref(),
                backdrop_ctx_opt.as_ref(),
                &mut backdrop_work_textures,
                &mut stencil_stack,
                &mut scissor_stack,
                &mut clip_kind_stack,
                self.scale_factor,
                self.physical_size,
                #[cfg(feature = "render_metrics")]
                &mut frame_pipeline_counts,
            );

            backdrop_work_textures.clear();
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
        self.scratch.clip_kind_stack = clip_kind_stack;
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
