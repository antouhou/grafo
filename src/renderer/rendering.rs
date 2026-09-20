use super::*;
#[cfg(feature = "render_metrics")]
use crate::renderer::metrics::{PhaseTimings, PipelineSwitchCounts, ShapeEffectCacheMetrics};
use crate::renderer::passes::{
    apply_effect_passes, render_segments, EffectPassRunConfig, SegmentRenderTarget,
};
use crate::renderer::traversal::{
    compute_node_depth, plan_traversal_in_place, subtree_has_backdrop_effects,
};
use crate::renderer::types::RenderError;

impl<'a> Renderer<'a> {
    pub(super) fn render_to_texture_view(
        &mut self,
        texture_view: &wgpu::TextureView,
        output_texture: Option<&wgpu::Texture>,
    ) {
        let render_to_texture_view_started_at = std::time::Instant::now();

        if self.state.draw_tree.is_empty() {
            self.state.scratch.shape_effect_leaves.clear();
            let _collected_shape_effect_results = self.state.shape_effect_cache.end_frame();
            let _collected_shape_effect_masks = self.state.shape_effect_mask_cache.end_frame();
            #[cfg(feature = "render_metrics")]
            {
                self.state.pipeline_switch_counts = PipelineSwitchCounts::default();
                self.state.shape_effect_cache_metrics = ShapeEffectCacheMetrics {
                    collected_results: _collected_shape_effect_results as u64,
                    collected_masks: _collected_shape_effect_masks as u64,
                    ..Default::default()
                };
            }
            self.state.shape_resources.tessellation_cache.end_frame();
            self.last_render_to_texture_view_cpu_time = render_to_texture_view_started_at.elapsed();
            return;
        }

        let mut traversal_scratch = std::mem::take(&mut self.state.scratch.traversal_scratch);
        let mut effect_results = std::mem::take(&mut self.state.scratch.effect_results);
        let mut effect_node_ids = std::mem::take(&mut self.state.scratch.effect_node_ids);
        let mut textures_to_recycle = std::mem::take(&mut self.state.scratch.textures_to_recycle);
        let mut effect_output_textures =
            std::mem::take(&mut self.state.scratch.effect_output_textures);

        let has_group_effects = !self.state.group_effects.is_empty();
        let has_backdrop_effects = !self.state.backdrop_effects.is_empty();
        let has_shape_effects = !self.state.shape_effects.is_empty();

        if has_group_effects || has_backdrop_effects {
            self.ensure_composite_pipeline();
        }
        if has_group_effects || has_backdrop_effects || has_shape_effects {
            self.ensure_effect_sampler();
        }
        if has_backdrop_effects {
            self.ensure_texture_blit_pipeline();
            self.ensure_backdrop_layer_composite_pipeline();
            self.ensure_stencil_only_pipeline();
            self.ensure_backdrop_color_pipeline();
            self.ensure_backdrop_color_gradient_pipeline();
        }

        if self.depth_stencil_view.is_none() {
            self.recreate_depth_stencil_texture();
        }

        #[cfg(feature = "render_metrics")]
        {
            self.state.pipeline_switch_counts = PipelineSwitchCounts::default();
            self.state.shape_effect_cache_metrics = ShapeEffectCacheMetrics::default();
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Command Encoder"),
            });

        if has_shape_effects {
            self.resolve_shape_effects(&mut encoder, &mut textures_to_recycle);
        }

        let pipeline_resources = &self.pipeline_resources;
        let backdrop_context = if has_backdrop_effects {
            let backdrop_composite = pipeline_resources
                .backdrop_layer_composite_resources
                .as_ref()
                .unwrap();
            Some(types::BackdropContext {
                loaded_effects: &self.loaded_effects,
                effect_sampler: pipeline_resources.effect_sampler.as_ref().unwrap(),
                gradient_ramp_sampler: &pipeline_resources.shapes.gradient_ramp_sampler,
                texture_blit_pipeline: pipeline_resources.texture_blit_pipeline.as_ref().unwrap(),
                backdrop_layer_composite_pipeline: &backdrop_composite.pipeline,
                backdrop_layer_composite_bind_group_layout: &backdrop_composite.bind_group_layout,
                stencil_only_pipeline: pipeline_resources.stencil_only_pipeline.as_ref().unwrap(),
                backdrop_color_pipeline: pipeline_resources
                    .backdrop_color_pipeline
                    .as_ref()
                    .unwrap(),
                backdrop_color_gradient_pipeline: pipeline_resources
                    .backdrop_color_gradient_pipeline
                    .as_ref()
                    .unwrap(),
                device: &self.device,
                queue: &self.queue,
                config_format: self.config.format,
                max_texture_dimension_2d: self.device.limits().max_texture_dimension_2d,
                backdrop_texture_bind_group_layout: &pipeline_resources
                    .shapes
                    .backdrop_texture_bind_group_layout,
                default_backdrop_texture_bind_group: &pipeline_resources
                    .shapes
                    .default_backdrop_texture_bind_group,
                backdrop_gradient_bind_group_layout: &pipeline_resources
                    .shapes
                    .backdrop_gradient_bind_group_layout,
            })
        } else {
            None
        };

        let state = &mut self.state;

        if has_group_effects {
            effect_node_ids.clear();
            for &node_id in state.group_effects.keys() {
                if state.draw_tree.get(node_id).is_some() {
                    let depth = compute_node_depth(&state.draw_tree, node_id);
                    effect_node_ids.push((node_id, depth));
                }
            }
            effect_node_ids.sort_by_key(|right| std::cmp::Reverse(right.1));

            let (width, height) = state.physical_size;

            for &(node_id, _depth) in &effect_node_ids {
                let effect_instance = match state.group_effects.get(&node_id) {
                    Some(instance) => instance,
                    None => continue,
                };
                let effect_id = effect_instance.effect_id;
                if !self.loaded_effects.contains_key(&effect_id) {
                    continue;
                }

                let subtree_texture = state.texture_pool.acquire_with_depth(
                    &self.device,
                    width,
                    height,
                    self.config.format,
                    self.msaa_sample_count,
                );

                let subtree_needs_backdrop_effects = subtree_has_backdrop_effects(
                    &state.draw_tree,
                    &state.backdrop_effects,
                    node_id,
                );

                // Backdrops inside the group need the scene painted before the group.
                let behind_texture = if subtree_needs_backdrop_effects {
                    let behind_tex = state.texture_pool.acquire_color_only(
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

                    plan_traversal_in_place(
                        &mut state.draw_tree,
                        &effect_results,
                        &state.scratch.shape_effect_leaves,
                        None,
                        Some(node_id),
                        &mut traversal_scratch,
                    );
                    render_segments(
                        &mut encoder,
                        traversal_scratch.events(),
                        &effect_results,
                        SegmentRenderTarget {
                            color_view: behind_color_view,
                            color_resolve_target: behind_resolve_target,
                            depth_stencil_view: &behind_depth_view,
                            backdrop_source: None,
                            backdrop_context: None,
                        },
                        pipeline_resources,
                        state,
                    );
                    Some(behind_tex)
                } else {
                    None
                };

                plan_traversal_in_place(
                    &mut state.draw_tree,
                    &effect_results,
                    &state.scratch.shape_effect_leaves,
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

                let backdrop_source = behind_texture.as_ref().map(|texture| {
                    let base_texture = if texture.sample_count > 1 {
                        texture.resolve_texture.as_ref().unwrap()
                    } else {
                        &texture.color_texture
                    };
                    let foreground_view = if subtree_texture.sample_count > 1 {
                        subtree_texture.resolve_view.as_ref().unwrap()
                    } else {
                        &subtree_texture.color_view
                    };
                    types::BackdropSource::Layered {
                        base_texture,
                        foreground_view,
                    }
                });

                render_segments(
                    &mut encoder,
                    traversal_scratch.events(),
                    &effect_results,
                    SegmentRenderTarget {
                        color_view: subtree_color_view,
                        color_resolve_target: subtree_resolve_target,
                        depth_stencil_view: subtree_texture.depth_stencil_view.as_ref().expect(
                            "subtree render targets must include a depth/stencil attachment",
                        ),
                        backdrop_source,
                        backdrop_context: backdrop_context
                            .as_ref()
                            .filter(|_| subtree_needs_backdrop_effects),
                    },
                    pipeline_resources,
                    state,
                );

                effect_output_textures.append(&mut state.scratch.backdrop_work_textures);
                if let Some(behind_tex) = behind_texture {
                    textures_to_recycle.push(behind_tex);
                }

                let source_view = if subtree_texture.sample_count > 1 {
                    subtree_texture.resolve_view.as_ref().unwrap()
                } else {
                    &subtree_texture.color_view
                };

                let loaded_effect = self.loaded_effects.get(&effect_id).unwrap();
                let effect_instance = state
                    .group_effects
                    .get(&node_id)
                    .expect("group effect remains attached during rendering");
                let effect_output = apply_effect_passes(
                    &self.device,
                    &mut encoder,
                    &mut state.texture_pool,
                    EffectPassRunConfig {
                        loaded_effect,
                        params_bind_group: effect_instance
                            .parameter_resources
                            .as_ref()
                            .map(|resources| &resources.bind_group),
                        source_view,
                        effect_sampler: pipeline_resources.effect_sampler.as_ref().unwrap(),
                        composite_bind_group_layout: &pipeline_resources
                            .composite_resources
                            .as_ref()
                            .unwrap()
                            .bind_group_layout,
                        create_composite_bind_group: true,
                        width,
                        height,
                        texture_format: self.config.format,
                        label: "group_effect",
                    },
                );

                let composite_bind_group = effect_output
                    .push_work_textures_into(&mut effect_output_textures)
                    .expect("group effects must create a composite bind group");
                effect_results.insert(node_id, composite_bind_group);
                textures_to_recycle.push(subtree_texture);
            }
        }

        {
            let depth_texture_view = self.depth_stencil_view.as_ref().unwrap();

            plan_traversal_in_place(
                &mut state.draw_tree,
                &effect_results,
                &state.scratch.shape_effect_leaves,
                None,
                None,
                &mut traversal_scratch,
            );

            let (output_color_view, output_resolve_target) =
                if let Some(msaa_view) = self.msaa_color_texture_view.as_ref() {
                    (
                        msaa_view as &wgpu::TextureView,
                        Some(texture_view as &wgpu::TextureView),
                    )
                } else {
                    (texture_view as &wgpu::TextureView, None)
                };

            let backdrop_source = if has_backdrop_effects {
                Some(types::BackdropSource::Flattened {
                    texture: output_texture.expect("output_texture required for backdrop effects"),
                })
            } else {
                None
            };

            render_segments(
                &mut encoder,
                traversal_scratch.events(),
                &effect_results,
                SegmentRenderTarget {
                    color_view: output_color_view,
                    color_resolve_target: output_resolve_target,
                    depth_stencil_view: depth_texture_view,
                    backdrop_source,
                    backdrop_context: backdrop_context.as_ref(),
                },
                pipeline_resources,
                state,
            );
        }

        self.queue.submit(std::iter::once(encoder.finish()));

        self.last_render_to_texture_view_cpu_time = render_to_texture_view_started_at.elapsed();

        effect_output_textures.append(&mut state.scratch.backdrop_work_textures);
        textures_to_recycle.append(&mut effect_output_textures);
        state.texture_pool.recycle(&mut textures_to_recycle);

        state
            .draw_tree
            .iter_mut()
            .for_each(|(_node_id, draw_command)| {
                draw_command.clear_frame_state();
            });

        state.scratch.shape_effect_leaves.clear();

        state.scratch.traversal_scratch = traversal_scratch;
        state.scratch.effect_results = effect_results;
        state.scratch.effect_node_ids = effect_node_ids;
        state.scratch.textures_to_recycle = textures_to_recycle;
        state.scratch.effect_output_textures = effect_output_textures;
        let _collected_shape_effect_results = state.shape_effect_cache.end_frame();
        let _collected_shape_effect_masks = state.shape_effect_mask_cache.end_frame();
        state.shape_resources.tessellation_cache.end_frame();

        #[cfg(feature = "render_metrics")]
        {
            state.shape_effect_cache_metrics.collected_results =
                _collected_shape_effect_results as u64;
            state.shape_effect_cache_metrics.collected_masks = _collected_shape_effect_masks as u64;
        }
    }

    /// Returns an error if geometry preparation or surface acquisition fails.
    pub fn render(&mut self) -> Result<(), RenderError> {
        #[cfg(feature = "render_metrics")]
        let frame_render_loop_started_at = std::time::Instant::now();
        self.prepare_render()?;

        #[cfg(feature = "render_metrics")]
        let after_prepare = std::time::Instant::now();

        let surface = self
            .surface
            .as_ref()
            .expect("Cannot call render() on a headless renderer; use render_to_buffer()");
        let output = surface.get_current_texture()?;
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
            // Measure the remaining wait for GPU work after presentation.
            let _ = self.device.poll(wgpu::MaintainBase::Wait);
            let after_gpu_wait = std::time::Instant::now();

            let prepare_dur = after_prepare.saturating_duration_since(frame_render_loop_started_at);
            let encode_submit_dur = after_submit.saturating_duration_since(after_prepare);
            let present_dur = after_present.saturating_duration_since(after_submit);
            let gpu_wait_dur = after_gpu_wait.saturating_duration_since(after_present);
            let total_dur = after_gpu_wait.saturating_duration_since(frame_render_loop_started_at);
            self.last_phase_timings = PhaseTimings {
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
