use super::*;
#[cfg(feature = "render_metrics")]
use crate::renderer::metrics::{PhaseTimings, PipelineSwitchCounts, ShapeEffectCacheMetrics};
use crate::renderer::passes::{apply_effect_passes, render_segments, EffectPassRunConfig};
use crate::renderer::traversal::{
    compute_node_depth, plan_traversal_in_place, subtree_has_backdrop_effects,
};

impl<'a> Renderer<'a> {
    /// Waits for this renderer's known submission when a caller needs CPU-side completion.
    /// Queue-ordered GPU resource reuse does not require this wait.
    pub fn wait_for_submitted_work(&mut self) -> Result<(), wgpu::PollError> {
        if let Some(submission) = self.submitted_work.take() {
            self.device
                .poll(wgpu::PollType::WaitForSubmissionIndex(submission))?;
        }
        Ok(())
    }

    pub(super) fn render_to_texture_view(
        &mut self,
        texture_view: &wgpu::TextureView,
        output_texture: Option<&wgpu::Texture>,
        deadline: Option<std::time::Instant>,
    ) -> bool {
        let render_to_texture_view_started_at = std::time::Instant::now();

        // Nothing to render when the draw queue is empty.
        if self.draw_tree.is_empty() {
            self.scratch_mut().shape_effect_leaves.clear();
            let _collected_shape_effect_results = self.shape_effect_cache.end_frame();
            let _collected_shape_effect_masks = self.shape_effect_mask_cache.end_frame();
            #[cfg(feature = "render_metrics")]
            {
                self.last_shape_effect_cache_metrics = ShapeEffectCacheMetrics {
                    collected_results: _collected_shape_effect_results as u64,
                    collected_masks: _collected_shape_effect_masks as u64,
                    ..Default::default()
                };
            }
            self.buffers_pool_manager.tessellation_cache.end_frame();
            self.last_render_to_texture_view_cpu_time = render_to_texture_view_started_at.elapsed();
            return true;
        }

        let RendererScratch {
            mut traversal_scratch,
            mut effect_results,
            mut shape_effect_leaves,
            mut effect_node_ids,
            mut textures_to_recycle,
            mut effect_output_textures,
            mut stencil_stack,
            skipped_stack,
            mut scissor_stack,
            mut clip_kind_stack,
            mut backdrop_work_textures,
            readback_bytes,
        } = self
            .scratch
            .take()
            .expect("rendering owns the reusable scratch storage");

        let has_group_effects = !self.group_effects.is_empty();
        let has_backdrop_effects = !self.backdrop_effects.is_empty();
        let has_shape_effects = !self.shape_effects.is_empty();

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

        // O1: Ensure depth/stencil texture exists (lazy init on first frame)
        if self.depth_stencil_view.is_none() {
            self.recreate_depth_stencil_texture();
        }

        #[cfg(feature = "render_metrics")]
        let mut frame_pipeline_counts = PipelineSwitchCounts::default();
        #[cfg(feature = "render_metrics")]
        let mut shape_effect_cache_metrics = ShapeEffectCacheMetrics::default();

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Command Encoder"),
            });

        self.texture_manager.encode_uploads(&mut encoder);

        if has_shape_effects {
            self.resolve_shape_effects(
                &mut encoder,
                &mut shape_effect_leaves,
                &mut textures_to_recycle,
                #[cfg(feature = "render_metrics")]
                &mut shape_effect_cache_metrics,
            );
        }

        let pipelines = types::Pipelines {
            and_pipeline: &self.and_pipeline,
            and_gradient_pipeline: &self.and_gradient_pipeline,
            and_bind_group: &self.and_bind_group,
            decrementing_pipeline: &self.decrementing_pipeline,
            decrementing_bind_group: &self.decrementing_bind_group,
            leaf_draw_pipeline: &self.leaf_draw_pipeline,
            leaf_draw_gradient_pipeline: &self.leaf_draw_gradient_pipeline,
            shape_texture_bind_group_layout_background: &self
                .shape_texture_bind_group_layout_background,
            shape_texture_bind_group_layout_foreground: &self
                .shape_texture_bind_group_layout_foreground,
            default_shape_texture_bind_groups: &self.default_shape_texture_bind_groups,
            texture_manager: &self.texture_manager,
        };

        let buffers = types::Buffers {
            aggregated_vertex_buffer: self.aggregated_vertex_buffer.as_ref(),
            aggregated_index_buffer: self.aggregated_index_buffer.as_ref(),
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
            effect_node_ids.sort_by_key(|right| std::cmp::Reverse(right.1));

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

                let subtree_texture = self.offscreen_texture_pool.acquire_with_depth(
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
                    let behind_tex = self.offscreen_texture_pool.acquire_color_only(
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
                        &shape_effect_leaves,
                        None,
                        Some(node_id),
                        &mut traversal_scratch,
                    );
                    render_segments(
                        &mut self.draw_tree,
                        &mut encoder,
                        traversal_scratch.events(),
                        &effect_results,
                        &mut shape_effect_leaves,
                        &self.group_effects,
                        &mut self.backdrop_effects,
                        behind_color_view,
                        behind_resolve_target,
                        &behind_depth_view,
                        None,
                        true,
                        &pipelines,
                        &buffers,
                        &mut self.buffers_pool_manager.gradient_cache,
                        &mut self.offscreen_texture_pool,
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
                        #[cfg(feature = "render_metrics")]
                        &mut shape_effect_cache_metrics,
                    );
                    Some(behind_tex)
                } else {
                    None
                };

                // --- Subtree rendering (unified: always use plan_traversal + render_segments) ---
                plan_traversal_in_place(
                    &mut self.draw_tree,
                    &effect_results,
                    &shape_effect_leaves,
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
                    Some(types::BackdropContext {
                        loaded_effects: &self.loaded_effects,
                        composite_bgl: self.composite_bgl.as_ref().unwrap(),
                        effect_sampler: self.effect_sampler.as_ref().unwrap(),
                        gradient_ramp_sampler: &self.gradient_ramp_sampler,
                        texture_blit_pipeline: self.texture_blit_pipeline.as_ref().unwrap(),
                        backdrop_layer_composite_pipeline: self
                            .backdrop_layer_composite_pipeline
                            .as_ref()
                            .unwrap(),
                        backdrop_layer_composite_bind_group_layout: self
                            .backdrop_layer_composite_bind_group_layout
                            .as_ref()
                            .unwrap(),
                        stencil_only_pipeline: self.stencil_only_pipeline.as_ref().unwrap(),
                        backdrop_color_pipeline: self.backdrop_color_pipeline.as_ref().unwrap(),
                        backdrop_color_gradient_pipeline: self
                            .backdrop_color_gradient_pipeline
                            .as_ref()
                            .unwrap(),
                        device: &self.device,
                        queue: &self.queue,
                        config_format: self.config.format,
                        max_texture_dimension_2d: self.device.limits().max_texture_dimension_2d,
                        backdrop_texture_bind_group_layout: &self
                            .backdrop_texture_bind_group_layout,
                        default_backdrop_texture_bind_group: &self
                            .default_backdrop_texture_bind_group,
                        backdrop_gradient_bind_group_layout: &self
                            .backdrop_gradient_bind_group_layout,
                    })
                } else {
                    None
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
                    &mut self.draw_tree,
                    &mut encoder,
                    traversal_scratch.events(),
                    &effect_results,
                    &mut shape_effect_leaves,
                    &self.group_effects,
                    &mut self.backdrop_effects,
                    subtree_color_view,
                    subtree_resolve_target,
                    subtree_texture
                        .depth_stencil_view
                        .as_ref()
                        .expect("subtree render targets must include a depth/stencil attachment"),
                    backdrop_source,
                    true,
                    &pipelines,
                    &buffers,
                    &mut self.buffers_pool_manager.gradient_cache,
                    &mut self.offscreen_texture_pool,
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
                    #[cfg(feature = "render_metrics")]
                    &mut shape_effect_cache_metrics,
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
                    &mut self.offscreen_texture_pool,
                    EffectPassRunConfig {
                        loaded_effect,
                        params_bind_group: effect_instance.params_bind_group.as_ref(),
                        source_view,
                        effect_sampler: self.effect_sampler.as_ref().unwrap(),
                        composite_bind_group_layout: self.composite_bgl.as_ref().unwrap(),
                        create_composite_bind_group: true,
                        width,
                        height,
                        texture_format: self.config.format,
                        label_prefix: "group_effect",
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

            // Unified main-scene rendering: always plan_traversal + render_segments.
            plan_traversal_in_place(
                &mut self.draw_tree,
                &effect_results,
                &shape_effect_leaves,
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
                Some(types::BackdropContext {
                    loaded_effects: &self.loaded_effects,
                    composite_bgl: self.composite_bgl.as_ref().unwrap(),
                    effect_sampler: self.effect_sampler.as_ref().unwrap(),
                    gradient_ramp_sampler: &self.gradient_ramp_sampler,
                    texture_blit_pipeline: self.texture_blit_pipeline.as_ref().unwrap(),
                    backdrop_layer_composite_pipeline: self
                        .backdrop_layer_composite_pipeline
                        .as_ref()
                        .unwrap(),
                    backdrop_layer_composite_bind_group_layout: self
                        .backdrop_layer_composite_bind_group_layout
                        .as_ref()
                        .unwrap(),
                    stencil_only_pipeline: self.stencil_only_pipeline.as_ref().unwrap(),
                    backdrop_color_pipeline: self.backdrop_color_pipeline.as_ref().unwrap(),
                    backdrop_color_gradient_pipeline: self
                        .backdrop_color_gradient_pipeline
                        .as_ref()
                        .unwrap(),
                    device: &self.device,
                    queue: &self.queue,
                    config_format: self.config.format,
                    max_texture_dimension_2d: self.device.limits().max_texture_dimension_2d,
                    backdrop_texture_bind_group_layout: &self.backdrop_texture_bind_group_layout,
                    default_backdrop_texture_bind_group: &self.default_backdrop_texture_bind_group,
                    backdrop_gradient_bind_group_layout: &self.backdrop_gradient_bind_group_layout,
                })
            } else {
                None
            };

            let backdrop_source = if has_backdrop_effects {
                Some(types::BackdropSource::Flattened {
                    texture: output_texture.expect("output_texture required for backdrop effects"),
                })
            } else {
                None
            };

            render_segments(
                &mut self.draw_tree,
                &mut encoder,
                traversal_scratch.events(),
                &effect_results,
                &mut shape_effect_leaves,
                &self.group_effects,
                &mut self.backdrop_effects,
                phase2_color_view,
                phase2_resolve_target,
                depth_texture_view,
                backdrop_source,
                true,
                &pipelines,
                &buffers,
                &mut self.buffers_pool_manager.gradient_cache,
                &mut self.offscreen_texture_pool,
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
                #[cfg(feature = "render_metrics")]
                &mut shape_effect_cache_metrics,
            );
        }

        let command_buffer = encoder.finish();
        let submitted = !deadline.is_some_and(|deadline| std::time::Instant::now() >= deadline);
        if submitted {
            self.submitted_work = Some(self.queue.submit(std::iter::once(command_buffer)));
        } else {
            self.shape_effect_cache.retain(|_, _| false);
            self.shape_effect_mask_cache.retain(|_, _| false);
            self.texture_manager.restore_pending_uploads();
        }

        self.last_render_to_texture_view_cpu_time = render_to_texture_view_started_at.elapsed();

        effect_output_textures.append(&mut backdrop_work_textures);
        textures_to_recycle.append(&mut effect_output_textures);
        self.offscreen_texture_pool
            .recycle(&mut textures_to_recycle);

        self.draw_tree
            .iter_mut()
            .for_each(|(_node_id, draw_command)| {
                draw_command.clear_frame_state();
            });

        shape_effect_leaves.clear();

        self.scratch = Some(RendererScratch {
            traversal_scratch,
            effect_results,
            shape_effect_leaves,
            effect_node_ids,
            textures_to_recycle,
            effect_output_textures,
            stencil_stack,
            skipped_stack,
            scissor_stack,
            clip_kind_stack,
            backdrop_work_textures,
            readback_bytes,
        });
        let _collected_shape_effect_results = self.shape_effect_cache.end_frame();
        let _collected_shape_effect_masks = self.shape_effect_mask_cache.end_frame();
        self.buffers_pool_manager.tessellation_cache.end_frame();

        // println!("Tesselation cache size: {}", self.buffers_pool_manager.tessellation_cache.len());

        #[cfg(feature = "render_metrics")]
        {
            shape_effect_cache_metrics.collected_results = _collected_shape_effect_results as u64;
            shape_effect_cache_metrics.collected_masks = _collected_shape_effect_masks as u64;
            self.last_pipeline_switch_counts = frame_pipeline_counts;
            self.last_shape_effect_cache_metrics = shape_effect_cache_metrics;
        }
        submitted
    }

    /// Consumes preparation, then acquires, submits, and presents the selected image.
    /// An expired deadline cancels only before acquisition. Acquired images must be presented,
    /// even if acquisition or encoding overruns, because Vulkan cannot safely discard them.
    /// The hook runs immediately before presentation, for platform pre-present notification.
    pub fn commit(&mut self, deadline: Option<std::time::Instant>) -> Result<(), RenderError> {
        let submission_started_at = std::time::Instant::now();
        let output = self.acquire_and_submit(deadline);
        self.last_submission_duration = submission_started_at.elapsed();
        let output = output?;

        #[cfg(feature = "render_metrics")]
        let after_submit = std::time::Instant::now();

        if let Some(callback) = &self.pre_present_callback {
            callback();
        }
        output.present();
        #[cfg(feature = "render_metrics")]
        {
            let after_present = std::time::Instant::now();
            // Force GPU completion to measure actual GPU execution time.
            let _ = self.device.poll(wgpu::MaintainBase::Wait);
            let after_gpu_wait = std::time::Instant::now();

            let prepare_dur = self.preparation_cpu_time;
            let encode_submit_dur = self.last_submission_duration;
            let present_dur = after_present.saturating_duration_since(after_submit);
            let gpu_wait_dur = after_gpu_wait.saturating_duration_since(after_present);
            let total_dur =
                prepare_dur + after_gpu_wait.saturating_duration_since(submission_started_at);
            self.last_phase_timings = PhaseTimings {
                prepare: prepare_dur,
                encode_and_submit: encode_submit_dur,
                present_or_readback: present_dur,
                gpu_wait: gpu_wait_dur,
                total: total_dur,
            };
            self.render_loop_metrics_tracker
                .record_presented_frame(submission_started_at, after_gpu_wait);
        }
        Ok(())
    }

    fn acquire_and_submit(
        &mut self,
        deadline: Option<std::time::Instant>,
    ) -> Result<wgpu::SurfaceTexture, RenderError> {
        if self.prepared_buffers.is_none() {
            return Err(RenderError::NotPrepared);
        }
        if deadline.is_some_and(|deadline| std::time::Instant::now() >= deadline) {
            self.discard_preparation();
            return Err(RenderError::DeadlineMissed);
        }

        let acquisition_started_at = std::time::Instant::now();
        let output = match self.surface.as_ref() {
            None => {
                self.discard_preparation();
                return Err(RenderError::Headless);
            }
            Some(surface) => match surface.get_current_texture() {
                Ok(output) => output,
                Err(error) => {
                    self.discard_preparation();
                    return Err(error.into());
                }
            },
        };
        tracing::debug!(
            acquisition_duration = ?acquisition_started_at.elapsed(),
            "Surface drawable acquired"
        );
        self.submit_acquired_texture(&output.texture, deadline)?;
        Ok(output)
    }

    fn submit_acquired_texture(
        &mut self,
        output: &wgpu::Texture,
        deadline: Option<std::time::Instant>,
    ) -> Result<(), RenderError> {
        self.upload_prepared_buffers()?;
        let output_texture_view = output.create_view(&wgpu::TextureViewDescriptor::default());

        // wgpu 25's Vulkan discard does not release an acquired swapchain image.
        self.render_to_texture_view(&output_texture_view, Some(output), None);
        if let Some(deadline) = deadline {
            tracing::debug!(
                deadline_overrun = ?std::time::Instant::now().saturating_duration_since(deadline),
                "Acquired surface drawable submitted"
            );
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::pipeline::{create_readback_buffer, encode_copy_texture_to_buffer};
    use crate::{
        Renderer, RendererCreationError, Shape, ShapeDrawCommandOptions, ShapeTextureOptions,
        Stroke,
    };
    use futures::executor::block_on;
    use std::time::{Duration, Instant};

    #[test]
    fn queued_resource_reuse_preserves_earlier_images_without_cpu_completion_waits() {
        let mut first = match block_on(Renderer::try_new_headless((16, 16), 1.0)) {
            Ok(renderer) => renderer,
            Err(RendererCreationError::AdapterNotAvailable(_)) => {
                println!("Skipping test: no suitable GPU adapter available.");
                return;
            }
            Err(error) => panic!("renderer creation failed: {error}"),
        };
        let mut second = Renderer::try_new_headless_with_context(
            first.context().isolated_resources(),
            (16, 16),
            1.0,
        )
        .unwrap();
        let output = first.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: 16,
                height: 16,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: first.config.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let mut captures = Vec::new();
        for _ in 0..16 {
            captures.push((
                submit_textured_capture(&mut first, &output, [255, 0, 0, 255]),
                [0, 0, 255, 255],
            ));
            captures.push((
                submit_textured_capture(&mut second, &output, [0, 255, 0, 255]),
                [0, 255, 0, 255],
            ));
            captures.push((
                submit_textured_capture(&mut first, &output, [0, 0, 255, 255]),
                [255, 0, 0, 255],
            ));
        }
        // All writes, draws and captures are submitted before the first CPU wait.
        let mut bytes = Vec::new();
        for (buffer, expected) in captures {
            Renderer::map_readback_buffer_into(&first.device, &buffer, &mut bytes);
            let center = 8 * 256 + 8 * 4;
            assert_eq!(&bytes[center..center + 4], &expected);
        }
    }

    fn submit_textured_capture(
        renderer: &mut Renderer<'_>,
        output: &wgpu::Texture,
        pixels: [u8; 4],
    ) -> wgpu::Buffer {
        prepare_textured_scene(renderer, pixels);
        renderer.upload_prepared_buffers().unwrap();
        let view = output.create_view(&wgpu::TextureViewDescriptor::default());
        assert!(renderer.render_to_texture_view(&view, Some(output), None));
        capture_texture(renderer, output)
    }

    fn prepare_textured_scene(renderer: &mut Renderer<'_>, pixels: [u8; 4]) {
        renderer.clear_draw_queue();
        renderer
            .texture_manager()
            .allocate_texture_with_data(7, (1, 1), &pixels);
        renderer
            .add_shape(
                Shape::rect([(0.0, 0.0), (16.0, 16.0)], Stroke::default()),
                None,
                None,
                ShapeDrawCommandOptions::new().background_texture(ShapeTextureOptions::new(7)),
            )
            .unwrap();
        renderer.prepare();
    }

    fn capture_texture(renderer: &Renderer<'_>, output: &wgpu::Texture) -> wgpu::Buffer {
        let buffer = create_readback_buffer(&renderer.device, None, 256 * 16);
        let mut encoder = renderer
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encode_copy_texture_to_buffer(&mut encoder, output, &buffer, 16, 16, 256);
        renderer.queue.submit([encoder.finish()]);
        buffer
    }

    #[test]
    fn acquired_texture_is_submitted_even_after_the_deadline() {
        let mut renderer = match block_on(Renderer::try_new_headless((16, 16), 1.0)) {
            Ok(renderer) => renderer,
            Err(RendererCreationError::AdapterNotAvailable(_)) => {
                println!("Skipping test: no suitable GPU adapter available.");
                return;
            }
            Err(error) => panic!("renderer creation failed: {error}"),
        };
        let output = renderer.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: 16,
                height: 16,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: renderer.config.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        prepare_textured_scene(&mut renderer, [255, 0, 0, 255]);
        renderer
            .submit_acquired_texture(&output, Some(Instant::now()))
            .unwrap();
        assert!(renderer.submitted_work.is_some());
        let buffer = capture_texture(&renderer, &output);
        let mut bytes = Vec::new();
        Renderer::map_readback_buffer_into(&renderer.device, &buffer, &mut bytes);
        let center = 8 * 256 + 8 * 4;
        assert_eq!(&bytes[center..center + 4], &[0, 0, 255, 255]);
    }

    #[test]
    fn missed_submission_restores_staged_texture_uploads_for_the_next_scene() {
        let mut renderer = match block_on(Renderer::try_new_headless((16, 16), 1.0)) {
            Ok(renderer) => renderer,
            Err(RendererCreationError::AdapterNotAvailable(_)) => {
                println!("Skipping test: no suitable GPU adapter available.");
                return;
            }
            Err(error) => panic!("renderer creation failed: {error}"),
        };
        renderer
            .texture_manager()
            .allocate_texture_with_data(7, (1, 1), &[255, 0, 0, 255]);
        renderer
            .add_shape(
                Shape::rect([(0.0, 0.0), (16.0, 16.0)], Stroke::default()),
                None,
                None,
                ShapeDrawCommandOptions::new().background_texture(ShapeTextureOptions::new(7)),
            )
            .unwrap();
        renderer.prepare();
        renderer.upload_prepared_buffers().unwrap();
        let output = renderer.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: 16,
                height: 16,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: renderer.config.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = output.create_view(&wgpu::TextureViewDescriptor::default());
        assert!(!renderer.render_to_texture_view(&view, Some(&output), Some(Instant::now())));
        assert!(renderer.submitted_work.is_none());
        renderer.prepare();
        renderer.last_submission_duration = Duration::MAX;
        assert!(matches!(
            renderer.commit(Some(Instant::now())),
            Err(crate::RenderError::DeadlineMissed)
        ));
        assert_ne!(renderer.last_submission_duration(), Duration::MAX);
        assert!(renderer.prepared_buffers.is_none());
        let mut pixels = Vec::new();
        renderer.render_to_buffer(&mut pixels);
        let center = (8 * 16 + 8) * 4;
        assert_eq!(&pixels[center..center + 4], &[0, 0, 255, 255]);
    }
}
