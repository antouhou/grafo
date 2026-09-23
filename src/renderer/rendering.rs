use super::*;
use crate::renderer::commands::BackdropCaptureSource;
use crate::renderer::execution::effects::{apply_effect_passes, EffectPassRunConfig};
use crate::renderer::execution::segments::{
    execute_segments, SegmentExecutionContext, SegmentExecutionResources, SegmentRenderTarget,
};
use crate::renderer::execution::shape_effects::{
    execute_shape_effects, ShapeEffectExecutionResources,
};
use crate::renderer::execution::targets::RenderTarget;
use crate::renderer::execution::textures::IntermediateTexture;
#[cfg(feature = "render_metrics")]
use crate::renderer::metrics::{PhaseTimings, PipelineSwitchCounts, ShapeEffectCacheMetrics};
use crate::renderer::plan::draws::{DrawPlanningInput, DrawTreeSelection};
use crate::renderer::traversal::{compute_node_depth, subtree_has_backdrop_effects};
use crate::renderer::types::RenderError;
use wgpu::CommandEncoder;

fn render_planned_draws(
    encoder: &mut CommandEncoder,
    selection: DrawTreeSelection,
    effect_results: &HashMap<usize, IntermediateTextureId>,
    backdrop_source: Option<BackdropCaptureSource>,
    target: SegmentRenderTarget<'_>,
    execution_context: &SegmentExecutionContext<'_>,
    state: &mut RendererState,
) {
    state.scratch.draw_planner.plan(
        DrawPlanningInput {
            tree: &state.draw_tree,
            selection,
            effect_results,
            shape_effects: &state.scratch.shape_effect_plan.composites,
            group_effects: &state.group_effects,
            backdrop_effects: &state.backdrop_effects,
            backdrop_source,
            scale_factor: state.scale_factor,
            physical_size: state.physical_size.into(),
            max_capture_dimension: target
                .backdrop_context
                .map(|context| context.max_texture_dimension_2d),
        },
        &mut state.scratch.draw_plan,
    );
    let _metrics = execute_segments(
        encoder,
        &state.scratch.draw_plan,
        target,
        SegmentExecutionResources {
            context: execution_context,
            buffers: &state.buffers,
            shapes: &mut state.shape_execution,
            effects: &mut state.effect_execution,
            textures: &mut state.textures,
        },
    );
    #[cfg(feature = "render_metrics")]
    {
        state
            .pipeline_switch_counts
            .accumulate(&_metrics.pipeline_switches);
    }
}

impl<'a> Renderer<'a> {
    pub(super) fn render_to_texture_view(
        &mut self,
        texture_view: &wgpu::TextureView,
        output_texture: Option<&wgpu::Texture>,
    ) {
        let render_to_texture_view_started_at = std::time::Instant::now();
        self.state.shape_execution.texture_materials.begin_render();
        self.state.effect_execution.begin_render();
        self.state.shape_execution.composites.begin_render();

        if self.state.draw_tree.is_empty() {
            self.state.shape_execution.texture_materials.finish_render();
            self.state.effect_execution.finish_render();
            self.state.scratch.shape_effect_plan.clear();
            let (_collected_shape_effect_results, _collected_shape_effect_masks) =
                self.state.textures.collect_unused_shape_effects();
            self.state.textures.work_textures.clear();
            self.state.textures.pool.clear();
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

        let mut effect_results = std::mem::take(&mut self.state.scratch.effect_results);
        let mut effect_node_ids = std::mem::take(&mut self.state.scratch.effect_node_ids);

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
            self.ensure_backdrop_pipelines();
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
            execute_shape_effects(
                &mut encoder,
                &self.state.scratch.shape_effect_plan.commands,
                ShapeEffectExecutionResources {
                    device: &self.device,
                    queue: &self.queue,
                    registry: &self.effect_registry,
                    sampler: self
                        .pipeline_resources
                        .effect_sampler
                        .as_ref()
                        .expect("shape effect sampler was initialized"),
                    composite_layout: &self
                        .pipeline_resources
                        .shapes
                        .shape_texture_bind_group_layout_background,
                    format: self.config.format,
                    pipelines: &self.pipeline_resources.shape_effects,
                    buffers: &self.state.buffers,
                    shapes: &self.state.shape_execution,
                    effects: &mut self.state.effect_execution,
                    textures: &mut self.state.textures,
                    #[cfg(feature = "render_metrics")]
                    metrics: &mut self.state.shape_effect_cache_metrics,
                },
            );
        }

        let pipeline_resources = &self.pipeline_resources;
        let execution_context = SegmentExecutionContext {
            device: &self.device,
            queue: &self.queue,
            pipelines: pipeline_resources,
        };
        let backdrop_context = if has_backdrop_effects {
            let backdrops = pipeline_resources
                .backdrops
                .as_ref()
                .expect("backdrop pipelines were initialized above");
            let backdrop_composite = &backdrops.layer_composite_resources;
            Some(types::BackdropContext {
                effect_registry: &self.effect_registry,
                effect_sampler: pipeline_resources.effect_sampler.as_ref().unwrap(),
                texture_blit_pipeline: &backdrops.texture_blit_pipeline,
                composite_bind_group_layout: &pipeline_resources
                    .composite_resources
                    .as_ref()
                    .expect("backdrop rendering requires composite resources")
                    .bind_group_layout,
                backdrop_layer_composite_pipeline: &backdrop_composite.pipeline,
                backdrop_layer_composite_bind_group_layout: &backdrop_composite.bind_group_layout,
                device: &self.device,
                queue: &self.queue,
                config_format: self.config.format,
                max_texture_dimension_2d: self.device.limits().max_texture_dimension_2d,
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
                let mut subtree_texture = state.textures.pool.acquire_with_depth(
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
                    let behind_tex = state.textures.pool.acquire_with_depth(
                        &self.device,
                        width,
                        height,
                        self.config.format,
                        self.msaa_sample_count,
                    );
                    render_planned_draws(
                        &mut encoder,
                        DrawTreeSelection {
                            excluded_subtree: Some(node_id),
                            ..Default::default()
                        },
                        &effect_results,
                        None,
                        SegmentRenderTarget {
                            output: RenderTarget::for_texture(&behind_tex),
                            capture_texture: None,
                            backdrop_context: None,
                        },
                        &execution_context,
                        state,
                    );
                    Some(state.textures.insert_transient(IntermediateTexture {
                        texture: behind_tex,
                        bind_group: None,
                    }))
                } else {
                    None
                };

                render_planned_draws(
                    &mut encoder,
                    DrawTreeSelection {
                        subtree_root: Some(node_id),
                        ..Default::default()
                    },
                    &effect_results,
                    behind_texture.map(|base| BackdropCaptureSource::Layered { base }),
                    SegmentRenderTarget {
                        output: RenderTarget::for_texture(&subtree_texture),
                        capture_texture: None,
                        backdrop_context: backdrop_context
                            .as_ref()
                            .filter(|_| subtree_needs_backdrop_effects),
                    },
                    &execution_context,
                    state,
                );

                if let Some(behind_tex) = behind_texture {
                    state.textures.finish_transient(behind_tex);
                }

                let effect_instance = state
                    .group_effects
                    .get(&node_id)
                    .expect("group effect remains attached during rendering");
                let source_bind_group = subtree_texture.input_bind_group(
                    &self.device,
                    self.effect_registry.input_bind_group_layout(),
                    pipeline_resources.effect_sampler.as_ref().unwrap(),
                );
                let effect_output = apply_effect_passes(
                    &self.effect_registry,
                    &self.device,
                    &self.queue,
                    &mut state.effect_execution.parameters,
                    &mut encoder,
                    &mut state.textures.pool,
                    EffectPassRunConfig {
                        effect_id: effect_instance.effect_id,
                        params: &effect_instance.params,
                        source_bind_group,
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

                let (texture, bind_group) =
                    effect_output.into_final_output(&mut state.textures.work_textures);
                let texture_id = state.textures.insert_transient(IntermediateTexture {
                    texture,
                    bind_group,
                });
                effect_results.insert(node_id, texture_id);
                state.textures.work_textures.push(subtree_texture);
            }
        }

        {
            let depth_texture_view = self.depth_stencil_view.as_ref().unwrap();

            render_planned_draws(
                &mut encoder,
                DrawTreeSelection::default(),
                &effect_results,
                has_backdrop_effects.then_some(BackdropCaptureSource::Target),
                SegmentRenderTarget {
                    output: RenderTarget::for_output(
                        texture_view,
                        self.msaa_color_texture_view.as_ref(),
                        depth_texture_view,
                    ),
                    capture_texture: output_texture,
                    backdrop_context: backdrop_context.as_ref(),
                },
                &execution_context,
                state,
            );
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        state.shape_execution.texture_materials.finish_render();
        state.effect_execution.finish_render();

        self.last_render_to_texture_view_cpu_time = render_to_texture_view_started_at.elapsed();

        state.scratch.shape_effect_plan.clear();

        state.scratch.effect_node_ids = effect_node_ids;
        let (_collected_shape_effect_results, _collected_shape_effect_masks) =
            state.textures.collect_unused_shape_effects();
        state
            .textures
            .recycle_submitted(effect_results.drain().map(|(_, texture_id)| texture_id));
        state.scratch.effect_results = effect_results;
        state.shape_resources.tessellation_cache.end_frame();

        #[cfg(feature = "render_metrics")]
        {
            state.shape_effect_cache_metrics.collected_results =
                _collected_shape_effect_results as u64;
            state.shape_effect_cache_metrics.collected_masks = _collected_shape_effect_masks as u64;
        }
    }

    /// Returns an error if surface acquisition fails.
    pub fn render(&mut self) -> Result<(), RenderError> {
        #[cfg(feature = "render_metrics")]
        let frame_render_loop_started_at = std::time::Instant::now();
        self.prepare_render();

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
