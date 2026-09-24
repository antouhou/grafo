use super::WgpuBackend;
use crate::commands::RenderPlan;
use crate::renderer::backend::execution::effects::EffectContext;
use crate::renderer::backend::execution::segments::{
    execute_segments, SegmentExecutionContext, SegmentExecutionResources, SegmentRenderTarget,
};
use crate::renderer::backend::execution::shape_effects::{
    execute_shape_effects, ShapeEffectExecutionResources,
};
use crate::renderer::backend::execution::targets::RenderTarget;
#[cfg(feature = "render_metrics")]
use crate::renderer::metrics::{PhaseTimings, ShapeEffectCacheMetrics};
use crate::renderer::types::{BackdropContext, RenderError};
use crate::renderer::RenderBackend;
use std::{iter, time::Instant};
#[cfg(feature = "render_metrics")]
use wgpu::MaintainBase;
use wgpu::{CommandEncoderDescriptor, Texture, TextureView, TextureViewDescriptor};

impl WgpuBackend {
    pub(in crate::renderer) fn render_to_texture_view(
        &mut self,
        commands: &RenderPlan,
        texture_view: &TextureView,
        output_texture: Option<&Texture>,
    ) {
        let render_to_texture_view_started_at = Instant::now();
        self.resources
            .shape_execution
            .texture_materials
            .begin_render();
        self.resources.effect_execution.begin_render();
        self.resources.shape_execution.composites.begin_render();

        let needs_scene_effects = commands.scene.texture_count != 0;
        let has_backdrop_effects = commands.scene.has_backdrop_captures;
        let has_shape_effects = !commands.shape_effects.segments.is_empty();

        if needs_scene_effects {
            self.ensure_composite_pipeline();
        }
        if needs_scene_effects || has_shape_effects {
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
            self.resources.shape_effect_cache_metrics = ShapeEffectCacheMetrics::default();
        }

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Render Command Encoder"),
            });

        if has_shape_effects {
            execute_shape_effects(
                &mut encoder,
                &commands.shape_effects,
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
                    buffers: &self.resources.buffers,
                    shapes: &self.resources.shape_execution,
                    effects: &mut self.resources.effect_execution,
                    textures: &mut self.resources.textures,
                    #[cfg(feature = "render_metrics")]
                    metrics: &mut self.resources.shape_effect_cache_metrics,
                },
            );
        }

        let pipeline_resources = &self.pipeline_resources;
        let effects = needs_scene_effects.then(|| EffectContext {
            device: &self.device,
            queue: &self.queue,
            registry: &self.effect_registry,
            sampler: pipeline_resources
                .effect_sampler
                .as_ref()
                .expect("effect sampler was initialized"),
            composite_layout: &pipeline_resources
                .composite_resources
                .as_ref()
                .expect("effect composites were initialized")
                .bind_group_layout,
            format: self.config.format,
        });
        let backdrops = has_backdrop_effects.then(|| {
            let backdrops = pipeline_resources
                .backdrops
                .as_ref()
                .expect("backdrop pipelines were initialized");
            BackdropContext {
                effects: effects.expect("backdrops require effect resources"),
                texture_blit_pipeline: &backdrops.texture_blit_pipeline,
                backdrop_layer_composite_pipeline: &backdrops.layer_composite_resources.pipeline,
                backdrop_layer_composite_bind_group_layout: &backdrops
                    .layer_composite_resources
                    .bind_group_layout,
            }
        });
        let execution_context = SegmentExecutionContext {
            device: &self.device,
            queue: &self.queue,
            pipelines: pipeline_resources,
            effects,
            backdrops,
            format: self.config.format,
            sample_count: self.msaa_sample_count,
        };
        let resources = &mut self.resources;
        let _metrics = execute_segments(
            &mut encoder,
            &commands.scene,
            SegmentRenderTarget {
                output: RenderTarget::for_output(
                    texture_view,
                    self.msaa_color_texture_view.as_ref(),
                    self.depth_stencil_view
                        .as_ref()
                        .expect("depth stencil target was initialized"),
                ),
                capture_texture: output_texture,
            },
            SegmentExecutionResources {
                context: &execution_context,
                buffers: &resources.buffers,
                shapes: &mut resources.shape_execution,
                effects: &mut resources.effect_execution,
                textures: &mut resources.textures,
            },
        );
        #[cfg(feature = "render_metrics")]
        {
            resources.pipeline_switch_counts = _metrics.pipeline_switches;
        }

        self.queue.submit(iter::once(encoder.finish()));
        resources.shape_execution.texture_materials.finish_render();
        resources.effect_execution.finish_render();

        self.last_render_to_texture_view_cpu_time = render_to_texture_view_started_at.elapsed();

        let (_collected_shape_effect_results, _collected_shape_effect_masks) =
            resources.textures.collect_unused_shape_effects();
        resources.textures.recycle_submitted();

        #[cfg(feature = "render_metrics")]
        {
            resources.shape_effect_cache_metrics.collected_results =
                _collected_shape_effect_results as u64;
            resources.shape_effect_cache_metrics.collected_masks =
                _collected_shape_effect_masks as u64;
        }
    }

    pub(in crate::renderer) fn prepare_resources(&mut self) {
        self.resources.shape_execution.upload(
            &self.device,
            &self.queue,
            &mut self.resources.buffers,
        );
    }
}

impl<'surface> RenderBackend<'surface> for WgpuBackend {
    type Surface = Option<wgpu::Surface<'surface>>;
    type Error = RenderError;

    /// Returns an error if surface acquisition fails.
    fn render(
        &mut self,
        commands: &RenderPlan,
        surface: &mut Self::Surface,
    ) -> Result<(), RenderError> {
        #[cfg(feature = "render_metrics")]
        let frame_render_loop_started_at = Instant::now();
        self.prepare_resources();

        #[cfg(feature = "render_metrics")]
        let after_prepare = Instant::now();

        let surface = surface
            .as_ref()
            .expect("Cannot call render() on a headless renderer; use render_to_buffer()");
        let output = surface.get_current_texture()?;
        let output_texture_view = output
            .texture
            .create_view(&TextureViewDescriptor::default());

        self.render_to_texture_view(commands, &output_texture_view, Some(&output.texture));

        #[cfg(feature = "render_metrics")]
        let after_submit = Instant::now();

        output.present();
        #[cfg(feature = "render_metrics")]
        {
            let after_present = Instant::now();
            // Measure the remaining wait for GPU work after presentation.
            let _ = self.device.poll(MaintainBase::Wait);
            let after_gpu_wait = Instant::now();

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
        }
        Ok(())
    }
}
