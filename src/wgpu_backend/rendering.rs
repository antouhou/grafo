use super::{SurfaceError, WgpuBackend, WgpuSurfaceState};
use crate::commands::RenderCommand;
use crate::core::UnsignedPhysicalRect;
use crate::render_backend::render_target::Surface;
use crate::wgpu_backend::execution::effects::EffectContext;
use crate::wgpu_backend::execution::instructions::{
    execute_commands, ExecutionContext, ExecutionResources,
};
use crate::wgpu_backend::execution::targets::{RenderTarget, SurfaceTarget};
#[cfg(feature = "render_metrics")]
use crate::wgpu_backend::metrics::{PhaseTimings, ShapeMaskEffectCacheMetrics};
use crate::wgpu_backend::retained_output::RetainedOutput;
use crate::wgpu_backend::types::BackdropContext;
use std::iter;
#[cfg(feature = "render_metrics")]
use std::time::Instant;
#[cfg(feature = "render_metrics")]
use wgpu::PollType;
use wgpu::{CommandEncoderDescriptor, CurrentSurfaceTexture, TextureViewDescriptor};

impl WgpuBackend {
    /// Updates the retained image within the planned redraw bounds.
    pub(in crate::wgpu_backend) fn update_retained_output<'scene>(
        &mut self,
        root_scissor: Option<UnsignedPhysicalRect>,
        commands: impl Iterator<Item = RenderCommand<'scene>>,
    ) {
        #[cfg(feature = "render_metrics")]
        let update_started_at = Instant::now();
        #[cfg(feature = "render_metrics")]
        {
            self.resources.pipeline_switch_counts = Default::default();
            self.resources.shape_mask_effect_cache_metrics = ShapeMaskEffectCacheMetrics::default();
        }
        let retained = self.retained_output.take().unwrap_or_else(|| {
            RetainedOutput::new(
                &self.device,
                &self.context.retained_output_resources,
                self.viewport.physical_size,
                self.render_format,
                self.msaa_sample_count,
            )
        });
        if let Some(scissor) = root_scissor {
            self.render_dirty_region(commands, &retained, scissor);
        }
        self.retained_output = Some(retained);
        #[cfg(feature = "render_metrics")]
        {
            self.last_retained_output_update_cpu_time = update_started_at.elapsed();
        }
    }

    fn render_dirty_region<'scene>(
        &mut self,
        commands: impl Iterator<Item = RenderCommand<'scene>>,
        retained: &RetainedOutput,
        root_scissor: UnsignedPhysicalRect,
    ) {
        self.resources
            .shape_execution
            .texture_materials
            .begin_render();
        self.resources.effect_execution.begin_render();

        if self.msaa_sample_count > 1 && self.msaa_color_texture_view.is_none() {
            self.recreate_msaa_texture();
        }
        if self.depth_stencil_view.is_none() {
            self.recreate_depth_stencil_texture();
        }

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Render Command Encoder"),
            });

        let is_full_redraw =
            root_scissor == UnsignedPhysicalRect::from_size(self.viewport.physical_size.into());
        let pipeline_resources = &self.pipeline_resources;
        let effects = EffectContext {
            device: &self.device,
            queue: &self.queue,
            registry: &self.effect_registry,
            sampler: &pipeline_resources.effect_sampler,
            composite_layout: &pipeline_resources.composite_resources.bind_group_layout,
            format: self.render_format,
        };
        let backdrops = &pipeline_resources.backdrops;
        let backdrops = BackdropContext {
            effects,
            texture_blit_pipeline: &backdrops.texture_blit_pipeline,
            backdrop_layer_composite_pipeline: &backdrops.layer_composite_resources.pipeline,
            backdrop_layer_composite_bind_group_layout: &backdrops
                .layer_composite_resources
                .bind_group_layout,
        };
        let execution_context = ExecutionContext {
            viewport: self.viewport,
            device: &self.device,
            queue: &self.queue,
            pipelines: pipeline_resources,
            effects,
            backdrops,
            format: self.render_format,
            sample_count: self.msaa_sample_count,
        };
        let resources = &mut self.resources;
        let _metrics = execute_commands(
            &mut encoder,
            commands,
            SurfaceTarget {
                root_scissor,
                output: RenderTarget::for_output(
                    &retained.view,
                    self.msaa_color_texture_view.as_ref(),
                    self.depth_stencil_view
                        .as_ref()
                        .expect("depth stencil target was initialized"),
                    (!is_full_redraw).then(|| retained.region_clear(root_scissor)),
                ),
                capture_texture: Some(&retained.texture),
            },
            ExecutionResources {
                context: &execution_context,
                buffers: &resources.buffers,
                shapes: &mut resources.shape_execution,
                effects: &mut resources.effect_execution,
                textures: &mut resources.textures,
                #[cfg(feature = "render_metrics")]
                shape_mask_effect_metrics: &mut resources.shape_mask_effect_cache_metrics,
            },
        );
        #[cfg(feature = "render_metrics")]
        {
            resources.pipeline_switch_counts = _metrics.pipeline_switches;
        }

        self.queue.submit(iter::once(encoder.finish()));
        resources.shape_execution.texture_materials.finish_render();
        resources.shape_execution.coverage_cache.finish_render();
        if let Some(cache) = &mut resources.shape_execution.clip_coverage {
            cache.finish_render();
        }
        resources.effect_execution.finish_render();

        let (_collected_shape_mask_effect_results, _collected_shape_masks) =
            resources.textures.collect_unused_shape_mask_effects();
        resources.textures.recycle_submitted();

        #[cfg(feature = "render_metrics")]
        {
            resources.shape_mask_effect_cache_metrics.collected_results =
                _collected_shape_mask_effect_results as u64;
            resources.shape_mask_effect_cache_metrics.collected_masks =
                _collected_shape_masks as u64;
        }
    }

    pub(in crate::wgpu_backend) fn prepare_resources(&mut self) {
        self.resources.shape_execution.coverage_cache.upload(
            &self.device,
            &self.queue,
            &self.pipeline_resources.shapes,
        );
        self.resources.shape_execution.upload(
            &self.device,
            &self.queue,
            &mut self.resources.buffers,
        );
    }
}

impl WgpuBackend {
    /// Returns an error if surface acquisition fails.
    pub(super) fn render_surface<'scene>(
        &mut self,
        root_scissor: Option<UnsignedPhysicalRect>,
        commands: impl Iterator<Item = RenderCommand<'scene>>,
        surface: &mut Surface<WgpuSurfaceState>,
    ) -> Result<(), SurfaceError> {
        #[cfg(feature = "render_metrics")]
        let frame_render_loop_started_at = Instant::now();
        self.prepare_resources();

        #[cfg(feature = "render_metrics")]
        let after_prepare = Instant::now();

        let output = match surface.resource().handle()?.get_current_texture() {
            CurrentSurfaceTexture::Success(output) => output,
            CurrentSurfaceTexture::Suboptimal(output) => {
                surface.invalidate();
                output
            }
            CurrentSurfaceTexture::Outdated => {
                surface.invalidate();
                return Err(SurfaceError::Outdated);
            }
            CurrentSurfaceTexture::Lost => return Err(SurfaceError::Lost),
            CurrentSurfaceTexture::Timeout => return Err(SurfaceError::Timeout),
            CurrentSurfaceTexture::Occluded => return Err(SurfaceError::Occluded),
            CurrentSurfaceTexture::Validation => return Err(SurfaceError::Validation),
        };
        let output_texture_view = output
            .texture
            .create_view(&TextureViewDescriptor::default());

        self.update_retained_output(root_scissor, commands);
        let retained = self
            .retained_output
            .as_ref()
            .expect("retained output was initialized before presentation");
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("present_retained_output"),
            });
        let overlay = root_scissor.filter(|_| self.is_dirty_region_overlay_enabled);
        surface.resource().presentation.present(
            &mut encoder,
            retained,
            &output_texture_view,
            overlay,
        );
        self.queue.submit(iter::once(encoder.finish()));

        #[cfg(feature = "render_metrics")]
        let after_submit = Instant::now();

        self.queue.present(output);
        #[cfg(feature = "render_metrics")]
        {
            let after_present = Instant::now();
            // Measure the remaining wait for GPU work after presentation.
            let _ = self.device.poll(PollType::wait_indefinitely());
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
