use super::WgpuBackend;
use crate::commands::RenderPlan;
use crate::core::UnsignedPhysicalRect;
use crate::wgpu_backend::execution::effects::EffectContext;
use crate::wgpu_backend::execution::instructions::{
    execute_commands, ExecutionContext, ExecutionResources,
};
use crate::wgpu_backend::execution::targets::{RenderTarget, SurfaceTarget};
#[cfg(feature = "render_metrics")]
use crate::wgpu_backend::metrics::{PhaseTimings, ShapeEffectCacheMetrics};
use crate::wgpu_backend::retained_output::RetainedOutput;
use crate::wgpu_backend::types::BackdropContext;
use std::iter;
#[cfg(feature = "render_metrics")]
use std::time::Instant;
#[cfg(feature = "render_metrics")]
use wgpu::MaintainBase;
use wgpu::{CommandEncoderDescriptor, Surface, SurfaceError, TextureViewDescriptor};

impl WgpuBackend {
    /// Updates the clean scene and returns the redrawn bounds.
    pub(in crate::wgpu_backend) fn update_retained_output(
        &mut self,
        root_scissor: Option<UnsignedPhysicalRect>,
        commands: &RenderPlan,
    ) -> Option<UnsignedPhysicalRect> {
        #[cfg(feature = "render_metrics")]
        let update_started_at = Instant::now();
        #[cfg(feature = "render_metrics")]
        {
            self.resources.pipeline_switch_counts = Default::default();
            self.resources.shape_effect_cache_metrics = ShapeEffectCacheMetrics::default();
        }
        let is_new = self.retained_output.is_none();
        let retained = self.retained_output.take().unwrap_or_else(|| {
            RetainedOutput::new(
                &self.device,
                self.viewport.physical_size,
                self.format,
                self.msaa_sample_count,
            )
        });
        let root_scissor = if is_new {
            Some(UnsignedPhysicalRect::from_size(
                self.viewport.physical_size.into(),
            ))
        } else {
            root_scissor
        };
        if let Some(scissor) = root_scissor {
            self.render_dirty_region(commands, &retained, scissor);
        }
        self.retained_output = Some(retained);
        #[cfg(feature = "render_metrics")]
        {
            self.last_retained_output_update_cpu_time = update_started_at.elapsed();
        }
        root_scissor
    }

    fn render_dirty_region(
        &mut self,
        commands: &RenderPlan,
        retained: &RetainedOutput,
        root_scissor: UnsignedPhysicalRect,
    ) {
        self.resources
            .shape_execution
            .texture_materials
            .begin_render();
        self.resources.effect_execution.begin_render();

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
        if !is_full_redraw {
            retained.clear_region(
                &mut encoder,
                self.msaa_color_texture_view.as_ref(),
                self.depth_stencil_view
                    .as_ref()
                    .expect("depth stencil target was initialized"),
                root_scissor,
            );
        }

        let pipeline_resources = &self.pipeline_resources;
        let effects = EffectContext {
            device: &self.device,
            queue: &self.queue,
            registry: &self.effect_registry,
            sampler: &pipeline_resources.effect_sampler,
            composite_layout: &pipeline_resources.composite_resources.bind_group_layout,
            format: self.format,
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
            device: &self.device,
            queue: &self.queue,
            pipelines: pipeline_resources,
            effects,
            backdrops,
            format: self.format,
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
                    is_full_redraw,
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
                shape_effect_metrics: &mut resources.shape_effect_cache_metrics,
            },
        );
        #[cfg(feature = "render_metrics")]
        {
            resources.pipeline_switch_counts = _metrics.pipeline_switches;
        }

        self.queue.submit(iter::once(encoder.finish()));
        resources.shape_execution.texture_materials.finish_render();
        resources.effect_execution.finish_render();

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

    pub(in crate::wgpu_backend) fn prepare_resources(&mut self) {
        self.resources.shape_execution.upload(
            &self.device,
            &self.queue,
            &mut self.resources.buffers,
        );
    }
}

impl WgpuBackend {
    /// Returns an error if surface acquisition fails.
    pub(super) fn render_surface(
        &mut self,
        root_scissor: Option<UnsignedPhysicalRect>,
        commands: &RenderPlan,
        surface: &Surface<'_>,
    ) -> Result<(), SurfaceError> {
        #[cfg(feature = "render_metrics")]
        let frame_render_loop_started_at = Instant::now();
        self.prepare_resources();

        #[cfg(feature = "render_metrics")]
        let after_prepare = Instant::now();

        let output = surface.get_current_texture()?;
        let output_texture_view = output
            .texture
            .create_view(&TextureViewDescriptor::default());

        let root_scissor = self.update_retained_output(root_scissor, commands);
        let retained = self
            .retained_output
            .as_ref()
            .expect("retained output was initialized before presentation");
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("present_retained_output"),
            });
        retained.present(&mut encoder, &output_texture_view);
        self.queue.submit(iter::once(encoder.finish()));
        if let Some(scissor) = root_scissor.filter(|_| self.is_dirty_region_overlay_enabled) {
            let mut encoder = self
                .device
                .create_command_encoder(&CommandEncoderDescriptor {
                    label: Some("draw_dirty_region_overlay"),
                });
            retained.draw_dirty_region_overlay(&mut encoder, &output_texture_view, scissor);
            self.queue.submit(iter::once(encoder.finish()));
        }

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
