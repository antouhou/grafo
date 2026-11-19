use super::retained_output::SurfacePresentation;
use super::{BackendCreationError, SurfaceError, WgpuBackend, WgpuBackendError, WgpuContext};
use crate::core::util::to_logical;
use crate::core::Viewport;
use crate::render_backend::render_target::{RenderTargetError, Surface};
use crate::wgpu_backend::pipeline::{create_and_depth_texture, create_msaa_color_texture};
use std::sync::Arc;
use tracing::warn;
use wgpu::{PresentMode, Surface as WgpuSurface, SurfaceConfiguration, SurfaceTarget};

/// WGPU handle, configuration, and presentation resources for a Grafo surface.
pub struct WgpuSurfaceState {
    pub(super) presentation: SurfacePresentation,
    pub(super) surface: Option<WgpuSurface<'static>>,
    pub(super) configuration: SurfaceConfiguration,
    pub(super) context: Arc<WgpuContext>,
    pub(super) transparent: bool,
}

impl WgpuSurfaceState {
    pub(super) fn handle(&self) -> Result<&WgpuSurface<'static>, SurfaceError> {
        self.surface.as_ref().ok_or(SurfaceError::Lost)
    }
}

impl Surface<WgpuSurfaceState> {
    /// Recreates a lost surface for the supplied target using the current rendering settings.
    pub fn recreate(
        &mut self,
        target: impl Into<SurfaceTarget<'static>>,
    ) -> Result<(), BackendCreationError> {
        // Release the lost surface before binding the target again.
        drop(self.resource_mut().surface.take());
        self.invalidate();
        let resource = self.resource();
        *self = resource.context.create_surface(
            target,
            self.size(),
            self.vsync(),
            resource.transparent,
        )?;
        Ok(())
    }
}

impl WgpuBackend {
    pub(in crate::wgpu_backend) fn resize(&mut self, viewport: Viewport) {
        if self.context.ensure_available().is_err() {
            return;
        }
        let should_rebuild_coverage = self.viewport.physical_size != viewport.physical_size
            || self.viewport.scale_factor != viewport.scale_factor;
        self.viewport = viewport;
        let new_physical_size = viewport.physical_size;

        let pipelines = &mut self.pipeline_resources.shapes;
        let logical_size = to_logical(new_physical_size, self.viewport.scale_factor);
        pipelines.and_uniforms.canvas_size = [logical_size.0, logical_size.1];
        pipelines.and_uniforms.scale_factor = self.viewport.scale_factor as f32;

        pipelines.decrementing_uniforms.canvas_size = [logical_size.0, logical_size.1];
        pipelines.decrementing_uniforms.scale_factor = self.viewport.scale_factor as f32;

        self.queue.write_buffer(
            &pipelines.and_uniform_buffer,
            0,
            bytemuck::cast_slice(&[pipelines.and_uniforms]),
        );
        self.queue.write_buffer(
            &pipelines.decrementing_uniform_buffer,
            0,
            bytemuck::cast_slice(&[pipelines.decrementing_uniforms]),
        );
        if should_rebuild_coverage {
            let shapes = &mut self.resources.shape_execution;
            shapes.coverage_cache.reset_for_viewport(viewport);
            for resources in shapes.draws.values_mut() {
                if let Some(coverage) = &mut resources.path_coverage {
                    *coverage = shapes.coverage_cache.prepare(
                        &coverage.tessellation,
                        coverage.transform,
                        viewport,
                    );
                    if let Some(location) = resources.geometry_buffer_location {
                        shapes.instance_metadata[location.instance_index].coverage_translation =
                            coverage.translation;
                    }
                }
            }
        }

        self.recreate_msaa_texture();
        self.recreate_depth_stencil_texture();
    }

    pub fn set_msaa_samples(&mut self, samples: u32) {
        let sample_count = Self::normalize_msaa_sample_count(samples);
        if sample_count == self.msaa_sample_count {
            return;
        }

        self.msaa_sample_count = sample_count;
        self.recreate_pipelines();
        self.recreate_msaa_texture();
        self.recreate_depth_stencil_texture();
    }

    pub(in crate::wgpu_backend) fn normalize_msaa_sample_count(requested: u32) -> u32 {
        match requested {
            0 | 1 => 1,
            2..=4 => 4,
            _ => {
                warn!(
                    "Requested MSAA sample count {} is not widely supported, clamping to 4",
                    requested
                );
                4
            }
        }
    }

    pub(in crate::wgpu_backend) fn recreate_msaa_texture(&mut self) {
        self.retained_output = None;
        if self.msaa_sample_count > 1 {
            let texture = create_msaa_color_texture(
                &self.device,
                self.viewport.physical_size,
                self.render_format,
                self.msaa_sample_count,
            );
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            self.msaa_color_texture = Some(texture);
            self.msaa_color_texture_view = Some(view);
        } else {
            self.msaa_color_texture = None;
            self.msaa_color_texture_view = None;
        }

        self.resources.textures.pool.clear();
    }

    /// Recreate the cached depth/stencil texture to match current physical size and MSAA settings.
    pub(in crate::wgpu_backend) fn recreate_depth_stencil_texture(&mut self) {
        let texture = create_and_depth_texture(
            &self.device,
            self.viewport.physical_size,
            self.msaa_sample_count,
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.depth_stencil_texture = Some(texture);
        self.depth_stencil_view = Some(view);
    }

    pub(super) fn prepare_surface(
        &mut self,
        surface: &mut Surface<WgpuSurfaceState>,
    ) -> Result<(), WgpuBackendError> {
        let resource = surface.resource();
        if !Arc::ptr_eq(&self.context, &resource.context) {
            return Err(RenderTargetError::IncompatibleSurface.into());
        }
        self.set_output_format(resource.configuration.format);
        if !surface.needs_configuration() {
            return Ok(());
        }
        let (width, height) = surface.size();
        let present_mode = if surface.vsync() {
            PresentMode::AutoVsync
        } else {
            PresentMode::AutoNoVsync
        };
        let resource = surface.resource_mut();
        resource.configuration.width = width;
        resource.configuration.height = height;
        resource.configuration.present_mode = present_mode;
        resource
            .handle()?
            .configure(&self.device, &resource.configuration);
        surface.finish_configuration();
        Ok(())
    }
}
