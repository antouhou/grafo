use super::WgpuBackend;
use crate::core::util::to_logical;
use crate::pipeline::{create_and_depth_texture, create_msaa_color_texture};
use crate::renderer::Viewport;
use tracing::warn;
use wgpu::SurfaceTarget;

impl WgpuBackend {
    pub(in crate::renderer) fn resize(
        &mut self,
        surface: &mut Option<wgpu::Surface<'_>>,
        viewport: Viewport,
        fringe_width: f32,
    ) {
        self.viewport = viewport;
        self.fringe_width = fringe_width;
        let new_physical_size = viewport.physical_size;
        self.config.width = new_physical_size.0;
        self.config.height = new_physical_size.1;

        let pipelines = &mut self.pipeline_resources.shapes;
        let logical_size = to_logical(new_physical_size, self.viewport.scale_factor);
        pipelines.and_uniforms.canvas_size = [logical_size.0, logical_size.1];
        pipelines.and_uniforms.scale_factor = self.viewport.scale_factor as f32;
        pipelines.and_uniforms.fringe_width = self.fringe_width;

        pipelines.decrementing_uniforms.canvas_size = [logical_size.0, logical_size.1];
        pipelines.decrementing_uniforms.scale_factor = self.viewport.scale_factor as f32;
        pipelines.decrementing_uniforms.fringe_width = self.fringe_width;

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

        if let Some(surface) = surface {
            surface.configure(&self.device, &self.config);
        }
        self.recreate_msaa_texture();
        self.recreate_depth_stencil_texture();
    }

    pub fn msaa_samples(&self) -> u32 {
        self.msaa_sample_count
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

    pub(in crate::renderer) fn normalize_msaa_sample_count(requested: u32) -> u32 {
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

    pub(in crate::renderer) fn recreate_msaa_texture(&mut self) {
        if self.msaa_sample_count > 1 {
            let texture = create_msaa_color_texture(
                &self.device,
                self.viewport.physical_size,
                self.config.format,
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
    pub(in crate::renderer) fn recreate_depth_stencil_texture(&mut self) {
        let texture = create_and_depth_texture(
            &self.device,
            self.viewport.physical_size,
            self.msaa_sample_count,
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.depth_stencil_texture = Some(texture);
        self.depth_stencil_view = Some(view);
    }

    pub(in crate::renderer) fn set_surface(
        &mut self,
        output: &mut Option<wgpu::Surface<'_>>,
        window: impl Into<SurfaceTarget<'static>>,
    ) {
        let surface = self
            .instance
            .create_surface(window)
            .expect("Failed to create surface");
        surface.configure(&self.device, &self.config);
        *output = Some(surface);
    }

    pub(in crate::renderer) fn set_vsync(
        &mut self,
        surface: &mut Option<wgpu::Surface<'_>>,
        vsync: bool,
    ) {
        self.config.present_mode = if vsync {
            wgpu::PresentMode::AutoVsync
        } else {
            wgpu::PresentMode::AutoNoVsync
        };
        if let Some(surface) = surface {
            surface.configure(&self.device, &self.config);
        }
    }
}
