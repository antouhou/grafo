use super::*;
use std::num::NonZeroU32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SampleCountChange {
    Unchanged,
    Changed(u32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PresentModeChange {
    Unchanged,
    Changed(wgpu::PresentMode),
}

impl<'a> Renderer<'a> {
    /// Sets wgpu's presentation queue-latency hint. Lower values can serialize work on some backends.
    pub fn set_maximum_frame_latency(&mut self, latency: NonZeroU32) {
        if self.config.desired_maximum_frame_latency == latency.get() {
            return;
        }
        self.discard_preparation();
        self.config.desired_maximum_frame_latency = latency.get();
        if let Some(surface) = &self.surface {
            surface.configure(&self.device, &self.config);
        }
    }

    pub fn maximum_frame_latency(&self) -> u32 {
        self.config.desired_maximum_frame_latency
    }

    /// Installs the platform notification once when creating or replacing a surface.
    pub fn set_pre_present_callback(&mut self, callback: impl Fn() + Send + Sync + 'a) {
        self.discard_preparation();
        self.pre_present_callback = Some(Box::new(callback));
    }

    /// Returns the shared GPU context used by this renderer.
    pub fn context(&self) -> &RendererContext {
        &self.context
    }

    pub fn size(&self) -> (u32, u32) {
        self.physical_size
    }

    pub fn change_scale_factor(&mut self, new_scale_factor: f64) {
        self.discard_preparation();
        self.scale_factor = new_scale_factor;
        self.resize(self.physical_size)
    }

    pub fn scale_factor(&self) -> f64 {
        self.scale_factor
    }

    pub fn set_fringe_width(&mut self, fringe_width: f32) {
        self.discard_preparation();
        self.fringe_width = fringe_width;
        self.resize(self.physical_size);
    }

    pub fn fringe_width(&self) -> f32 {
        self.fringe_width
    }

    pub fn resize(&mut self, new_physical_size: (u32, u32)) {
        self.discard_preparation();
        self.physical_size = new_physical_size;
        if new_physical_size.0 == 0 || new_physical_size.1 == 0 {
            return;
        }
        self.config.width = new_physical_size.0;
        self.config.height = new_physical_size.1;

        let logical_size = to_logical(new_physical_size, self.scale_factor);
        self.and_uniforms.canvas_size = [logical_size.0, logical_size.1];
        self.and_uniforms.scale_factor = self.scale_factor as f32;
        self.and_uniforms.fringe_width = self.fringe_width;

        self.decrementing_uniforms.canvas_size = [logical_size.0, logical_size.1];
        self.decrementing_uniforms.scale_factor = self.scale_factor as f32;
        self.decrementing_uniforms.fringe_width = self.fringe_width;

        self.queue.write_buffer(
            &self.and_uniform_buffer,
            0,
            bytemuck::cast_slice(&[self.and_uniforms]),
        );
        self.queue.write_buffer(
            &self.decrementing_uniform_buffer,
            0,
            bytemuck::cast_slice(&[self.decrementing_uniforms]),
        );

        if let Some(surface) = &self.surface {
            surface.configure(&self.device, &self.config);
        }
        self.recreate_msaa_texture();
        self.recreate_depth_stencil_texture();

        self.offscreen_texture_pool.trim(
            new_physical_size.0,
            new_physical_size.1,
            self.msaa_sample_count,
        );
    }

    pub fn msaa_samples(&self) -> u32 {
        self.msaa_sample_count
    }

    pub fn set_msaa_samples(&mut self, samples: u32) {
        let SampleCountChange::Changed(validated) =
            Self::sample_count_change(self.msaa_sample_count, samples)
        else {
            return;
        };

        self.discard_preparation();
        self.msaa_sample_count = validated;
        self.recreate_pipelines();
        self.recreate_msaa_texture();
        self.recreate_depth_stencil_texture();
    }

    pub(super) fn validate_sample_count_static(requested: u32) -> u32 {
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

    fn sample_count_change(current: u32, requested: u32) -> SampleCountChange {
        let validated = Self::validate_sample_count_static(requested);
        if validated == current {
            SampleCountChange::Unchanged
        } else {
            SampleCountChange::Changed(validated)
        }
    }

    pub(super) fn recreate_msaa_texture(&mut self) {
        if self.physical_size.0 == 0 || self.physical_size.1 == 0 {
            return;
        }
        if self.msaa_sample_count > 1 {
            let texture = create_msaa_color_texture(
                &self.device,
                self.physical_size,
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

        self.texture_blit_pipeline = None;
        self.stencil_only_pipeline = None;
        self.backdrop_color_pipeline = None;
        self.backdrop_color_gradient_pipeline = None;

        self.offscreen_texture_pool.trim(
            self.physical_size.0,
            self.physical_size.1,
            self.msaa_sample_count,
        );
    }

    /// Recreate the cached depth/stencil texture to match current physical size and MSAA settings.
    pub(super) fn recreate_depth_stencil_texture(&mut self) {
        if self.physical_size.0 == 0 || self.physical_size.1 == 0 {
            return;
        }
        let texture =
            create_and_depth_texture(&self.device, self.physical_size, self.msaa_sample_count);
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.depth_stencil_texture = Some(texture);
        self.depth_stencil_view = Some(view);
    }

    pub fn set_surface(&mut self, window: impl Into<SurfaceTarget<'static>>) {
        self.discard_preparation();
        let surface = self
            .instance
            .create_surface(window)
            .expect("Failed to create surface");
        surface.configure(&self.device, &self.config);
        self.surface = Some(surface);
    }

    pub fn set_vsync(&mut self, vsync: bool) {
        let PresentModeChange::Changed(present_mode) =
            Self::present_mode_change(self.config.present_mode, vsync)
        else {
            return;
        };
        self.discard_preparation();
        self.config.present_mode = present_mode;
        if let Some(surface) = &self.surface {
            surface.configure(&self.device, &self.config);
        }
        tracing::debug!(
            vsync,
            present_mode = ?self.config.present_mode,
            maximum_frame_latency = self.config.desired_maximum_frame_latency,
            "Surface presentation mode configured"
        );
    }

    fn present_mode_change(current: wgpu::PresentMode, vsync: bool) -> PresentModeChange {
        let requested = if vsync {
            wgpu::PresentMode::AutoVsync
        } else {
            wgpu::PresentMode::AutoNoVsync
        };
        if requested == current {
            PresentModeChange::Unchanged
        } else {
            PresentModeChange::Changed(requested)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{PresentModeChange, Renderer, SampleCountChange};

    #[test]
    fn equivalent_msaa_requests_do_not_require_renderer_mutation() {
        assert_eq!(
            Renderer::sample_count_change(1, 0),
            SampleCountChange::Unchanged
        );
        assert_eq!(
            Renderer::sample_count_change(4, 2),
            SampleCountChange::Unchanged
        );
        assert_eq!(
            Renderer::sample_count_change(1, 4),
            SampleCountChange::Changed(4)
        );
    }

    #[test]
    fn equivalent_vsync_requests_do_not_require_renderer_mutation() {
        assert_eq!(
            Renderer::present_mode_change(wgpu::PresentMode::AutoVsync, true),
            PresentModeChange::Unchanged
        );
        assert_eq!(
            Renderer::present_mode_change(wgpu::PresentMode::AutoNoVsync, false),
            PresentModeChange::Unchanged
        );
        assert_eq!(
            Renderer::present_mode_change(wgpu::PresentMode::AutoVsync, false),
            PresentModeChange::Changed(wgpu::PresentMode::AutoNoVsync)
        );
    }
}
