use super::{Renderer, RendererContext};
use std::sync::Arc;
use wgpu::SurfaceTarget;

impl<'surface> Renderer<'surface> {
    /// Returns a context sharing this renderer's GPU resources and loaded shapes.
    pub fn context(&self) -> RendererContext {
        RendererContext {
            gpu: Arc::clone(&self.backend.context),
            loaded_shapes: Arc::clone(&self.planner.loaded_shapes),
        }
    }

    pub fn size(&self) -> (u32, u32) {
        self.viewport.physical_size
    }
    pub fn scale_factor(&self) -> f64 {
        self.viewport.scale_factor
    }
    pub fn fringe_width(&self) -> f32 {
        self.planner.fringe_width
    }

    pub fn change_scale_factor(&mut self, new_scale_factor: f64) {
        self.viewport.scale_factor = new_scale_factor;
        self.resize(self.viewport.physical_size);
    }

    pub fn set_fringe_width(&mut self, fringe_width: f32) {
        self.planner.fringe_width = fringe_width;
        self.resize(self.viewport.physical_size);
    }

    pub fn resize(&mut self, new_physical_size: (u32, u32)) {
        self.viewport.physical_size = new_physical_size;
        self.backend
            .resize(&mut self.surface, self.viewport, self.planner.fringe_width);
    }

    pub fn msaa_samples(&self) -> u32 {
        self.backend.msaa_samples()
    }
    pub fn set_msaa_samples(&mut self, samples: u32) {
        self.backend.set_msaa_samples(samples);
    }

    pub fn set_surface(&mut self, window: impl Into<SurfaceTarget<'static>>) {
        self.backend.set_surface(&mut self.surface, window);
    }

    pub fn set_vsync(&mut self, vsync: bool) {
        self.backend.set_vsync(&mut self.surface, vsync);
    }
}
