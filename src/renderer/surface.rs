use super::{RenderBackend, Renderer};

impl<'surface, B: RenderBackend<'surface>> Renderer<'surface, B> {
    pub fn size(&self) -> (u32, u32) {
        self.viewport.physical_size
    }
    pub fn scale_factor(&self) -> f64 {
        self.viewport.scale_factor
    }
    pub fn fringe_width(&self) -> f32 {
        self.fringe_width
    }

    pub fn change_scale_factor(&mut self, new_scale_factor: f64) {
        self.viewport.scale_factor = new_scale_factor;
        self.resize(self.viewport.physical_size);
    }

    pub fn set_fringe_width(&mut self, fringe_width: f32) {
        self.fringe_width = fringe_width;
        self.resize(self.viewport.physical_size);
    }

    pub fn resize(&mut self, new_physical_size: (u32, u32)) {
        self.viewport.physical_size = new_physical_size;
        self.backend
            .resize(&mut self.surface, self.viewport, self.fringe_width);
    }

    pub fn set_msaa_samples(&mut self, samples: u32) {
        self.backend.set_msaa_samples(samples);
    }

    /// Configures and replaces the backend output without changing the scene.
    pub fn set_surface(&mut self, mut surface: B::Surface) {
        self.backend.configure_surface(&mut surface);
        self.surface = surface;
    }

    pub fn set_vsync(&mut self, vsync: bool) {
        self.backend.set_vsync(&mut self.surface, vsync);
    }
}
