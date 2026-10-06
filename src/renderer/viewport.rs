use super::{RenderBackend, Renderer};
use crate::core::UnsignedPhysicalRect;
use crate::scene::SceneError;

impl<B: RenderBackend> Renderer<B> {
    pub fn size(&self) -> (u32, u32) {
        self.scene.viewport().physical_size
    }

    pub fn scale_factor(&self) -> f64 {
        self.scene.viewport().scale_factor
    }

    pub fn fringe_width(&self) -> f32 {
        self.scene.fringe_width()
    }

    /// Returns an error and preserves settings if attached shape effects cannot use the new scale.
    pub fn change_scale_factor(&mut self, new_scale_factor: f64) -> Result<(), SceneError> {
        self.update_raster_settings(new_scale_factor, self.scene.fringe_width())
    }

    /// Returns an error and preserves settings if attached shape effects cannot use the new fringe.
    pub fn set_fringe_width(&mut self, fringe_width: f32) -> Result<(), SceneError> {
        self.update_raster_settings(self.scene.viewport().scale_factor, fringe_width)
    }

    fn update_raster_settings(
        &mut self,
        scale_factor: f64,
        fringe_width: f32,
    ) -> Result<(), SceneError> {
        self.scene.update_raster_settings(
            scale_factor,
            fringe_width,
            self.backend.maximum_texture_dimension(),
        )?;
        self.update_backend_viewport();
        Ok(())
    }

    pub fn resize(&mut self, new_physical_size: (u32, u32)) {
        self.scene
            .resize(new_physical_size, self.backend.maximum_texture_dimension());
        self.update_backend_viewport();
    }

    fn update_backend_viewport(&mut self) {
        let viewport = self.scene.viewport();
        self.dirty_bounds = Some(UnsignedPhysicalRect::from_size(
            viewport.physical_size.into(),
        ));
        self.backend.resize(viewport, self.scene.fringe_width());
    }

    pub fn set_msaa_samples(&mut self, samples: u32) {
        self.backend.set_msaa_samples(samples);
        self.dirty_bounds = Some(UnsignedPhysicalRect::from_size(
            self.scene.viewport().physical_size.into(),
        ));
    }
}
