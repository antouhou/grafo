use super::{RenderBackend, Renderer};
use crate::core::UnsignedPhysicalRect;
use crate::scene::SceneError;

impl<B: RenderBackend> Renderer<B> {
    pub fn size(&self) -> (u32, u32) {
        self.viewport.physical_size
    }

    pub fn scale_factor(&self) -> f64 {
        self.viewport.scale_factor
    }

    pub fn fringe_width(&self) -> f32 {
        self.fringe_width
    }

    /// Returns an error and preserves settings if attached shape effects cannot use the new scale.
    pub fn change_scale_factor(&mut self, new_scale_factor: f64) -> Result<(), SceneError> {
        if let Err(error) = self
            .scene
            .refresh_shape_effect_bounds(new_scale_factor, self.fringe_width)
        {
            self.scene
                .refresh_shape_effect_bounds(self.viewport.scale_factor, self.fringe_width)?;
            return Err(error);
        }
        self.viewport.scale_factor = new_scale_factor;
        self.resize(self.viewport.physical_size);
        Ok(())
    }

    /// Returns an error and preserves settings if attached shape effects cannot use the new fringe.
    pub fn set_fringe_width(&mut self, fringe_width: f32) -> Result<(), SceneError> {
        if let Err(error) = self
            .scene
            .refresh_shape_effect_bounds(self.viewport.scale_factor, fringe_width)
        {
            self.scene
                .refresh_shape_effect_bounds(self.viewport.scale_factor, self.fringe_width)?;
            return Err(error);
        }
        self.fringe_width = fringe_width;
        self.resize(self.viewport.physical_size);
        Ok(())
    }

    pub fn resize(&mut self, new_physical_size: (u32, u32)) {
        self.viewport.physical_size = new_physical_size;
        self.dirty_bounds = Some(UnsignedPhysicalRect::from_size(new_physical_size.into()));
        self.backend.resize(self.viewport, self.fringe_width);
        self.scene.refresh_backdrop_capture_regions(
            self.viewport,
            self.fringe_width,
            self.backend.maximum_texture_dimension(),
        );
    }

    pub fn set_msaa_samples(&mut self, samples: u32) {
        self.backend.set_msaa_samples(samples);
        self.dirty_bounds = Some(UnsignedPhysicalRect::from_size(
            self.viewport.physical_size.into(),
        ));
    }
}
