use super::texture_manager::WgpuTextureManager;
use super::{WgpuBackend, WgpuBackendError};
use crate::commands::{RenderPlan, ShapeDrawId};
use crate::core::shape::{CachedShapeHandle, ShapeInstance};
use crate::core::Viewport;
use crate::render_backend::RenderBackend;
use std::sync::Arc;

impl<'surface> RenderBackend<'surface> for WgpuBackend {
    type Surface = Option<wgpu::Surface<'surface>>;
    type Error = WgpuBackendError;
    type TextureManager = WgpuTextureManager;

    fn register_shape(
        &mut self,
        id: ShapeDrawId,
        shape: &ShapeInstance,
    ) -> Result<(), Self::Error> {
        let resources = self.resources.shape_execution.prepare_draw(
            shape,
            &self.device,
            &self.queue,
            &self.pipeline_resources.shapes,
            self.viewport.scale_factor,
        )?;
        self.resources.shape_execution.draws.insert(id.0, resources);
        Ok(())
    }

    fn clear_draw_queue(&mut self) {
        self.resources.shape_execution.clear_draw_queue();
    }

    fn texture_manager(&self) -> &Self::TextureManager {
        self.texture_manager()
    }

    fn maximum_texture_dimension(&self) -> u32 {
        self.device.limits().max_texture_dimension_2d
    }

    fn viewport(&self) -> Viewport {
        self.viewport
    }

    fn fringe_width(&self) -> f32 {
        self.fringe_width
    }

    fn load_effect(&mut self, effect_id: u64, pass_sources: &[&str]) -> Result<bool, Self::Error> {
        self.effect_registry
            .load(&self.device, self.config.format, effect_id, pass_sources)
            .map_err(WgpuBackendError::Effect)
    }

    fn validate_effect_params(&self, effect_id: u64, params: &[u8]) -> Result<(), Self::Error> {
        self.effect_registry
            .validate_params(effect_id, params)
            .map_err(WgpuBackendError::Effect)
    }

    fn unload_effect(&mut self, effect_id: u64) {
        self.effect_registry.unload(effect_id);
    }

    fn invalidate_effect(&mut self, effect_id: u64) {
        self.resources.textures.invalidate_shape_effect(effect_id);
    }

    fn set_shape_effect_geometry(&mut self, id: ShapeDrawId, shape: &CachedShapeHandle) {
        self.resources
            .shape_execution
            .draws
            .get_mut(&id.0)
            .expect("shape registered before effects are attached")
            .mask_tessellation
            .get_or_insert_with(|| Arc::clone(&shape.tessellation));
    }

    fn remove_shape_effect(&mut self, id: ShapeDrawId) {
        if let Some(resources) = self.resources.shape_execution.draws.get_mut(&id.0) {
            resources.mask_tessellation = None;
        }
    }

    fn remove_backdrop_effect(&mut self, id: ShapeDrawId) {
        if let Some(resources) = self.resources.shape_execution.draws.get_mut(&id.0) {
            resources.clear_under_fill_binding();
        }
    }

    fn resize(&mut self, surface: &mut Self::Surface, viewport: Viewport, fringe_width: f32) {
        self.resize(surface, viewport, fringe_width);
    }

    fn set_msaa_samples(&mut self, samples: u32) {
        self.set_msaa_samples(samples);
    }

    fn configure_surface(&mut self, surface: &mut Self::Surface) {
        Self::configure_surface(self, surface);
    }

    fn set_vsync(&mut self, surface: &mut Self::Surface, vsync: bool) {
        self.set_vsync(surface, vsync);
    }

    fn render(
        &mut self,
        commands: &RenderPlan,
        surface: &mut Self::Surface,
    ) -> Result<(), Self::Error> {
        self.render(commands, surface)
            .map_err(WgpuBackendError::Surface)
    }

    fn render_to_buffer(
        &mut self,
        commands: &RenderPlan,
        buffer: &mut Vec<u8>,
    ) -> Result<(), Self::Error> {
        self.render_to_buffer(commands, buffer)
            .map_err(WgpuBackendError::Readback)
    }

    fn render_to_argb32(
        &mut self,
        commands: &RenderPlan,
        pixels: &mut [u32],
    ) -> Result<(), Self::Error> {
        self.render_to_argb32(commands, pixels)
            .map_err(WgpuBackendError::Readback)
    }
}
