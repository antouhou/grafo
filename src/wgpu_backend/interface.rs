use super::texture_manager::WgpuTextureManager;
use super::{WgpuBackend, WgpuBackendError, WgpuSurface};
use crate::commands::{RenderPlan, ShapeDrawId};
use crate::core::shape::{CachedShapeHandle, ShapeInstance};
use crate::core::Viewport;
use crate::render_backend::render_target::{PixelFormat, RenderTarget};
use crate::render_backend::RenderBackend;
use std::sync::Arc;
use wgpu::TextureFormat;

impl RenderBackend for WgpuBackend {
    type Surface = WgpuSurface;
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
            .load(&self.device, self.format, effect_id, pass_sources)
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

    fn resize(&mut self, viewport: Viewport, fringe_width: f32) {
        self.resize(viewport, fringe_width);
    }

    fn set_msaa_samples(&mut self, samples: u32) {
        self.set_msaa_samples(samples);
    }

    fn render(
        &mut self,
        commands: &RenderPlan,
        target: RenderTarget<'_, Self::Surface>,
    ) -> Result<(), Self::Error> {
        let size = target.validate_size(self.maximum_texture_dimension())?;
        if self.viewport.physical_size != size {
            self.resize(
                Viewport {
                    physical_size: size,
                    ..self.viewport
                },
                self.fringe_width,
            );
        }
        match target {
            RenderTarget::Surface(surface) => {
                self.prepare_surface(surface)?;
                self.render_surface(commands, &surface.resource().surface)
                    .map_err(WgpuBackendError::Surface)
            }
            RenderTarget::Pixmap(mut pixels) => {
                let format = match pixels.layout().format() {
                    PixelFormat::Bgra8 | PixelFormat::Argb32 => TextureFormat::Bgra8UnormSrgb,
                    PixelFormat::Rgba8 => TextureFormat::Rgba8UnormSrgb,
                };
                self.set_format(format);
                if pixels.layout().format() == PixelFormat::Argb32 {
                    self.render_argb_pixels(commands, &mut pixels)?;
                } else {
                    self.render_byte_pixels(commands, &mut pixels)?;
                }
                Ok(())
            }
        }
    }
}
