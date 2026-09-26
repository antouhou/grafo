//! Contracts shared by the renderer coordinator and backend implementations.

use crate::commands::{RenderPlan, ShapeDrawId};
use crate::core::{CachedShapeHandle, ShapeInstance, Viewport};

/// Source textures addressed by ID. Dimensions are `(width, height)` in pixels.
pub trait TextureManager {
    type Error;

    /// Removes all source textures and their cached bindings.
    fn clear(&self);

    /// Allocates an RGBA8 sRGB texture, replacing any texture with the same ID.
    /// Upload pixels with [`Self::load_data_into_texture`].
    fn allocate_texture(&self, texture_id: u64, texture_dimensions: (u32, u32));

    /// Allocates and uploads a texture, replacing any texture with the same ID.
    /// See [`Self::load_data_into_texture`] for the required pixel format.
    fn allocate_texture_with_data(
        &self,
        texture_id: u64,
        texture_dimensions: (u32, u32),
        texture_data: &[u8],
    );

    /// Uploads RGBA8 sRGB pixels to the top-left corner of an allocated texture.
    ///
    /// The upload dimensions must fit inside the texture. Supply four bytes per pixel
    /// with no padding between rows. RGB must be premultiplied by alpha in linear space,
    /// then encoded as sRGB. Use
    /// [`premultiply_rgba8_srgb_inplace`](crate::core::premultiply_rgba8_srgb_inplace)
    /// to convert straight-alpha input before uploading.
    ///
    /// Returns an error if the texture ID has not been allocated.
    fn load_data_into_texture(
        &self,
        texture_id: u64,
        texture_dimensions: (u32, u32),
        texture_data: &[u8],
    ) -> Result<(), Self::Error>;

    /// Removes the texture identified by `texture_id` and its cached bindings.
    fn remove_texture(&self, texture_id: u64);

    /// Returns whether the ID has an allocated texture, even if no pixels were uploaded.
    fn is_texture_loaded(&self, texture_id: u64) -> bool;
}

/// Owns execution resources and interprets completed commands, without scene access.
/// Resource preparation happens while queuing; rendering consumes the finished plan.
pub trait RenderBackend<'surface> {
    /// Backend-defined output; it need not represent a window.
    type Surface;
    /// Error type shared by all fallible backend operations.
    type Error;
    type TextureManager: TextureManager<Error = Self::Error>;

    /// Prepares and registers resources for a borrowed CPU instance.
    /// An error must leave the ID unregistered.
    fn register_shape(&mut self, id: ShapeDrawId, shape: &ShapeInstance)
        -> Result<(), Self::Error>;
    /// Releases queued references, retaining reusable storage and resource caches.
    fn clear_draw_queue(&mut self);

    fn texture_manager(&self) -> &Self::TextureManager;

    fn maximum_texture_dimension(&self) -> u32;

    fn viewport(&self) -> Viewport;

    fn fringe_width(&self) -> f32;

    /// Returns true when sources changed and existing attachments must be removed.
    fn load_effect(&mut self, effect_id: u64, pass_sources: &[&str]) -> Result<bool, Self::Error>;

    fn validate_effect_params(&self, effect_id: u64, params: &[u8]) -> Result<(), Self::Error>;

    fn unload_effect(&mut self, effect_id: u64);

    fn invalidate_effect(&mut self, effect_id: u64);

    fn set_shape_effect_geometry(&mut self, id: ShapeDrawId, shape: &CachedShapeHandle);

    fn remove_shape_effect(&mut self, id: ShapeDrawId);

    fn remove_backdrop_effect(&mut self, id: ShapeDrawId);

    fn resize(&mut self, surface: &mut Self::Surface, viewport: Viewport, fringe_width: f32);

    fn set_msaa_samples(&mut self, samples: u32);
    /// Configures a replacement output using the current viewport and backend settings.
    fn configure_surface(&mut self, surface: &mut Self::Surface);

    fn set_vsync(&mut self, surface: &mut Self::Surface, vsync: bool);

    fn render(
        &mut self,
        commands: &RenderPlan,
        surface: &mut Self::Surface,
    ) -> Result<(), Self::Error>;

    fn render_to_buffer(
        &mut self,
        commands: &RenderPlan,
        buffer: &mut Vec<u8>,
    ) -> Result<(), Self::Error>;

    fn render_to_argb32(
        &mut self,
        commands: &RenderPlan,
        pixels: &mut [u32],
    ) -> Result<(), Self::Error>;
}
