//! Contracts shared by the scene coordinator and backend implementations.

use crate::commands::{RenderCommand, ShapeDrawId};
use crate::core::{CachedShapeHandle, ShapeInstance, Viewport};
use crate::UnsignedPhysicalRect;
use render_target::{RenderTarget, RenderTargetError};
use texture_manager::TextureManager;

pub mod render_target;
pub mod texture_manager;

/// Owns execution resources and interprets incoming commands.
/// Shapes are registered while queuing; rendering consumes commands in order.
pub trait RenderBackend {
    /// Backend resources stored in a `Surface`.
    type Surface;
    /// Error type shared by all fallible backend operations.
    type Error: From<RenderTargetError>;
    type TextureManager: TextureManager<Error = Self::Error>;

    /// Registers or replaces resources for a borrowed CPU instance.
    /// An error must preserve the previous registration, if any.
    fn register_shape(&mut self, id: ShapeDrawId, shape: &ShapeInstance)
        -> Result<(), Self::Error>;
    /// Updates texture coordinates using current texture dimensions and viewport scale.
    /// `shape` must describe the registered instance. Preserves geometry and attached effects.
    fn refresh_uv(&mut self, id: ShapeDrawId, shape: &ShapeInstance);
    /// Releases the given instances and their effect bindings.
    /// Preserve other instances and resource caches.
    fn unregister_shapes(&mut self, ids: &[ShapeDrawId]);
    /// Releases queued references, retaining reusable storage and resource caches.
    fn clear(&mut self);

    fn texture_manager(&self) -> &Self::TextureManager;

    fn maximum_texture_dimension(&self) -> u32;

    fn viewport(&self) -> Viewport;

    /// Returns true when sources changed and existing attachments must be removed.
    fn load_effect(&mut self, effect_id: u64, pass_sources: &[&str]) -> Result<bool, Self::Error>;

    fn validate_effect_params(&self, effect_id: u64, params: &[u8]) -> Result<(), Self::Error>;

    fn unload_effect(&mut self, effect_id: u64);

    fn invalidate_effect(&mut self, effect_id: u64);

    fn set_shape_mask_effect_geometry(&mut self, id: ShapeDrawId, shape: &CachedShapeHandle);

    fn remove_shape_mask_effect(&mut self, id: ShapeDrawId);

    fn remove_backdrop_effect(&mut self, id: ShapeDrawId);

    fn resize(&mut self, viewport: Viewport);

    fn set_msaa_samples(&mut self, samples: u32);

    /// Controls the transient red redraw overlay on surfaces. Disabled by default.
    /// This setting must not invalidate the retained scene or affect pixmap pixels.
    /// The default implementation ignores this setting.
    fn set_dirty_region_overlay_enabled(&mut self, _enabled: bool) {}

    /// Returns whether surface redraw bounds are highlighted. Defaults to false.
    fn is_dirty_region_overlay_enabled(&self) -> bool {
        false
    }

    /// Renders `commands` at the target's dimensions. The commands must have been
    /// planned for those dimensions.
    /// Preserve the previous image outside `root_scissor`. Clear and redraw
    /// inside it; None leaves the image unchanged. A newly allocated output needs a
    /// full redraw, for which the iterator still yields the entire scene.
    ///
    /// Pixmap pixels must be ready on success and remain unchanged on error.
    /// Surface rendering must submit and present the frame.
    fn render<'scene>(
        &mut self,
        root_scissor: Option<UnsignedPhysicalRect>,
        commands: impl Iterator<Item = RenderCommand<'scene>>,
        target: RenderTarget<'_, Self::Surface>,
    ) -> Result<(), Self::Error>;
}
