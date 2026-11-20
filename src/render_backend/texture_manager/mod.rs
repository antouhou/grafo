use ahash::HashSet;
pub use pixels::{AlphaMode, RgbEncoding, TexturePixelFormat, TexturePixels, TextureStorageFormat};
use std::sync::{Arc, Mutex, Weak};

mod pixels;

/// Textures are addressed by ID. Dimensions are `(width, height)` in pixels.
/// Use the storage formats, source pixel layouts, and write origins supported by the backend.
/// Allocation returns an error for invalid dimensions or source layouts and preserves
/// any existing texture at that ID on failure.
pub trait TextureManager {
    type Error;

    /// Removes all textures and their cached bindings.
    fn clear(&self);

    /// Allocates a texture of the given size and storage format, replacing the entry at this ID.
    fn allocate_texture_with_format(
        &self,
        texture_id: u64,
        size: (u32, u32),
        storage: TextureStorageFormat,
    ) -> Result<(), Self::Error>;

    /// Allocates a texture the size of `pixels` and initializes it, replacing the entry at this ID.
    fn allocate_texture_with_pixels(
        &self,
        texture_id: u64,
        storage: TextureStorageFormat,
        pixels: TexturePixels<'_>,
    ) -> Result<(), Self::Error>;

    /// Allocates an RGBA8 sRGB texture, replacing the entry at this ID.
    fn allocate_texture(&self, texture_id: u64, size: (u32, u32)) -> Result<(), Self::Error> {
        self.allocate_texture_with_format(texture_id, size, TextureStorageFormat::Rgba8Srgb)
    }

    /// Allocates an RGBA8 sRGB texture and fills it from tightly packed rows.
    /// RGB must be premultiplied in linear space, then encoded as sRGB. For straight-alpha
    /// input, use [`premultiply_rgba8_srgb_inplace`](crate::core::color::premultiply_rgba8_srgb_inplace).
    fn allocate_texture_with_data(
        &self,
        texture_id: u64,
        size: (u32, u32),
        bytes: &[u8],
    ) -> Result<(), Self::Error> {
        let pixels = TexturePixels::rgba8_srgb(bytes, size, AlphaMode::PremultipliedInLinearSpace);
        self.allocate_texture_with_pixels(texture_id, TextureStorageFormat::Rgba8Srgb, pixels)
    }

    /// Writes `pixels` at the given top-left destination in pixel coordinates.
    /// The rectangle must fit inside the allocated texture.
    fn load_data_into_texture(
        &self,
        texture_id: u64,
        origin: (u32, u32),
        pixels: TexturePixels<'_>,
    ) -> Result<(), Self::Error>;

    /// Removes the texture identified by `texture_id` and its cached bindings.
    fn remove_texture(&self, texture_id: u64);

    /// Returns whether the ID has an allocated texture, even if no pixels were uploaded.
    fn is_texture_loaded(&self, texture_id: u64) -> bool;
}

type PendingTextureUpdates = Mutex<HashSet<u64>>;

/// An independent collection of changed source texture IDs.
pub(crate) struct TextureUpdateSubscription {
    pending: Arc<PendingTextureUpdates>,
}

impl TextureUpdateSubscription {
    pub(crate) fn clear_caches(&self) {
        self.pending
            .lock()
            .expect("pending texture updates lock poisoned")
            .shrink_to_fit();
    }

    /// Replaces `texture_ids` with pending changes, consuming them from this subscription.
    /// Repeated changes to an ID are coalesced. ID order is unspecified.
    pub(crate) fn drain_into(&mut self, texture_ids: &mut Vec<u64>) {
        texture_ids.clear();
        texture_ids.extend(
            self.pending
                .lock()
                .expect("pending texture updates lock poisoned")
                .drain(),
        );
    }
}

/// Delivers source texture changes to subscribers.
/// Cloned publishers share subscriptions; dropping a subscription releases its pending changes.
#[derive(Clone, Default)]
pub(crate) struct TextureUpdatePublisher {
    subscriptions: Arc<Mutex<Vec<Weak<PendingTextureUpdates>>>>,
}

impl TextureUpdatePublisher {
    pub(crate) fn clear_caches(&self) {
        let mut subscriptions = self
            .subscriptions
            .lock()
            .expect("texture subscriptions lock poisoned");
        subscriptions.retain(|subscription| subscription.strong_count() != 0);
        subscriptions.shrink_to_fit();
        for pending in subscriptions.iter().filter_map(Weak::upgrade) {
            pending
                .lock()
                .expect("pending texture updates lock poisoned")
                .shrink_to_fit();
        }
    }

    /// Subscribes to changes published after this call
    pub(crate) fn subscribe(&self) -> TextureUpdateSubscription {
        let pending = Arc::new(Mutex::new(HashSet::default()));
        let mut subscriptions = self
            .subscriptions
            .lock()
            .expect("texture subscriptions lock poisoned");
        subscriptions.retain(|subscription| subscription.strong_count() != 0);
        subscriptions.push(Arc::downgrade(&pending));
        TextureUpdateSubscription { pending }
    }

    /// Records a change for every current subscriber
    pub(crate) fn publish(&self, texture_id: u64) {
        self.subscriptions
            .lock()
            .expect("texture subscriptions lock poisoned")
            .retain(|subscription| {
                let Some(pending) = subscription.upgrade() else {
                    return false;
                };
                pending
                    .lock()
                    .expect("pending texture updates lock poisoned")
                    .insert(texture_id);
                true
            });
    }
}

#[cfg(test)]
mod tests;
