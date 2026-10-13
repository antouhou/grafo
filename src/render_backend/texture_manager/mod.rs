use ahash::HashSet;
use std::sync::{Arc, Mutex, Weak};

/// Textures are addressed by ID. Dimensions are `(width, height)` in pixels
pub trait TextureManager {
    type Error;

    /// Removes all textures and their cached bindings.
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
    /// The upload dimensions must fit inside the texture. RGB must be premultiplied by alpha in
    /// linear space, then encoded as sRGB. Use
    /// [`premultiply_rgba8_srgb_inplace`](crate::core::premultiply_rgba8_srgb_inplace)
    /// to convert straight-alpha input before uploading.
    ///
    /// Returns an error if the texture ID has not been allocated
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

type PendingTextureUpdates = Mutex<HashSet<u64>>;

/// An independent collection of changed source texture IDs.
pub(crate) struct TextureUpdateSubscription {
    pending: Arc<PendingTextureUpdates>,
}

impl TextureUpdateSubscription {
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
