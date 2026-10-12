use ahash::HashSet;
use std::sync::{Arc, Mutex, Weak};

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

/// Delivers source texture changes to independent subscribers.
/// Cloned publishers share subscriptions; dropping a subscription releases its pending changes.
#[derive(Clone, Default)]
pub(crate) struct TextureUpdatePublisher {
    subscriptions: Arc<Mutex<Vec<Weak<PendingTextureUpdates>>>>,
}

impl TextureUpdatePublisher {
    /// Subscribes to changes published after this call.
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

    /// Records a change for every current subscriber.
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
