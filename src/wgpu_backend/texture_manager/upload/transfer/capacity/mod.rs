use super::{TextureTransferError, MAX_PENDING_UPLOAD_BATCHES};
use futures_util::FutureExt;
use std::pin::pin;
use std::sync::Arc;
use tokio::sync::{OwnedSemaphorePermit, Semaphore, SetOnce};
#[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
use tokio::task::coop;

pub(super) struct UploadCapacity {
    permits: Arc<Semaphore>,
    failure: Arc<SetOnce<TextureTransferError>>,
}

impl UploadCapacity {
    pub(super) fn new(failure: Arc<SetOnce<TextureTransferError>>) -> Arc<Self> {
        Arc::new(Self {
            permits: Arc::new(Semaphore::new(MAX_PENDING_UPLOAD_BATCHES)),
            failure,
        })
    }

    pub(super) async fn acquire_async(&self) -> Result<OwnedSemaphorePermit, TextureTransferError> {
        if let Some(error) = self.failure.get() {
            return Err(error.clone());
        }
        let acquire = Arc::clone(&self.permits).acquire_owned();
        // A blocking executor nested inside Tokio cannot yield to renew the task's budget.
        #[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
        let acquire = coop::unconstrained(acquire);
        let permit = acquire.await;
        match self.failure.get() {
            Some(error) => Err(error.clone()),
            None => Ok(permit.expect("upload capacity closes only after a failure")),
        }
    }

    pub(super) fn acquire(&self) -> Result<OwnedSemaphorePermit, TextureTransferError> {
        let mut acquire = pin!(self.acquire_async());
        // Only a pending acquisition needs a blocking waker.
        acquire
            .as_mut()
            .now_or_never()
            .unwrap_or_else(|| pollster::block_on(acquire))
    }

    pub(super) fn fail(&self, error: TextureTransferError) {
        let _ = self.failure.set(error);
        self.permits.close();
    }
}

#[cfg(test)]
mod tests;
