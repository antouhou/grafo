use super::{TextureManagerError, TextureTransferError};
use crate::wgpu_backend::WgpuBackendError;
use futures_util::future::{self, Either};
use std::collections::HashMap;
use std::future::Future;
use std::pin::{pin, Pin};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};
use tokio::sync::SetOnce;

#[derive(Default)]
struct CancellationState {
    is_cancelled: bool,
    waker: Option<Waker>,
}

#[derive(Default)]
struct TextureUploadCancellation {
    state: Mutex<CancellationState>,
}

impl TextureUploadCancellation {
    fn is_cancelled(&self) -> bool {
        self.state
            .lock()
            .expect("upload cancellation lock poisoned")
            .is_cancelled
    }

    fn cancel(&self) -> Option<Waker> {
        let mut state = self
            .state
            .lock()
            .expect("upload cancellation lock poisoned");
        state.is_cancelled = true;
        state.waker.take()
    }
}

type PendingUploads = HashMap<u64, Arc<TextureUploadCancellation>>;

#[derive(Clone, Default)]
pub(super) struct TextureUploadOperations {
    pending: Arc<Mutex<PendingUploads>>,
    failure: Arc<SetOnce<TextureTransferError>>,
}

pub(super) struct TextureUploadOperation {
    texture_id: u64,
    operations: TextureUploadOperations,
    cancellation: Arc<TextureUploadCancellation>,
    is_finished: bool,
}

pub(super) struct TextureUploadCancelled<'a> {
    cancellation: &'a TextureUploadCancellation,
}

impl TextureUploadOperations {
    pub(super) fn failure_signal(&self) -> Arc<SetOnce<TextureTransferError>> {
        Arc::clone(&self.failure)
    }

    pub(super) fn begin(
        &self,
        texture_id: u64,
    ) -> Result<TextureUploadOperation, TextureManagerError> {
        let mut pending = self.pending.lock().expect("pending uploads lock poisoned");
        if pending.contains_key(&texture_id) {
            return Err(TextureManagerError::TextureBusy(texture_id));
        }
        let cancellation = Arc::new(TextureUploadCancellation::default());
        pending.insert(texture_id, Arc::clone(&cancellation));
        Ok(TextureUploadOperation {
            texture_id,
            operations: self.clone(),
            cancellation,
            is_finished: false,
        })
    }

    pub(super) fn remove<T>(&self, texture_id: u64, remove: impl FnOnce() -> T) -> T {
        let mut pending = self.pending.lock().expect("pending uploads lock poisoned");
        let waker = pending
            .remove(&texture_id)
            .and_then(|operation| operation.cancel());
        let result = remove();
        drop(pending);
        if let Some(waker) = waker {
            waker.wake();
        }
        result
    }

    pub(super) fn clear<T>(&self, clear: impl FnOnce() -> T) -> T {
        let mut pending = self.pending.lock().expect("pending uploads lock poisoned");
        let wakers: Vec<_> = pending
            .drain()
            .filter_map(|(_, operation)| operation.cancel())
            .collect();
        let result = clear();
        drop(pending);
        for waker in wakers {
            waker.wake();
        }
        result
    }
}

impl TextureUploadOperation {
    pub(super) fn check(&self) -> Result<(), WgpuBackendError> {
        if let Some(error) = self.operations.failure.get() {
            Err(error.clone().into())
        } else if self.cancellation.is_cancelled() {
            Err(TextureManagerError::UploadCancelled.into())
        } else {
            Ok(())
        }
    }

    pub(super) fn cancelled(&self) -> TextureUploadCancelled<'_> {
        TextureUploadCancelled {
            cancellation: &self.cancellation,
        }
    }

    pub(super) async fn wait<T>(
        &self,
        work: impl Future<Output = T>,
    ) -> Result<T, WgpuBackendError> {
        self.check()?;
        let work = pin!(work);
        let cancelled = pin!(self.cancelled());
        let failed = pin!(self.operations.failure.wait());
        let interrupted = pin!(future::select(cancelled, failed));
        let result = future::select(work, interrupted).await;
        self.check()?;
        match result {
            Either::Left((result, _)) => Ok(result),
            Either::Right(_) => Err(TextureManagerError::UploadCancelled.into()),
        }
    }

    pub(super) fn commit<T>(
        mut self,
        commit: impl FnOnce() -> Result<T, WgpuBackendError>,
    ) -> Result<T, WgpuBackendError> {
        let mut pending = self
            .operations
            .pending
            .lock()
            .expect("pending uploads lock poisoned");
        self.check()?;
        let result = commit();
        pending.remove(&self.texture_id);
        self.is_finished = true;
        result
    }
}

impl Drop for TextureUploadOperation {
    fn drop(&mut self) {
        if self.is_finished {
            return;
        }
        let mut pending = self
            .operations
            .pending
            .lock()
            .expect("pending uploads lock poisoned");
        if !self.cancellation.is_cancelled() {
            pending.remove(&self.texture_id);
        }
    }
}

impl Future for TextureUploadCancelled<'_> {
    type Output = ();

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<()> {
        let mut state = self
            .cancellation
            .state
            .lock()
            .expect("upload cancellation lock poisoned");
        if state.is_cancelled {
            Poll::Ready(())
        } else {
            state.waker = Some(context.waker().clone());
            Poll::Pending
        }
    }
}

#[cfg(test)]
mod tests;
