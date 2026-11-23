use super::{
    BindGroupCache, ManagedTexture, TextureManagerError, TextureTransferError, WgpuTextureManager,
};
use crate::render_backend::texture_manager::TextureUpdatePublisher;
use crate::wgpu_backend::WgpuBackendError;
use futures_util::future::{self, Either};
use std::collections::HashMap;
use std::future::Future;
use std::pin::pin;
use std::sync::{Arc, Mutex, RwLock};
use std::task::{Poll, Waker};
use tokio::sync::SetOnce;
use wgpu::{CommandEncoderDescriptor, Extent3d, Origin3d, Texture};

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

    pub(super) fn remove(
        &self,
        texture_id: u64,
        texture_storage: &RwLock<HashMap<u64, ManagedTexture>>,
        bind_group_cache: &RwLock<BindGroupCache>,
        updates: &TextureUpdatePublisher,
    ) {
        let mut pending = self.pending.lock().expect("pending uploads lock poisoned");
        let waker = pending
            .remove(&texture_id)
            .and_then(|operation| operation.cancel());
        {
            let mut texture_storage = texture_storage
                .write()
                .expect("texture storage lock poisoned");
            let mut bind_group_cache = bind_group_cache
                .write()
                .expect("shape bind-group cache lock poisoned");
            bind_group_cache.remove(&texture_id);
            if texture_storage.remove(&texture_id).is_some() {
                updates.publish(texture_id);
            }
        }
        drop(pending);
        if let Some(waker) = waker {
            waker.wake();
        }
    }

    pub(super) fn clear(
        &self,
        texture_storage: &RwLock<HashMap<u64, ManagedTexture>>,
        bind_group_cache: &RwLock<BindGroupCache>,
        updates: &TextureUpdatePublisher,
    ) {
        let mut pending = self.pending.lock().expect("pending uploads lock poisoned");
        let wakers: Vec<_> = pending
            .drain()
            .filter_map(|(_, operation)| operation.cancel())
            .collect();
        {
            let mut texture_storage = texture_storage
                .write()
                .expect("texture storage lock poisoned");
            let mut bind_group_cache = bind_group_cache
                .write()
                .expect("shape bind-group cache lock poisoned");
            for &texture_id in texture_storage.keys() {
                updates.publish(texture_id);
            }
            texture_storage.clear();
            bind_group_cache.clear();
        }
        drop(pending);
        for waker in wakers {
            waker.wake();
        }
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

    pub(super) fn cancelled(&self) -> impl Future<Output = ()> + '_ {
        future::poll_fn(|context| {
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
        })
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

    pub(super) fn commit_allocation(
        mut self,
        manager: &WgpuTextureManager,
        texture: ManagedTexture,
    ) -> Result<(), WgpuBackendError> {
        let mut pending = self
            .operations
            .pending
            .lock()
            .expect("pending uploads lock poisoned");
        self.check()?;
        manager.ensure_available()?;
        let mut texture_storage = manager.texture_storage.write().unwrap();
        let mut bind_group_cache = manager.shape_bind_group_cache.write().unwrap();
        // Invalidate old bindings while both locks exclude concurrent cache insertion.
        bind_group_cache.remove(&self.texture_id);
        texture_storage.insert(self.texture_id, texture);
        manager.updates.publish(self.texture_id);
        pending.remove(&self.texture_id);
        self.is_finished = true;
        Ok(())
    }

    pub(super) fn commit_update(
        mut self,
        manager: &WgpuTextureManager,
        destination: &Texture,
        patch: &Texture,
        origin: (u32, u32),
    ) -> Result<(), WgpuBackendError> {
        self.check()?;
        manager.ensure_available()?;
        // GPU callbacks can cancel other uploads, so submission must run without the pending lock.
        let mut encoder = manager
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("commit_texture_upload"),
            });
        let mut target = destination.as_image_copy();
        target.origin = Origin3d {
            x: origin.0,
            y: origin.1,
            z: 0,
        };
        encoder.copy_texture_to_texture(
            patch.as_image_copy(),
            target,
            Extent3d {
                width: patch.width(),
                height: patch.height(),
                depth_or_array_layers: 1,
            },
        );
        manager.queue.submit([encoder.finish()]);

        let mut pending = self
            .operations
            .pending
            .lock()
            .expect("pending uploads lock poisoned");
        self.check()?;
        manager.ensure_available()?;
        manager.updates.publish(self.texture_id);
        pending.remove(&self.texture_id);
        self.is_finished = true;
        Ok(())
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

#[cfg(test)]
mod tests;
