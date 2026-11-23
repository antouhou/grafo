use super::TextureUploadBatch;
use capacity::UploadCapacity;
use completion::TextureUploadCompletion;
use std::io;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::SetOnce;
use wgpu::{
    Device, Extent3d, Origin3d, PollError, Queue, SubmissionIndex, TexelCopyBufferLayout, Texture,
};

const MAX_PENDING_UPLOAD_BATCHES: usize = 2;

mod capacity;
mod completion;

#[derive(Clone, Debug, Error)]
pub enum TextureTransferError {
    #[error("GPU device lost during texture upload")]
    DeviceLost,
    #[error("Texture upload completion worker stopped")]
    WorkerStopped,
    #[error("Could not start the texture upload completion worker: {0}")]
    WorkerStart(#[source] Arc<io::Error>),
    #[error("Could not wait for a texture upload: {0}")]
    GpuWait(#[source] Arc<PollError>),
}

pub(in super::super) struct TextureUploadQueue {
    queue: Arc<Queue>,
    device_lost: Arc<AtomicBool>,
    capacity: Arc<UploadCapacity>,
    completion: TextureUploadCompletion,
}

impl TextureUploadQueue {
    pub(in super::super) fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        device_lost: Arc<AtomicBool>,
        failure_signal: Arc<SetOnce<TextureTransferError>>,
    ) -> Self {
        let capacity = UploadCapacity::new(failure_signal);
        let completion = TextureUploadCompletion::new(
            device,
            Arc::clone(&queue),
            Arc::clone(&device_lost),
            Arc::clone(&capacity),
        );
        Self {
            queue,
            device_lost,
            capacity,
            completion,
        }
    }

    pub(in crate::wgpu_backend) fn device_loss_notifier(
        &self,
    ) -> impl Fn() + Send + Sync + 'static {
        let capacity = Arc::downgrade(&self.capacity);
        move || {
            if let Some(capacity) = capacity.upgrade() {
                capacity.fail(TextureTransferError::DeviceLost);
            }
        }
    }

    fn submit_batch(
        &self,
        texture: &Texture,
        origin: (u32, u32),
        batch: TextureUploadBatch<'_>,
    ) -> Result<SubmissionIndex, TextureTransferError> {
        if self.device_lost.load(Ordering::Acquire) {
            self.capacity.fail(TextureTransferError::DeviceLost);
            return Err(TextureTransferError::DeviceLost);
        }
        self.capacity.check()?;
        let mut destination = texture.as_image_copy();
        destination.origin = Origin3d {
            x: origin.0 + batch.origin.0,
            y: origin.1 + batch.origin.1,
            z: 0,
        };
        self.queue.write_texture(
            destination,
            batch.bytes,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(batch.bytes_per_row),
                rows_per_image: None,
            },
            Extent3d {
                width: batch.size.0,
                height: batch.size.1,
                depth_or_array_layers: 1,
            },
        );
        Ok(self.queue.submit([]))
    }

    pub(in super::super) fn write(
        &self,
        texture: &Texture,
        origin: (u32, u32),
        batch: TextureUploadBatch<'_>,
    ) -> Result<(), TextureTransferError> {
        let reservation = self.completion.reserve()?;
        let submission = self.submit_batch(texture, origin, batch)?;
        self.completion.submitted(submission, reservation)
    }

    pub(in super::super) async fn write_async(
        &self,
        texture: &Texture,
        origin: (u32, u32),
        batch: TextureUploadBatch<'_>,
    ) -> Result<(), TextureTransferError> {
        let reservation = self.completion.reserve_async().await?;
        let submission = self.submit_batch(texture, origin, batch)?;
        self.completion.submitted(submission, reservation)
    }
}
