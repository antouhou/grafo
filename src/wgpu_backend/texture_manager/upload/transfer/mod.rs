use super::TextureUploadBatch;
use capacity::UploadCapacity;
use std::io;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use thiserror::Error;
use tokio::sync::{OwnedSemaphorePermit, SetOnce};
use wgpu::{
    Device, Extent3d, Origin3d, PollError, PollType, Queue, SubmissionIndex, TexelCopyBufferLayout,
    Texture,
};

const MAX_PENDING_UPLOAD_BATCHES: usize = 2;

mod capacity;

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

struct PendingUploadCompletion {
    submission: SubmissionIndex,
    capacity_permit: OwnedSemaphorePermit,
}

struct UploadCompletionWorker {
    device: Arc<Device>,
    device_lost: Arc<AtomicBool>,
    capacity: Arc<UploadCapacity>,
    receiver: Receiver<PendingUploadCompletion>,
}

impl UploadCompletionWorker {
    fn run(self) {
        while let Ok(pending) = self.receiver.recv() {
            let result = self.device.poll(PollType::Wait {
                submission_index: Some(pending.submission),
                timeout: None,
            });
            if self.device_lost.load(Ordering::Acquire) {
                self.capacity.fail(TextureTransferError::DeviceLost);
                break;
            }
            if let Err(error) = result {
                self.capacity
                    .fail(TextureTransferError::GpuWait(Arc::new(error)));
                break;
            }
            drop(pending.capacity_permit);
        }
    }
}

impl Drop for UploadCompletionWorker {
    fn drop(&mut self) {
        self.capacity.fail(TextureTransferError::WorkerStopped);
    }
}

pub(in super::super) struct TextureUploadQueue {
    device: Arc<Device>,
    queue: Arc<Queue>,
    device_lost: Arc<AtomicBool>,
    capacity: Arc<UploadCapacity>,
    completion_sender: Mutex<Option<Sender<PendingUploadCompletion>>>,
}

impl TextureUploadQueue {
    pub(in super::super) fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        device_lost: Arc<AtomicBool>,
        failure_signal: Arc<SetOnce<TextureTransferError>>,
    ) -> Self {
        Self {
            device,
            queue,
            device_lost,
            capacity: UploadCapacity::new(failure_signal),
            completion_sender: Mutex::new(None),
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

    fn completion_sender(&self) -> Result<Sender<PendingUploadCompletion>, TextureTransferError> {
        if self.device_lost.load(Ordering::Acquire) {
            self.capacity.fail(TextureTransferError::DeviceLost);
            return Err(TextureTransferError::DeviceLost);
        }
        let mut sender = self
            .completion_sender
            .lock()
            .expect("upload completion sender lock poisoned");
        if let Some(sender) = sender.as_ref() {
            return Ok(sender.clone());
        }

        let (new_sender, receiver) = mpsc::channel();
        let device = Arc::clone(&self.device);
        let device_lost = Arc::clone(&self.device_lost);
        let capacity = Arc::clone(&self.capacity);
        if let Err(error) = thread::Builder::new()
            .name("grafo-texture-uploads".into())
            .spawn(move || {
                UploadCompletionWorker {
                    device,
                    device_lost,
                    capacity,
                    receiver,
                }
                .run();
            })
        {
            let error = TextureTransferError::WorkerStart(Arc::new(error));
            self.capacity.fail(error.clone());
            return Err(error);
        }
        *sender = Some(new_sender.clone());
        Ok(new_sender)
    }

    fn submit_batch(
        &self,
        texture: &Texture,
        origin: (u32, u32),
        batch: TextureUploadBatch<'_>,
        permit: OwnedSemaphorePermit,
        sender: Sender<PendingUploadCompletion>,
    ) -> Result<(), TextureTransferError> {
        if self.device_lost.load(Ordering::Acquire) {
            self.capacity.fail(TextureTransferError::DeviceLost);
            return Err(TextureTransferError::DeviceLost);
        }
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
        let submission = self.queue.submit([]);
        match sender.send(PendingUploadCompletion {
            submission,
            capacity_permit: permit,
        }) {
            Ok(()) => Ok(()),
            Err(pending) => {
                self.capacity.fail(TextureTransferError::WorkerStopped);
                drop(pending);
                Err(TextureTransferError::WorkerStopped)
            }
        }
    }

    pub(in super::super) fn write(
        &self,
        texture: &Texture,
        origin: (u32, u32),
        batch: TextureUploadBatch<'_>,
    ) -> Result<(), TextureTransferError> {
        let sender = self.completion_sender()?;
        let permit = self.capacity.acquire()?;
        self.submit_batch(texture, origin, batch, permit, sender)
    }

    pub(in super::super) async fn write_async(
        &self,
        texture: &Texture,
        origin: (u32, u32),
        batch: TextureUploadBatch<'_>,
    ) -> Result<(), TextureTransferError> {
        let sender = self.completion_sender()?;
        let permit = self.capacity.acquire_async().await?;
        self.submit_batch(texture, origin, batch, permit, sender)
    }
}
