use super::super::capacity::UploadCapacity;
use super::super::TextureTransferError;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use tokio::sync::OwnedSemaphorePermit;
use wgpu::{Device, PollType, Queue, SubmissionIndex};

struct PendingUploadCompletion {
    submission: SubmissionIndex,
    capacity_permit: OwnedSemaphorePermit,
}

pub(in super::super) struct UploadReservation {
    sender: Sender<PendingUploadCompletion>,
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

pub(in super::super) struct TextureUploadCompletion {
    device: Arc<Device>,
    device_lost: Arc<AtomicBool>,
    capacity: Arc<UploadCapacity>,
    completion_sender: Mutex<Option<Sender<PendingUploadCompletion>>>,
}

impl TextureUploadCompletion {
    pub(in super::super) fn new(
        device: Arc<Device>,
        _queue: Arc<Queue>,
        device_lost: Arc<AtomicBool>,
        capacity: Arc<UploadCapacity>,
    ) -> Self {
        Self {
            device,
            device_lost,
            capacity,
            completion_sender: Mutex::new(None),
        }
    }

    fn completion_sender(&self) -> Result<Sender<PendingUploadCompletion>, TextureTransferError> {
        if self.device_lost.load(Ordering::Acquire) {
            self.capacity.fail(TextureTransferError::DeviceLost);
            return Err(TextureTransferError::DeviceLost);
        }
        self.capacity.check()?;
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

    pub(in super::super) fn reserve(&self) -> Result<UploadReservation, TextureTransferError> {
        let sender = self.completion_sender()?;
        let capacity_permit = self.capacity.acquire()?;
        Ok(UploadReservation {
            sender,
            capacity_permit,
        })
    }

    pub(in super::super) async fn reserve_async(
        &self,
    ) -> Result<UploadReservation, TextureTransferError> {
        let sender = self.completion_sender()?;
        let capacity_permit = self.capacity.acquire_async().await?;
        Ok(UploadReservation {
            sender,
            capacity_permit,
        })
    }

    pub(in super::super) fn submitted(
        &self,
        submission: SubmissionIndex,
        reservation: UploadReservation,
    ) -> Result<(), TextureTransferError> {
        match reservation.sender.send(PendingUploadCompletion {
            submission,
            capacity_permit: reservation.capacity_permit,
        }) {
            Ok(()) => Ok(()),
            Err(pending) => {
                self.capacity.fail(TextureTransferError::WorkerStopped);
                drop(pending);
                Err(TextureTransferError::WorkerStopped)
            }
        }
    }
}
