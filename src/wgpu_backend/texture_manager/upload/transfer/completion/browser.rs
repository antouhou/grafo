use super::super::capacity::UploadCapacity;
use super::super::TextureTransferError;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use tokio::sync::OwnedSemaphorePermit;
use wgpu::{Device, Queue, SubmissionIndex};

pub(in super::super) struct TextureUploadCompletion {
    queue: Arc<Queue>,
    capacity: Arc<UploadCapacity>,
}

impl TextureUploadCompletion {
    pub(in super::super) fn new(
        _device: Arc<Device>,
        queue: Arc<Queue>,
        _device_lost: Arc<AtomicBool>,
        capacity: Arc<UploadCapacity>,
    ) -> Self {
        Self { queue, capacity }
    }

    pub(in super::super) fn reserve(
        &self,
    ) -> Result<Option<OwnedSemaphorePermit>, TextureTransferError> {
        self.capacity.check()?;
        Ok(None)
    }

    pub(in super::super) async fn reserve_async(
        &self,
    ) -> Result<Option<OwnedSemaphorePermit>, TextureTransferError> {
        self.capacity.acquire_async().await.map(Some)
    }

    pub(in super::super) fn submitted(
        &self,
        _submission: SubmissionIndex,
        reservation: Option<OwnedSemaphorePermit>,
    ) -> Result<(), TextureTransferError> {
        if let Some(permit) = reservation {
            self.queue.on_submitted_work_done(move || drop(permit));
        }
        Ok(())
    }
}
