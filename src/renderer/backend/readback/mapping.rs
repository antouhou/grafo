use super::ReadbackError;
use std::sync::{mpsc, Mutex};
use wgpu::BufferAsyncError;

/// Report a dropped callback as well as a completed mapping.
pub(super) struct MappingCompletion {
    sender: Option<mpsc::SyncSender<Result<(), ReadbackError>>>,
}

impl MappingCompletion {
    pub(super) fn finish(mut self, result: Result<(), BufferAsyncError>) {
        if let Some(sender) = self.sender.take() {
            let _ = sender.send(result.map_err(ReadbackError::BufferMap));
        }
    }
}

impl Drop for MappingCompletion {
    fn drop(&mut self) {
        if let Some(sender) = self.sender.take() {
            let _ = sender.send(Err(ReadbackError::MapCallbackDropped));
        }
    }
}

/// One bounded completion slot per target. Only one mapping may be pending.
pub(super) struct ReadbackMapping {
    sender: mpsc::SyncSender<Result<(), ReadbackError>>,
    receiver: Mutex<mpsc::Receiver<Result<(), ReadbackError>>>,
}

impl ReadbackMapping {
    pub(super) fn new() -> Self {
        let (sender, receiver) = mpsc::sync_channel(1);
        Self {
            sender,
            receiver: Mutex::new(receiver),
        }
    }

    pub(super) fn completion(&self) -> MappingCompletion {
        MappingCompletion {
            sender: Some(self.sender.clone()),
        }
    }

    pub(super) fn wait(&self) -> Result<(), ReadbackError> {
        self.receiver
            .lock()
            .expect("readback receiver is only locked while waiting")
            .recv()
            .map_err(|_| ReadbackError::MapCallbackDropped)?
    }
}
