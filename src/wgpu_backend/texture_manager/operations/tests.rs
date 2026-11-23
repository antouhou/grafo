use super::{TextureManagerError, TextureTransferError, TextureUploadOperations};
use crate::wgpu_backend::WgpuBackendError;
use std::future::Future;
use std::pin::pin;
use std::sync::{mpsc, Arc};
use std::task::{Context, Poll, Wake, Waker};
use std::time::Duration;
use std::{future, thread};

struct CancellationWake(mpsc::Sender<()>);

impl Wake for CancellationWake {
    fn wake(self: Arc<Self>) {
        self.0.send(()).unwrap();
    }
}

#[test]
fn abandoned_upload_releases_ownership_without_publishing() {
    let operations = TextureUploadOperations::default();
    let pending = operations.begin(42).unwrap();
    assert!(matches!(
        operations.begin(42),
        Err(TextureManagerError::TextureBusy(42))
    ));
    drop(pending);
    let replacement = operations.begin(42).unwrap();
    let published = replacement.commit(|| Ok(123)).unwrap();
    assert_eq!(published, 123);
}

#[test]
fn failure_notification_wakes_work_waiting_for_input() {
    let operations = TextureUploadOperations::default();
    let operation = operations.begin(42).unwrap();
    let (sender, receiver) = mpsc::channel();
    let waker = Waker::from(Arc::new(CancellationWake(sender)));
    let mut context = Context::from_waker(&waker);
    let mut waiting = pin!(operation.wait(future::pending::<()>()));
    assert!(waiting.as_mut().poll(&mut context).is_pending());
    operations
        .failure
        .set(TextureTransferError::DeviceLost)
        .unwrap();
    receiver.recv_timeout(Duration::from_secs(5)).unwrap();
    assert!(matches!(
        waiting.as_mut().poll(&mut context),
        Poll::Ready(Err(WgpuBackendError::DeviceLost))
    ));
}

#[test]
fn failure_callback_can_run_inside_commit_without_lock_reentrancy() {
    let (sender, receiver) = mpsc::channel();
    let worker = thread::spawn(move || {
        let operations = TextureUploadOperations::default();
        let operation = operations.begin(42).unwrap();
        let result = operation.commit(|| {
            operations
                .failure
                .set(TextureTransferError::DeviceLost)
                .unwrap();
            Err::<(), _>(WgpuBackendError::DeviceLost)
        });
        assert!(operations.begin(42).is_ok());
        sender.send(result).unwrap();
    });
    let result = receiver
        .recv_timeout(Duration::from_secs(5))
        .expect("failure callback blocked on commit lock");
    worker.join().unwrap();
    assert!(matches!(result, Err(WgpuBackendError::DeviceLost)));
}

#[test]
fn a_late_transfer_failure_does_not_retroactively_fail_a_commit() {
    let operations = TextureUploadOperations::default();
    let operation = operations.begin(42).unwrap();
    let mut published = false;
    let result = operation.commit(|| {
        published = true;
        operations
            .failure
            .set(TextureTransferError::WorkerStopped)
            .unwrap();
        Ok(())
    });
    assert!(result.is_ok());
    assert!(published);
    assert!(matches!(
        operations.begin(43).unwrap().check(),
        Err(WgpuBackendError::Texture(TextureManagerError::Transfer(
            TextureTransferError::WorkerStopped
        )))
    ));
}
