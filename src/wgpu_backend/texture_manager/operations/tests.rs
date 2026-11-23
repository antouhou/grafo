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
fn cancellation_wakes_pending_source_and_releases_the_id() {
    let operations = TextureUploadOperations::default();
    let operation = operations.begin(42).unwrap();
    let (sender, receiver) = mpsc::channel();
    let waker = Waker::from(Arc::new(CancellationWake(sender)));
    let mut context = Context::from_waker(&waker);
    let mut cancelled = pin!(operation.cancelled());
    assert_eq!(cancelled.as_mut().poll(&mut context), Poll::Pending);

    operations.remove(42, || {});
    receiver.recv().unwrap();
    assert_eq!(cancelled.as_mut().poll(&mut context), Poll::Ready(()));
    assert!(operations.begin(42).is_ok());
}

#[test]
fn removed_upload_cannot_publish_or_release_a_new_upload() {
    let operations = TextureUploadOperations::default();
    let old = operations.begin(42).unwrap();
    operations.remove(42, || {});
    let replacement = operations.begin(42).unwrap();
    assert!(matches!(
        old.commit::<()>(|| panic!("removed upload committed")),
        Err(WgpuBackendError::Texture(
            TextureManagerError::UploadCancelled
        ))
    ));
    assert!(matches!(
        operations.begin(42),
        Err(TextureManagerError::TextureBusy(42))
    ));
    replacement.commit(|| Ok(())).unwrap();
    assert!(operations.begin(42).is_ok());
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
fn clearing_cancels_every_upload_before_new_ones_can_commit() {
    let operations = TextureUploadOperations::default();
    let first = operations.begin(1).unwrap();
    let second = operations.begin(2).unwrap();
    operations.clear(|| {});
    let replacement = operations.begin(1).unwrap();
    drop(first);
    assert!(matches!(
        second.commit::<()>(|| panic!("cleared upload committed")),
        Err(WgpuBackendError::Texture(
            TextureManagerError::UploadCancelled
        ))
    ));
    assert!(matches!(
        operations.begin(1),
        Err(TextureManagerError::TextureBusy(1))
    ));
    replacement.commit(|| Ok(())).unwrap();
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
