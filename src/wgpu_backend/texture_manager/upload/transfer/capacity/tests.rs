use super::{TextureTransferError, UploadCapacity, MAX_PENDING_UPLOAD_BATCHES};
use crate::wgpu_backend::texture_manager::operations::TextureUploadOperations;
use crate::wgpu_backend::WgpuBackendError;
#[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
use futures::executor;
use futures_util::FutureExt;
use std::future::{self, Future};
use std::pin::{pin, Pin};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::task::{Context, Poll, Wake, Waker};
use std::thread;
use std::time::Duration;
#[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
use tokio::{runtime, task};

#[derive(Default)]
struct WakeCounter(AtomicUsize);

impl WakeCounter {
    fn poll<F: Future>(self: &Arc<Self>, future: Pin<&mut F>) -> Poll<F::Output> {
        let waker = Waker::from(Arc::clone(self));
        future.poll(&mut Context::from_waker(&waker))
    }

    fn count(&self) -> usize {
        self.0.load(Ordering::Relaxed)
    }
}

impl Wake for WakeCounter {
    fn wake(self: Arc<Self>) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
}

struct CancelUploadOnWake<F> {
    upload: Mutex<Option<F>>,
}

impl<F: Send> Wake for CancelUploadOnWake<F> {
    fn wake(self: Arc<Self>) {
        self.upload.lock().unwrap().take();
    }
}

#[test]
fn completion_wakes_only_the_first_waiter_and_cancellation_preserves_queue_order() {
    let capacity = UploadCapacity::new(Arc::default());
    let first_permit = capacity.acquire().unwrap();
    let second_permit = capacity.acquire().unwrap();
    let mut first_wait = pin!(capacity.acquire_async());
    let mut canceled_wait = Box::pin(capacity.acquire_async());
    let mut last_wait = pin!(capacity.acquire_async());
    let first_counter = Arc::new(WakeCounter::default());
    let canceled_counter = Arc::new(WakeCounter::default());
    let last_counter = Arc::new(WakeCounter::default());

    assert!(first_counter.poll(first_wait.as_mut()).is_pending());
    assert!(canceled_counter.poll(canceled_wait.as_mut()).is_pending());
    assert!(last_counter.poll(last_wait.as_mut()).is_pending());
    drop(canceled_wait);
    drop(first_permit);
    assert_eq!(first_counter.count(), 1);
    assert_eq!(canceled_counter.count(), 0);
    assert_eq!(last_counter.count(), 0);
    assert!(last_counter.poll(last_wait.as_mut()).is_pending());

    let first_permit = first_wait.now_or_never().unwrap().unwrap();
    drop(first_permit);
    assert_eq!(last_counter.count(), 1);
    let last_permit = last_wait.now_or_never().unwrap().unwrap();
    assert_eq!(capacity.permits.available_permits(), 0);
    drop(second_permit);
    drop(last_permit);
    assert_eq!(
        capacity.permits.available_permits(),
        MAX_PENDING_UPLOAD_BATCHES
    );
}

#[test]
fn dropping_a_notified_waiter_hands_its_reserved_capacity_to_the_next_waiter() {
    let capacity = UploadCapacity::new(Arc::default());
    let first_permit = capacity.acquire().unwrap();
    let _second_permit = capacity.acquire().unwrap();
    let mut canceled_wait = Box::pin(capacity.acquire_async());
    let mut following_wait = pin!(capacity.acquire_async());
    let canceled_counter = Arc::new(WakeCounter::default());
    let following_counter = Arc::new(WakeCounter::default());
    assert!(canceled_counter.poll(canceled_wait.as_mut()).is_pending());
    assert!(following_counter.poll(following_wait.as_mut()).is_pending());

    drop(first_permit);
    assert_eq!(canceled_counter.count(), 1);
    assert_eq!(following_counter.count(), 0);
    drop(canceled_wait);
    assert_eq!(following_counter.count(), 1);
    assert!(matches!(following_wait.now_or_never(), Some(Ok(_))));
}

#[test]
fn canceling_an_async_reservation_allows_a_blocking_upload_to_proceed() {
    let capacity = UploadCapacity::new(Arc::default());
    let first_permit = capacity.acquire().unwrap();
    let _second_permit = capacity.acquire().unwrap();
    let mut canceled_wait = Box::pin(capacity.acquire_async());
    assert!(canceled_wait.as_mut().now_or_never().is_none());
    drop(first_permit);
    assert_eq!(capacity.permits.available_permits(), 0);

    let waiting_capacity = Arc::clone(&capacity);
    let (started_sender, started) = mpsc::channel();
    let (result_sender, result) = mpsc::channel();
    let worker = thread::spawn(move || {
        started_sender.send(()).unwrap();
        result_sender.send(waiting_capacity.acquire()).unwrap();
    });
    started.recv().unwrap();
    drop(canceled_wait);
    let _permit = result
        .recv_timeout(Duration::from_secs(5))
        .unwrap()
        .unwrap();
    worker.join().unwrap();
}

#[test]
fn first_failure_wakes_source_and_capacity_waiters_without_releasing_pending_work() {
    let operations = TextureUploadOperations::default();
    let capacity = UploadCapacity::new(operations.failure_signal());
    let _first_permit = capacity.acquire().unwrap();
    let _second_permit = capacity.acquire().unwrap();
    let operation = operations.begin(42).unwrap();
    let mut source_wait = pin!(operation.wait(future::pending::<()>()));
    let mut capacity_wait = pin!(capacity.acquire_async());
    let source_counter = Arc::new(WakeCounter::default());
    let capacity_counter = Arc::new(WakeCounter::default());
    assert!(source_counter.poll(source_wait.as_mut()).is_pending());
    assert!(capacity_counter.poll(capacity_wait.as_mut()).is_pending());

    let waiting_capacity = Arc::clone(&capacity);
    let (started_sender, started) = mpsc::channel();
    let (result_sender, result) = mpsc::channel();
    let worker = thread::spawn(move || {
        started_sender.send(()).unwrap();
        result_sender.send(waiting_capacity.acquire()).unwrap();
    });
    started.recv().unwrap();
    capacity.fail(TextureTransferError::DeviceLost);
    capacity.fail(TextureTransferError::WorkerStopped);
    assert_eq!(source_counter.count(), 1);
    assert_eq!(capacity_counter.count(), 1);
    assert!(matches!(
        source_wait.now_or_never(),
        Some(Err(WgpuBackendError::DeviceLost))
    ));
    assert!(matches!(
        capacity_wait.now_or_never(),
        Some(Err(TextureTransferError::DeviceLost))
    ));
    assert!(matches!(
        result.recv_timeout(Duration::from_secs(5)),
        Ok(Err(TextureTransferError::DeviceLost))
    ));
    worker.join().unwrap();
    assert!(matches!(
        capacity.acquire(),
        Err(TextureTransferError::DeviceLost)
    ));
    assert_eq!(capacity.permits.available_permits(), 0);
}

#[test]
fn failure_can_wake_an_executor_that_cancels_another_pending_upload() {
    let (finished_sender, finished) = mpsc::channel();
    let worker = thread::spawn(move || {
        let capacity = UploadCapacity::new(Arc::default());
        let _first_permit = capacity.acquire().unwrap();
        let _second_permit = capacity.acquire().unwrap();
        let other_capacity = Arc::clone(&capacity);
        let mut other_upload = Box::pin(async move { other_capacity.acquire_async().await });
        assert!(other_upload.as_mut().now_or_never().is_none());
        let cancel = Arc::new(CancelUploadOnWake {
            upload: Mutex::new(Some(other_upload)),
        });
        let waker = Waker::from(Arc::clone(&cancel));
        let mut upload = pin!(capacity.acquire_async());
        assert!(upload
            .as_mut()
            .poll(&mut Context::from_waker(&waker))
            .is_pending());

        capacity.fail(TextureTransferError::DeviceLost);
        assert!(cancel.upload.lock().unwrap().is_none());
        assert!(matches!(
            upload.now_or_never(),
            Some(Err(TextureTransferError::DeviceLost))
        ));
        finished_sender.send(()).unwrap();
    });
    finished
        .recv_timeout(Duration::from_secs(5))
        .expect("failure blocked while canceling an upload");
    worker.join().unwrap();
}

#[test]
fn completion_before_wait_registration_does_not_lose_available_capacity() {
    let capacity = UploadCapacity::new(Arc::default());
    let completed_permit = capacity.acquire().unwrap();
    let _second_permit = capacity.acquire().unwrap();
    let waiting_upload = capacity.acquire_async();
    drop(completed_permit);
    assert!(matches!(waiting_upload.now_or_never(), Some(Ok(_))));
}

#[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
#[test]
fn sync_and_foreign_executor_uploads_can_acquire_with_an_exhausted_tokio_budget() {
    let (result_sender, result) = mpsc::channel();
    let worker = thread::spawn(move || {
        let runtime = runtime::Builder::new_current_thread().build().unwrap();
        runtime.block_on(async {
            task::spawn(async move {
                let capacity = UploadCapacity::new(Arc::default());
                while task::coop::has_budget_remaining() {
                    task::coop::consume_budget().await;
                }
                let _blocking_permit = capacity.acquire().unwrap();
                result_sender
                    .send(executor::block_on(capacity.acquire_async()))
                    .unwrap();
            })
            .await
            .unwrap();
        });
    });
    let _permit = result
        .recv_timeout(Duration::from_secs(5))
        .unwrap()
        .unwrap();
    worker.join().unwrap();
}
