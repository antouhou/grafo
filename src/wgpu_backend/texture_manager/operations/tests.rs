use super::{TextureTransferError, TextureUploadOperations};
use crate::wgpu_backend::WgpuBackendError;
use std::future::{self, Future};
use std::pin::pin;
use std::sync::{mpsc, Arc};
use std::task::{Context, Poll, Wake, Waker};
use std::time::Duration;

struct CancellationWake(mpsc::Sender<()>);

impl Wake for CancellationWake {
    fn wake(self: Arc<Self>) {
        self.0.send(()).unwrap();
    }
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
