use std::time::{Duration, Instant};
use winit::event::StartCause;
use winit::event_loop::{ActiveEventLoop, ControlFlow};
use winit::window::Window;

const SURFACE_TIMEOUT_RETRY_DELAY: Duration = Duration::from_millis(50);

/// Retries a timed-out render when winit delivers the requested deadline event.
#[derive(Default)]
pub struct RedrawRetry {
    is_pending: bool,
}

impl RedrawRetry {
    pub fn schedule(&mut self, event_loop: &ActiveEventLoop) {
        self.is_pending = true;
        event_loop.set_control_flow(ControlFlow::WaitUntil(
            Instant::now() + SURFACE_TIMEOUT_RETRY_DELAY,
        ));
    }

    pub fn cancel(&mut self, event_loop: &ActiveEventLoop) {
        self.is_pending = false;
        event_loop.set_control_flow(ControlFlow::Wait);
    }

    pub fn new_events(
        &mut self,
        event_loop: &ActiveEventLoop,
        cause: StartCause,
        window: Option<&Window>,
    ) {
        if self.is_pending && matches!(cause, StartCause::ResumeTimeReached { .. }) {
            self.cancel(event_loop);
            if let Some(window) = window {
                window.request_redraw();
            }
        }
    }
}
