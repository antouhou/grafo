/// Visual confirmation example — renders the shared visual-regression tile grid.
///
/// Run with:    cargo run --example visual_test_grid
///
/// The window shows the exact same scene that the headless visual-regression
/// test validates with pixel-level assertions.
use futures::executor::block_on;
use grafo_test_scenes::{build_main_scene, CANVAS_HEIGHT, CANVAS_WIDTH};
use std::sync::Arc;
use std::time::{Duration, Instant};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

/// How long to wait before retrying a frame that was skipped because the surface
/// reported it was not visible (`Occluded`/`Timeout`).
const OCCLUDED_RETRY_DELAY: Duration = Duration::from_millis(50);

#[derive(Default)]
struct App<'a> {
    window: Option<Arc<Window>>,
    renderer: Option<grafo::Renderer<'a>>,
    /// Pending retry of a frame skipped because the window was not visible.
    redraw_retry_at: Option<Instant>,
}

impl<'a> ApplicationHandler for App<'a> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_inner_size(winit::dpi::PhysicalSize::new(CANVAS_WIDTH, CANVAS_HEIGHT))
                        .with_title("Visual Test Grid — grafo")
                        .with_resizable(false),
                )
                .unwrap(),
        );

        let window_size = window.inner_size();
        let physical_size = (window_size.width, window_size.height);

        let mut renderer = block_on(grafo::Renderer::new(
            window.clone(),
            physical_size,
            1.0,   // scale_factor — match test expectations
            true,  // vsync
            false, // transparent
            1,     // msaa_samples — match test expectations
        ));

        build_main_scene(&mut renderer);

        self.window = Some(window);
        self.renderer = Some(renderer);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(window) = &self.window else { return };
        let Some(renderer) = &mut self.renderer else {
            return;
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(physical_size) => {
                let new_size = (physical_size.width, physical_size.height);
                renderer.resize(new_size);
                window.request_redraw();
            }
            WindowEvent::RedrawRequested => {
                renderer.clear_draw_queue();
                build_main_scene(renderer);

                match renderer.render() {
                    Ok(_) => {
                        self.redraw_retry_at = None;
                    }
                    Err(
                        wgpu::CurrentSurfaceTexture::Lost | wgpu::CurrentSurfaceTexture::Outdated,
                    ) => renderer.resize(renderer.size()),
                    Err(
                        wgpu::CurrentSurfaceTexture::Timeout
                        | wgpu::CurrentSurfaceTexture::Occluded,
                    ) => {
                        // The window is not visible yet (still appearing, minimized, or fully
                        // covered). Retry shortly instead of busy-looping redraws — winit does
                        // not request one when the window becomes visible again. `WaitUntil`
                        // wakes the event loop without spinning while the window stays hidden.
                        let retry_at = Instant::now() + OCCLUDED_RETRY_DELAY;
                        self.redraw_retry_at = Some(retry_at);
                        event_loop.set_control_flow(ControlFlow::WaitUntil(retry_at));
                    }
                    Err(e) => eprintln!("{e:?}"),
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let Some(retry_at) = self.redraw_retry_at else {
            // Clear a stale WaitUntil deadline left behind when a successful
            // render cancelled the pending retry before it fired.
            event_loop.set_control_flow(ControlFlow::Wait);
            return;
        };
        if Instant::now() >= retry_at {
            self.redraw_retry_at = None;
            if let Some(window) = &self.window {
                window.request_redraw();
            }
            event_loop.set_control_flow(ControlFlow::Wait);
        } else {
            event_loop.set_control_flow(ControlFlow::WaitUntil(retry_at));
        }
    }
}

pub fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().expect("Failed to create event loop");

    let mut app = App::default();
    let _ = event_loop.run_app(&mut app);
}
