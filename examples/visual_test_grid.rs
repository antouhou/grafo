/// Visual confirmation example — renders all 34 test tiles in a 6×6 grid.
///
/// Run with:    cargo run --example visual_test_grid
///
/// The window shows the exact same scene that the headless visual-regression
/// test validates with pixel-level assertions.
use futures::executor::block_on;
use grafo_test_scenes::{build_main_scene, CANVAS_HEIGHT, CANVAS_WIDTH};
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

#[derive(Default)]
struct App<'a> {
    window: Option<Arc<Window>>,
    renderer: Option<grafo::Renderer<'a>>,
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
                    Ok(_) => {}
                    Err(wgpu::SurfaceError::Lost) => renderer.resize(renderer.size()),
                    Err(wgpu::SurfaceError::OutOfMemory) => event_loop.exit(),
                    Err(e) => eprintln!("{e:?}"),
                }
            }
            _ => {}
        }
    }
}

pub fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().expect("Failed to create event loop");

    let mut app = App::default();
    let _ = event_loop.run_app(&mut app);
}
