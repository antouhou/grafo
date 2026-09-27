//! Displays the tile grid checked by the headless visual regression test.
//! Run with `cargo run --example visual_test_grid`.

use futures::executor::block_on;
use grafo::{RendererContext, Surface};
use grafo_test_scenes::{build_main_scene, CANVAS_HEIGHT, CANVAS_WIDTH};
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

mod window_rendering;

#[derive(Default)]
struct App {
    window: Option<Arc<Window>>,
    renderer: Option<grafo::Renderer>,
    surface: Option<Surface>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_inner_size(winit::dpi::PhysicalSize::new(CANVAS_WIDTH, CANVAS_HEIGHT))
                        .with_title("Visual regression test pattern")
                        .with_resizable(false),
                )
                .unwrap(),
        );

        let window_size = window.inner_size();
        let physical_size = (window_size.width, window_size.height);

        let context = block_on(RendererContext::new());
        let surface = Surface::new(&context, window.clone(), physical_size, true, false)
            .expect("Failed to create surface");
        let mut renderer = grafo::Renderer::new_with_context(context, physical_size, 1.0, 1);

        build_main_scene(&mut renderer);

        self.window = Some(window);
        self.renderer = Some(renderer);
        self.surface = Some(surface);
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
        let Some(surface) = &mut self.surface else {
            return;
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(physical_size) => {
                let new_size = (physical_size.width, physical_size.height);
                surface.resize(new_size);
                window.request_redraw();
            }
            WindowEvent::RedrawRequested => {
                renderer.clear_draw_queue();
                build_main_scene(renderer);

                window_rendering::render(renderer, surface, event_loop);
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
