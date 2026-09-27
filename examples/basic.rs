use futures::executor::block_on;
use grafo::{Color, Renderer, Shape, ShapeDrawCommandOptions};
use grafo::{RendererContext, Surface};
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

#[path = "../examples/window_rendering/mod.rs"]
mod window_rendering;

#[derive(Default)]
struct App {
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    surface: Option<Surface>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes())
                .unwrap(),
        );

        let window_size = window.inner_size();
        let scale_factor = window.scale_factor();
        let physical_size = (window_size.width, window_size.height);

        let context = block_on(RendererContext::new());
        let surface = Surface::new(&context, window.clone(), physical_size, true, false)
            .expect("Failed to create surface");
        let mut renderer = Renderer::new_with_context(context, physical_size, scale_factor, 1);

        let rect = Shape::rect([(100.0, 100.0), (300.0, 200.0)]);
        renderer
            .add_shape(
                rect,
                None,
                None,
                ShapeDrawCommandOptions::new().color(Color::rgb(0, 128, 255)),
            )
            .unwrap();

        let rect = Shape::rect([(500.0, 100.0), (600.0, 200.0)]);
        renderer
            .add_shape(
                rect,
                None,
                None,
                ShapeDrawCommandOptions::new().color(Color::rgb(0, 128, 0)),
            )
            .unwrap();

        window.request_redraw();
        self.window = Some(window);
        self.renderer = Some(renderer);
        self.surface = Some(surface);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(window) = &self.window else { return };
        let Some(renderer) = &mut self.renderer else {
            return;
        };
        let Some(surface) = &mut self.surface else {
            return;
        };

        if window_id != window.id() {
            return;
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(physical_size) => {
                let new_size = (physical_size.width, physical_size.height);
                surface.resize(new_size);
                window.request_redraw();
            }
            WindowEvent::RedrawRequested => {
                // Keep the static scene queued so later redraws render the same shapes.
                window_rendering::render(renderer, surface, event_loop);
            }
            _ => {}
        }
    }
}

pub fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().expect("To create the event loop");

    let mut app = App::default();
    let _ = event_loop.run_app(&mut app);
}
