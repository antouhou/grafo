use futures::executor::block_on;
use grafo::Shape;
use grafo::{Color, Stroke};
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
                .create_window(Window::default_attributes())
                .unwrap(),
        );

        let window_size = window.inner_size();
        let scale_factor = window.scale_factor();
        let physical_size = (window_size.width, window_size.height);

        // Initialize the renderer
        let mut renderer = block_on(grafo::Renderer::new(
            window.clone(),
            physical_size,
            scale_factor,
            true,  // vsync
            false, // transparent
        ));

        // Define a simple rectangle shape
        let rect = Shape::rect(
            [(100.0, 100.0), (300.0, 200.0)],
            Color::rgb(0, 128, 255),        // Blue fill
            Stroke::new(2.0, Color::BLACK), // Black stroke with width 2.0
        );
        renderer.add_shape(rect, None, (0.0, 0.0), None);

        self.window = Some(window);
        self.renderer = Some(renderer);
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

        if window_id != window.id() {
            return;
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(physical_size) => {
                let new_size = (physical_size.width, physical_size.height);
                renderer.resize(new_size);
                window.request_redraw();
            }
            WindowEvent::RedrawRequested => match renderer.render() {
                Ok(_) => {
                    renderer.clear_draw_queue();
                }
                Err(wgpu::SurfaceError::Lost) => renderer.resize(renderer.size()),
                Err(wgpu::SurfaceError::OutOfMemory) => event_loop.exit(),
                Err(e) => eprintln!("{e:?}"),
            },
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
