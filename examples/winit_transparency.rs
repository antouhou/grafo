use futures::executor::block_on;
use grafo::{BorderRadii, Color, Shape, Stroke};
use std::sync::Arc;
use std::time::Instant;
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
                        .with_title("Grafo Winit Test")
                        .with_transparent(true)
                        .with_inner_size(winit::dpi::LogicalSize::new(800, 600)),
                )
                .unwrap(),
        );

        let window_size = window.inner_size();
        let scale_factor = window.scale_factor();
        let physical_size = (window_size.width, window_size.height);

        let renderer = block_on(grafo::Renderer::new_transparent(
            window.clone(),
            physical_size,
            scale_factor,
            true, // vsync
        ));

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
            WindowEvent::RedrawRequested => {
                let timer = Instant::now();

                // Create a simple rectangle to test rendering
                let rect = Shape::rect(
                    [(100.0, 100.0), (300.0, 200.0)],
                    Color::rgb(255, 100, 50),       // Orange fill
                    Stroke::new(3.0, Color::BLACK), // Black stroke
                );
                renderer.add_shape(rect, None, (0.0, 0.0), None);

                // Create a rounded rectangle to test different shapes
                let rounded_rect = Shape::rounded_rect(
                    [(350.0, 250.0), (450.0, 350.0)],
                    BorderRadii::new(50.0),    // Makes it circular
                    Color::rgb(100, 200, 255), // Light blue fill
                    Stroke::new(2.0, Color::rgb(0, 100, 200)), // Darker blue stroke
                );
                renderer.add_shape(rounded_rect, None, (0.0, 0.0), None);

                // Render the frame
                match renderer.render() {
                    Ok(_) => {
                        renderer.clear_draw_queue();
                        println!("Render time: {:?}", timer.elapsed());
                    }
                    Err(wgpu::SurfaceError::Lost) => {
                        println!("Surface lost, resizing...");
                        renderer.resize(renderer.size())
                    }
                    Err(wgpu::SurfaceError::OutOfMemory) => {
                        println!("Out of memory, exiting...");
                        event_loop.exit()
                    }
                    Err(e) => eprintln!("Render error: {:?}", e),
                }
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
