use futures::executor::block_on;
use grafo::Shape;
use grafo::{Color, Stroke};
use std::sync::Arc;
use winit::event::{Event, WindowEvent};
use winit::event_loop::EventLoop;
use winit::window::WindowBuilder;

pub fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().expect("To create the event loop");
    let window = Arc::new(WindowBuilder::new().build(&event_loop).unwrap());

    let window_size = window.inner_size();
    let scale_factor = window.scale_factor();
    let physical_size = (window_size.width, window_size.height);

    // Initialize the renderer
    let mut renderer = block_on(grafo::Renderer::new(
        window.clone(),
        physical_size,
        scale_factor,
    ));

    // Define a simple rectangle shape
    let rect = Shape::rect(
        [(100.0, 100.0), (300.0, 200.0)],
        Color::rgb(0, 128, 255),        // Blue fill
        Stroke::new(2.0, Color::BLACK), // Black stroke with width 2.0
    );
    renderer.add_shape(rect, None, (0.0, 0.0), None);

    // Start the event loop
    let _ = event_loop.run(move |event, event_loop_window_target| match event {
        Event::WindowEvent {
            ref event,
            window_id,
        } if window_id == window.id() => match event {
            WindowEvent::CloseRequested => event_loop_window_target.exit(),
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
                Err(wgpu::SurfaceError::OutOfMemory) => event_loop_window_target.exit(),
                Err(e) => eprintln!("{:?}", e),
            },
            _ => {}
        },
        _ => {}
    });
}
