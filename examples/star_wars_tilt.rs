use euclid::{default::Transform3D, Angle};
use futures::executor::block_on;
use grafo::{Color, Shape, Stroke};
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

// Helper to convert euclid Transform3D to grafo's GPU instance layout
fn transform_instance_from_euclid(
    m: Transform3D<f32>,
    perspective_distance: f32,
    offset: [f32; 2],
) -> grafo::TransformInstance {
    grafo::TransformInstance {
        col0: [m.m11, m.m21, m.m31, m.m14],
        col1: [m.m12, m.m22, m.m32, m.m24],
        col2: [m.m13, m.m23, m.m33, m.m34],
        col3: [m.m41, m.m42, m.m43, m.m44],
        perspective_distance,
        offset,
        _padding: 0.0,
    }
}

struct App<'a> {
    window: Option<Arc<Window>>,
    renderer: Option<grafo::Renderer<'a>>,
    rect_id: Option<usize>,
}

impl<'a> App<'a> {
    fn new() -> Self {
        Self {
            window: None,
            renderer: None,
            rect_id: None,
        }
    }
}

impl<'a> ApplicationHandler for App<'a> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let window_attrs = winit::window::Window::default_attributes()
            .with_title("Star Wars Tilt - grafo")
            .with_inner_size(winit::dpi::LogicalSize::new(800, 600));

        let window = Arc::new(event_loop.create_window(window_attrs).unwrap());
        let window_size = window.inner_size();
        let scale_factor = window.scale_factor();
        let physical_size = (window_size.width, window_size.height);

        self.window = Some(window.clone());

        let mut renderer = block_on(grafo::Renderer::new(
            window.clone(),
            physical_size,
            scale_factor,
            true,
            false,
        ));

        // Create a 100x100 rectangle (matching the HTML)
        // Gold color (#FFD700 = rgb(255, 215, 0)) with white border (2px)
        let rect_shape = Shape::rect(
            [(0.0, 0.0), (100.0, 100.0)],
            Stroke::new(2.0, Color::WHITE),
        );

        let rect_id = renderer.add_shape(rect_shape, None, None);
        
        // Set the fill color to gold
        renderer.set_shape_color(rect_id, Some(Color::rgb(255, 215, 0)));

        // Position the rectangle at center of screen (400, 300)
        // In the HTML, the rectangle is rotated 45 degrees around the X axis (rotateX(45deg))
        // and the container has perspective: 500px
        //
        // To replicate this:
        // 1. Center the rectangle's origin (translate by -50, -50 to center the 100x100 rect)
        // 2. Apply rotateX(45deg)
        // 3. Set perspective_distance to 500
        // 4. Use offset to position at screen center

        let center_shape = Transform3D::translation(-50.0, -50.0, 0.0);
        let rotate_x = Transform3D::rotation(1.0, 0.0, 0.0, Angle::degrees(45.0));

        let transform = center_shape.then(&rotate_x);

        // Set perspective distance to 500 (matching CSS perspective: 500px)
        // Use offset to position the shape at screen center after perspective is applied
        let perspective_distance = 500.0;
        let (width, height) = renderer.size();
        let offset = [width as f32 / 2.0, height as f32 / 2.0];

        let instance = transform_instance_from_euclid(transform, perspective_distance, offset);
        renderer.set_shape_transform(rect_id, instance);

        self.rect_id = Some(rect_id);
        self.renderer = Some(renderer);

        window.request_redraw();
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                if let Some(renderer) = &mut self.renderer {
                    renderer.render().unwrap();
                }
            }
            WindowEvent::Resized(new_size) => {
                if let (Some(renderer), Some(rect_id)) = (&mut self.renderer, self.rect_id) {
                    renderer.resize((new_size.width, new_size.height));
                    
                    // Update transform with new center position
                    let center_shape = Transform3D::translation(-50.0, -50.0, 0.0);
                    let rotate_x = Transform3D::rotation(1.0, 0.0, 0.0, Angle::degrees(45.0));
                    let transform = center_shape.then(&rotate_x);
                    
                    let perspective_distance = 500.0;
                    let (width, height) = renderer.size();
                    let offset = [width as f32 / 2.0, height as f32 / 2.0];
                    
                    let instance = transform_instance_from_euclid(transform, perspective_distance, offset);
                    renderer.set_shape_transform(rect_id, instance);
                    
                    if let Some(window) = &self.window {
                        window.request_redraw();
                    }
                }
            }
            _ => {}
        }
    }
}

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}
