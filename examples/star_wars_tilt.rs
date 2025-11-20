use euclid::{default::Transform3D, Angle};
use futures::executor::block_on;
use grafo::{transformator, Color, InstanceRenderParams, Shape, Stroke};
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

// Helper to convert euclid Transform3D to grafo's GPU instance layout
fn transform_instance_from_euclid(
    m: Transform3D<f32>,
) -> grafo::TransformInstance {
    grafo::TransformInstance {
        col0: [m.m11, m.m21, m.m31, m.m14],
        col1: [m.m12, m.m22, m.m32, m.m24],
        col2: [m.m13, m.m23, m.m33, m.m34],
        col3: [m.m41, m.m42, m.m43, m.m44],
    }
}

struct App<'a> {
    window: Option<Arc<Window>>,
    renderer: Option<grafo::Renderer<'a>>,
    rect_id: Option<usize>,
    inner_rect_1: Option<usize>,
    inner_rect_2: Option<usize>,
}

impl<'a> App<'a> {
    fn new() -> Self {
        Self {
            window: None,
            renderer: None,
            rect_id: None,
            inner_rect_1: None,
            inner_rect_2: None,
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

        let renderer = block_on(grafo::Renderer::new(
            window.clone(),
            physical_size,
            scale_factor,
            true,
            false,
        ));

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
                    let (width, height) = renderer.size();
                    // Create shape if it doesn't exist yet
                    if self.rect_id.is_none() {
                        // We need some background TODO: mention this in the docs
                        renderer.add_shape(Shape::rect(
                            [(0.0, 0.0), (width as f32, height as f32)],
                            Stroke::new(2.0, Color::BLACK),
                        ), None, None);

                        // Create a 100x100 rectangle (matching the HTML)
                        // Gold color (#FFD700 = rgb(255, 215, 0)) with white border (2px)
                        let rect_shape = Shape::rect(
                            [(0.0, 0.0), (100.0, 100.0)],
                            Stroke::new(2.0, Color::WHITE),
                        );

                        let rect_id = renderer.add_shape(rect_shape, None, None);

                        // Set the fill color to gold
                        renderer.set_shape_color(rect_id, Some(Color::rgb(255, 215, 0)));

                        self.rect_id = Some(rect_id);

                        let inner_rect_shape = Shape::rect(
                            [(0.0, 0.0), (35.0, 80.0)],
                            Stroke::new(1.0, Color::BLACK),
                        );

                        // TODO: clip
                        let inner_rect_1 = renderer.add_shape(inner_rect_shape.clone(), None, None);
                        renderer.set_shape_color(inner_rect_1, Some(Color::rgb(125, 0, 0)));
                        self.inner_rect_1 = Some(inner_rect_1);

                        // TODO: clip
                        let inner_rect_2 = renderer.add_shape(inner_rect_shape, None, None);
                        renderer.set_shape_color(inner_rect_2, Some(Color::rgb(0, 125, 0)));
                        self.inner_rect_2 = Some(inner_rect_2);
                    }

                    let scale_factor = renderer.scale_factor();
                    let (width, height) = renderer.size();
                    let (width, height) = (
                        width as f32 / scale_factor as f32,
                        height as f32 / scale_factor as f32,
                    );
                    let viewport_center = (width as f32 / 2.0, height as f32 / 2.0);
                    
                    let rect_id = self.rect_id.unwrap();
                    let inner_rect_1 = self.inner_rect_1.unwrap();
                    let inner_rect_2 = self.inner_rect_2.unwrap();
                    // Position the rectangle at center of screen (400, 300)
                    // In the HTML, the rectangle is rotated 45 degrees around the X axis (rotateX(45deg))
                    // and the container has perspective: 500px
                    //
                    // To replicate this:
                    // 1. Center the rectangle's origin
                    // 2. Apply rotateX(45deg)
                    // 3. Set perspective_distance to 500
                    // 4. Use offset to position at screen center

                    // Parent transform replicating the CSS 3D container
                    let parent_local = transformator::Transform::new()
                        .with_position_relative_to_parent(viewport_center.0 - 50.0, viewport_center.1 - 50.0)
                        .with_camera_perspective_origin(viewport_center.0, viewport_center.1)
                        .with_perspective_distance(500.0)
                        .with_origin(50.0, 50.0)
                        .then_rotate_x(45.0)
                        .then_rotate_y(30.0)
                        .compose_2(&transformator::Transform::new());

                    renderer.set_transformator(rect_id, &parent_local);

                    // Inner rectangles inherit parent transform and sit inside with 10px padding.
                    // Layout: padding(10) + rect(35) + gap(10) + rect(35) + padding(10) = 100 total width.
                    // Vertical: padding(10) + height(80) + padding(10) = 100 total height.

                    let pos_absoulte = (viewport_center.0 - 50.0 + 10.0, viewport_center.1 - 50.0 + 10.0);

                    let child1 = transformator::Transform::new()
                        .with_position_relative_to_parent(10.0, 10.0)
                        // .with_camera_perspective_origin(pos_absoulte.0 + 17.5, pos_absoulte.1 + 40.0)
                        // .with_origin(17.5, 40.0)
                        // .with_perspective_distance(500.0)
                        .then_rotate_y(30.0)
                        .compose_2(&parent_local);
                    renderer.set_transformator(inner_rect_1, &child1);

                    let child2 = transformator::Transform::new()
                        .with_position_relative_to_parent(55.0, 10.0) // 10 + 35 + 10
                        .compose_2(&parent_local);
                    renderer.set_transformator(inner_rect_2, &child2);

                    renderer.render().unwrap();
                }
            }
            WindowEvent::Resized(new_size) => {
                if let Some(renderer) = &mut self.renderer {
                    renderer.resize((new_size.width, new_size.height));
                    
                    if let Some(window) = &self.window {
                        window.request_redraw();
                    }
                }
            }
            WindowEvent::ScaleFactorChanged {
                scale_factor, inner_size_writer
            } => {
                if let Some(renderer) = &mut self.renderer {
                    println!("Change scale factor to {}", scale_factor);
                    renderer.change_scale_factor(scale_factor);

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
