use euclid::{Point2D, UnknownUnit};
use futures::executor::block_on;
use grafo::{transformator, Color, Shape, Stroke};
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

struct App<'a> {
    window: Option<Arc<Window>>,
    renderer: Option<grafo::Renderer<'a>>,
    rect_id: Option<usize>,
    inner_rect_1: Option<usize>,
    inner_rect_2: Option<usize>,
    mouse_position: Point2D<f32, UnknownUnit>,
    // Track which shapes are being hovered
    parent_hovered: bool,
    child1_hovered: bool,
    child2_hovered: bool,
}

impl<'a> App<'a> {
    fn new() -> Self {
        Self {
            window: None,
            renderer: None,
            rect_id: None,
            inner_rect_1: None,
            inner_rect_2: None,
            mouse_position: Point2D::new(0.0, 0.0),
            parent_hovered: false,
            child1_hovered: false,
            child2_hovered: false,
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
                        let background = renderer.add_shape(
                            Shape::rect(
                                [(0.0, 0.0), (width as f32, height as f32)],
                                Stroke::new(2.0, Color::TRANSPARENT),
                            ),
                            None,
                            None,
                        );
                        renderer.set_shape_color(background, Some(Color::rgb(30, 30, 30)));

                        // Create a 100x100 rectangle (matching the HTML)
                        // Gold color (#FFD700 = rgb(255, 215, 0)) with white border (2px)
                        let rect_shape = Shape::rect(
                            [(0.0, 0.0), (100.0, 100.0)],
                            Stroke::new(2.0, Color::TRANSPARENT),
                        );

                        let rect_id = renderer.add_shape(rect_shape, None, None);

                        // Set the fill color to gold
                        renderer.set_shape_color(rect_id, Some(Color::TRANSPARENT));

                        self.rect_id = Some(rect_id);

                        let inner_rect_shape =
                            Shape::rect([(0.0, 0.0), (35.0, 80.0)], Stroke::new(1.0, Color::BLACK));

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

                    // Mouse position in logical coordinates
                    let mouse_x = self.mouse_position.x / scale_factor as f32;
                    let mouse_y = self.mouse_position.y / scale_factor as f32;
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
                        .with_position_relative_to_parent(
                            viewport_center.0 - 50.0,
                            viewport_center.1 - 50.0,
                        )
                        .with_parent_container_perspective(
                            500.0,
                            viewport_center.0,
                            viewport_center.1,
                        )
                        // TODO: important! Order of rotations matters!
                        .then_rotate_y(30.0)
                        .then_rotate_x(45.0)
                        .with_origin(50.0, 50.0)
                        .compose_2(&transformator::Transform::new());

                    renderer.set_transformator(rect_id, &parent_local);

                    // Inner rectangles inherit parent transform and sit inside with 10px padding.
                    // Layout: padding(10) + rect(35) + gap(10) + rect(35) + padding(10) = 100 total width.
                    // Vertical: padding(10) + height(80) + padding(10) = 100 total height.

                    let child1 = transformator::Transform::new()
                        .with_position_relative_to_parent(10.0, 10.0)
                        .then_rotate_y(20.0)
                        .with_origin(17.5, 40.0)
                        .compose_2(&parent_local);
                    renderer.set_transformator(inner_rect_1, &child1);

                    let child2 = transformator::Transform::new()
                        .with_position_relative_to_parent(55.0, 10.0)
                        .then_rotate_y(20.0)
                        .with_origin(17.5, 40.0)
                        .compose_2(&parent_local);
                    renderer.set_transformator(inner_rect_2, &child2);

                    // Hit testing: transform mouse position to local coordinates for each shape
                    // Check children first (they're on top)
                    let child1_local = child1.project_screen_point_to_local_2d((mouse_x, mouse_y));
                    let child1_hit = if let Some((lx, ly)) = child1_local {
                        lx >= 0.0 && lx <= 35.0 && ly >= 0.0 && ly <= 80.0
                    } else {
                        false
                    };

                    let child2_local = child2.project_screen_point_to_local_2d((mouse_x, mouse_y));
                    let child2_hit = if let Some((lx, ly)) = child2_local {
                        lx >= 0.0 && lx <= 35.0 && ly >= 0.0 && ly <= 80.0
                    } else {
                        false
                    };

                    let parent_local_coords =
                        parent_local.project_screen_point_to_local_2d((mouse_x, mouse_y));
                    let parent_hit = if !child1_hit && !child2_hit {
                        if let Some((lx, ly)) = parent_local_coords {
                            lx >= 0.0 && lx <= 100.0 && ly >= 0.0 && ly <= 100.0
                        } else {
                            false
                        }
                    } else {
                        false
                    };

                    // Update hover states
                    self.parent_hovered = parent_hit;
                    self.child1_hovered = child1_hit;
                    self.child2_hovered = child2_hit;

                    // Update colors based on hover state
                    if self.parent_hovered {
                        renderer.set_shape_color(rect_id, Some(Color::rgba(255, 235, 100, 200)));
                    // Brighter gold
                    } else {
                        renderer.set_shape_color(rect_id, Some(Color::rgba(255, 215, 0, 200)));
                        // Normal gold
                    }

                    if self.child1_hovered {
                        renderer.set_shape_color(inner_rect_1, Some(Color::rgb(200, 50, 50)));
                    // Brighter red
                    } else {
                        renderer.set_shape_color(inner_rect_1, Some(Color::rgb(125, 0, 0)));
                        // Normal red
                    }

                    if self.child2_hovered {
                        renderer.set_shape_color(inner_rect_2, Some(Color::rgb(50, 200, 50)));
                    // Brighter green
                    } else {
                        renderer.set_shape_color(inner_rect_2, Some(Color::rgb(0, 125, 0)));
                        // Normal green
                    }

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
                scale_factor,
                inner_size_writer: _,
            } => {
                if let Some(renderer) = &mut self.renderer {
                    println!("Change scale factor to {}", scale_factor);
                    renderer.change_scale_factor(scale_factor);

                    if let Some(window) = &self.window {
                        window.request_redraw();
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.mouse_position = Point2D::new(position.x as f32, position.y as f32);
                if let Some(window) = &self.window {
                    window.request_redraw();
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
