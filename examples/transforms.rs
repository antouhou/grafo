use futures::executor::block_on;
use grafo::Shape;
use grafo::{Color, Stroke};
use euclid::{default::Transform3D, Angle};

// Local converter from euclid to grafo's GPU instance layout so we keep euclid out of the main crate.
fn transform_instance_from_euclid(m: Transform3D<f32>) -> grafo::TransformInstance {
    // Euclid stores translation in the last ROW (m41, m42, m43) with column-major notation.
    // Our WGSL uses column-major matrices multiplied by column vectors: model * vec4(x, y, z, 1).
    // Therefore, translation must be in the LAST COLUMN (col3.xyz). Map as follows:
    //   col0 = [m11, m21, m31, m14]
    //   col1 = [m12, m22, m32, m24]
    //   col2 = [m13, m23, m33, m34]
    //   col3 = [m41, m42, m43, m44]
    // For typical affine transforms, m14/m24/m34 are 0.
    grafo::TransformInstance {
        col0: [m.m11, m.m21, m.m31, m.m14],
        col1: [m.m12, m.m22, m.m32, m.m24],
        col2: [m.m13, m.m23, m.m33, m.m34],
        col3: [m.m41, m.m42, m.m43, m.m44],
    }
}
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

#[derive(Default)]
struct App<'a> {
    window: Option<Arc<Window>>,
    renderer: Option<grafo::Renderer<'a>>,
    angle: f32,
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
        let renderer = block_on(grafo::Renderer::new(
            window.clone(),
            physical_size,
            scale_factor,
            true,  // vsync
            false, // transparent
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
                let background = Shape::rect(
                    [
                        (0.0, 0.0),
                        (window.inner_size().width as f32, window.inner_size().height as f32),
                    ],
                    Color::BLACK, // Magenta background
                    Stroke::new(1.0, Color::rgb(0, 0, 0)), // Black stroke
                );
                renderer.add_shape(background, None, (0.0, 0.0), None);

                // Re-add shapes each frame and apply per-shape transforms
                let red = renderer.add_shape(
                    Shape::rect(
                        [(0.0, 0.0), (200.0, 100.0)],
                        Color::rgb(200, 60, 60),
                        Stroke::new(2.0, Color::BLACK),
                    ),
                    None,
                    (0.0, 0.0),
                    None,
                );
                let green = renderer.add_shape(
                    Shape::rect(
                        [(0.0, 0.0), (200.0, 100.0)],
                        Color::rgb(60, 200, 60),
                        Stroke::new(2.0, Color::BLACK),
                    ),
                    None,
                    (0.0, 0.0),
                    None,
                );
                let blue = renderer.add_shape(
                    Shape::rect(
                        [(0.0, 0.0), (200.0, 100.0)],
                        Color::rgb(60, 60, 200),
                        Stroke::new(2.0, Color::BLACK),
                    ),
                    None,
                    (0.0, 0.0),
                    None,
                );

                renderer.set_shape_transform(red, transform_instance_from_euclid(
                    Transform3D::
                        rotation(0.0, 0.0, 1.0, Angle::degrees(45.0))
                        .then(&Transform3D::translation(100.0, 100.0, 0.0))
                ));
                renderer.set_shape_transform(green, transform_instance_from_euclid(Transform3D::scale(0.5, 0.5, 1.0).then(&Transform3D::translation(400.0, 100.0, 0.0))));
                renderer.set_shape_transform(blue, transform_instance_from_euclid(Transform3D::translation(100.0, 300.0, 0.0)));

                // Advance animation angle
                self.angle = (self.angle + 0.02) % (std::f32::consts::TAU);

                match renderer.render() {
                    Ok(_) => {
                        renderer.clear_draw_queue();
                        window.request_redraw();
                    }
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
    let event_loop = EventLoop::new().expect("To create the event loop");

    let mut app = App::default();
    let _ = event_loop.run_app(&mut app);
}
