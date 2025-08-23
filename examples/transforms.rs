use euclid::{default::Transform3D, Angle};
use futures::executor::block_on;
use grafo::Shape;
use grafo::{Color, Stroke};
use lyon::algorithms::hit_test::hit_test_path;
use lyon::algorithms::math::point as algo_point;
use lyon::geom::point;
use lyon::path::FillRule;
use lyon::path::Path;

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

// Convert world point (in pixels) into shape-local coordinates using the same 2D affine
// that the shader applies: world = A * local + t, where
//   A = [[m11, m12], [m21, m22]] and t = [m41, m42].
// Returns None if the transform is not invertible.
fn world_to_local_2d(tx: &Transform3D<f32>, world: (f32, f32)) -> Option<(f32, f32)> {
    // In our vertex shader we transform local (x,y,0,1) by the 4x4 model, then divide by w
    // and treat (x/w, y/w) as pixel-space before canvas normalization. This induces a 2D
    // homography on the z=0 plane: [px, py, pw]^T = H * [x, y, 1]^T with
    //   H = [[m11, m12, m41],
    //        [m21, m22, m42],
    //        [m14, m24, m44]].
    // To hit-test, invert H and map [mx, my, 1] back to local, then divide by w.
    let m = tx;
    let h11 = m.m11;
    let h12 = m.m12;
    let h13 = m.m41;
    let h21 = m.m21;
    let h22 = m.m22;
    let h23 = m.m42;
    let h31 = m.m14;
    let h32 = m.m24;
    let h33 = m.m44;

    // Compute inverse of 3x3 H using adjugate/determinant
    let c11 = h22 * h33 - h23 * h32;
    let c12 = h23 * h31 - h21 * h33;
    let c13 = h21 * h32 - h22 * h31;
    let c21 = h13 * h32 - h12 * h33;
    let c22 = h11 * h33 - h13 * h31;
    let c23 = h12 * h31 - h11 * h32;
    let c31 = h12 * h23 - h13 * h22;
    let c32 = h13 * h21 - h11 * h23;
    let c33 = h11 * h22 - h12 * h21;

    let det = h11 * c11 + h12 * c12 + h13 * c13;
    if det.abs() < 1e-6 {
        return None;
    }
    let inv_det = 1.0 / det;

    // adj(H)^T times inv_det gives H^{-1}
    let i11 = c11 * inv_det;
    let i12 = c21 * inv_det;
    let i13 = c31 * inv_det;
    let i21 = c12 * inv_det;
    let i22 = c22 * inv_det;
    let i23 = c32 * inv_det;
    let i31 = c13 * inv_det;
    let i32 = c23 * inv_det;
    let i33 = c33 * inv_det;

    let mx = world.0;
    let my = world.1;
    let lx = i11 * mx + i12 * my + i13 * 1.0;
    let ly = i21 * mx + i22 * my + i23 * 1.0;
    let lw = i31 * mx + i32 * my + i33 * 1.0;
    if lw.abs() < 1e-6 {
        return None;
    }
    Some((lx / lw, ly / lw))
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
    // Last mouse position in physical pixels (window space)
    last_mouse_pos: Option<(f32, f32)>,
    // Window scale factor
    scale_factor: f64,
    // Lyon paths for our rectangles (local space, origin at (0,0))
    red_path: Path,
    green_path: Path,
    blue_path: Path,
    heart_path: Path,
    perspective_path: Path,
    // Base and hover colors
    red_color: (Color, Color),
    green_color: (Color, Color),
    blue_color: (Color, Color),
    heart_color: (Color, Color),
    perspective_color: (Color, Color),
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

        // Prepare lyon paths for rectangles (200x100 at local origin)
        let build_rect_path = || {
            let mut pb = lyon::path::Path::builder();
            pb.begin(point(0.0, 0.0));
            pb.line_to(point(200.0, 0.0));
            pb.line_to(point(200.0, 100.0));
            pb.line_to(point(0.0, 100.0));
            pb.close();
            pb.build()
        };

        self.red_path = build_rect_path();
        self.green_path = build_rect_path();
        self.blue_path = build_rect_path();
    // Build a heart shape path centered near (0,0)
        let mut hb = lyon::path::Path::builder();
        hb.begin(point(0.0, 30.0));
        hb.cubic_bezier_to(point(0.0, 0.0), point(50.0, 0.0), point(50.0, 30.0));
        hb.cubic_bezier_to(point(50.0, 55.0), point(25.0, 77.0), point(0.0, 100.0));
        hb.cubic_bezier_to(point(-25.0, 77.0), point(-50.0, 55.0), point(-50.0, 30.0));
        hb.cubic_bezier_to(point(-50.0, 0.0), point(0.0, 0.0), point(0.0, 30.0));
        hb.close();
        self.heart_path = hb.build();
    // Simple diamond to showcase perspective
    let mut pb = lyon::path::Path::builder();
    pb.begin(point(0.0, -60.0));
    pb.line_to(point(60.0, 0.0));
    pb.line_to(point(0.0, 60.0));
    pb.line_to(point(-60.0, 0.0));
    pb.close();
    self.perspective_path = pb.build();
        self.red_color = (Color::rgb(200, 60, 60), Color::rgb(255, 120, 120));
        self.green_color = (Color::rgb(60, 200, 60), Color::rgb(120, 255, 120));
        self.blue_color = (Color::rgb(60, 60, 200), Color::rgb(120, 120, 255));
        self.heart_color = (Color::rgb(220, 0, 90), Color::rgb(255, 80, 150));
    self.perspective_color = (Color::rgb(255, 180, 0), Color::rgb(255, 220, 120));

        self.scale_factor = scale_factor;
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
            WindowEvent::CursorMoved { position, .. } => {
                // Convert physical coords to logical to match the renderer's logical space
                let (x, y) = (
                    position.x as f32 / self.scale_factor as f32,
                    position.y as f32 / self.scale_factor as f32,
                );
                self.last_mouse_pos = Some((x, y));
                window.request_redraw();
            }
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                self.scale_factor = scale_factor;
                // Propagate DPI change to the renderer so normalization uses the new logical size
                renderer.change_scale_factor(scale_factor);
                window.request_redraw();
            }
            WindowEvent::Resized(physical_size) => {
                let new_size = (physical_size.width, physical_size.height);
                renderer.resize(new_size);
                window.request_redraw();
            }
            WindowEvent::RedrawRequested => {
                // Background in logical coordinates (renderer normalizes using logical canvas size)
                let logical_w = window.inner_size().width as f32 / self.scale_factor as f32;
                let logical_h = window.inner_size().height as f32 / self.scale_factor as f32;
                let background = Shape::rect(
                    [(0.0, 0.0), (logical_w, logical_h)],
                    Color::BLACK,
                    Stroke::new(1.0, Color::rgb(0, 0, 0)),
                );
                renderer.add_shape(background, None, (0.0, 0.0), None);

                // Compute transforms in euclid space (same as before)
                let red_tx = Transform3D::rotation(0.0, 0.0, 1.0, Angle::degrees(45.0))
                    .then(&Transform3D::translation(100.0, 100.0, 0.0));
                let green_tx = Transform3D::scale(0.5, 0.5, 1.0)
                    .then(&Transform3D::translation(400.0, 100.0, 0.0));

                // Perspective origin at mouse (logical pixels); if no mouse yet, use canvas center.
                let (origin_x, origin_y) = match self.last_mouse_pos {
                    Some((mx, my)) => (mx, my),
                    None => (logical_w * 0.5, logical_h * 0.5),
                };
                let d = 1000.0;
                // Build a perspective "with origin": T(+origin) · P(d) · T(-origin)
                let perspective_with_origin = Transform3D::translation(origin_x, origin_y, 0.0)
                    .then(&Transform3D::perspective(d))
                    .then(&Transform3D::translation(-origin_x, -origin_y, 0.0));

                let blue_tx = Transform3D::rotation(0.0, 1.0, 0.0, Angle::degrees(25.0))
                    .then_rotate(1.0, 0.0, 0.0, Angle::degrees(25.0))
                    .then(&Transform3D::translation(100.0, 300.0, 0.0))
                    .then(&perspective_with_origin);

                // Hover detection: transform mouse point back into local space and test against path's bbox
                let mouse = self.last_mouse_pos;
                let is_hover =
                    |path: &Path, tx: &Transform3D<f32>, mouse: Option<(f32, f32)>| -> bool {
                        let Some((mx, my)) = mouse else {
                            return false;
                        };
                        if let Some((lx, ly)) = world_to_local_2d(tx, (mx, my)) {
                            return hit_test_path(
                                &algo_point(lx, ly),
                                path.iter(),
                                FillRule::NonZero,
                                0.01,
                            );
                        }
                        false
                    };

                let red_hover = is_hover(&self.red_path, &red_tx, mouse);
                let green_hover = is_hover(&self.green_path, &green_tx, mouse);
                let blue_hover = is_hover(&self.blue_path, &blue_tx, mouse);
                // Heart transform: scale and rotate a bit, then translate
                let heart_tx = Transform3D::scale(1.8, 1.8, 1.0)
                    .then(&Transform3D::rotation(0.0, 0.0, 1.0, Angle::degrees(-20.0)))
                    .then(&Transform3D::translation(450.0, 300.0, 0.0));
                let heart_hover = is_hover(&self.heart_path, &heart_tx, mouse);
                // Perspective demo: simulate a camera by giving the model a w-affecting row
                // Start with a tilt around X and some translation in Z, plus tiny perspective.
                let persp = Transform3D::perspective(600.0);
                let tilt = Transform3D::rotation(1.0, 0.0, 0.0, Angle::degrees(60.0));
                let model = tilt.then(&Transform3D::translation(0.0, 0.0, 200.0));
                let perspective_tx = model.then(&Transform3D::translation(500.0, 420.0, 0.0)).then(&persp);
                let perspective_hover = is_hover(&self.perspective_path, &perspective_tx, mouse);

                // Re-add shapes each frame using lyon paths for hit testing and dynamic color
                let red_shape = Shape::Path(grafo::PathShape::new(
                    self.red_path.clone(),
                    if red_hover {
                        self.red_color.1
                    } else {
                        self.red_color.0
                    },
                    Stroke::new(2.0, Color::BLACK),
                ));
                let green_shape = Shape::Path(grafo::PathShape::new(
                    self.green_path.clone(),
                    if green_hover {
                        self.green_color.1
                    } else {
                        self.green_color.0
                    },
                    Stroke::new(2.0, Color::BLACK),
                ));
                let blue_shape = Shape::Path(grafo::PathShape::new(
                    self.blue_path.clone(),
                    if blue_hover {
                        self.blue_color.1
                    } else {
                        self.blue_color.0
                    },
                    Stroke::new(2.0, Color::BLACK),
                ));
                let heart_shape = Shape::Path(grafo::PathShape::new(
                    self.heart_path.clone(),
                    if heart_hover {
                        self.heart_color.1
                    } else {
                        self.heart_color.0
                    },
                    Stroke::new(2.0, Color::BLACK),
                ));
                let perspective_shape = Shape::Path(grafo::PathShape::new(
                    self.perspective_path.clone(),
                    if perspective_hover { self.perspective_color.1 } else { self.perspective_color.0 },
                    Stroke::new(2.0, Color::BLACK),
                ));

                let red = renderer.add_shape(red_shape, None, (0.0, 0.0), None);
                let green = renderer.add_shape(green_shape, None, (0.0, 0.0), None);
                let blue = renderer.add_shape(blue_shape, None, (0.0, 0.0), None);
                let heart = renderer.add_shape(heart_shape, None, (0.0, 0.0), None);
                let perspective = renderer.add_shape(perspective_shape, None, (0.0, 0.0), None);

                renderer.set_shape_transform(red, transform_instance_from_euclid(red_tx));
                renderer.set_shape_transform(green, transform_instance_from_euclid(green_tx));
                renderer.set_shape_transform(blue, transform_instance_from_euclid(blue_tx));
                renderer.set_shape_transform(heart, transform_instance_from_euclid(heart_tx));
                renderer.set_shape_transform(perspective, transform_instance_from_euclid(perspective_tx));

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

    // Initialize app with defaults and placeholders; actual renderer/window set on resume
    let mut app = App {
        window: None,
        renderer: None,
        angle: 0.0,
        last_mouse_pos: None,
        scale_factor: 1.0,
        red_path: Path::new(),
        green_path: Path::new(),
        blue_path: Path::new(),
        heart_path: Path::new(),
    perspective_path: Path::new(),
        red_color: (Color::rgb(200, 60, 60), Color::rgb(255, 120, 120)),
        green_color: (Color::rgb(60, 200, 60), Color::rgb(120, 255, 120)),
        blue_color: (Color::rgb(60, 60, 200), Color::rgb(120, 120, 255)),
        heart_color: (Color::rgb(220, 0, 90), Color::rgb(255, 80, 150)),
    perspective_color: (Color::rgb(255, 180, 0), Color::rgb(255, 220, 120)),
    };
    let _ = event_loop.run_app(&mut app);
}
