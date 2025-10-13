use euclid::{default::Transform3D, Angle};
use futures::executor::block_on;
use grafo::{premultiply_rgba8_srgb_inplace, Shape};
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
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

// Simple helpers to build a few paths for the demo
fn build_rect_path(w: f32, h: f32) -> Path {
    let mut pb = Path::builder();
    pb.begin(point(0.0, 0.0));
    pb.line_to(point(w, 0.0));
    pb.line_to(point(w, h));
    pb.line_to(point(0.0, h));
    pb.close();
    pb.build()
}

fn build_heart_path() -> Path {
    // A rough heart shape centered around (0,0) extending mostly in +Y
    let mut hb = Path::builder();
    hb.begin(point(0.0, 30.0));
    hb.cubic_bezier_to(point(0.0, 0.0), point(50.0, 0.0), point(50.0, 30.0));
    hb.cubic_bezier_to(point(50.0, 55.0), point(25.0, 77.0), point(0.0, 92.0));
    hb.cubic_bezier_to(point(-25.0, 77.0), point(-50.0, 55.0), point(-50.0, 30.0));
    hb.cubic_bezier_to(point(-50.0, 0.0), point(0.0, 0.0), point(0.0, 30.0));
    hb.close();
    hb.build()
}

fn build_perspective_demo_path() -> Path {
    // A simple trapezoid to suggest perspective
    let mut pb = Path::builder();
    pb.begin(point(-60.0, 0.0));
    pb.line_to(point(60.0, 0.0));
    pb.line_to(point(100.0, 40.0));
    pb.line_to(point(-100.0, 40.0));
    pb.close();
    pb.build()
}

#[derive(Default)]
struct App<'a> {
    window: Option<Arc<Window>>,
    renderer: Option<grafo::Renderer<'a>>,
    angle: f32,
    // Last mouse position in physical pixels (window space)
    last_mouse_pos: Option<(f32, f32)>,
    // Accumulated orbit angles in degrees (mouse-driven)
    orbit_yaw_deg: f32,
    orbit_pitch_deg: f32,
    // Orbit control decoupling: update yaw/pitch only during drag
    orbit_dragging: bool,
    orbit_last_mouse_pos: Option<(f32, f32)>,
    // User-tweakable settings
    orbit_sensitivity: f32,  // degrees per logical pixel
    blue_perspective_d: f32, // perspective distance for blue shape
    blue_follow_mouse: bool, // whether perspective origin follows mouse
    blue_pos: (f32, f32),    // world position (top-left) of blue rect
    blue_size: (f32, f32),   // local size of blue rect
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
    // Jelly wobble demo
    jelly_path: Path,
    jelly_color: (Color, Color),
    rust_logo_png_dimensions: (u32, u32),
    rust_logo_png_bytes: Vec<u8>,
    rust_logo_texture_id: u64,
    perspective_color: (Color, Color),
}

// Keyboard controls in this example:
// - Arrow Left/Right: orbit yaw -/+ (rotate camera around blue shape)
// - Arrow Up/Down: orbit pitch -/+ (tilt camera around blue shape)
// - [ / ]: decrease / increase the blue shape perspective distance (strength)
// - F: toggle following mouse for camera origin (on/off)
// - R: reset orbit (yaw=0, pitch=0)

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

        // Load a demo texture (Rust logo) and upload it once
        let rust_logo_png_bytes = include_bytes!("assets/rust-logo-256x256-blk.png");
        let rust_logo_png = image::ImageReader::new(std::io::Cursor::new(rust_logo_png_bytes))
            .with_guessed_format()
            .unwrap()
            .decode()
            .unwrap();
        let rust_logo_rgba = rust_logo_png.as_rgba8().unwrap();
        let rust_logo_png_dimensions = rust_logo_rgba.dimensions();
        let mut rust_logo_png_bytes = rust_logo_rgba.to_vec();
        // Premultiply to avoid fringes when minifying near transparent edges
        premultiply_rgba8_srgb_inplace(&mut rust_logo_png_bytes);
        let rust_logo_texture_id = 1u64;
        renderer
            .texture_manager()
            .allocate_texture_with_data(
                rust_logo_texture_id,
                rust_logo_png_dimensions,
                &rust_logo_png_bytes,
            );

        // Build demo paths
        self.red_path = build_rect_path(200.0, 100.0);
        self.green_path = build_rect_path(200.0, 100.0);
        self.blue_path = build_rect_path(self.blue_size.0, self.blue_size.1);
        self.jelly_path = build_rect_path(200.0, 100.0);
        self.heart_path = build_heart_path();
        self.perspective_path = build_perspective_demo_path();

        // Colors for hover states
        self.red_color = (Color::rgb(200, 60, 60), Color::rgb(255, 120, 120));
        self.green_color = (Color::rgb(60, 200, 60), Color::rgb(120, 255, 120));
        self.blue_color = (Color::rgb(60, 60, 200), Color::rgb(120, 120, 255));
        self.heart_color = (Color::rgb(220, 0, 90), Color::rgb(255, 80, 150));
        self.perspective_color = (Color::rgb(255, 180, 0), Color::rgb(255, 220, 120));

        // Save state
        self.scale_factor = scale_factor;
        self.rust_logo_png_dimensions = rust_logo_png_dimensions;
        self.rust_logo_png_bytes = rust_logo_png_bytes;
        self.rust_logo_texture_id = rust_logo_texture_id;
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
                // Only update orbit while dragging; use a separate last position to avoid
                // interfering with the perspective origin tracking.
                if self.orbit_dragging {
                    if let Some((px, py)) = self.orbit_last_mouse_pos {
                        let dx = x - px;
                        let dy = y - py;
                        let sens = self.orbit_sensitivity; // degrees per logical pixel
                        self.orbit_yaw_deg = (self.orbit_yaw_deg + dx * sens) % 360.0;
                        self.orbit_pitch_deg =
                            (self.orbit_pitch_deg + dy * sens).clamp(-80.0, 80.0);
                    }
                    self.orbit_last_mouse_pos = Some((x, y));
                }
                window.request_redraw();
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if button == winit::event::MouseButton::Left {
                    match state {
                        winit::event::ElementState::Pressed => {
                            self.orbit_dragging = true;
                            // Start orbit deltas from current cursor pos if available
                            self.orbit_last_mouse_pos = self.last_mouse_pos;
                        }
                        winit::event::ElementState::Released => {
                            self.orbit_dragging = false;
                            self.orbit_last_mouse_pos = None;
                        }
                    }
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                use winit::event::ElementState;
                if event.state == ElementState::Pressed {
                    let yaw_step = 3.0f32;
                    let pitch_step = 3.0f32;
                    let dist_step = 100.0f32;
                    let key = event.logical_key.clone();
                    match key {
                        Key::Named(NamedKey::ArrowLeft) => {
                            self.orbit_yaw_deg = (self.orbit_yaw_deg - yaw_step).rem_euclid(360.0);
                        }
                        Key::Named(NamedKey::ArrowRight) => {
                            self.orbit_yaw_deg = (self.orbit_yaw_deg + yaw_step).rem_euclid(360.0);
                        }
                        Key::Named(NamedKey::ArrowUp) => {
                            self.orbit_pitch_deg =
                                (self.orbit_pitch_deg - pitch_step).clamp(-80.0, 80.0);
                        }
                        Key::Named(NamedKey::ArrowDown) => {
                            self.orbit_pitch_deg =
                                (self.orbit_pitch_deg + pitch_step).clamp(-80.0, 80.0);
                        }
                        Key::Character(ch) if ch == "[" => {
                            self.blue_perspective_d =
                                (self.blue_perspective_d - dist_step).max(1.0);
                        }
                        Key::Character(ch) if ch == "]" => {
                            self.blue_perspective_d =
                                (self.blue_perspective_d + dist_step).max(1.0);
                        }
                        Key::Character(ch) if ch.eq_ignore_ascii_case("f") => {
                            self.blue_follow_mouse = !self.blue_follow_mouse;
                        }
                        Key::Character(ch) if ch.eq_ignore_ascii_case("r") => {
                            self.orbit_yaw_deg = 0.0;
                            self.orbit_pitch_deg = 0.0;
                        }
                        _ => {}
                    }
                    // Trigger redraw after parameter change
                    window.request_redraw();
                }
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
                    Stroke::new(1.0, Color::rgb(0, 0, 0)),
                );
                let background_id = renderer.add_shape(background, None, None);
                renderer.set_shape_color(background_id, Some(Color::BLACK));

                // Compute transforms in euclid space (same as before)
                let red_tx = Transform3D::rotation(0.0, 0.0, 1.0, Angle::degrees(45.0))
                    .then(&Transform3D::translation(100.0, 100.0, 0.0));
                let green_tx = Transform3D::scale(0.5, 0.5, 1.0)
                    .then(&Transform3D::translation(400.0, 100.0, 0.0));

                // Blue shape: rotate around Y by base 45° + mouse-driven yaw, and around X by
                // a mouse-driven pitch; also simulate a per-shape "camera" by sliding the
                // perspective origin in both X and Y with the mouse.
                let d = self.blue_perspective_d; // perspective distance (bigger = subtler perspective)
                let blue_pos = self.blue_pos;
                let blue_size = self.blue_size; // local rect path size
                let blue_center_local = (blue_size.0 * 0.5, blue_size.1 * 0.5); // local pivot
                let blue_center = (
                    blue_pos.0 + blue_size.0 * 0.5,
                    blue_pos.1 + blue_size.1 * 0.5,
                );
                let (origin_x_for_blue, origin_y_for_blue) = if self.blue_follow_mouse {
                    match self.last_mouse_pos {
                        Some((mx, my)) => (mx, my), // react to both horizontal and vertical motion
                        None => blue_center,
                    }
                } else {
                    blue_center
                };
                let blue_perspective =
                    Transform3D::translation(origin_x_for_blue, origin_y_for_blue, 0.0)
                        .then(&Transform3D::perspective(d))
                        .then(&Transform3D::translation(
                            -origin_x_for_blue,
                            -origin_y_for_blue,
                            0.0,
                        ));

                let yaw = 45.0 + self.orbit_yaw_deg; // base + yaw
                let pitch = self.orbit_pitch_deg; // pitch

                // Rotate around the shape's local center to simulate orbiting
                let blue_tx =
                    Transform3D::translation(-blue_center_local.0, -blue_center_local.1, 0.0)
                        .then(&Transform3D::rotation(0.0, 1.0, 0.0, Angle::degrees(yaw)))
                        .then(&Transform3D::rotation(1.0, 0.0, 0.0, Angle::degrees(pitch)))
                        .then(&Transform3D::translation(
                            blue_center_local.0,
                            blue_center_local.1,
                            0.0,
                        ))
                        .then(&Transform3D::translation(blue_pos.0, blue_pos.1, 0.0))
                        .then(&blue_perspective);

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
                // Jelly wobble: bottom-anchored rectangle that squashes and rotates slightly
                let jelly_pos = (750.0, 120.0); // world position (top-left approx)
                let jelly_local_size = (200.0, 100.0);
                let jelly_pivot = (jelly_local_size.0 * 0.5, jelly_local_size.1); // bottom-center
                let s = (self.angle * 3.0).sin();
                let wobble_x = 1.0 + 0.14 * s;
                let wobble_y = 1.0 - 0.14 * s;
                let wobble_rot = 5.0 * (self.angle * 2.0).sin(); // degrees
                let jelly_tx = Transform3D::translation(-jelly_pivot.0, -jelly_pivot.1, 0.0)
                    .then(&Transform3D::scale(wobble_x, wobble_y, 1.0))
                    .then(&Transform3D::rotation(
                        0.0,
                        0.0,
                        1.0,
                        Angle::degrees(wobble_rot),
                    ))
                    .then(&Transform3D::translation(jelly_pivot.0, jelly_pivot.1, 0.0))
                    .then(&Transform3D::translation(jelly_pos.0, jelly_pos.1, 0.0));
                let jelly_hover = is_hover(&self.jelly_path, &jelly_tx, mouse);
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
                let perspective_tx = model
                    .then(&Transform3D::translation(500.0, 420.0, 0.0))
                    .then(&persp);
                let perspective_hover = is_hover(&self.perspective_path, &perspective_tx, mouse);

                // Re-add shapes each frame using lyon paths for hit testing and dynamic color
                let red_shape = Shape::Path(grafo::PathShape::new(
                    self.red_path.clone(),
                    Stroke::new(2.0, Color::BLACK),
                ));
                let green_shape = Shape::Path(grafo::PathShape::new(
                    self.green_path.clone(),
                    Stroke::new(2.0, Color::BLACK),
                ));
                let blue_shape = Shape::Path(grafo::PathShape::new(
                    self.blue_path.clone(),
                    Stroke::new(2.0, Color::BLACK),
                ));
                let jelly_shape = Shape::Path(grafo::PathShape::new(
                    self.jelly_path.clone(),
                    Stroke::new(2.0, Color::BLACK),
                ));
                let heart_shape = Shape::Path(grafo::PathShape::new(
                    self.heart_path.clone(),
                    Stroke::new(2.0, Color::BLACK),
                ));
                let perspective_shape = Shape::Path(grafo::PathShape::new(
                    self.perspective_path.clone(),
                    Stroke::new(2.0, Color::BLACK),
                ));

                let red = renderer.add_shape(red_shape, None, None);
                let green = renderer.add_shape(green_shape, None, None);
                let blue = renderer.add_shape(blue_shape, None, None);
                let jelly = renderer.add_shape(jelly_shape, None, None);
                let heart = renderer.add_shape(heart_shape, None, None);
                let perspective = renderer.add_shape(perspective_shape, None, None);

                // Set per-instance colors
                renderer.set_shape_color(
                    red,
                    Some(if red_hover { self.red_color.1 } else { self.red_color.0 }),
                );
                renderer.set_shape_color(
                    green,
                    Some(if green_hover { self.green_color.1 } else { self.green_color.0 }),
                );
                renderer.set_shape_color(
                    blue,
                    Some(if blue_hover { self.blue_color.1 } else { self.blue_color.0 }),
                );
                renderer.set_shape_color(
                    jelly,
                    Some(if jelly_hover { self.jelly_color.1 } else { self.jelly_color.0 }),
                );
                renderer.set_shape_color(
                    heart,
                    Some(if heart_hover { self.heart_color.1 } else { self.heart_color.0 }),
                );
                renderer.set_shape_color(
                    perspective,
                    Some(if perspective_hover {
                        self.perspective_color.1
                    } else {
                        self.perspective_color.0
                    }),
                );

                // Give the blue shape a texture
                renderer.set_shape_texture(blue, Some(self.rust_logo_texture_id));

                renderer.set_shape_transform(red, transform_instance_from_euclid(red_tx));
                renderer.set_shape_transform(green, transform_instance_from_euclid(green_tx));
                renderer.set_shape_transform(blue, transform_instance_from_euclid(blue_tx));
                renderer.set_shape_transform(jelly, transform_instance_from_euclid(jelly_tx));
                renderer.set_shape_transform(heart, transform_instance_from_euclid(heart_tx));
                renderer.set_shape_transform(
                    perspective,
                    transform_instance_from_euclid(perspective_tx),
                );

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

// Helper methods to tweak orbit/camera and blue shape settings for experimentation.
impl<'a> App<'a> {
    /// Set absolute yaw in degrees (adds to the base 45°).
    pub fn set_orbit_yaw_deg(&mut self, deg: f32) {
        self.orbit_yaw_deg = deg;
    }

    /// Set absolute pitch in degrees (will be clamped to [-80, 80]).
    pub fn set_orbit_pitch_deg(&mut self, deg: f32) {
        self.orbit_pitch_deg = deg.clamp(-80.0, 80.0);
    }

    /// Increment yaw by delta degrees.
    pub fn add_orbit_yaw_deg(&mut self, delta: f32) {
        self.orbit_yaw_deg = (self.orbit_yaw_deg + delta) % 360.0;
    }

    /// Increment pitch by delta degrees (clamped to [-80, 80]).
    pub fn add_orbit_pitch_deg(&mut self, delta: f32) {
        self.orbit_pitch_deg = (self.orbit_pitch_deg + delta).clamp(-80.0, 80.0);
    }

    /// Reset yaw/pitch to 0.
    pub fn reset_orbit(&mut self) {
        self.orbit_yaw_deg = 0.0;
        self.orbit_pitch_deg = 0.0;
    }

    /// Set orbit mouse sensitivity in degrees per logical pixel.
    pub fn set_orbit_sensitivity(&mut self, sensitivity: f32) {
        self.orbit_sensitivity = sensitivity.max(0.0);
    }

    /// Set perspective distance for the blue shape (smaller => stronger perspective).
    pub fn set_blue_perspective_distance(&mut self, d: f32) {
        // Avoid zero/negative distances which would break the projection
        self.blue_perspective_d = d.max(1.0);
    }

    /// Follow mouse for perspective origin (true) or use the shape center (false).
    pub fn set_blue_follow_mouse(&mut self, enabled: bool) {
        self.blue_follow_mouse = enabled;
    }

    /// Set the world position (top-left) of the blue rectangle.
    pub fn set_blue_position(&mut self, x: f32, y: f32) {
        self.blue_pos = (x, y);
    }

    /// Set the local size of the blue rectangle.
    pub fn set_blue_size(&mut self, width: f32, height: f32) {
        self.blue_size = (width.max(1.0), height.max(1.0));
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
        orbit_yaw_deg: 0.0,
        orbit_pitch_deg: 0.0,
        orbit_dragging: false,
        orbit_last_mouse_pos: None,
        orbit_sensitivity: 0.08,
        blue_perspective_d: 2000.0,
        blue_follow_mouse: true,
        blue_pos: (100.0, 300.0),
        blue_size: (200.0, 100.0),
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
        jelly_path: Path::new(),
        jelly_color: (Color::rgb(90, 200, 255), Color::rgb(140, 235, 255)),
        perspective_color: (Color::rgb(255, 180, 0), Color::rgb(255, 220, 120)),
        rust_logo_png_dimensions: (0, 0),
        rust_logo_png_bytes: Vec::new(),
        rust_logo_texture_id: 0,
    };
    let _ = event_loop.run_app(&mut app);
}
