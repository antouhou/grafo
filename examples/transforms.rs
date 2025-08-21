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
        let mut renderer = block_on(grafo::Renderer::new(
            window.clone(),
            physical_size,
            scale_factor,
            true,  // vsync
            false, // transparent
        ));

        // Base rectangle
        let rect = Shape::rect(
            [(100.0, 100.0), (300.0, 200.0)],
            Color::rgb(200, 60, 60),         // Red-ish
            Stroke::new(2.0, Color::BLACK),  // Black stroke
        );
        let _ = renderer.add_shape(rect, None, (0.0, 0.0), None);

        // Second rectangle to demonstrate rotation around its center (set in redraw)
        let rect2 = Shape::rect(
            [(350.0, 100.0), (550.0, 200.0)],
            Color::rgb(60, 200, 60),
            Stroke::new(2.0, Color::BLACK),
        );
        let _ = renderer.add_shape(rect2, None, (0.0, 0.0), None);

        // Third rectangle for scale demonstration (set in redraw)
        let rect3 = Shape::rect(
            [(100.0, 250.0), (300.0, 350.0)],
            Color::rgb(60, 60, 200),
            Stroke::new(2.0, Color::BLACK),
        );
        let _ = renderer.add_shape(rect3, None, (0.0, 0.0), None);

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
                let id1 = renderer.add_shape(
                    Shape::rect(
                        [(100.0, 100.0), (300.0, 200.0)],
                        Color::rgb(200, 60, 60),
                        Stroke::new(2.0, Color::BLACK),
                    ),
                    None,
                    (0.0, 0.0),
                    None,
                );

                let id2 = renderer.add_shape(
                    Shape::rect(
                        [(350.0, 100.0), (550.0, 200.0)],
                        Color::rgb(60, 200, 60),
                        Stroke::new(2.0, Color::BLACK),
                    ),
                    None,
                    (0.0, 0.0),
                    None,
                );

                let id3 = renderer.add_shape(
                    Shape::rect(
                        [(100.0, 250.0), (300.0, 350.0)],
                        Color::rgb(60, 60, 200),
                        Stroke::new(2.0, Color::BLACK),
                    ),
                    None,
                    (0.0, 0.0),
                    None,
                );

                // Compute logical canvas size to convert pixels to NDC
                let (pw, ph) = renderer.size();
                let sf = renderer.scale_factor() as f32;
                let (cw, ch) = (pw as f32 / sf, ph as f32 / sf);

                // 1) Translate first rect by a small oscillating amount in NDC
                let dx_px = 30.0 * self.angle.sin();
                let dy_px = 0.0;
                let dx_ndc = 2.0 * dx_px / cw;
                let dy_ndc = -2.0 * dy_px / ch; // Y axis is flipped in NDC
                renderer.set_shape_transform_cols(
                    id1,
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [dx_ndc, dy_ndc, 0.0, 1.0],
                    ],
                );

                // 2) Rotate second rect around its center
                let cx2 = (350.0 + 550.0) * 0.5;
                let cy2 = (100.0 + 200.0) * 0.5;
                let c2_ndc_x = 2.0 * cx2 / cw - 1.0;
                let c2_ndc_y = 1.0 - 2.0 * cy2 / ch;
                let (s, c) = self.angle.sin_cos();
                // Rotation around point c: R*p + (c - R*c)
                let rx_cx = c * c2_ndc_x + (-s) * c2_ndc_y;
                let ry_cy = s * c2_ndc_x + c * c2_ndc_y;
                let tx = c2_ndc_x - rx_cx;
                let ty = c2_ndc_y - ry_cy;
                renderer.set_shape_transform_cols(
                    id2,
                    [
                        [c, s, 0.0, 0.0],
                        [-s, c, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [tx, ty, 0.0, 1.0],
                    ],
                );

                // 3) Scale third rect around its center
                let cx3 = (100.0 + 300.0) * 0.5;
                let cy3 = (250.0 + 350.0) * 0.5;
                let c3_ndc_x = 2.0 * cx3 / cw - 1.0;
                let c3_ndc_y = 1.0 - 2.0 * cy3 / ch;
                let sx = 1.0 + 0.3 * self.angle.sin();
                let sy = 1.0 + 0.3 * self.angle.cos();
                let tx3 = (1.0 - sx) * c3_ndc_x;
                let ty3 = (1.0 - sy) * c3_ndc_y;
                renderer.set_shape_transform_cols(
                    id3,
                    [
                        [sx, 0.0, 0.0, 0.0],
                        [0.0, sy, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [tx3, ty3, 0.0, 1.0],
                    ],
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

pub fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().expect("To create the event loop");

    let mut app = App::default();
    let _ = event_loop.run_app(&mut app);
}
