//! Example demonstrating multi-texturing (background + foreground) on a single shape.
//! Run with: `cargo run --example multi_texture`

use grafo::{Color, Renderer, Shape, ShapeDrawCommandOptions, Stroke};
use std::sync::Arc;
use std::time::{Duration, Instant};
use winit::application::ApplicationHandler;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::Window;

/// How long to wait before retrying a frame that was skipped because the surface
/// reported it was not visible (`Occluded`/`Timeout`).
const OCCLUDED_RETRY_DELAY: Duration = Duration::from_millis(50);

struct App {
    window: Option<Arc<Window>>,
    renderer: Option<Renderer<'static>>,
    bg_tex_id: u64,
    fg_tex_id: u64,
    /// Pending retry of a frame skipped because the window was not visible.
    redraw_retry_at: Option<Instant>,
}

impl Default for App {
    fn default() -> Self {
        Self {
            window: None,
            renderer: None,
            bg_tex_id: 100,
            fg_tex_id: 101,
            redraw_retry_at: None,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes())
                .unwrap(),
        );
        let physical_size = (800, 600);
        let scale_factor = 1.0;
        let mut renderer = futures::executor::block_on(Renderer::new(
            window.clone(),
            physical_size,
            scale_factor,
            true,
            false,
            1, // msaa_samples
        ));

        // Allocate two textures (background checker, foreground circle mask for demo)
        let tex_mgr = renderer.texture_manager();

        let w = 256u32;
        let h = 256u32;
        // Background: simple 2-color checkerboard premultiplied
        let mut bg = vec![0u8; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                let idx = ((y * w + x) * 4) as usize;
                let checker = ((x / 32) + (y / 32)) % 2 == 0;
                let c = if checker { 60 } else { 180 };
                bg[idx] = c;
                bg[idx + 1] = c;
                bg[idx + 2] = c;
                bg[idx + 3] = 255;
            }
        }
        tex_mgr.allocate_texture_with_data(self.bg_tex_id, (w, h), &bg);

        // Foreground: white circle with soft edge over transparent
        let mut fg = vec![0u8; (w * h * 4) as usize];
        let cx = w as f32 / 2.0;
        let cy = h as f32 / 2.0;
        let r = w.min(h) as f32 * 0.4;
        for y in 0..h {
            for x in 0..w {
                let idx = ((y * w + x) * 4) as usize;
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let d = (dx * dx + dy * dy).sqrt();
                if d <= r {
                    let a = ((r - d) / r).clamp(0.0, 1.0);
                    let alpha = (a * 255.0) as u8;
                    fg[idx] = 255;
                    fg[idx + 1] = 255;
                    fg[idx + 2] = 255;
                    fg[idx + 3] = alpha;
                }
            }
        }
        tex_mgr.allocate_texture_with_data(self.fg_tex_id, (w, h), &fg);

        renderer
            .add_shape(
                Shape::rect(
                    [(100.0, 100.0), (500.0, 400.0)],
                    Stroke::new(1.0, Color::BLACK),
                ),
                None,
                None,
                ShapeDrawCommandOptions::new()
                    .color(Color::rgb(200, 200, 200))
                    .background_texture_id(self.bg_tex_id)
                    .foreground_texture_id(self.fg_tex_id),
            )
            .unwrap();
        self.window = Some(window);
        self.renderer = Some(renderer);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        use winit::event::WindowEvent;

        let (Some(window), Some(renderer)) = (&self.window, &mut self.renderer) else {
            return;
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(physical_size) => {
                renderer.resize((physical_size.width, physical_size.height));
                window.request_redraw();
            }
            WindowEvent::RedrawRequested => {
                // The draw queue is populated once in `resumed` and persists across frames.
                match renderer.render() {
                    Ok(_) => {
                        self.redraw_retry_at = None;
                    }
                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                        let size = renderer.size();
                        renderer.resize(size);
                    }
                    Err(wgpu::SurfaceError::Timeout) => {
                        // The window is not visible yet (still appearing, minimized, or fully
                        // covered). Retry shortly instead of busy-looping redraws — winit does
                        // not request one when the window becomes visible again. `WaitUntil`
                        // wakes the event loop without spinning while the window stays hidden.
                        let retry_at = Instant::now() + OCCLUDED_RETRY_DELAY;
                        self.redraw_retry_at = Some(retry_at);
                        event_loop.set_control_flow(ControlFlow::WaitUntil(retry_at));
                    }
                    Err(e) => eprintln!("{e:?}"),
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let Some(retry_at) = self.redraw_retry_at else {
            // Clear a stale WaitUntil deadline left behind when a successful
            // render cancelled the pending retry before it fired.
            event_loop.set_control_flow(ControlFlow::Wait);
            return;
        };
        if Instant::now() >= retry_at {
            self.redraw_retry_at = None;
            if let Some(window) = &self.window {
                window.request_redraw();
            }
            event_loop.set_control_flow(ControlFlow::Wait);
        } else {
            event_loop.set_control_flow(ControlFlow::WaitUntil(retry_at));
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::default();
    event_loop.run_app(&mut app).unwrap();
}
