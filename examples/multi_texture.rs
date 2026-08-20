//! Example demonstrating multi-texturing (background + foreground) on a single shape.
//! Run with: `cargo run --example multi_texture`

use grafo::{Color, Renderer, Shape, ShapeDrawCommandOptions, Stroke};
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::Window;

struct App {
    window: Option<Arc<Window>>,
    renderer: Option<Renderer<'static>>,
    bg_tex_id: u64,
    fg_tex_id: u64,
}

impl Default for App {
    fn default() -> Self {
        Self {
            window: None,
            renderer: None,
            bg_tex_id: 100,
            fg_tex_id: 101,
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
                    Ok(_) => {}
                    Err(
                        wgpu::CurrentSurfaceTexture::Lost | wgpu::CurrentSurfaceTexture::Outdated,
                    ) => {
                        let size = renderer.size();
                        renderer.resize(size);
                    }
                    Err(
                        wgpu::CurrentSurfaceTexture::Timeout
                        | wgpu::CurrentSurfaceTexture::Occluded,
                    ) => {
                        // The window is not visible yet (still appearing, minimized, or fully
                        // covered). Ask for another redraw instead of dropping the frame for
                        // good — winit does not request one when the window becomes visible.
                        window.request_redraw();
                    }
                    Err(e) => eprintln!("{e:?}"),
                }
            }
            _ => {}
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::default();
    event_loop.run_app(&mut app).unwrap();
}
