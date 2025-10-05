//! Example demonstrating multi-texturing (background + foreground) on a single shape.
//! Run with: `cargo run --example multi_texture`

use grafo::{Color, Renderer, Shape, Stroke, TextureLayer};
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::Window;

struct App {
    renderer: Option<Renderer<'static>>,
    bg_tex_id: u64,
    fg_tex_id: u64,
    shape_id: Option<usize>,
}

impl Default for App {
    fn default() -> Self {
        Self {
            renderer: None,
            bg_tex_id: 100,
            fg_tex_id: 101,
            shape_id: None,
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
            window,
            physical_size,
            scale_factor,
            true,
            false,
        ));

        // Create a simple rectangle shape
        let shape_id = renderer.add_shape(
            Shape::rect(
                [(100.0, 100.0), (500.0, 400.0)],
                Color::rgb(200, 200, 200),
                Stroke::new(1.0, Color::BLACK),
            ),
            None,
            None,
        );

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

        // Assign textures to background + foreground layers
        renderer.set_shape_texture_on(shape_id, TextureLayer::Background, Some(self.bg_tex_id));
        renderer.set_shape_texture_on(shape_id, TextureLayer::Foreground, Some(self.fg_tex_id));

        self.shape_id = Some(shape_id);
        self.renderer = Some(renderer);

        if let Some(r) = self.renderer.as_mut() {
            let _ = r.render();
            r.clear_draw_queue();
        }
    }

    fn window_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _id: winit::window::WindowId,
        _event: winit::event::WindowEvent,
    ) {
        // No interactivity in this minimal example; a real app would handle events here.
    }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::default();
    event_loop.run_app(&mut app).unwrap();
}
