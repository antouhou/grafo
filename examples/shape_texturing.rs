use futures::executor::block_on;
use grafo::{BorderRadii, Shape};
use grafo::{Color, Stroke};
use std::sync::Arc;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

use image::ImageReader;

struct App<'a> {
    window: Option<Arc<Window>>,
    renderer: Option<grafo::Renderer<'a>>,
    rust_logo_png_bytes: Vec<u8>,
    rust_logo_png_dimensions: (u32, u32),
}

impl<'a> Default for App<'a> {
    fn default() -> Self {
        let rust_logo_png_bytes = include_bytes!("assets/rust-logo-256x256-blk.png");
        let rust_logo_png = ImageReader::new(std::io::Cursor::new(rust_logo_png_bytes))
            .with_guessed_format()
            .unwrap()
            .decode()
            .unwrap();
        let rust_logo_rgba = rust_logo_png.as_rgba8().unwrap();
        let rust_logo_png_dimensions = rust_logo_rgba.dimensions();
        let rust_logo_png_bytes = rust_logo_rgba.to_vec();

        Self {
            window: None,
            renderer: None,
            rust_logo_png_bytes,
            rust_logo_png_dimensions,
        }
    }
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

        let renderer = block_on(grafo::Renderer::new(
            window.clone(),
            physical_size,
            scale_factor,
            true,
            true,
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
                let window_size = window.inner_size();

                // Background
                let background = Shape::rect(
                    [
                        (0.0, 0.0),
                        (window_size.width as f32, window_size.height as f32),
                    ],
                    Color::rgb(30, 30, 30),
                    Stroke::new(0.0, Color::rgb(0, 0, 0)),
                );

                // A white rounded rect that we will texture
                let textured_rect = Shape::rounded_rect(
                    [(0.0, 0.0), (300.0, 300.0)],
                    BorderRadii::new(20.0),
                    Color::rgb(255, 255, 255), // white so texture shows un-tinted
                    Stroke::new(2.0, Color::rgb(200, 200, 200)),
                );

                let background_id = renderer.add_shape(background, None, (0.0, 0.0), None);
                let rect_id =
                    renderer.add_shape(textured_rect, Some(background_id), (100.0, 100.0), None);

                // Upload texture once per frame here for demo purposes. In a real app, do this once.
                let texture_id = 100u64;
                renderer
                    .texture_manager()
                    .allocate_texture(texture_id, self.rust_logo_png_dimensions);
                renderer
                    .texture_manager()
                    .load_data_into_texture(
                        texture_id,
                        self.rust_logo_png_dimensions,
                        &self.rust_logo_png_bytes,
                    )
                    .unwrap();

                // Associate the uploaded texture with our shape
                renderer.set_shape_texture(rect_id, Some(texture_id));

                // Draw also the raw image for comparison (not clipped)
                let dims = self.rust_logo_png_dimensions;
                renderer.add_texture_draw_to_queue(
                    texture_id,
                    [
                        (450.0, 100.0),
                        (450.0 + dims.0 as f32, 100.0 + dims.1 as f32),
                    ],
                    None,
                );

                let timer = Instant::now();
                match renderer.render() {
                    Ok(_) => {
                        renderer.clear_draw_queue();
                    }
                    Err(wgpu::SurfaceError::Lost) => renderer.resize(renderer.size()),
                    Err(wgpu::SurfaceError::OutOfMemory) => event_loop.exit(),
                    Err(e) => eprintln!("{e:?}"),
                }
                println!("Render time: {:?}", timer.elapsed());
            }
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                renderer.change_scale_factor(scale_factor);
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
