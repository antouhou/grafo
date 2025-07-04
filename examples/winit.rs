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
    rust_logo_png_dimensions_f32: (f32, f32),
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
        let rust_logo_png_dimensions_f32 = (
            rust_logo_png_dimensions.0 as f32,
            rust_logo_png_dimensions.1 as f32,
        );
        let rust_logo_png_bytes = rust_logo_rgba.to_vec();

        Self {
            window: None,
            renderer: None,
            rust_logo_png_bytes,
            rust_logo_png_dimensions,
            rust_logo_png_dimensions_f32,
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

                let background = Shape::rect(
                    [
                        (0.0, 0.0),
                        (window_size.width as f32, window_size.height as f32),
                    ],
                    Color::rgb(255, 255, 255),
                    Stroke::new(1.0, Color::rgb(0, 0, 0)),
                );

                let red = Shape::rounded_rect(
                    [(0.0, 0.0), (200.0, 200.0)],
                    BorderRadii::new(0.0),
                    Color::rgb(255, 0, 0),
                    Stroke::new(1.0, Color::rgb(0, 0, 0)),
                );

                let green = Shape::rounded_rect(
                    [(0.0, 0.0), (200.0, 200.0)],
                    BorderRadii::new(0.0),
                    Color::rgb(0, 255, 0),
                    Stroke::new(1.0, Color::rgb(0, 0, 0)),
                );

                let blue = Shape::rounded_rect(
                    [(0.0, 0.0), (200.0, 200.0)],
                    BorderRadii::new(10.0),
                    Color::rgb(0, 0, 255),
                    Stroke::new(1.0, Color::rgb(0, 0, 0)),
                );

                let yellow = Shape::rounded_rect(
                    [(0.0, 0.0), (150.0, 150.0)],
                    BorderRadii::new(0.0),
                    Color::rgb(255, 255, 0),
                    Stroke::new(1.0, Color::rgb(0, 0, 0)),
                );

                let white = Shape::rounded_rect(
                    [(0.0, 0.0), (20.0, 20.0)],
                    BorderRadii::new(0.0),
                    Color::rgb(255, 255, 255),
                    Stroke::new(1.0, Color::rgb(0, 0, 0)),
                );

                let shape_that_doesnt_fit = Shape::rounded_rect(
                    [(0.0, 0.0), (20.0, 20.0)],
                    BorderRadii::new(0.0),
                    Color::rgb(255, 255, 255),
                    Stroke::new(1.0, Color::rgb(0, 0, 0)),
                );

                let background_id = renderer.add_shape(background, None, (0.0, 0.0), None);
                let red_id = renderer.add_shape(red, Some(background_id), (0.0, 0.0), None);
                let green_id = renderer.add_shape(green, Some(red_id), (100.0, 100.0), None);
                let blue_id = renderer.add_shape(blue, Some(green_id), (150.0, 150.0), None);
                renderer.add_shape(yellow, Some(green_id), (0.0, 0.0), None);
                renderer.add_shape(white, Some(red_id), (0.0, 0.0), None);
                renderer.add_shape(shape_that_doesnt_fit, Some(blue_id), (0.0, 0.0), None);

                renderer.add_rgba_image(
                    &self.rust_logo_png_bytes,
                    self.rust_logo_png_dimensions,
                    [
                        (100.0, 100.0),
                        (
                            100.0 + self.rust_logo_png_dimensions_f32.0,
                            100.0 + self.rust_logo_png_dimensions_f32.1,
                        ),
                    ],
                    Some(red_id),
                );

                renderer.add_rgba_image(
                    &self.rust_logo_png_bytes,
                    self.rust_logo_png_dimensions,
                    [
                        (200.0, 200.0),
                        (
                            200.0 + self.rust_logo_png_dimensions_f32.0,
                            200.0 + self.rust_logo_png_dimensions_f32.1,
                        ),
                    ],
                    Some(background_id),
                );

                renderer.add_rgba_image(
                    &self.rust_logo_png_bytes,
                    self.rust_logo_png_dimensions,
                    [
                        (400.0, 400.0),
                        (
                            400.0 + self.rust_logo_png_dimensions_f32.0,
                            400.0 + self.rust_logo_png_dimensions_f32.0,
                        ),
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
