use futures::executor::block_on;
use grafo::{BorderRadii, Shape, TextureManager};
use grafo::{Color, ShapeDrawCommandOptions};
use grafo::{RendererContext, Surface};
use image::ImageReader;
use std::sync::Arc;
use std::time::Instant;
use tracing::warn;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

mod window_rendering;

const RUST_LOGO_TEXTURE_ID: u64 = 100;

struct App {
    window: Option<Arc<Window>>,
    renderer: Option<grafo::Renderer>,
    surface: Option<Surface>,
    rust_logo_png_bytes: Vec<u8>,
    rust_logo_png_dimensions: (u32, u32),
}

impl Default for App {
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
            surface: None,
            rust_logo_png_bytes,
            rust_logo_png_dimensions,
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

        let window_size = window.inner_size();
        let scale_factor = window.scale_factor();
        let physical_size = (window_size.width, window_size.height);

        let context = block_on(RendererContext::new());
        let surface = Surface::new(&context, window.clone(), physical_size, true, true)
            .expect("Failed to create surface");
        let renderer = grafo::Renderer::new_with_context(context, physical_size, scale_factor, 1);
        renderer.texture_manager().allocate_texture_with_data(
            RUST_LOGO_TEXTURE_ID,
            self.rust_logo_png_dimensions,
            &self.rust_logo_png_bytes,
        );

        self.window = Some(window);
        self.renderer = Some(renderer);
        self.surface = Some(surface);
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
        let Some(surface) = &mut self.surface else {
            return;
        };

        if window_id != window.id() {
            return;
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(physical_size) => {
                let new_size = (physical_size.width, physical_size.height);
                surface.resize(new_size);
                window.request_redraw();
            }
            WindowEvent::RedrawRequested => {
                renderer.clear_draw_queue();
                let window_size = window.inner_size();

                // Background
                let background = Shape::rect([
                    (0.0, 0.0),
                    (window_size.width as f32, window_size.height as f32),
                ]);
                let background_id = renderer
                    .add_shape(
                        background,
                        None,
                        None,
                        ShapeDrawCommandOptions::new().color(Color::rgb(30, 30, 30)),
                    )
                    .unwrap();

                // A white rounded rect that we will texture
                let textured_rect =
                    Shape::rounded_rect([(0.0, 0.0), (300.0, 300.0)], BorderRadii::new(20.0));
                renderer
                    .add_shape(
                        textured_rect,
                        Some(background_id),
                        None,
                        ShapeDrawCommandOptions::new()
                            .color(Color::rgb(255, 255, 255))
                            .background_texture_id(RUST_LOGO_TEXTURE_ID)
                            .transform(grafo::TransformInstance::translation(100.0, 100.0)),
                    )
                    .unwrap();

                let timer = Instant::now();
                window_rendering::render(renderer, surface, event_loop);
                println!("Render time: {:?}", timer.elapsed());
            }
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                if let Err(error) = renderer.change_scale_factor(scale_factor) {
                    warn!(%error, scale_factor, "scale factor change rejected");
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
