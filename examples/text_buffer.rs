use futures::executor::block_on;
use glyphon::cosmic_text;
use grafo::{fontdb, Color, NoopTextDataIter, Stroke};
use grafo::{MathRect, Shape};
use std::sync::{Arc, RwLock};
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::PhysicalKey;
use winit::window::{Window, WindowId};

struct App {
    window: Option<Arc<Window>>,
    renderer: Option<grafo::Renderer<'static>>,
    scroll_state: Arc<RwLock<(f32, f32)>>,
}

impl Default for App {
    fn default() -> Self {
        Self {
            window: None,
            renderer: None,
            scroll_state: Arc::new(RwLock::new((0.0, 0.0))),
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

        let mut renderer = block_on(grafo::Renderer::new(
            window.clone(),
            physical_size,
            scale_factor,
            false, // vsync
            false, // transparent
        ));

        // Load the font
        let roboto_font_ttf = include_bytes!("assets/Roboto-Regular.ttf");
        let roboto_font_source = fontdb::Source::Binary(Arc::new(roboto_font_ttf.to_vec()));
        renderer.load_fonts([roboto_font_source].into_iter());

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
            WindowEvent::KeyboardInput {
                device_id: _,
                event,
                is_synthetic: _,
            } => match event.physical_key {
                PhysicalKey::Unidentified(_) => {}
                PhysicalKey::Code(key_code) => {
                    match key_code {
                        winit::keyboard::KeyCode::ArrowLeft => {
                            self.scroll_state.write().unwrap().0 += 10.0;
                        }
                        winit::keyboard::KeyCode::ArrowRight => {
                            self.scroll_state.write().unwrap().0 -= 10.0;
                        }
                        winit::keyboard::KeyCode::ArrowUp => {
                            self.scroll_state.write().unwrap().1 += 1.0;
                        }
                        winit::keyboard::KeyCode::ArrowDown => {
                            self.scroll_state.write().unwrap().1 -= 1.0;
                        }
                        _ => {}
                    }

                    window.request_redraw();
                }
            },
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(physical_size) => {
                let new_size = (physical_size.width, physical_size.height);
                renderer.resize(new_size);
                window.request_redraw();
            }
            WindowEvent::RedrawRequested => {
                let window_size = window.inner_size();

                // Main window background
                let background = Shape::rect(
                    [
                        (0.0, 0.0),
                        (window_size.width as f32, window_size.height as f32),
                    ],
                    Color::rgb(255, 0, 255),
                    Stroke::new(1.0, Color::rgb(0, 0, 0)),
                );
                let background_shape_id = renderer.add_shape(background, None, None);

                // First text instance
                let background_min = (10.0, 10.0);
                let background_max = (210.0, 32.0);
                let first_text_background = Shape::rect(
                    [background_min, background_max],
                    Color::rgb(200, 200, 200),
                    Stroke::new(1.0, Color::rgb(0, 0, 0)),
                );
                let first_text_background_id = renderer
                    .add_shape(first_text_background, Some(background_shape_id), None);
                let text = "Hello, world! This is a long text that will be clipped if it overflows the bounds.";

                let text_buffer_id = 123;
                let font_size = 20.0;
                let line_height = 1.5;
                let mut font_system = cosmic_text::FontSystem::new();
                let mut swash_cache = cosmic_text::SwashCache::new();
                let text_metrics = cosmic_text::Metrics::new(font_size, font_size * line_height);
                let mut text_buffer = cosmic_text::Buffer::new(&mut font_system, text_metrics);
                // No wrap, so if the text overflows the bounds, it will be clipped
                text_buffer.set_wrap(&mut font_system, cosmic_text::Wrap::None);

                let scroll_value = *self.scroll_state.read().unwrap();
                let scroll = cosmic_text::Scroll::new(0, scroll_value.1, scroll_value.0);

                let text_area_size = (
                    background_max.0 - background_min.0,
                    background_max.1 - background_min.1,
                );

                println!("text_area_size: {text_area_size:?}");

                text_buffer.set_text(
                    &mut font_system,
                    text,
                    &cosmic_text::Attrs::new()
                        .family(cosmic_text::Family::SansSerif)
                        .metadata(text_buffer_id)
                        .color(cosmic_text::Color::rgb(0, 0, 255)),
                    cosmic_text::Shaping::Advanced,
                );

                text_buffer.set_scroll(scroll);
                println!("Scroll: x: {}, y: {}", scroll_value.0, scroll_value.1);
                text_buffer.set_size(
                    &mut font_system,
                    Some(text_area_size.0 * renderer.scale_factor() as f32),
                    Some(text_area_size.1 * renderer.scale_factor() as f32),
                );
                text_buffer.shape_until_scroll(&mut font_system, true);

                println!("Text buffer scroll: {:?}", text_buffer.scroll());

                let text_bounds = MathRect::new(background_min.into(), background_max.into());
                let vertical_offset = 10.0;

                renderer.add_text_buffer(
                    text_buffer,
                    text_bounds,
                    Color::WHITE,
                    vertical_offset,
                    text_buffer_id,
                    Some(first_text_background_id),
                );

                let timer = Instant::now();
                match renderer.render_with_custom_font_system(
                    &mut font_system,
                    &mut swash_cache,
                    None::<NoopTextDataIter>,
                ) {
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
    event_loop.run_app(&mut app).unwrap();
}
