/// MSAA example: Toggle MSAA with spacebar to see anti-aliasing effect.
///
/// Renders a triangle and a rounded rectangle so you can visually
/// compare edge quality between MSAA off (1x) and MSAA on (4x).
use futures::executor::block_on;
use grafo::{BorderRadii, Shape};
use grafo::{Color, Stroke};
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

struct App<'a> {
    window: Option<Arc<Window>>,
    renderer: Option<grafo::Renderer<'a>>,
    msaa_enabled: bool,
}

impl<'a> Default for App<'a> {
    fn default() -> Self {
        Self {
            window: None,
            renderer: None,
            msaa_enabled: true,
        }
    }
}

impl<'a> ApplicationHandler for App<'a> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Grafo MSAA Demo — Press SPACE to toggle"),
                )
                .unwrap(),
        );

        let window_size = window.inner_size();
        let scale_factor = window.scale_factor();
        let physical_size = (window_size.width, window_size.height);

        let msaa_samples = if self.msaa_enabled { 4 } else { 1 };

        let renderer = block_on(grafo::Renderer::new(
            window.clone(),
            physical_size,
            scale_factor,
            true,  // vsync
            false, // transparent
            msaa_samples,
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
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key: Key::Named(NamedKey::Space),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => {
                self.msaa_enabled = !self.msaa_enabled;
                let new_samples = if self.msaa_enabled { 4 } else { 1 };
                renderer.set_msaa_samples(new_samples);
                println!(
                    "MSAA: {} (sample count: {})",
                    if self.msaa_enabled { "ON" } else { "OFF" },
                    renderer.msaa_samples()
                );
                window.request_redraw();
            }
            WindowEvent::RedrawRequested => {
                let background = Shape::rect(
                    [(0.0, 0.0), (800.0, 600.0)],
                    Stroke::new(2.0, Color::rgb(255, 0, 0)),
                );
                let back_id = renderer.add_shape(background, None, None);
                renderer.set_shape_color(back_id, Some(Color::BLACK));

                // Draw a triangle — diagonal edges show aliasing clearly
                let triangle = Shape::builder()
                    .stroke(Stroke::new(2.0, Color::BLACK))
                    .begin((200.0, 80.0))
                    .line_to((80.0, 350.0))
                    .line_to((320.0, 350.0))
                    .close()
                    .build();
                let id = renderer.add_shape(triangle, Some(0), None);
                renderer.set_shape_color(id, Some(Color::rgb(0, 128, 255)));

                // Draw a rounded rectangle — curved edges benefit from MSAA
                let rounded_rect = Shape::rounded_rect(
                    [(380.0, 100.0), (620.0, 330.0)],
                    BorderRadii::new(30.0),
                    Stroke::new(2.0, Color::rgb(255, 0, 0)),
                );
                let id2 = renderer.add_shape(rounded_rect, Some(0), None);
                renderer.set_shape_color(id2, Some(Color::rgb(255, 100, 50)));

                // Draw a circle (approximated with a rounded rect)
                let circle = Shape::rounded_rect(
                    [(100.0, 270.0), (260.0, 430.0)],
                    BorderRadii::new(80.0),
                    Stroke::new(2.0, Color::BLACK),
                );
                let id3 = renderer.add_shape(circle, Some(0), None);
                renderer.set_shape_color(id3, Some(Color::rgb(50, 200, 100)));

                // Draw a small detailed shape - thin diagonal lines are great for MSAA testing
                let small_rect = Shape::rect(
                    [(400.0, 280.0), (550.0, 420.0)],
                    Stroke::new(1.0, Color::rgb(100, 0, 150)),
                );
                let id4 = renderer.add_shape(small_rect, None, None);
                renderer.set_shape_color(id4, Some(Color::rgb(200, 200, 255)));

                match renderer.render() {
                    Ok(_) => {
                        renderer.clear_draw_queue();
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
    println!("MSAA Demo: Press SPACE to toggle MSAA on/off");
    let event_loop = EventLoop::new().expect("To create the event loop");

    let mut app = App::default();
    let _ = event_loop.run_app(&mut app);
}
