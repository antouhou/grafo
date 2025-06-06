use futures::executor::block_on;
use grafo::{fontdb, Color, FontFamily, Stroke};
use grafo::{MathRect, Shape, TextAlignment, TextLayout};
use std::sync::Arc;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

#[derive(Default)]
struct App<'a> {
    window: Option<Arc<Window>>,
    renderer: Option<grafo::Renderer<'a>>,
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

        let mut renderer = block_on(grafo::Renderer::new(
            window.clone(),
            physical_size,
            scale_factor,
            true,  // vsync
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
                let background_shape_id = renderer.add_shape(background, None, (0.0, 0.0), None);

                // First text instance
                let first_background_min = (10.0, 10.0);
                let first_background_max = (210.0, 32.0);
                let first_text_background = Shape::rect(
                    [first_background_min, first_background_max],
                    Color::rgb(200, 200, 200),
                    Stroke::new(1.0, Color::rgb(0, 0, 0)),
                );
                let first_text_background_id = renderer.add_shape(
                    first_text_background,
                    Some(background_shape_id),
                    (0.0, 0.0),
                    None,
                );
                let text = "Hello, world!";
                let text_layout = TextLayout {
                    font_size: 20.0,
                    line_height: 22.0,
                    color: Color::rgb(255, 0, 0),
                    area: MathRect::new(first_background_min.into(), first_background_max.into()),
                    horizontal_alignment: TextAlignment::Start,
                    vertical_alignment: TextAlignment::Start,
                };
                renderer.add_text(
                    text,
                    text_layout,
                    FontFamily::SansSerif,
                    Some(first_text_background_id),
                );

                // Second text instance
                let second_background_min = (10.0, 40.0);
                // TODO: IMPORTANT: TEXT WON'T BE LAYED OUT IF LINE HEIGHT IS SMALLER THAN
                //  THE TEXT AREA HEIGHT
                let second_background_max = (300.0, 80.0);
                let second_text_background = Shape::rect(
                    [second_background_min, second_background_max],
                    Color::rgb(200, 200, 200),
                    Stroke::new(1.0, Color::rgb(0, 0, 0)),
                );
                let second_text_background_id = renderer.add_shape(
                    second_text_background,
                    Some(background_shape_id),
                    (0.0, 0.0),
                    None,
                );
                let text_layout2 = TextLayout {
                    font_size: 40.0,
                    line_height: 40.0,
                    color: Color::rgb(255, 0, 0),
                    area: MathRect::new(second_background_min.into(), second_background_max.into()),
                    horizontal_alignment: TextAlignment::End,
                    vertical_alignment: TextAlignment::Start,
                };
                renderer.add_text(
                    text,
                    text_layout2,
                    FontFamily::SansSerif,
                    Some(second_text_background_id),
                );

                // Example with font loaded from file
                let third_background_min = (10.0, 90.0);
                // TODO: IMPORTANT: TEXT WON'T BE LAYED OUT IF LINE HEIGHT IS SMALLER THAN
                //  THE TEXT AREA HEIGHT
                let third_background_max = (300.0, 130.0);
                let third_text_background = Shape::rect(
                    [third_background_min, third_background_max],
                    Color::rgb(200, 200, 200),
                    Stroke::new(1.0, Color::rgb(0, 0, 0)),
                );
                let third_text_background_id = renderer.add_shape(
                    third_text_background,
                    Some(background_shape_id),
                    (0.0, 0.0),
                    None,
                );
                let text_layout3 = TextLayout {
                    font_size: 40.0,
                    line_height: 40.0,
                    color: Color::rgb(255, 0, 0),
                    area: MathRect::new(third_background_min.into(), third_background_max.into()),
                    horizontal_alignment: TextAlignment::End,
                    vertical_alignment: TextAlignment::Start,
                };
                renderer.add_text(
                    "This it Roboto",
                    text_layout3,
                    FontFamily::Name("Roboto"),
                    Some(third_text_background_id),
                );

                // Example with system font
                let fourth_background_min = (10.0, 140.0);
                // TODO: IMPORTANT: TEXT WON'T BE LAYED OUT IF LINE HEIGHT IS SMALLER THAN
                //  THE TEXT AREA HEIGHT
                let fourth_background_max = (300.0, 180.0);
                let fourth_text_background = Shape::rect(
                    [fourth_background_min, fourth_background_max],
                    Color::rgb(200, 200, 200),
                    Stroke::new(1.0, Color::rgb(0, 0, 0)),
                );
                let fourth_text_background_id = renderer.add_shape(
                    fourth_text_background,
                    Some(background_shape_id),
                    (0.0, 0.0),
                    None,
                );
                let text_layout4 = TextLayout {
                    font_size: 40.0,
                    line_height: 40.0,
                    color: Color::rgb(255, 0, 0),
                    area: MathRect::new(fourth_background_min.into(), fourth_background_max.into()),
                    horizontal_alignment: TextAlignment::Center,
                    vertical_alignment: TextAlignment::Start,
                };
                renderer.add_text(
                    "This is Papyrus",
                    text_layout4,
                    FontFamily::Name("Papyrus"),
                    Some(fourth_text_background_id),
                );

                let timer = Instant::now();
                match renderer.render() {
                    Ok(_) => {
                        renderer.clear_draw_queue();
                    }
                    Err(wgpu::SurfaceError::Lost) => renderer.resize(renderer.size()),
                    Err(wgpu::SurfaceError::OutOfMemory) => event_loop.exit(),
                    Err(e) => eprintln!("{:?}", e),
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

fn main() {
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::default();
    event_loop.run_app(&mut app).unwrap();
}
