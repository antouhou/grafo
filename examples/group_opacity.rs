//! Applies 50% and 80% opacity to two groups of shapes.
//! Each group composites its parent and children into one translucent layer.

use futures::executor::block_on;
use grafo::wgpu::SurfaceError;
use grafo::RenderError;
use grafo::Shape;
use grafo::{Color, ShapeDrawCommandOptions, Stroke};
use redraw_retry::RedrawRetry;
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::{StartCause, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

mod redraw_retry;

const OPACITY_EFFECT: u64 = 1;

#[derive(Default)]
struct App<'a> {
    window: Option<Arc<Window>>,
    renderer: Option<grafo::Renderer<'a>>,
    effect_loaded: bool,
    redraw_retry: RedrawRetry,
}

impl<'a> ApplicationHandler for App<'a> {
    fn new_events(&mut self, event_loop: &ActiveEventLoop, cause: StartCause) {
        self.redraw_retry
            .new_events(event_loop, cause, self.window.as_deref());
    }

    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes().with_title("Grafo – Group Opacity Effect"),
                )
                .unwrap(),
        );

        let window_size = window.inner_size();
        let scale_factor = window.scale_factor();
        let physical_size = (window_size.width, window_size.height);

        let mut renderer = block_on(grafo::Renderer::new(
            window.clone(),
            physical_size,
            scale_factor,
            true,
            false,
            1,
        ));

        // Load the opacity effect shader once
        let opacity_wgsl = r#"
            struct Params {
                opacity: f32,
            }
            @group(1) @binding(0) var<uniform> params: Params;

            @fragment
            fn effect_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
                let color = textureSample(t_input, s_input, uv);
                return color * params.opacity;
            }
        "#;

        renderer
            .load_effect(OPACITY_EFFECT, &[opacity_wgsl])
            .expect("Failed to compile opacity effect");
        self.effect_loaded = true;

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
                // Background without an effect
                let bg = Shape::rect(
                    [(50.0, 50.0), (750.0, 550.0)],
                    Stroke::new(2.0_f32, Color::BLACK),
                );
                let _bg_id = renderer
                    .add_shape(
                        bg,
                        None,
                        None,
                        ShapeDrawCommandOptions::new().color(Color::rgb(200, 200, 220)),
                    )
                    .unwrap();

                // Group 1: 50% opacity
                let group1_bg = Shape::rect(
                    [(100.0, 100.0), (400.0, 350.0)],
                    Stroke::new(0.0_f32, Color::TRANSPARENT),
                );
                let group1 = renderer
                    .add_shape(
                        group1_bg,
                        None,
                        None,
                        ShapeDrawCommandOptions::new().color(Color::rgb(255, 100, 100)),
                    )
                    .unwrap();

                // Child 1: overlapping blue rectangle
                let child1 = Shape::rect(
                    [(120.0, 120.0), (300.0, 250.0)],
                    Stroke::new(2.0_f32, Color::BLACK),
                );
                renderer
                    .add_shape(
                        child1,
                        Some(group1),
                        None,
                        ShapeDrawCommandOptions::new().color(Color::rgb(50, 100, 255)),
                    )
                    .unwrap();

                // Child 2: overlapping green rectangle
                let child2 = Shape::rect(
                    [(200.0, 180.0), (380.0, 320.0)],
                    Stroke::new(2.0_f32, Color::BLACK),
                );
                renderer
                    .add_shape(
                        child2,
                        Some(group1),
                        None,
                        ShapeDrawCommandOptions::new().color(Color::rgb(50, 200, 50)),
                    )
                    .unwrap();

                // Attach 50% opacity to group1
                let opacity: f32 = 0.5;
                renderer
                    .set_group_effect(group1, OPACITY_EFFECT, bytemuck::bytes_of(&opacity))
                    .expect("Failed to set effect");

                // Group 2: 80% opacity
                let group2_bg = Shape::rect(
                    [(350.0, 100.0), (700.0, 350.0)],
                    Stroke::new(0.0_f32, Color::TRANSPARENT),
                );
                let group2 = renderer
                    .add_shape(
                        group2_bg,
                        None,
                        None,
                        ShapeDrawCommandOptions::new().color(Color::rgb(255, 200, 50)),
                    )
                    .unwrap();

                let child3 = Shape::rect(
                    [(370.0, 130.0), (680.0, 320.0)],
                    Stroke::new(2.0_f32, Color::BLACK),
                );
                renderer
                    .add_shape(
                        child3,
                        Some(group2),
                        None,
                        ShapeDrawCommandOptions::new().color(Color::rgb(128, 0, 200)),
                    )
                    .unwrap();

                let opacity2: f32 = 0.8;
                renderer
                    .set_group_effect(group2, OPACITY_EFFECT, bytemuck::bytes_of(&opacity2))
                    .expect("Failed to set effect");

                match renderer.render() {
                    Ok(_) => {
                        self.redraw_retry.cancel(event_loop);
                        renderer.clear_draw_queue();
                    }
                    Err(RenderError::Surface(SurfaceError::Lost | SurfaceError::Outdated)) => {
                        renderer.resize(renderer.size())
                    }

                    Err(RenderError::Surface(SurfaceError::Timeout)) => {
                        renderer.clear_draw_queue();
                        self.redraw_retry.schedule(event_loop);
                    }
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
