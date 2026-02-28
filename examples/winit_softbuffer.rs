use futures::executor::block_on;
use grafo::Shape;
use grafo::{Color, Stroke};
use std::num::NonZeroU32;
use std::sync::{Arc, RwLock};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

#[derive(Default)]
struct App<'a> {
    window: Option<Arc<Window>>,
    renderer: Option<Arc<RwLock<grafo::Renderer<'a>>>>,
    softbuffer_context: Option<softbuffer::Context<Arc<Window>>>,
    softbuffer_surface: Option<softbuffer::Surface<Arc<Window>, Arc<Window>>>,
    pending_resize: Option<(u32, u32)>,
    frame_count: u64,
    argb_buffer: Vec<u32>,
}

impl<'a> ApplicationHandler for App<'a> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes().with_title("Grafo + Softbuffer Hybrid Test"),
                )
                .unwrap(),
        );

        let window_size = window.inner_size();
        let scale_factor = window.scale_factor();
        let physical_size = (window_size.width, window_size.height);

        // Create GPU renderer (renders offscreen)
        let renderer = block_on(grafo::Renderer::new(
            window.clone(),
            physical_size,
            scale_factor,
            false, // vsync doesn't matter for offscreen rendering
            false, // not transparent
            1,     // msaa_samples
        ));

        println!("\n=== Grafo + Softbuffer Hybrid Resize Test ===");
        println!("This renders with GPU to an offscreen texture,");
        println!("then copies to CPU and presents via softbuffer.");
        println!("If resize jitter disappears, the issue is GPU presentation timing.\n");

        // Create softbuffer for CPU presentation
        let softbuffer_context = softbuffer::Context::new(window.clone()).unwrap();
        let mut softbuffer_surface =
            softbuffer::Surface::new(&softbuffer_context, window.clone()).unwrap();
        softbuffer_surface
            .resize(
                NonZeroU32::new(physical_size.0).unwrap(),
                NonZeroU32::new(physical_size.1).unwrap(),
            )
            .unwrap();

        let renderer = Arc::new(RwLock::new(renderer));

        self.window = Some(window);
        self.renderer = Some(renderer);
        self.softbuffer_context = Some(softbuffer_context);
        self.softbuffer_surface = Some(softbuffer_surface);
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
        let Some(softbuffer_surface) = &mut self.softbuffer_surface else {
            return;
        };

        if window_id != window.id() {
            return;
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(physical_size) => {
                println!(
                    "Resize event to ({}, {})",
                    physical_size.width, physical_size.height
                );
                self.pending_resize = Some((physical_size.width, physical_size.height));

                // Resize softbuffer surface immediately
                if physical_size.width > 0 && physical_size.height > 0 {
                    softbuffer_surface
                        .resize(
                            NonZeroU32::new(physical_size.width).unwrap(),
                            NonZeroU32::new(physical_size.height).unwrap(),
                        )
                        .unwrap();
                }

                window.request_redraw();
            }
            WindowEvent::RedrawRequested => {
                // Apply any pending resize to GPU renderer
                if let Some(pending) = self.pending_resize.take() {
                    println!("Applying resize to GPU renderer: {:?}", pending);
                    renderer.write().unwrap().resize(pending);
                }

                self.frame_count += 1;

                let mut renderer_guard = renderer.write().unwrap();
                renderer_guard.clear_draw_queue();

                let window_size = window.inner_size();

                // Create shapes to render
                let background = Shape::rect(
                    [
                        (0.0, 0.0),
                        (window_size.width as f32, window_size.height as f32),
                    ],
                    Stroke::new(1.0, Color::rgb(0, 0, 0)),
                );
                let bg_id = renderer_guard.add_shape(background, None, None);
                renderer_guard.set_shape_color(bg_id, Some(Color::rgb(255, 255, 200)));

                let red_id = renderer_guard.add_shape(
                    Shape::rect(
                        [(0.0, 0.0), (200.0, 200.0)],
                        Stroke::new(1.0, Color::rgb(0, 0, 0)),
                    ),
                    None,
                    None,
                );
                renderer_guard.set_shape_color(red_id, Some(Color::rgb(255, 0, 0)));
                renderer_guard.set_shape_transform(red_id, grafo::TransformInstance::identity());

                let blue_id = renderer_guard.add_shape(
                    Shape::rect(
                        [(0.0, 0.0), (200.0, 200.0)],
                        Stroke::new(1.0, Color::rgb(0, 0, 0)),
                    ),
                    None,
                    None,
                );
                renderer_guard.set_shape_color(blue_id, Some(Color::rgb(0, 0, 255)));
                renderer_guard.set_shape_transform(
                    blue_id,
                    grafo::TransformInstance::translation(220.0, 0.0),
                );

                // Render to GPU offscreen texture and get ARGB32 pixels
                let render_start = std::time::Instant::now();
                // Ensure buffer capacity for current size
                let needed_len = (window_size.width as usize) * (window_size.height as usize);
                if self.argb_buffer.len() < needed_len {
                    self.argb_buffer.resize(needed_len, 0);
                }
                renderer_guard.render_to_argb32(&mut self.argb_buffer);
                let render_time = render_start.elapsed();

                // Present ARGB u32s via softbuffer
                let copy_start = std::time::Instant::now();
                let mut buffer = softbuffer_surface.buffer_mut().unwrap();
                let count = buffer.len().min(self.argb_buffer.len());
                buffer[..count].copy_from_slice(&self.argb_buffer[..count]);

                buffer.present().unwrap();
                let copy_time = copy_start.elapsed();

                if self.frame_count.is_multiple_of(60) {
                    println!(
                        "Frame {}: GPU render: {:?}, CPU copy+present: {:?}, Total: {:?}",
                        self.frame_count,
                        render_time,
                        copy_time,
                        render_start.elapsed()
                    );
                }
            }
            _ => {}
        }
    }
}

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().expect("to start an event loop");
    let mut app = App::default();
    event_loop.run_app(&mut app).unwrap();
}
