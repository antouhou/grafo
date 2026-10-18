//! Left: cosmic-text's rasterizer in one texture. Right: glyph outlines as Grafo shapes.
//! Run with `cargo run --example text_comparison`. Uses the system sans-serif font.
//! Both sides use the same shaping, hinting, and subpixel positions, with MSAA off.

use cosmic_text::{
    Attrs, Buffer, CacheKey, Color as TextColor, Command, Family, FontSystem, Metrics,
};
use cosmic_text::{Shaping, SwashCache};
use futures::executor::block_on;
use grafo::{Color, DrawCommandError, Renderer, RendererContext, Shape, ShapeDrawCommandOptions};
use grafo::{Surface, TextureManager, TransformInstance};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{error, info};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

mod window_rendering;

const TEXT_TEXTURE_ID: u64 = 1;
const FONT_SIZES: [f32; 5] = [12.0, 16.0, 24.0, 36.0, 48.0];
const SAMPLE_TEXT: &str = "Hamburgefontsiv 0123456789\nAV fi fl ffi O0 Il1 @%&()[]{}";

#[derive(Default)]
struct PreparationTimings {
    scene_clear: Duration,
    shaping: Duration,
    rasterization: Duration,
    outline_extraction: Duration,
    path_construction: Duration,
    grafo_shape_preparation: Duration,
    texture_upload: Duration,
    total: Duration,
    glyph_count: usize,
}

impl PreparationTimings {
    fn other(&self) -> Duration {
        self.total.saturating_sub(
            self.scene_clear
                + self.shaping
                + self.rasterization
                + self.outline_extraction
                + self.path_construction
                + self.grafo_shape_preparation
                + self.texture_upload,
        )
    }
}

fn glyph_geometry_id(cache_key: CacheKey) -> u64 {
    // Subpixel positions are applied by each glyph's instance transform.
    let mut hasher = DefaultHasher::new();
    (
        cache_key.font_id,
        cache_key.glyph_id,
        cache_key.font_size_bits,
        cache_key.font_weight,
        cache_key.flags,
    )
        .hash(&mut hasher);
    hasher.finish()
}

fn outline_shape(commands: &[Command]) -> Shape {
    let mut builder = Shape::builder();
    for command in commands {
        // Font outlines have Y pointing up; window coordinates have Y pointing down.
        builder = match *command {
            Command::MoveTo(point) => builder.begin((point.x, -point.y)),
            Command::LineTo(point) => builder.line_to((point.x, -point.y)),
            Command::QuadTo(control, point) => {
                builder.quadratic_bezier_to((control.x, -control.y), (point.x, -point.y))
            }
            Command::CurveTo(first, second, point) => builder.cubic_bezier_to(
                (first.x, -first.y),
                (second.x, -second.y),
                (point.x, -point.y),
            ),
            Command::Close => builder.close(),
        };
    }
    builder.build()
}

fn build_comparison(
    renderer: &mut Renderer,
    font_system: &mut FontSystem,
    size: (u32, u32),
    scale_factor: f32,
) -> Result<PreparationTimings, DrawCommandError> {
    let preparation_started_at = Instant::now();
    let mut timings = PreparationTimings::default();
    let clear_started_at = Instant::now();
    renderer.clear_draw_queue();
    timings.scene_clear = clear_started_at.elapsed();
    if size.0 < 2 || size.1 == 0 {
        timings.total = preparation_started_at.elapsed();
        return Ok(timings);
    }

    // All drawing uses physical pixels so the left texture is sampled exactly 1:1.
    let half_width = size.0 / 2;
    let width = size.0 as f32;
    let height = size.1 as f32;
    let padding = (24.0 * scale_factor).round() as i32;
    let mut top = padding;
    let mut pixels = vec![0_u8; half_width as usize * size.1 as usize * 4];
    let mut swash_cache = SwashCache::new();

    renderer.add_shape(
        Shape::rect([(0.0, 0.0), (width, height)]),
        None,
        None,
        ShapeDrawCommandOptions::new().color(Color::WHITE),
    )?;
    let right_panel = renderer.add_clipping_rect(
        [(half_width as f32, 0.0), (width, height)],
        None,
        None::<TransformInstance>,
        true,
    )?;

    for font_size in FONT_SIZES {
        if top >= size.1 as i32 {
            break;
        }
        let shaping_started_at = Instant::now();
        let metrics = Metrics::new(font_size * scale_factor, font_size * scale_factor * 1.3);
        let mut buffer = Buffer::new(font_system, metrics);
        buffer.set_size(
            Some((half_width as f32 - 2.0 * padding as f32).max(1.0)),
            None,
        );
        buffer.set_text(
            &format!("{font_size}px: {SAMPLE_TEXT}"),
            &Attrs::new().family(Family::SansSerif),
            Shaping::Advanced,
            None,
        );
        buffer.shape_until_scroll(font_system, false);
        timings.shaping += shaping_started_at.elapsed();

        let rasterization_started_at = Instant::now();
        buffer.draw(
            font_system,
            &mut swash_cache,
            TextColor::rgb(0, 0, 0),
            |x, y, pixel_width, pixel_height, color| {
                for y in (top + y).max(0)..(top + y + pixel_height as i32).min(size.1 as i32) {
                    for x in (padding + x).max(0)
                        ..(padding + x + pixel_width as i32).min(half_width as i32)
                    {
                        let alpha =
                            &mut pixels[(y as usize * half_width as usize + x as usize) * 4 + 3];
                        // Black RGB is already premultiplied. Composite overlapping coverage.
                        let coverage = u16::from(color.a());
                        *alpha =
                            (coverage + (u16::from(*alpha) * (255 - coverage) + 127) / 255) as u8;
                    }
                }
            },
        );
        timings.rasterization += rasterization_started_at.elapsed();

        let mut text_height = 0.0_f32;
        for run in buffer.layout_runs() {
            text_height = text_height.max(run.line_top + run.line_height);
            for glyph in run.glyphs {
                let physical = glyph.physical((0.0, run.line_y), 1.0);
                let outline_started_at = Instant::now();
                let commands =
                    swash_cache.get_outline_commands_uncached(font_system, physical.cache_key);
                timings.outline_extraction += outline_started_at.elapsed();
                let Some(commands) = commands else {
                    continue;
                };
                if commands.is_empty() {
                    continue;
                }
                let path_started_at = Instant::now();
                let shape = outline_shape(&commands);
                timings.path_construction += path_started_at.elapsed();
                let x = half_width as f32
                    + padding as f32
                    + physical.x as f32
                    + physical.cache_key.x_bin.as_float();
                let y = top as f32 + physical.y as f32 - physical.cache_key.y_bin.as_float();
                let shape_started_at = Instant::now();
                let geometry_id = glyph_geometry_id(physical.cache_key);
                renderer.add_shape(
                    shape,
                    Some(right_panel),
                    Some(geometry_id),
                    ShapeDrawCommandOptions::new()
                        .color(Color::BLACK)
                        .transform(TransformInstance::translation(x, y)),
                )?;
                timings.grafo_shape_preparation += shape_started_at.elapsed();
                timings.glyph_count += 1;
            }
        }
        top += text_height.ceil() as i32 + padding;
    }

    let upload_started_at = Instant::now();
    renderer.texture_manager().allocate_texture_with_data(
        TEXT_TEXTURE_ID,
        (half_width, size.1),
        &pixels,
    );
    timings.texture_upload = upload_started_at.elapsed();
    renderer.add_shape(
        Shape::rect([(0.0, 0.0), (half_width as f32, height)]),
        None,
        None,
        ShapeDrawCommandOptions::new().background_texture_id(TEXT_TEXTURE_ID),
    )?;
    renderer.add_shape(
        Shape::rect([(half_width as f32, 0.0), (half_width as f32 + 1.0, height)]),
        None,
        None,
        ShapeDrawCommandOptions::new().color(Color::rgb(180, 180, 180)),
    )?;
    timings.total = preparation_started_at.elapsed();
    Ok(timings)
}

struct ComparisonWindow {
    window: Arc<Window>,
    renderer: Renderer,
    surface: Surface,
    font_system: FontSystem,
    needs_scene_rebuild: bool,
}

impl ComparisonWindow {
    fn new(event_loop: &ActiveEventLoop) -> Self {
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Left: cosmic-text rasterizer | Right: Grafo glyph outlines")
                        .with_inner_size(LogicalSize::new(1200.0, 800.0)),
                )
                .expect("Failed to create comparison window"),
        );
        let size = window.inner_size();
        let size = (size.width, size.height);
        let context = block_on(RendererContext::new());
        let surface = Surface::new(&context, window.clone(), size, true, false)
            .expect("Failed to create comparison surface");
        let renderer = Renderer::new_with_context(context, size, 1.0, 1);
        Self {
            window,
            renderer,
            surface,
            font_system: FontSystem::new(),
            needs_scene_rebuild: true,
        }
    }

    fn redraw(&mut self, event_loop: &ActiveEventLoop) {
        let frame_started_at = Instant::now();
        let scene_rebuilt = self.needs_scene_rebuild;
        let mut resize_duration = Duration::ZERO;
        let mut timings = PreparationTimings::default();
        if self.needs_scene_rebuild {
            let size = self.window.inner_size();
            let size = (size.width, size.height);
            let resize_started_at = Instant::now();
            self.surface.resize(size);
            self.renderer.resize(size);
            resize_duration = resize_started_at.elapsed();
            timings = match build_comparison(
                &mut self.renderer,
                &mut self.font_system,
                size,
                self.window.scale_factor() as f32,
            ) {
                Ok(timings) => timings,
                Err(error) => {
                    error!(%error, "Failed to build text comparison");
                    event_loop.exit();
                    return;
                }
            };
            self.needs_scene_rebuild = false;
        }
        let render_started_at = Instant::now();
        if window_rendering::render(&mut self.renderer, &mut self.surface, event_loop) {
            let render_duration = render_started_at.elapsed();
            info!(
                "Frame CPU preparation and GPU submission, scene_rebuilt={scene_rebuilt}, glyphs={}\n\
                 Total:                   {:9.3} ms\n\
                 Resize:                  {:9.3} ms\n\
                 Scene clear:             {:9.3} ms\n\
                 Shaping:                 {:9.3} ms\n\
                 Cosmic rasterization:    {:9.3} ms\n\
                 Outline extraction:      {:9.3} ms\n\
                 Path construction:       {:9.3} ms\n\
                 Grafo shape preparation: {:9.3} ms\n\
                 Texture upload:          {:9.3} ms\n\
                 Other preparation:       {:9.3} ms\n\
                 Render call:             {:9.3} ms",
                timings.glyph_count,
                frame_started_at.elapsed().as_secs_f64() * 1000.0,
                resize_duration.as_secs_f64() * 1000.0,
                timings.scene_clear.as_secs_f64() * 1000.0,
                timings.shaping.as_secs_f64() * 1000.0,
                timings.rasterization.as_secs_f64() * 1000.0,
                timings.outline_extraction.as_secs_f64() * 1000.0,
                timings.path_construction.as_secs_f64() * 1000.0,
                timings.grafo_shape_preparation.as_secs_f64() * 1000.0,
                timings.texture_upload.as_secs_f64() * 1000.0,
                timings.other().as_secs_f64() * 1000.0,
                render_duration.as_secs_f64() * 1000.0,
            );
        }
    }
}

#[derive(Default)]
struct App {
    comparison: Option<ComparisonWindow>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.comparison.is_none() {
            let comparison = ComparisonWindow::new(event_loop);
            comparison.window.request_redraw();
            self.comparison = Some(comparison);
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(comparison) = &mut self.comparison else {
            return;
        };
        if window_id != comparison.window.id() {
            return;
        }
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(_) | WindowEvent::ScaleFactorChanged { .. } => {
                comparison.needs_scene_rebuild = true;
                comparison.window.request_redraw();
            }
            WindowEvent::RedrawRequested => comparison.redraw(event_loop),
            _ => {}
        }
    }
}

fn main() {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("warn,text_comparison=info"),
    )
    .init();
    let event_loop = EventLoop::new().expect("Failed to create event loop");
    event_loop
        .run_app(&mut App::default())
        .expect("Failed to run text comparison");
}
