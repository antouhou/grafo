//! Compares rendering an unchanged draw queue with rebuilding it from cached shapes
//! before each frame. Both cases present to a window with vsync disabled.
//!
//! Render samples measure elapsed time in `render()`, including CPU work. Queue
//! rebuild time is measured separately. Throughput and average frame time include
//! time between redraws.
//!
//! Run with:
//! ```
//! cargo run --example bench_render_loop --features render_metrics --release
//! ```
use futures::executor::block_on;
use grafo::{
    AlphaMode, Color, Scene, Shape, ShapeDrawOptions, TextureManager, TexturePixels,
    TransformInstance,
};
use grafo::{SceneContext, Surface};
use std::sync::Arc;
use std::time::{Duration, Instant};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

mod window_rendering;

/// Fixed benchmark parameters
const BENCH_WIDTH: u32 = 2560;
const BENCH_HEIGHT: u32 = 1600;
const WARMUP_FRAMES: u64 = 100;
const BENCH_FRAMES: u64 = 2000;

/// Scene layout: CONTAINERS * ROWS_PER_CONTAINER * CELLS_PER_ROW leaf shapes,
/// plus container and row shapes as interior nodes.
const CONTAINERS: usize = 5;
const ROWS_PER_CONTAINER: usize = 4;
const CELLS_PER_ROW: usize = 5;
const CIRCLES_IN_SIDEBAR: usize = 4;

/// Shapes with separate texture allocations.
const TEXTURED_ELEMENTS: usize = 25;
const TEXTURE_SIZE: u32 = 250;

// Cache keys for geometry
const CACHE_KEY_CONTAINER: u64 = 1;
const CACHE_KEY_ROW: u64 = 2;
const CACHE_KEY_CELL: u64 = 3;
const CACHE_KEY_SIDEBAR: u64 = 4;
const CACHE_KEY_CIRCLE: u64 = 5;
const CACHE_KEY_TEXTURED: u64 = 6;

/// Base texture ID; actual IDs are TEXTURE_ID_BASE..TEXTURE_ID_BASE+TEXTURED_ELEMENTS.
const TEXTURE_ID_BASE: u64 = 100;

/// Create procedural textures and load the textured shape geometry.
fn load_textures_and_shapes(scene: &mut Scene) {
    // Generate a checkerboard RGBA texture.
    let tex_w = TEXTURE_SIZE;
    let tex_h = TEXTURE_SIZE;
    let mut rgba = vec![0u8; (tex_w * tex_h * 4) as usize];
    for y in 0..tex_h {
        for x in 0..tex_w {
            let idx = ((y * tex_w + x) * 4) as usize;
            let checker = ((x / 16) + (y / 16)) % 2 == 0;
            let v = if checker { 200u8 } else { 80u8 };
            rgba[idx] = v;
            rgba[idx + 1] = v / 2;
            rgba[idx + 2] = 255 - v;
            rgba[idx + 3] = 255;
        }
    }

    // Each texture has its own ID and allocation but uses the same pixels.
    for i in 0..TEXTURED_ELEMENTS {
        let tex_id = TEXTURE_ID_BASE + i as u64;
        scene
            .texture_manager()
            .allocate_texture(tex_id, (tex_w, tex_h))
            .expect("allocate benchmark texture");
        scene
            .texture_manager()
            .load_data_into_texture(
                tex_id,
                (0, 0),
                TexturePixels::rgba8_srgb(&rgba, (tex_w, tex_h), AlphaMode::Opaque),
            )
            .unwrap();
    }

    // A 250x250 textured rectangle.
    let textured_rect = Shape::rect([(0.0, 0.0), (TEXTURE_SIZE as f32, TEXTURE_SIZE as f32)]);
    scene
        .load_shape(textured_rect, CACHE_KEY_TEXTURED, Some(CACHE_KEY_TEXTURED))
        .expect("load textured rectangle");
}

fn load_shape_geometries(scene: &mut Scene) {
    let container = Shape::rect([(0.0, 0.0), (240.0, 500.0)]);
    scene
        .load_shape(container, CACHE_KEY_CONTAINER, Some(CACHE_KEY_CONTAINER))
        .expect("load container geometry");

    let row = Shape::rect([(0.0, 0.0), (220.0, 110.0)]);
    scene
        .load_shape(row, CACHE_KEY_ROW, Some(CACHE_KEY_ROW))
        .expect("load row geometry");

    let cell = Shape::rect([(0.0, 0.0), (36.0, 90.0)]);
    scene
        .load_shape(cell, CACHE_KEY_CELL, Some(CACHE_KEY_CELL))
        .expect("load cell geometry");

    let sidebar = Shape::rect([(0.0, 0.0), (100.0, 500.0)]);
    scene
        .load_shape(sidebar, CACHE_KEY_SIDEBAR, Some(CACHE_KEY_SIDEBAR))
        .expect("load sidebar geometry");

    let circle = Shape::rounded_rect([(0.0, 0.0), (40.0, 40.0)], grafo::BorderRadii::new(20.0));
    scene
        .load_shape(circle, CACHE_KEY_CIRCLE, Some(CACHE_KEY_CIRCLE))
        .expect("tessellate benchmark circle");
}

fn build_scene(scene: &mut Scene) -> usize {
    let container_colors = [
        Color::rgb(30, 60, 120),
        Color::rgb(120, 30, 60),
        Color::rgb(60, 120, 30),
        Color::rgb(100, 80, 40),
        Color::rgb(40, 100, 80),
    ];

    let mut total_shapes = 0;

    for c in 0..CONTAINERS {
        let cx = 10.0 + c as f32 * 250.0;
        let container_id = scene
            .add_cached_shape(
                CACHE_KEY_CONTAINER,
                None,
                ShapeDrawOptions::new()
                    .color(container_colors[c % container_colors.len()])
                    .transform(TransformInstance::translation(cx, 10.0)),
            )
            .unwrap();
        total_shapes += 1;

        for r in 0..ROWS_PER_CONTAINER {
            let ry = 10.0 + r as f32 * 120.0;
            let row_id = scene
                .add_cached_shape(
                    CACHE_KEY_ROW,
                    Some(container_id),
                    ShapeDrawOptions::new()
                        .color(Color::rgb(200, 200, 210))
                        .transform(TransformInstance::translation(cx + 10.0, 10.0 + ry)),
                )
                .unwrap();
            total_shapes += 1;

            for cell in 0..CELLS_PER_ROW {
                let brightness = (100u8).saturating_add((cell * 30) as u8);
                let cellx = cx + 20.0 + cell as f32 * 42.0;
                let celly = 20.0 + ry + 10.0;
                scene
                    .add_cached_shape(
                        CACHE_KEY_CELL,
                        Some(row_id),
                        ShapeDrawOptions::new()
                            .color(Color::rgb(brightness, brightness, brightness))
                            .transform(TransformInstance::translation(cellx, celly)),
                    )
                    .unwrap();
                total_shapes += 1;
            }
        }
    }

    // Sidebar with circles
    let sidebar_x = 10.0 + CONTAINERS as f32 * 250.0;
    let sidebar_id = scene
        .add_cached_shape(
            CACHE_KEY_SIDEBAR,
            None,
            ShapeDrawOptions::new()
                .color(Color::rgb(50, 50, 70))
                .transform(TransformInstance::translation(sidebar_x, 10.0)),
        )
        .unwrap();
    total_shapes += 1;

    for i in 0..CIRCLES_IN_SIDEBAR {
        scene
            .add_cached_shape(
                CACHE_KEY_CIRCLE,
                Some(sidebar_id),
                ShapeDrawOptions::new()
                    .color(Color::rgb(220, 180, 50))
                    .transform(TransformInstance::translation(
                        sidebar_x + 30.0,
                        30.0 + i as f32 * 60.0,
                    )),
            )
            .unwrap();
        total_shapes += 1;
    }

    // Place textured shapes below the containers, with 30px overlap between neighbors.
    for i in 0..TEXTURED_ELEMENTS {
        let col = i % 5;
        let row = i / 5;
        let tx = 10.0 + col as f32 * 220.0;
        let ty = 520.0 + row as f32 * 220.0;
        scene
            .add_cached_shape(
                CACHE_KEY_TEXTURED,
                None,
                ShapeDrawOptions::new()
                    .background_texture_id(TEXTURE_ID_BASE + i as u64)
                    .transform(TransformInstance::translation(tx, ty)),
            )
            .unwrap();
        total_shapes += 1;
    }

    total_shapes
}

fn print_results(label: &str, frame_times: &mut [Duration], total_elapsed: Duration) {
    frame_times.sort();
    let count = frame_times.len();
    let avg = total_elapsed.as_secs_f64() / count as f64;
    let p50 = frame_times[count / 2];
    let p95 = frame_times[(count as f64 * 0.95) as usize];
    let p99 = frame_times[(count as f64 * 0.99) as usize];
    let min = frame_times[0];
    let max = frame_times[count - 1];

    println!("=== {} ===", label);
    println!("Frames rendered: {}", count);
    println!("Total time:      {:.2}s", total_elapsed.as_secs_f64());
    println!(
        "Throughput FPS:  {:.1}",
        count as f64 / total_elapsed.as_secs_f64()
    );
    println!("Average frame:   {:.3}ms", avg * 1000.0);
    println!("P50 frame:       {:.3}ms", p50.as_secs_f64() * 1000.0);
    println!("P95 frame:       {:.3}ms", p95.as_secs_f64() * 1000.0);
    println!("P99 frame:       {:.3}ms", p99.as_secs_f64() * 1000.0);
    println!("Min frame:       {:.3}ms", min.as_secs_f64() * 1000.0);
    println!("Max frame:       {:.3}ms", max.as_secs_f64() * 1000.0);
}

#[cfg(feature = "render_metrics")]
fn print_phase_breakdown(
    phase_prepare: &mut [Duration],
    phase_encode_submit: &mut [Duration],
    phase_present: &mut [Duration],
    phase_gpu_wait: &mut [Duration],
) {
    fn summarize(label: &str, durations: &mut [Duration]) {
        durations.sort();
        let n = durations.len();
        let sum: Duration = durations.iter().sum();
        let avg = sum.as_secs_f64() / n as f64;
        let p50 = durations[n / 2].as_secs_f64();
        let p95 = durations[(n as f64 * 0.95) as usize].as_secs_f64();
        println!(
            "  {:<20} avg {:.3}ms  P50 {:.3}ms  P95 {:.3}ms",
            label,
            avg * 1000.0,
            p50 * 1000.0,
            p95 * 1000.0,
        );
    }

    println!("--- WGPU Phase Breakdown ---");
    summarize("prepare:", phase_prepare);
    summarize("encode+submit:", phase_encode_submit);
    summarize("present:", phase_present);
    summarize("gpu_wait:", phase_gpu_wait);
}

#[cfg(feature = "render_metrics")]
fn print_metrics(scene: &mut Scene) {
    println!("--- render_metrics ---");
    println!("Rolling 1s FPS:  {:.1}", scene.rolling_frames_per_second());
    println!(
        "Rolling 1s avg:  {:.3}ms",
        scene.rolling_average_render_loop_duration().as_secs_f64() * 1000.0
    );
    println!(
        "Cumulative avg:  {:.3}ms",
        scene.average_render_loop_duration().as_secs_f64() * 1000.0
    );
    let pc = scene.backend().last_pipeline_switch_counts();
    println!("--- pipeline switches (last frame) ---");
    println!("  Total switches:   {}", pc.total_switches);
}

#[derive(Default, Clone, Copy, PartialEq, Eq)]
enum Benchmark {
    #[default]
    Static,
    Dynamic,
}

struct Measurement {
    started_at: Instant,
    frame_times: Vec<Duration>,
    rebuild_times: Vec<Duration>,
    #[cfg(feature = "render_metrics")]
    phase_prepare: Vec<Duration>,
    #[cfg(feature = "render_metrics")]
    phase_encode_submit: Vec<Duration>,
    #[cfg(feature = "render_metrics")]
    phase_present: Vec<Duration>,
    #[cfg(feature = "render_metrics")]
    phase_gpu_wait: Vec<Duration>,
}

impl Measurement {
    fn start() -> Self {
        let samples = || Vec::with_capacity(BENCH_FRAMES as usize);
        Self {
            frame_times: samples(),
            rebuild_times: samples(),
            #[cfg(feature = "render_metrics")]
            phase_prepare: samples(),
            #[cfg(feature = "render_metrics")]
            phase_encode_submit: samples(),
            #[cfg(feature = "render_metrics")]
            phase_present: samples(),
            #[cfg(feature = "render_metrics")]
            phase_gpu_wait: samples(),
            started_at: Instant::now(),
        }
    }

    fn print(&mut self, benchmark: Benchmark) {
        let label = match benchmark {
            Benchmark::Static => "Benchmark 1: Static Scene",
            Benchmark::Dynamic => "Benchmark 2: Dynamic Scene (re-add every frame)",
        };
        print_results(label, &mut self.frame_times, self.started_at.elapsed());
        if benchmark == Benchmark::Dynamic {
            self.rebuild_times.sort();
            let count = self.rebuild_times.len();
            let sum: Duration = self.rebuild_times.iter().sum();
            println!(
                "Rebuild queue:   avg {:.3}ms  P50 {:.3}ms  P95 {:.3}ms",
                sum.as_secs_f64() / count as f64 * 1000.0,
                self.rebuild_times[count / 2].as_secs_f64() * 1000.0,
                self.rebuild_times[(count as f64 * 0.95) as usize].as_secs_f64() * 1000.0,
            );
        }
        #[cfg(feature = "render_metrics")]
        print_phase_breakdown(
            &mut self.phase_prepare,
            &mut self.phase_encode_submit,
            &mut self.phase_present,
            &mut self.phase_gpu_wait,
        );
    }
}

enum Phase {
    Warmup(u64),
    Measure(Measurement),
    Done,
}

impl Default for Phase {
    fn default() -> Self {
        Self::Warmup(0)
    }
}

#[derive(Default)]
struct BenchApp {
    window: Option<Arc<Window>>,
    scene: Option<Scene>,
    surface: Option<Surface>,
    benchmark: Benchmark,
    phase: Phase,
    is_occluded: bool,
}

impl BenchApp {
    fn interrupt_measurement(&mut self) {
        if matches!(self.phase, Phase::Measure(_)) {
            self.phase = Phase::Warmup(0);
        }
    }

    fn redraw(&mut self, event_loop: &ActiveEventLoop) {
        if self.is_occluded || matches!(self.phase, Phase::Done) {
            return;
        }
        let window = self.window.as_ref().expect("window initialized");
        let scene = self.scene.as_mut().expect("scene initialized");
        let surface = self.surface.as_mut().expect("surface initialized");
        let rebuild_time = (self.benchmark == Benchmark::Dynamic).then(|| {
            let start = Instant::now();
            build_scene(scene);
            start.elapsed()
        });
        let frame_start = Instant::now();
        let outcome = window_rendering::render(scene, surface, window, event_loop);
        let frame_time = frame_start.elapsed();
        if rebuild_time.is_some() {
            scene.clear();
        }
        match outcome {
            window_rendering::RenderResult::Presented => {}
            window_rendering::RenderResult::Occluded => {
                self.interrupt_measurement();
                return;
            }
            window_rendering::RenderResult::Skipped => {
                eprintln!("Stopping benchmark after a failed redraw.");
                event_loop.exit();
                return;
            }
        }

        match &mut self.phase {
            Phase::Warmup(frames) => {
                *frames += 1;
                if *frames >= WARMUP_FRAMES {
                    eprintln!("  Warmup done, measuring...");
                    #[cfg(feature = "render_metrics")]
                    scene.reset_render_loop_metrics();
                    self.phase = Phase::Measure(Measurement::start());
                }
            }
            Phase::Measure(measurement) => {
                measurement.frame_times.push(frame_time);
                if let Some(duration) = rebuild_time {
                    measurement.rebuild_times.push(duration);
                }
                #[cfg(feature = "render_metrics")]
                {
                    let timings = scene.backend().last_phase_timings();
                    measurement.phase_prepare.push(timings.prepare);
                    measurement
                        .phase_encode_submit
                        .push(timings.encode_and_submit);
                    measurement.phase_present.push(timings.present_or_readback);
                    measurement.phase_gpu_wait.push(timings.gpu_wait);
                }
                if measurement.frame_times.len() == BENCH_FRAMES as usize {
                    measurement.print(self.benchmark);
                    #[cfg(feature = "render_metrics")]
                    print_metrics(scene);
                    println!();
                    if self.benchmark == Benchmark::Dynamic {
                        self.phase = Phase::Done;
                        event_loop.exit();
                        return;
                    }
                    scene.clear();
                    self.benchmark = Benchmark::Dynamic;
                    self.phase = Phase::Warmup(0);
                    eprintln!("  Dynamic: starting warmup...");
                }
            }
            Phase::Done => return,
        }
        window.request_redraw();
    }
}

impl ApplicationHandler for BenchApp {
    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        // Release the GPU surface before the event loop closes the display connection.
        self.scene = None;
        self.surface = None;
        self.window = None;
    }

    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attrs = Window::default_attributes()
            .with_inner_size(PhysicalSize::new(BENCH_WIDTH, BENCH_HEIGHT))
            .with_resizable(false)
            .with_visible(true)
            .with_title("grafo bench");
        let window = Arc::new(event_loop.create_window(attrs).unwrap());

        let physical_size = (BENCH_WIDTH, BENCH_HEIGHT);
        let scale_factor = window.scale_factor();

        let (context, surface) = block_on(SceneContext::try_new_with_surface(
            event_loop.owned_display_handle(),
            window.clone(),
            physical_size,
            false,
            false,
        ))
        .expect("create window rendering context and surface");
        let mut scene =
            Scene::try_new(context, physical_size, scale_factor, 1).expect("create scene");

        load_shape_geometries(&mut scene);
        load_textures_and_shapes(&mut scene);
        let total_shapes = build_scene(&mut scene);
        eprintln!(
            "Benchmark: {} shapes, {}x{}, {} warmup + {} measured frames per test",
            total_shapes, BENCH_WIDTH, BENCH_HEIGHT, WARMUP_FRAMES, BENCH_FRAMES,
        );

        self.scene = Some(scene);
        self.surface = Some(surface);
        self.window = Some(window.clone());
        self.benchmark = Benchmark::Static;
        self.phase = Phase::Warmup(0);
        self.is_occluded = false;

        window.request_redraw();
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Occluded(is_occluded) => {
                self.is_occluded = is_occluded;
                if is_occluded {
                    self.interrupt_measurement();
                } else if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            WindowEvent::RedrawRequested => self.redraw(event_loop),
            _ => {}
        }
    }
}

pub fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().expect("To create the event loop");
    let mut app = BenchApp::default();
    let _ = event_loop.run_app(&mut app);
}
