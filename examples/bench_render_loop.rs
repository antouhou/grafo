/// Renderer performance benchmark — two scenarios, both using the real `render()` path
/// (present to screen, vsync OFF).
///
/// **Benchmark 1 — Static scene:**
///   Build the scene once, then render() repeatedly. Measures pure GPU + present cost.
///
/// **Benchmark 2 — Dynamic scene (re-add every frame):**
///   Each frame: clear_draw_queue() → rebuild all cached shapes → render().
///   Simulates a real UI where the render queue is reconstructed each frame.
///
/// Build and run with:
/// ```
/// cargo run --example bench_render_loop --features render_metrics --release
/// ```
use futures::executor::block_on;
use grafo::{Color, Shape, Stroke, TransformInstance};
use std::sync::Arc;
use std::time::{Duration, Instant};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

/// Fixed benchmark parameters
const BENCH_WIDTH: u32 = 2560;
const BENCH_HEIGHT: u32 = 1600;
const WARMUP_FRAMES: u64 = 100;
const BENCH_FRAMES: u64 = 2000;

/// Scene layout: CONTAINERS × ROWS_PER_CONTAINER × CELLS_PER_ROW leaf shapes,
/// plus container and row shapes as interior nodes.
const CONTAINERS: usize = 5;
const ROWS_PER_CONTAINER: usize = 4;
const CELLS_PER_ROW: usize = 5;
const CIRCLES_IN_SIDEBAR: usize = 4;

/// Textured elements matching real-world usage.
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
fn load_textures_and_shapes(renderer: &mut grafo::Renderer<'_>) {
    // Generate a checkerboard RGBA texture (TEXTURE_SIZE × TEXTURE_SIZE)
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

    // Allocate TEXTURED_ELEMENTS distinct textures (same pixel data, different IDs)
    for i in 0..TEXTURED_ELEMENTS {
        let tex_id = TEXTURE_ID_BASE + i as u64;
        renderer
            .texture_manager()
            .allocate_texture(tex_id, (tex_w, tex_h));
        renderer
            .texture_manager()
            .load_data_into_texture(tex_id, (tex_w, tex_h), &rgba)
            .unwrap();
    }

    // Textured rect shape (250×250)
    let textured_rect = Shape::rect(
        [(0.0, 0.0), (TEXTURE_SIZE as f32, TEXTURE_SIZE as f32)],
        Stroke::default(),
    );
    renderer.load_shape(textured_rect, CACHE_KEY_TEXTURED, Some(CACHE_KEY_TEXTURED));
}

fn load_shape_geometries(renderer: &mut grafo::Renderer<'_>) {
    let container = Shape::rect([(0.0, 0.0), (240.0, 500.0)], Stroke::new(1.0, Color::BLACK));
    renderer.load_shape(container, CACHE_KEY_CONTAINER, Some(CACHE_KEY_CONTAINER));

    let row = Shape::rect([(0.0, 0.0), (220.0, 110.0)], Stroke::new(1.0, Color::BLACK));
    renderer.load_shape(row, CACHE_KEY_ROW, Some(CACHE_KEY_ROW));

    let cell = Shape::rect([(0.0, 0.0), (36.0, 90.0)], Stroke::new(1.0, Color::BLACK));
    renderer.load_shape(cell, CACHE_KEY_CELL, Some(CACHE_KEY_CELL));

    let sidebar = Shape::rect([(0.0, 0.0), (100.0, 500.0)], Stroke::new(1.0, Color::BLACK));
    renderer.load_shape(sidebar, CACHE_KEY_SIDEBAR, Some(CACHE_KEY_SIDEBAR));

    let circle = Shape::rounded_rect(
        [(0.0, 0.0), (40.0, 40.0)],
        grafo::BorderRadii::new(20.0),
        Stroke::new(1.0, Color::BLACK),
    );
    renderer.load_shape(circle, CACHE_KEY_CIRCLE, Some(CACHE_KEY_CIRCLE));
}

fn build_scene(renderer: &mut grafo::Renderer<'_>) -> usize {
    let container_colors = [
        Color::rgb(30, 60, 120),
        Color::rgb(120, 30, 60),
        Color::rgb(60, 120, 30),
        Color::rgb(100, 80, 40),
        Color::rgb(40, 100, 80),
    ];

    let mut total_shapes = 0;

    for c in 0..CONTAINERS {
        let container_id = renderer.add_cached_shape_to_the_render_queue(CACHE_KEY_CONTAINER, None);
        renderer.set_shape_color(
            container_id,
            Some(container_colors[c % container_colors.len()]),
        );
        let cx = 10.0 + c as f32 * 250.0;
        renderer.set_shape_transform(container_id, TransformInstance::translation(cx, 10.0));
        total_shapes += 1;

        for r in 0..ROWS_PER_CONTAINER {
            let row_id =
                renderer.add_cached_shape_to_the_render_queue(CACHE_KEY_ROW, Some(container_id));
            renderer.set_shape_color(row_id, Some(Color::rgb(200, 200, 210)));
            let ry = 10.0 + r as f32 * 120.0;
            renderer
                .set_shape_transform(row_id, TransformInstance::translation(cx + 10.0, 10.0 + ry));
            total_shapes += 1;

            for cell in 0..CELLS_PER_ROW {
                let cell_id =
                    renderer.add_cached_shape_to_the_render_queue(CACHE_KEY_CELL, Some(row_id));
                let brightness = 100 + (cell * 30) as u8;
                renderer.set_shape_color(
                    cell_id,
                    Some(Color::rgb(brightness, brightness, brightness)),
                );
                let cellx = cx + 20.0 + cell as f32 * 42.0;
                let celly = 20.0 + ry + 10.0;
                renderer.set_shape_transform(cell_id, TransformInstance::translation(cellx, celly));
                total_shapes += 1;
            }
        }
    }

    // Sidebar with circles
    let sidebar_x = 10.0 + CONTAINERS as f32 * 250.0;
    let sidebar_id = renderer.add_cached_shape_to_the_render_queue(CACHE_KEY_SIDEBAR, None);
    renderer.set_shape_color(sidebar_id, Some(Color::rgb(50, 50, 70)));
    renderer.set_shape_transform(sidebar_id, TransformInstance::translation(sidebar_x, 10.0));
    total_shapes += 1;

    for i in 0..CIRCLES_IN_SIDEBAR {
        let circle_id =
            renderer.add_cached_shape_to_the_render_queue(CACHE_KEY_CIRCLE, Some(sidebar_id));
        renderer.set_shape_color(circle_id, Some(Color::rgb(220, 180, 50)));
        renderer.set_shape_transform(
            circle_id,
            TransformInstance::translation(sidebar_x + 30.0, 30.0 + i as f32 * 60.0),
        );
        total_shapes += 1;
    }

    // Textured elements — 25 shapes with distinct textures, some overlapping
    // Laid out in a 5×5 grid starting below the containers, partially overlapping by 30px
    for i in 0..TEXTURED_ELEMENTS {
        let tex_id = renderer.add_cached_shape_to_the_render_queue(CACHE_KEY_TEXTURED, None);
        renderer.set_shape_texture(tex_id, Some(TEXTURE_ID_BASE + i as u64));
        let col = i % 5;
        let row = i / 5;
        // Overlap: offset by 220px instead of 250px so they overlap by 30px
        let tx = 10.0 + col as f32 * 220.0;
        let ty = 520.0 + row as f32 * 220.0;
        renderer.set_shape_transform(tex_id, TransformInstance::translation(tx, ty));
        total_shapes += 1;
    }

    total_shapes
}

// ── Reporting helpers ────────────────────────────────────────────────────────

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
    phase_prepare: Vec<Duration>,
    phase_encode_submit: Vec<Duration>,
    phase_present: Vec<Duration>,
    phase_gpu_wait: Vec<Duration>,
) {
    fn summarize(label: &str, mut durations: Vec<Duration>) {
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

    println!("--- Phase Breakdown ---");
    summarize("prepare:", phase_prepare);
    summarize("encode+submit:", phase_encode_submit);
    summarize("present:", phase_present);
    summarize("gpu_wait:", phase_gpu_wait);
}

#[cfg(feature = "render_metrics")]
fn print_metrics(renderer: &grafo::Renderer<'_>) {
    println!("--- render_metrics ---");
    println!(
        "Rolling 1s FPS:  {:.1}",
        renderer.rolling_one_second_frames_per_second()
    );
    println!(
        "Rolling 1s avg:  {:.3}ms",
        renderer
            .rolling_one_second_average_render_loop_duration()
            .as_secs_f64()
            * 1000.0
    );
    println!(
        "Cumulative avg:  {:.3}ms",
        renderer.average_render_loop_duration().as_secs_f64() * 1000.0
    );
    let pc = renderer.last_pipeline_switch_counts();
    println!("--- pipeline switches (last frame) ---");
    println!("  StencilIncrement: {}", pc.to_stencil_increment);
    println!("  StencilDecrement: {}", pc.to_stencil_decrement);
    println!("  LeafDraw:         {}", pc.to_leaf_draw);
    println!("  Composite:        {}", pc.to_composite);
    println!("  Total switches:   {}", pc.total_switches);
    println!("  Scissor clips:    {}", pc.scissor_clips);
}

// ── Event-loop–driven benchmark ──────────────────────────────────────────────
// Uses request_redraw() + RedrawRequested so macOS compositor doesn't throttle.

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    WarmupStatic,
    MeasureStatic,
    WarmupDynamic,
    MeasureDynamic,
    Done,
}

struct BenchApp<'a> {
    window: Option<Arc<Window>>,
    renderer: Option<grafo::Renderer<'a>>,
    phase: Phase,
    frame_counter: u64,
    total_shapes: usize,
    // Static bench data
    static_frame_times: Vec<Duration>,
    static_bench_start: Option<Instant>,
    #[cfg(feature = "render_metrics")]
    static_phase_prepare: Vec<Duration>,
    #[cfg(feature = "render_metrics")]
    static_phase_encode_submit: Vec<Duration>,
    #[cfg(feature = "render_metrics")]
    static_phase_present: Vec<Duration>,
    #[cfg(feature = "render_metrics")]
    static_phase_gpu_wait: Vec<Duration>,
    // Dynamic bench data
    dynamic_frame_times: Vec<Duration>,
    dynamic_rebuild_times: Vec<Duration>,
    dynamic_bench_start: Option<Instant>,
    #[cfg(feature = "render_metrics")]
    dynamic_phase_prepare: Vec<Duration>,
    #[cfg(feature = "render_metrics")]
    dynamic_phase_encode_submit: Vec<Duration>,
    #[cfg(feature = "render_metrics")]
    dynamic_phase_present: Vec<Duration>,
    #[cfg(feature = "render_metrics")]
    dynamic_phase_gpu_wait: Vec<Duration>,
}

impl<'a> Default for BenchApp<'a> {
    fn default() -> Self {
        Self {
            window: None,
            renderer: None,
            phase: Phase::WarmupStatic,
            frame_counter: 0,
            total_shapes: 0,
            static_frame_times: Vec::with_capacity(BENCH_FRAMES as usize),
            static_bench_start: None,
            #[cfg(feature = "render_metrics")]
            static_phase_prepare: Vec::with_capacity(BENCH_FRAMES as usize),
            #[cfg(feature = "render_metrics")]
            static_phase_encode_submit: Vec::with_capacity(BENCH_FRAMES as usize),
            #[cfg(feature = "render_metrics")]
            static_phase_present: Vec::with_capacity(BENCH_FRAMES as usize),
            #[cfg(feature = "render_metrics")]
            static_phase_gpu_wait: Vec::with_capacity(BENCH_FRAMES as usize),
            dynamic_frame_times: Vec::with_capacity(BENCH_FRAMES as usize),
            dynamic_rebuild_times: Vec::with_capacity(BENCH_FRAMES as usize),
            dynamic_bench_start: None,
            #[cfg(feature = "render_metrics")]
            dynamic_phase_prepare: Vec::with_capacity(BENCH_FRAMES as usize),
            #[cfg(feature = "render_metrics")]
            dynamic_phase_encode_submit: Vec::with_capacity(BENCH_FRAMES as usize),
            #[cfg(feature = "render_metrics")]
            dynamic_phase_present: Vec::with_capacity(BENCH_FRAMES as usize),
            #[cfg(feature = "render_metrics")]
            dynamic_phase_gpu_wait: Vec::with_capacity(BENCH_FRAMES as usize),
        }
    }
}

impl<'a> BenchApp<'a> {
    fn print_static_results(&mut self) {
        let total_elapsed = self.static_bench_start.unwrap().elapsed();
        print_results(
            "Benchmark 1: Static Scene",
            &mut self.static_frame_times,
            total_elapsed,
        );
        #[cfg(feature = "render_metrics")]
        {
            let renderer = self.renderer.as_ref().unwrap();
            print_phase_breakdown(
                std::mem::take(&mut self.static_phase_prepare),
                std::mem::take(&mut self.static_phase_encode_submit),
                std::mem::take(&mut self.static_phase_present),
                std::mem::take(&mut self.static_phase_gpu_wait),
            );
            print_metrics(renderer);
        }
        println!();
    }

    fn print_dynamic_results(&mut self) {
        let total_elapsed = self.dynamic_bench_start.unwrap().elapsed();
        print_results(
            "Benchmark 2: Dynamic Scene (re-add every frame)",
            &mut self.dynamic_frame_times,
            total_elapsed,
        );

        // Rebuild timing
        self.dynamic_rebuild_times.sort();
        let rn = self.dynamic_rebuild_times.len();
        let rebuild_sum: Duration = self.dynamic_rebuild_times.iter().sum();
        println!(
            "Rebuild queue:   avg {:.3}ms  P50 {:.3}ms  P95 {:.3}ms",
            rebuild_sum.as_secs_f64() / rn as f64 * 1000.0,
            self.dynamic_rebuild_times[rn / 2].as_secs_f64() * 1000.0,
            self.dynamic_rebuild_times[(rn as f64 * 0.95) as usize].as_secs_f64() * 1000.0,
        );

        #[cfg(feature = "render_metrics")]
        {
            let renderer = self.renderer.as_ref().unwrap();
            print_phase_breakdown(
                std::mem::take(&mut self.dynamic_phase_prepare),
                std::mem::take(&mut self.dynamic_phase_encode_submit),
                std::mem::take(&mut self.dynamic_phase_present),
                std::mem::take(&mut self.dynamic_phase_gpu_wait),
            );
            print_metrics(renderer);
        }
        println!();
    }
}

impl<'a> ApplicationHandler for BenchApp<'a> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attrs = Window::default_attributes()
            .with_inner_size(PhysicalSize::new(BENCH_WIDTH, BENCH_HEIGHT))
            .with_resizable(false)
            .with_visible(true)
            .with_title("grafo bench");
        let window = Arc::new(event_loop.create_window(attrs).unwrap());

        let physical_size = (BENCH_WIDTH, BENCH_HEIGHT);
        let scale_factor = window.scale_factor();

        let mut renderer = block_on(grafo::Renderer::new(
            window.clone(),
            physical_size,
            scale_factor,
            false, // vsync OFF
            false, // not transparent
            1,     // no MSAA
        ));

        load_shape_geometries(&mut renderer);
        load_textures_and_shapes(&mut renderer);
        self.total_shapes = build_scene(&mut renderer);
        eprintln!(
            "Benchmark: {} shapes, {}x{}, {} warmup + {} measured frames per test",
            self.total_shapes, BENCH_WIDTH, BENCH_HEIGHT, WARMUP_FRAMES, BENCH_FRAMES,
        );

        self.renderer = Some(renderer);
        self.window = Some(window.clone());
        self.phase = Phase::WarmupStatic;
        self.frame_counter = 0;

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
            WindowEvent::RedrawRequested => {
                let window = self.window.clone().unwrap();

                match self.phase {
                    Phase::WarmupStatic => {
                        let renderer = self.renderer.as_mut().unwrap();
                        renderer.render().expect("render failed");
                        self.frame_counter += 1;
                        if self.frame_counter >= WARMUP_FRAMES {
                            #[cfg(feature = "render_metrics")]
                            renderer.reset_render_loop_metrics();
                            self.frame_counter = 0;
                            self.static_bench_start = Some(Instant::now());
                            self.phase = Phase::MeasureStatic;
                            eprintln!("  Static: warmup done, measuring...");
                        }
                        window.request_redraw();
                    }
                    Phase::MeasureStatic => {
                        {
                            let renderer = self.renderer.as_mut().unwrap();
                            let frame_start = Instant::now();
                            renderer.render().expect("render failed");
                            self.static_frame_times.push(frame_start.elapsed());

                            #[cfg(feature = "render_metrics")]
                            {
                                let pt = renderer.last_phase_timings();
                                self.static_phase_prepare.push(pt.prepare);
                                self.static_phase_encode_submit.push(pt.encode_and_submit);
                                self.static_phase_present.push(pt.present_or_readback);
                                self.static_phase_gpu_wait.push(pt.gpu_wait);
                            }
                        }

                        self.frame_counter += 1;
                        if self.frame_counter >= BENCH_FRAMES {
                            self.print_static_results();
                            let renderer = self.renderer.as_mut().unwrap();
                            renderer.clear_draw_queue();
                            self.frame_counter = 0;
                            self.phase = Phase::WarmupDynamic;
                            eprintln!("  Dynamic: starting warmup...");
                        }
                        window.request_redraw();
                    }
                    Phase::WarmupDynamic => {
                        let renderer = self.renderer.as_mut().unwrap();
                        build_scene(renderer);
                        renderer.render().expect("render failed");
                        renderer.clear_draw_queue();
                        self.frame_counter += 1;
                        if self.frame_counter >= WARMUP_FRAMES {
                            #[cfg(feature = "render_metrics")]
                            renderer.reset_render_loop_metrics();
                            self.frame_counter = 0;
                            self.dynamic_bench_start = Some(Instant::now());
                            self.phase = Phase::MeasureDynamic;
                            eprintln!("  Dynamic: warmup done, measuring...");
                        }
                        window.request_redraw();
                    }
                    Phase::MeasureDynamic => {
                        {
                            let renderer = self.renderer.as_mut().unwrap();
                            let rebuild_start = Instant::now();
                            build_scene(renderer);
                            self.dynamic_rebuild_times.push(rebuild_start.elapsed());

                            let frame_start = Instant::now();
                            renderer.render().expect("render failed");
                            self.dynamic_frame_times.push(frame_start.elapsed());

                            #[cfg(feature = "render_metrics")]
                            {
                                let pt = renderer.last_phase_timings();
                                self.dynamic_phase_prepare.push(pt.prepare);
                                self.dynamic_phase_encode_submit.push(pt.encode_and_submit);
                                self.dynamic_phase_present.push(pt.present_or_readback);
                                self.dynamic_phase_gpu_wait.push(pt.gpu_wait);
                            }

                            renderer.clear_draw_queue();
                        }

                        self.frame_counter += 1;
                        if self.frame_counter >= BENCH_FRAMES {
                            self.print_dynamic_results();
                            self.phase = Phase::Done;
                            event_loop.exit();
                            return;
                        }
                        window.request_redraw();
                    }
                    Phase::Done => {}
                }
            }
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
