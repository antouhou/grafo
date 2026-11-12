use super::ExampleError;
use grafo::{Color, Scene, Shape, ShapeDrawOptions, TransformInstance};
use grafo_test_scenes::caption_font;
use std::fmt::Write;
use std::time::{Duration, Instant};

const FONT_CACHE_START: u64 = 1 << 32;
const PANEL_CACHE_KEY: u64 = FONT_CACHE_START - 1;

pub struct FrameStatistics {
    sample_started: Instant,
    sample_count: u32,
    render_elapsed: Duration,
    rebuild_elapsed: Duration,
    frames_per_second: f64,
    frame_ms: f64,
    render_ms: f64,
    rebuild_ms: f64,
}

impl Default for FrameStatistics {
    fn default() -> Self {
        Self {
            sample_started: Instant::now(),
            sample_count: 0,
            render_elapsed: Duration::ZERO,
            rebuild_elapsed: Duration::ZERO,
            frames_per_second: 0.0,
            frame_ms: 0.0,
            render_ms: 0.0,
            rebuild_ms: 0.0,
        }
    }
}

impl FrameStatistics {
    /// Publishes a new average twice a second, using completed render calls.
    pub fn record(&mut self, render: Duration, rebuild: Duration) -> bool {
        self.sample_count += 1;
        self.render_elapsed += render;
        self.rebuild_elapsed += rebuild;
        let now = Instant::now();
        let elapsed = now.duration_since(self.sample_started);
        if elapsed < Duration::from_millis(500) {
            return false;
        }
        let count = f64::from(self.sample_count);
        self.frames_per_second = count / elapsed.as_secs_f64();
        self.frame_ms = elapsed.as_secs_f64() * 1000.0 / count;
        self.render_ms = self.render_elapsed.as_secs_f64() * 1000.0 / count;
        self.rebuild_ms = self.rebuild_elapsed.as_secs_f64() * 1000.0 / count;
        self.sample_started = now;
        self.sample_count = 0;
        self.render_elapsed = Duration::ZERO;
        self.rebuild_elapsed = Duration::ZERO;
        true
    }
}

/// Keeps the caption geometry loaded independently of scene rebuilds.
pub struct StatsOverlay {
    text: String,
    node_id: Option<usize>,
}

impl StatsOverlay {
    pub fn new(scene: &mut Scene) -> Result<Self, ExampleError> {
        scene.load_shape(Shape::rect([(0.0, 0.0), (1.0, 1.0)]), PANEL_CACHE_KEY, None)?;
        for character in b' '..=b'Z' {
            let text = char::from(character).to_string();
            scene.load_shape(
                caption_font::shape(&text),
                FONT_CACHE_START + u64::from(character),
                None,
            )?;
        }
        Ok(Self {
            text: String::with_capacity(256),
            node_id: None,
        })
    }

    pub fn update_text(
        &mut self,
        stats: &FrameStatistics,
        should_rebuild_each_redraw: bool,
        zoom: f64,
    ) {
        self.text.clear();
        let mode = if should_rebuild_each_redraw {
            "REBUILD"
        } else {
            "RETAINED"
        };
        write!(
            self.text,
            "FPS        {:>7.1}\nFRAME MS   {:>7.2}\nRENDER MS  {:>7.2}\nREBUILD MS {:>7.2}\nZOOM       {:>6.0}%\nMODE {mode}\nSCROLL ZOOM / DRAG PAN\nSPACE TOGGLE REBUILD",
            stats.frames_per_second,
            stats.frame_ms,
            stats.render_ms,
            stats.rebuild_ms,
            zoom * 100.0,
        )
        .expect("writing to a String cannot fail");
    }

    pub fn queue(&mut self, scene: &mut Scene, viewport: (f64, f64)) -> Result<(), ExampleError> {
        let text_width = self
            .text
            .lines()
            .map(caption_font::width)
            .fold(0.0, f32::max);
        let panel_width = text_width * 2.0 + 24.0;
        let panel_height = self.text.lines().count() as f32 * 20.0 + 18.0;
        let left = (viewport.0 as f32 - panel_width - 16.0).floor().max(8.0);
        let top = (viewport.1 as f32 - panel_height - 16.0).floor().max(8.0);
        let group = scene.add_group(None, Default::default())?;
        self.node_id = Some(group);
        scene.add_cached_shape(
            PANEL_CACHE_KEY,
            Some(group),
            ShapeDrawOptions::new()
                .color(Color::rgba(20, 25, 32, 240))
                .transform(
                    TransformInstance::scale(panel_width, panel_height)
                        .then(&TransformInstance::translation(left, top)),
                ),
        )?;
        for (row, line) in self.text.lines().enumerate() {
            for (column, character) in line.bytes().enumerate() {
                if character == b' ' {
                    continue;
                }
                scene.add_cached_shape(
                    FONT_CACHE_START + u64::from(character),
                    Some(group),
                    ShapeDrawOptions::new()
                        .color(Color::rgb(235, 240, 245))
                        .transform(TransformInstance::scale(2.0, 2.0).then(
                            &TransformInstance::translation(
                                left + 12.0 + column as f32 * 12.0,
                                top + 12.0 + row as f32 * 20.0,
                            ),
                        )),
                )?;
            }
        }
        Ok(())
    }

    pub fn refresh(&mut self, scene: &mut Scene, viewport: (f64, f64)) -> Result<(), ExampleError> {
        if let Some(node) = self.node_id.take() {
            scene.remove_subtrees([node], |_| {});
        }
        self.queue(scene, viewport)
    }
}
