use crate::animation::{AnimationSample, RunStatus, Workload, RUN_CYCLES};
use crate::artwork::Artwork;
use crate::font;
use crate::interface::{self, APPS_PER_PAGE, MARGIN};
use crate::statistics::Statistics;
use crate::BenchmarkError;
use grafo::cosmic_text::{Align, Buffer, FontSystem};
use grafo::{Color, CosmicText, Scene, ShapeDrawOptions};
use std::fmt::Write;
use std::time::Duration;

const TABLE_COLUMNS: usize = 6;
const TABLE_TEXTS: usize = TABLE_COLUMNS * 2;
const TEXT_COUNT: usize = TABLE_TEXTS + 2;

struct OverlayText {
    buffer: Buffer,
    node: usize,
    text: String,
    options: ShapeDrawOptions,
    alignment: Option<Align>,
}

#[derive(Default)]
pub struct LiveTimings {
    pub update: Duration,
    pub render: Duration,
}

pub struct Overlay {
    labels: Vec<OverlayText>,
}

impl Overlay {
    pub fn new(
        scene: &mut Scene,
        fonts: &mut FontSystem,
        artwork: &mut Artwork,
    ) -> Result<Self, BenchmarkError> {
        let background = artwork.rounded(scene, [1176.0, 128.0], 18.0)?;
        let background = scene.add_cached_shape(
            background,
            None,
            interface::shape_options([MARGIN, 580.0], Color::rgb(8, 14, 25)),
        )?;
        let mut labels = Vec::with_capacity(TEXT_COUNT);
        for index in 0..TEXT_COUNT {
            let column = index % TABLE_COLUMNS;
            let is_hint = index >= TABLE_TEXTS;
            let (table, offset, y, width, size) = if is_hint {
                (index - TABLE_TEXTS, 0.0, 674.0, 556.0, 12.0)
            } else {
                let offsets = [0.0, 176.0, 252.0, 328.0, 404.0, 480.0];
                (
                    index / TABLE_COLUMNS,
                    offsets[column],
                    594.0,
                    if column == 0 { 160.0 } else { 64.0 },
                    13.0,
                )
            };
            let options = interface::shape_options(
                [MARGIN + 16.0 + table as f32 * 588.0 + offset, y],
                Color::rgb(205, 221, 239),
            )
            .clips_children(false);
            let mut buffer = font::shape(fonts, "", size, 20.0);
            buffer.set_size(Some(width), None);
            labels.push(OverlayText {
                node: scene.add_text(
                    CosmicText::new(&buffer, fonts, None),
                    Some(background),
                    options.clone(),
                )?,
                buffer,
                text: String::with_capacity(128),
                options,
                alignment: (!is_hint && column != 0).then_some(Align::Right),
            });
        }
        Ok(Self { labels })
    }

    pub fn refresh(
        &mut self,
        scene: &mut Scene,
        statistics: &Statistics,
        timings: &LiveTimings,
        sample: &AnimationSample,
        fonts: &mut FontSystem,
        vsync_requested: bool,
    ) -> Result<(), BenchmarkError> {
        for label in &mut self.labels {
            label.text.clear();
        }
        let phase = match sample.status {
            RunStatus::Running if sample.is_warming_up => "WARMUP",
            RunStatus::Running => "MEASURING",
            RunStatus::Stopped => "STOPPED",
            RunStatus::Complete => "COMPLETE",
        };
        // Formatting into a String cannot fail.
        let _ = writeln!(
            self.labels[0].text,
            "{phase} {}/{RUN_CYCLES}",
            sample.completed_cycles
        );
        let _ = writeln!(
            self.labels[TABLE_COLUMNS].text,
            "VSYNC REQ {}",
            if vsync_requested { "ON" } else { "OFF" },
        );
        for table in [0, TABLE_COLUMNS] {
            for (column, header) in ["fps", "mean ms", "p95 ms", "max ms", "n"]
                .into_iter()
                .enumerate()
            {
                let _ = writeln!(self.labels[table + column + 1].text, "{header}");
            }
        }
        for workload in Workload::ALL {
            let statistics = &statistics.workloads[workload.index()];
            let table = workload.index() / 3 * TABLE_COLUMNS;
            let _ = writeln!(self.labels[table].text, "{}", workload.label());
            let _ = writeln!(self.labels[table + 1].text, "{:.1}", statistics.fps());
            let _ = writeln!(self.labels[table + 2].text, "{:.1}", statistics.mean_ms());
            let _ = writeln!(self.labels[table + 3].text, "{}", statistics.p95_ms());
            let _ = writeln!(self.labels[table + 4].text, "{:.1}", statistics.max_ms());
            let _ = writeln!(self.labels[table + 5].text, "{}", statistics.samples);
        }
        self.labels[TABLE_TEXTS]
            .text
            .push_str("Enter start  Space stop  Esc exit");
        let _ = write!(
            self.labels[TABLE_TEXTS + 1].text,
            "{}x{}  {} apps/page   update {:.2} ms   render call {:.2} ms",
            scene.size().0,
            scene.size().1,
            APPS_PER_PAGE,
            timings.update.as_secs_f64() * 1000.0,
            timings.render.as_secs_f64() * 1000.0
        );
        for label in &mut self.labels {
            font::set_text(fonts, &mut label.buffer, &label.text, label.alignment);
            scene.replace_with_text(
                label.node,
                CosmicText::new(&label.buffer, fonts, None),
                label.options.clone(),
            )?;
        }
        Ok(())
    }
}
