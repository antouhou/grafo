use crate::animation::{AnimationSample, Workload};
use crate::artwork::Artwork;
use crate::font;
use crate::interface::{Interface, MARGIN};
use crate::statistics::Statistics;
use crate::BenchmarkError;
use grafo::cosmic_text::{Align, Buffer, FontSystem};
use grafo::{Color, CosmicText, Scene, ShapeDrawOptions, TransformInstance};
use std::time::Duration;
use std::{array, fmt::Write};

const TABLE_COLUMNS: usize = 6;
const TABLE_TEXTS: usize = TABLE_COLUMNS * 2;
const TEXT_COUNT: usize = TABLE_TEXTS + 2;

fn text_options(column: usize) -> ShapeDrawOptions {
    let (table, offset, y) = if column < TABLE_TEXTS {
        let offsets = [0.0, 176.0, 252.0, 328.0, 404.0, 480.0];
        (
            column / TABLE_COLUMNS,
            offsets[column % TABLE_COLUMNS],
            594.0,
        )
    } else {
        (column - TABLE_TEXTS, 0.0, 674.0)
    };
    ShapeDrawOptions::new()
        .color(Color::rgb(205, 221, 239))
        .clips_children(false)
        .transform(TransformInstance::translation(
            MARGIN + 16.0 + table as f32 * 588.0 + offset,
            y,
        ))
}

#[derive(Default)]
pub struct LiveTimings {
    pub update: Duration,
    pub render: Duration,
}

pub struct Overlay {
    buffers: [Buffer; TEXT_COUNT],
    nodes: [usize; TEXT_COUNT],
    text: [String; TEXT_COUNT],
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
            ShapeDrawOptions::new()
                .color(Color::rgb(8, 14, 25))
                .transform(TransformInstance::translation(MARGIN, 580.0)),
        )?;
        let buffers: [Buffer; TEXT_COUNT] = array::from_fn(|column| {
            let is_hint = column >= TABLE_TEXTS;
            let mut buffer = font::shape(fonts, "", if is_hint { 12.0 } else { 13.0 }, 20.0);
            let width = if is_hint {
                556.0
            } else if column % TABLE_COLUMNS == 0 {
                160.0
            } else {
                64.0
            };
            buffer.set_size(Some(width), None);
            buffer
        });
        let mut nodes = [0; TEXT_COUNT];
        for (column, buffer) in buffers.iter().enumerate() {
            nodes[column] = scene.add_text(
                CosmicText::new(buffer, fonts, None),
                Some(background),
                text_options(column),
            )?;
        }
        Ok(Self {
            buffers,
            nodes,
            text: array::from_fn(|_| String::with_capacity(128)),
        })
    }

    pub fn refresh(
        &mut self,
        scene: &mut Scene,
        statistics: &Statistics,
        timings: &LiveTimings,
        sample: &AnimationSample,
        interface: &mut Interface,
        vsync: bool,
    ) -> Result<(), BenchmarkError> {
        for text in &mut self.text {
            text.clear();
        }
        let phase = if sample.is_warming_up {
            "WARMUP"
        } else {
            "MEASURING"
        };
        // Formatting into a String cannot fail.
        let _ = writeln!(self.text[0], "{} / {}", sample.mode_label, phase);
        let _ = writeln!(
            self.text[TABLE_COLUMNS],
            "VSYNC {}",
            if vsync { "ON" } else { "OFF" },
        );
        for table in [0, TABLE_COLUMNS] {
            for (column, header) in ["fps", "mean ms", "p95 ms", "max ms", "n"]
                .into_iter()
                .enumerate()
            {
                let _ = writeln!(self.text[table + column + 1], "{header}");
            }
        }
        for workload in Workload::ALL {
            let statistics = &statistics.workloads[workload.index()];
            let table = workload.index() / 3 * TABLE_COLUMNS;
            let _ = writeln!(self.text[table], "{}", workload.label());
            for (column, value) in [
                statistics.fps(),
                statistics.mean_ms(),
                statistics.p95_ms(),
                statistics.max_ms(),
            ]
            .into_iter()
            .enumerate()
            {
                let _ = writeln!(self.text[table + column + 1], "{value:.1}");
            }
            let _ = writeln!(self.text[table + 5], "{}", statistics.samples);
        }
        self.text[TABLE_TEXTS].push_str(
            "A/P/S/C/O mode  Space pause  R reset  V vsync  D load  F fullscreen  Esc exit",
        );
        let _ = write!(
            self.text[TABLE_TEXTS + 1],
            "{}x{}  {} apps/page   update {:.2} ms   render call {:.2} ms",
            scene.size().0,
            scene.size().1,
            interface.apps_per_page,
            timings.update.as_secs_f64() * 1000.0,
            timings.render.as_secs_f64() * 1000.0
        );
        for (column, buffer) in self.buffers.iter_mut().enumerate() {
            let alignment =
                (column < TABLE_TEXTS && column % TABLE_COLUMNS != 0).then_some(Align::Right);
            font::set_text(&mut interface.fonts, buffer, &self.text[column], alignment);
            scene.replace_with_text(
                self.nodes[column],
                CosmicText::new(buffer, &mut interface.fonts, None),
                text_options(column),
            )?;
        }
        Ok(())
    }
}
