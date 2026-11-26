use crate::animation::{AnimationSample, Workload};
use crate::artwork::Artwork;
use crate::font;
use crate::interface::Interface;
use crate::statistics::Statistics;
use crate::BenchmarkError;
use grafo::cosmic_text::{Buffer, FontSystem};
use grafo::{Color, CosmicText, Scene, ShapeDrawOptions, TransformInstance};
use std::fmt::Write;
use std::time::Duration;

fn text_options(column: usize) -> ShapeDrawOptions {
    ShapeDrawOptions::new()
        .color(Color::rgb(205, 221, 239))
        .clips_children(false)
        .transform(TransformInstance::translation(
            if column == 0 { 44.0 } else { 645.0 },
            590.0,
        ))
}

#[derive(Default)]
pub struct LiveTimings {
    pub update: Duration,
    pub render: Duration,
}

pub struct Overlay {
    buffers: [Buffer; 2],
    nodes: [usize; 2],
    text: [String; 2],
}

impl Overlay {
    pub fn new(
        scene: &mut Scene,
        fonts: &mut FontSystem,
        artwork: &mut Artwork,
    ) -> Result<Self, BenchmarkError> {
        let background = artwork.rounded(scene, [1232.0, 128.0], 18.0)?;
        let background = scene.add_cached_shape(
            background,
            None,
            ShapeDrawOptions::new()
                .color(Color::rgb(8, 14, 25))
                .transform(TransformInstance::translation(24.0, 580.0)),
        )?;
        let buffers = [
            font::shape(fonts, "", 13.0, 23.0),
            font::shape(fonts, "", 13.0, 23.0),
        ];
        let mut nodes = [0; 2];
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
            text: [String::with_capacity(512), String::with_capacity(512)],
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
        writeln!(
            self.text[0],
            "{} / {}     fps / mean / p95 / max ms",
            sample.mode_label, phase
        )
        .expect("String writes cannot fail");
        writeln!(
            self.text[1],
            "VSYNC {} requested  {}x{}  {} apps/page",
            if vsync { "ON" } else { "OFF" },
            scene.size().0,
            scene.size().1,
            interface.apps_per_page
        )
        .expect("String writes cannot fail");
        for workload in Workload::ALL {
            let statistics = &statistics.workloads[workload.index()];
            let column = workload.index() / 3;
            writeln!(
                self.text[column],
                "{:<15} {:>5.1} fps  {:>5.1}/{:>5.1}/{:>5.1} ms  n={}",
                workload.label(),
                statistics.fps(),
                statistics.mean_ms(),
                statistics.p95_ms(),
                statistics.max_ms(),
                statistics.samples
            )
            .expect("String writes cannot fail");
        }
        write!(
            self.text[0],
            "A/P/S/C/O mode  Up/Down scroll  Space pause  R reset"
        )
        .expect("String writes cannot fail");
        write!(
            self.text[1],
            "Update {:.2} ms  render {:.2} ms   V vsync  D load  F fullscreen",
            timings.update.as_secs_f64() * 1000.0,
            timings.render.as_secs_f64() * 1000.0
        )
        .expect("String writes cannot fail");
        for (column, buffer) in self.buffers.iter_mut().enumerate() {
            font::set_text(&mut interface.fonts, buffer, &self.text[column]);
            scene.replace_with_text(
                self.nodes[column],
                CosmicText::new(buffer, &mut interface.fonts, None),
                text_options(column),
            )?;
        }
        Ok(())
    }
}
