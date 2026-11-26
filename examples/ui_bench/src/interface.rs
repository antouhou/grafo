use crate::animation::Pose;
use crate::artwork::{Artwork, APP_NAMES};
use crate::controls::{BUTTONS, BUTTON_SIZE};
use crate::font;
use crate::settings::Settings;
use crate::BenchmarkError;
use grafo::cosmic_text::{Buffer, FontSystem};
use grafo::{
    Color, CosmicText, DrawMode, Fill, Gradient, GradientCommonDesc, GradientStop,
    LinearGradientDesc, LinearGradientLine, Scene, Shape, ShapeDrawOptions, TransformInstance,
};

pub const WIDTH: f32 = 1280.0;
pub const HEIGHT: f32 = 720.0;
pub const APPS_PER_PAGE: usize = 3;
pub(super) const MARGIN: f32 = 52.0;
const PAGE_WIDTH: f32 = WIDTH - MARGIN * 2.0;
const PAGE_TOP: f32 = 112.0;
const PAGE_HEIGHT: f32 = 432.0;
pub(super) const TEXT: Color = Color::from_srgb_f32(237.0 / 255.0, 242.0 / 255.0, 250.0 / 255.0);
pub(super) const MUTED: Color = Color::from_srgb_f32(164.0 / 255.0, 178.0 / 255.0, 199.0 / 255.0);

fn gradient(start: Color, end: Color, size: [f32; 2]) -> Fill {
    Fill::Gradient(
        Gradient::linear(LinearGradientDesc {
            line: LinearGradientLine {
                start: [0.0, 0.0],
                end: size,
            },
            common: GradientCommonDesc::new([GradientStop::auto(start), GradientStop::auto(end)]),
        })
        .expect("finite two-stop gradient"),
    )
}

struct MovingShape {
    id: usize,
    key: u64,
    options: ShapeDrawOptions,
}

struct MovingLabel {
    id: usize,
    buffer: Buffer,
    options: ShapeDrawOptions,
}

#[derive(Default)]
pub struct Layer {
    shapes: Vec<MovingShape>,
    labels: Vec<MovingLabel>,
    translation: [f32; 2],
}

impl Layer {
    pub fn translate(
        &mut self,
        scene: &mut Scene,
        fonts: &mut FontSystem,
        translation: [f32; 2],
    ) -> Result<(), BenchmarkError> {
        if self.translation == translation {
            return Ok(());
        }
        let offset = TransformInstance::translation(translation[0], translation[1]);
        for shape in &self.shapes {
            let base = shape
                .options
                .transform
                .unwrap_or_else(TransformInstance::identity);
            scene.replace_with_cached_shape(
                shape.id,
                shape.key,
                shape.options.clone().transform(base.then(&offset)),
            )?;
        }
        for label in &self.labels {
            let base = label
                .options
                .transform
                .unwrap_or_else(TransformInstance::identity);
            scene.replace_with_text(
                label.id,
                CosmicText::new(&label.buffer, fonts, None),
                label.options.clone().transform(base.then(&offset)),
            )?;
        }
        self.translation = translation;
        Ok(())
    }
}

pub struct LayerBuilder<'a> {
    scene: &'a mut Scene,
    fonts: &'a mut FontSystem,
    layer: &'a mut Layer,
}

impl<'a> LayerBuilder<'a> {
    pub fn new(scene: &'a mut Scene, fonts: &'a mut FontSystem, layer: &'a mut Layer) -> Self {
        Self {
            scene,
            fonts,
            layer,
        }
    }

    pub fn add(
        &mut self,
        key: u64,
        parent: Option<usize>,
        options: ShapeDrawOptions,
    ) -> Result<usize, BenchmarkError> {
        let id = self.scene.add_cached_shape(key, parent, options.clone())?;
        self.layer.shapes.push(MovingShape { id, key, options });
        Ok(id)
    }

    pub fn shape(
        &mut self,
        key: u64,
        parent: Option<usize>,
        position: [f32; 2],
        color: Color,
    ) -> Result<usize, BenchmarkError> {
        self.add(
            key,
            parent,
            ShapeDrawOptions::new()
                .color(color)
                .transform(TransformInstance::translation(position[0], position[1])),
        )
    }

    fn text(
        &mut self,
        buffer: Buffer,
        parent: Option<usize>,
        position: [f32; 2],
        color: Color,
    ) -> Result<(), BenchmarkError> {
        let options = ShapeDrawOptions::new()
            .color(color)
            .clips_children(false)
            .transform(TransformInstance::translation(position[0], position[1]));
        let id = self.scene.add_text(
            CosmicText::new(&buffer, self.fonts, None),
            parent,
            options.clone(),
        )?;
        self.layer.labels.push(MovingLabel {
            id,
            buffer,
            options,
        });
        Ok(())
    }

    pub fn label(
        &mut self,
        text: &str,
        parent: Option<usize>,
        position: [f32; 2],
        (size, color): (f32, Color),
    ) -> Result<(), BenchmarkError> {
        let buffer = font::shape(self.fonts, text, size, size * 1.3);
        self.text(buffer, parent, position, color)
    }

    pub fn centered_label(
        &mut self,
        text: &str,
        parent: Option<usize>,
        bounds: [(f32, f32); 2],
        (size, color): (f32, Color),
    ) -> Result<(), BenchmarkError> {
        let line_height = size * 1.3;
        let buffer = font::shape(self.fonts, text, size, line_height);
        let width = buffer
            .layout_runs()
            .map(|run| run.line_w)
            .fold(0.0, f32::max);
        self.text(
            buffer,
            parent,
            [
                (bounds[0].0 + bounds[1].0 - width) * 0.5,
                (bounds[0].1 + bounds[1].1 - line_height) * 0.5,
            ],
            color,
        )
    }
}

pub struct Interface {
    pages: [Layer; 3],
    settings: Settings,
    page_indicator: Layer,
    pub fonts: FontSystem,
}

fn build_background(
    scene: &mut Scene,
    artwork: &mut Artwork,
    font: &mut FontSystem,
    layer: &mut Layer,
) -> Result<(), BenchmarkError> {
    let background = artwork.load(scene, Shape::rect([(0.0, 0.0), (WIDTH, HEIGHT)]))?;
    let mut builder = LayerBuilder::new(scene, font, layer);
    builder.add(
        background,
        None,
        ShapeDrawOptions::new()
            .fill(gradient(
                Color::rgb(10, 18, 38),
                Color::rgb(43, 30, 57),
                [WIDTH, HEIGHT],
            ))
            .clips_children(false),
    )?;
    let hill = artwork.load(
        builder.scene,
        Shape::builder()
            .begin((0.0, 530.0))
            .cubic_bezier_to((330.0, 330.0), (850.0, 740.0), (1280.0, 290.0))
            .line_to((1280.0, 720.0))
            .line_to((0.0, 720.0))
            .close()
            .build(),
    )?;
    builder.add(
        hill,
        None,
        ShapeDrawOptions::new()
            .fill(gradient(
                Color::rgba(32, 80, 108, 70),
                Color::rgba(157, 82, 96, 80),
                [WIDTH, HEIGHT],
            ))
            .clips_children(false),
    )?;
    builder.label("Home", None, [MARGIN, 38.5], (30.0, TEXT))?;
    let control = artwork.rounded(builder.scene, BUTTON_SIZE, 18.0)?;
    for button in &BUTTONS {
        builder.shape(
            control,
            None,
            [button.bounds[0].0, button.bounds[0].1],
            Color::rgba(220, 235, 255, 22),
        )?;
        builder.centered_label(button.label, None, button.bounds, (13.0, TEXT))?;
    }
    let status = artwork.rounded(builder.scene, [160.0, 48.0], 24.0)?;
    builder.shape(status, None, [1068.0, 34.0], Color::rgb(20, 30, 45))?;
    builder.centered_label(
        "12:45   Settings",
        None,
        [(1068.0, 34.0), (1228.0, 82.0)],
        (15.0, TEXT),
    )?;
    Ok(())
}

fn build_page(
    scene: &mut Scene,
    artwork: &mut Artwork,
    font: &mut FontSystem,
    page: usize,
    parent: usize,
) -> Result<Layer, BenchmarkError> {
    let mut layer = Layer::default();
    let size = 80.0;
    let tile = artwork.rounded(scene, [size, size], size * 0.24)?;
    let tile_inner = artwork.rounded(scene, [size - 6.0, size - 6.0], size * 0.21)?;
    let card = artwork.rounded(scene, [330.0, 240.0], 24.0)?;
    let mut builder = LayerBuilder::new(scene, font, &mut layer);
    let cell_width = 1128.0 / APPS_PER_PAGE as f32;
    let label_size = 22.0;
    let label_gap = 16.0;
    let content_height = size + label_gap + label_size * 1.3;
    let y = PAGE_TOP + (PAGE_HEIGHT - content_height) * 0.5;
    let palette = [
        Color::rgb(224, 80, 101),
        Color::rgb(137, 96, 232),
        Color::rgb(38, 153, 190),
        Color::rgb(228, 163, 68),
        Color::rgb(63, 162, 130),
        Color::rgb(63, 114, 216),
    ];
    for index in 0..APPS_PER_PAGE {
        let name_index = page * APPS_PER_PAGE + index;
        let x = MARGIN
            + 24.0
            + page as f32 * PAGE_WIDTH
            + index as f32 * cell_width
            + (cell_width - size) * 0.5;
        let parent = builder.shape(
            card,
            Some(parent),
            [x - 125.0, PAGE_TOP + (PAGE_HEIGHT - 240.0) * 0.5],
            Color::rgb(35, 42, 61),
        )?;
        let color = palette[(index + page * 2) % palette.len()];
        let icon = builder.shape(tile, Some(parent), [x, y], Color::rgba(240, 248, 255, 100))?;
        let interior = builder.shape(tile_inner, Some(icon), [x + 3.0, y + 3.0], color)?;
        builder.shape(
            artwork.circle,
            Some(interior),
            [x + size * 0.42, y - size * 0.4],
            Color::rgba(255, 255, 255, 35),
        )?;
        builder.shape(
            artwork.ribbon,
            Some(interior),
            [x, y],
            Color::rgba(255, 255, 255, 22),
        )?;
        let symbol = name_index % 6;
        builder.add(
            artwork.symbols[symbol],
            Some(interior),
            ShapeDrawOptions::new()
                .color(Color::rgba(255, 255, 255, 240))
                .draw_mode(if [1, 4, 5].contains(&symbol) {
                    DrawMode::Stroke
                } else {
                    DrawMode::Fill
                })
                .transform(TransformInstance::translation(x, y)),
        )?;
        let label_top = y + size + label_gap;
        builder.centered_label(
            APP_NAMES[name_index],
            Some(parent),
            [(x, label_top), (x + size, label_top + label_size * 1.3)],
            (label_size, TEXT),
        )?;
    }
    Ok(layer)
}

impl Interface {
    pub fn new(scene: &mut Scene) -> Result<(Self, Artwork), BenchmarkError> {
        let mut artwork = Artwork::new(scene)?;
        let mut font = font::load();
        let mut background = Layer::default();
        build_background(scene, &mut artwork, &mut font, &mut background)?;
        let viewport_key = artwork.rounded(scene, [PAGE_WIDTH, PAGE_HEIGHT], 28.0)?;
        let viewport = LayerBuilder::new(scene, &mut font, &mut background).shape(
            viewport_key,
            None,
            [MARGIN, PAGE_TOP],
            Color::rgba(12, 20, 34, 22),
        )?;
        let pages = [
            build_page(scene, &mut artwork, &mut font, 0, viewport)?,
            build_page(scene, &mut artwork, &mut font, 1, viewport)?,
            build_page(scene, &mut artwork, &mut font, 2, viewport)?,
        ];
        let dot_key = artwork.rounded(scene, [22.0, 6.0], 3.0)?;
        let mut builder = LayerBuilder::new(scene, &mut font, &mut background);
        for x in [603.0, 629.0, 655.0] {
            builder.shape(dot_key, None, [x, 556.0], Color::rgb(82, 93, 116))?;
        }
        let mut page_indicator = Layer::default();
        LayerBuilder::new(scene, &mut font, &mut page_indicator).shape(
            dot_key,
            None,
            [603.0, 556.0],
            TEXT,
        )?;
        let settings = Settings::new(scene, &mut artwork, &mut font)?;
        Ok((
            Self {
                pages,
                settings,
                page_indicator,
                fonts: font,
            },
            artwork,
        ))
    }

    pub fn update(&mut self, scene: &mut Scene, pose: Pose) -> Result<(), BenchmarkError> {
        for page in &mut self.pages {
            page.translate(scene, &mut self.fonts, [-pose.page * PAGE_WIDTH, 0.0])?;
        }
        self.settings.update(scene, &mut self.fonts, pose)?;
        self.page_indicator
            .translate(scene, &mut self.fonts, [pose.page * 26.0, 0.0])?;
        Ok(())
    }
}
