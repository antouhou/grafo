use crate::animation::Pose;
use crate::artwork::{Artwork, APP_NAMES};
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
const PAGE_WIDTH: f32 = 1176.0;
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
    pub fn add(
        &mut self,
        scene: &mut Scene,
        key: u64,
        parent: Option<usize>,
        options: ShapeDrawOptions,
    ) -> Result<usize, BenchmarkError> {
        let id = scene.add_cached_shape(key, parent, options.clone())?;
        self.shapes.push(MovingShape { id, key, options });
        Ok(id)
    }

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

    fn text(
        &mut self,
        scene: &mut Scene,
        fonts: &mut FontSystem,
        buffer: Buffer,
        parent: Option<usize>,
        options: ShapeDrawOptions,
    ) -> Result<(), BenchmarkError> {
        let id = scene.add_text(
            CosmicText::new(&buffer, fonts, None),
            parent,
            options.clone(),
        )?;
        self.labels.push(MovingLabel {
            id,
            buffer,
            options,
        });
        Ok(())
    }

    pub fn label(
        &mut self,
        scene: &mut Scene,
        font: &mut FontSystem,
        text: &str,
        position: [f32; 2],
        (size, color): (f32, Color),
        parent: Option<usize>,
    ) -> Result<(), BenchmarkError> {
        let buffer = font::shape(font, text, size, size * 1.3);
        self.text(
            scene,
            font,
            buffer,
            parent,
            ShapeDrawOptions::new()
                .color(color)
                .clips_children(false)
                .transform(TransformInstance::translation(position[0], position[1])),
        )?;
        Ok(())
    }
}

pub struct Interface {
    pages: [Layer; 3],
    settings: Settings,
    page_indicator: Layer,
    pub apps_per_page: usize,
    pub fonts: FontSystem,
}

pub(super) fn rectangle(
    layer: &mut Layer,
    scene: &mut Scene,
    key: u64,
    parent: Option<usize>,
    position: [f32; 2],
    color: Color,
) -> Result<usize, BenchmarkError> {
    layer.add(
        scene,
        key,
        parent,
        ShapeDrawOptions::new()
            .color(color)
            .transform(TransformInstance::translation(position[0], position[1])),
    )
}

fn build_background(
    scene: &mut Scene,
    artwork: &mut Artwork,
    font: &mut FontSystem,
    layer: &mut Layer,
) -> Result<(), BenchmarkError> {
    let background = artwork.load(scene, Shape::rect([(0.0, 0.0), (WIDTH, HEIGHT)]))?;
    layer.add(
        scene,
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
        scene,
        Shape::builder()
            .begin((0.0, 530.0))
            .cubic_bezier_to((330.0, 330.0), (850.0, 740.0), (1280.0, 290.0))
            .line_to((1280.0, 720.0))
            .line_to((0.0, 720.0))
            .close()
            .build(),
    )?;
    layer.add(
        scene,
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
    layer.label(scene, font, "Home", [52.0, 28.0], (30.0, TEXT), None)?;
    let control = artwork.rounded(scene, [80.0, 36.0], 18.0)?;
    for (index, name) in [
        "Auto", "Pages", "Settings", "Scroll", "Power", "Pause", "Load +",
    ]
    .iter()
    .enumerate()
    {
        let x = 408.0 + index as f32 * 88.0;
        rectangle(
            layer,
            scene,
            control,
            None,
            [x, 40.0],
            Color::rgba(220, 235, 255, 22),
        )?;
        layer.label(scene, font, name, [x + 12.0, 46.0], (13.0, TEXT), None)?;
    }
    let status = artwork.rounded(scene, [160.0, 48.0], 24.0)?;
    rectangle(
        layer,
        scene,
        status,
        None,
        [1068.0, 34.0],
        Color::rgb(20, 30, 45),
    )?;
    layer.label(
        scene,
        font,
        "12:45   Settings",
        [1087.0, 48.0],
        (15.0, TEXT),
        None,
    )?;
    Ok(())
}

fn build_page(
    scene: &mut Scene,
    artwork: &mut Artwork,
    font: &mut FontSystem,
    page: usize,
    columns: usize,
    rows: usize,
    parent: usize,
) -> Result<Layer, BenchmarkError> {
    let mut layer = Layer::default();
    let size = 80.0;
    let tile = artwork.rounded(scene, [size, size], size * 0.24)?;
    let tile_inner = artwork.rounded(scene, [size - 6.0, size - 6.0], size * 0.21)?;
    let card = artwork.rounded(scene, [330.0, 240.0], 24.0)?;
    let cell_width = 1128.0 / columns as f32;
    let cell_height = 408.0 / rows as f32;
    let palette = [
        Color::rgb(224, 80, 101),
        Color::rgb(137, 96, 232),
        Color::rgb(38, 153, 190),
        Color::rgb(228, 163, 68),
        Color::rgb(63, 162, 130),
        Color::rgb(63, 114, 216),
    ];
    for index in 0..columns * rows {
        let name_index = (page * columns * rows + index) % APP_NAMES.len();
        let x = 76.0
            + page as f32 * PAGE_WIDTH
            + (index % columns) as f32 * cell_width
            + (cell_width - size) * 0.5;
        let y = if rows == 1 {
            265.0
        } else {
            135.0 + (index / columns) as f32 * cell_height
        };
        let parent = if rows == 1 {
            rectangle(
                &mut layer,
                scene,
                card,
                Some(parent),
                [x - 125.0, 219.0],
                Color::rgb(35, 42, 61),
            )?
        } else {
            parent
        };
        let color = palette[(index + page * 2) % palette.len()];
        let icon = rectangle(
            &mut layer,
            scene,
            tile,
            Some(parent),
            [x, y],
            Color::rgba(240, 248, 255, 100),
        )?;
        let interior = rectangle(
            &mut layer,
            scene,
            tile_inner,
            Some(icon),
            [x + 3.0, y + 3.0],
            color,
        )?;
        let scale = size / 80.0;
        layer.add(
            scene,
            artwork.circle,
            Some(interior),
            ShapeDrawOptions::new()
                .color(Color::rgba(255, 255, 255, 35))
                .transform(TransformInstance::scale(scale, scale).then(
                    &TransformInstance::translation(x + size * 0.42, y - size * 0.4),
                )),
        )?;
        layer.add(
            scene,
            artwork.ribbon,
            Some(interior),
            ShapeDrawOptions::new()
                .color(Color::rgba(255, 255, 255, 22))
                .transform(
                    TransformInstance::scale(scale, scale)
                        .then(&TransformInstance::translation(x, y)),
                ),
        )?;
        let symbol = name_index % 6;
        layer.add(
            scene,
            artwork.symbols[symbol],
            Some(interior),
            ShapeDrawOptions::new()
                .color(Color::rgba(255, 255, 255, 240))
                .draw_mode(if [1, 4, 5].contains(&symbol) {
                    DrawMode::Stroke
                } else {
                    DrawMode::Fill
                })
                .transform(
                    TransformInstance::scale(scale, scale)
                        .then(&TransformInstance::translation(x, y)),
                ),
        )?;
        let label_size = if rows == 1 { 22.0 } else { 15.0 };
        let buffer = font::shape(font, APP_NAMES[name_index], label_size, label_size * 1.3);
        let width = buffer
            .layout_runs()
            .map(|run| run.line_w)
            .fold(0.0, f32::max);
        layer.text(
            scene,
            font,
            buffer,
            Some(parent),
            ShapeDrawOptions::new()
                .color(TEXT)
                .clips_children(false)
                .transform(TransformInstance::translation(
                    x + (size - width) * 0.5,
                    y + size + 18.0,
                )),
        )?;
    }
    Ok(layer)
}

impl Interface {
    pub fn new(scene: &mut Scene, dense: bool) -> Result<(Self, Artwork), BenchmarkError> {
        let mut artwork = Artwork::new(scene)?;
        let mut font = font::load();
        let mut background = Layer::default();
        build_background(scene, &mut artwork, &mut font, &mut background)?;
        let viewport_key = artwork.rounded(scene, [1176.0, 440.0], 28.0)?;
        let viewport = rectangle(
            &mut background,
            scene,
            viewport_key,
            None,
            [52.0, 119.5],
            Color::rgba(12, 20, 34, 22),
        )?;
        let (columns, rows) = if dense { (6, 3) } else { (3, 1) };
        let pages = [
            build_page(scene, &mut artwork, &mut font, 0, columns, rows, viewport)?,
            build_page(scene, &mut artwork, &mut font, 1, columns, rows, viewport)?,
            build_page(scene, &mut artwork, &mut font, 2, columns, rows, viewport)?,
        ];
        let dot_key = artwork.rounded(scene, [22.0, 6.0], 3.0)?;
        for x in [604.0, 630.0, 656.0] {
            rectangle(
                &mut background,
                scene,
                dot_key,
                None,
                [x, 560.0],
                Color::rgb(82, 93, 116),
            )?;
        }
        let mut page_indicator = Layer::default();
        rectangle(
            &mut page_indicator,
            scene,
            dot_key,
            None,
            [604.0, 560.0],
            TEXT,
        )?;
        let settings = Settings::new(scene, &mut artwork, &mut font)?;
        Ok((
            Self {
                pages,
                settings,
                page_indicator,
                apps_per_page: columns * rows,
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
