mod assets;

use self::assets::Assets;
use crate::animation::Pose;
use crate::artwork::{Artwork, Icon};
use crate::interface::{self, Layer, LayerBuilder, MUTED, TEXT};
use crate::{font, shadow, BenchmarkError};
use grafo::cosmic_text::FontSystem;
use grafo::{BorderRadii, Color, Scene, Shape, ShapeDrawOptions, TransformInstance};

const X: f32 = 858.0;
const Y: f32 = 96.0;
const WIDTH: f32 = 370.0;
const HEIGHT: f32 = 452.0;
const PADDING: f32 = 24.0;
const CHIP_SIZE: f32 = 36.0;
const ICON_SIZE: f32 = 28.0;
const CONTENT_LEFT: f32 = X + PADDING;
const CONTENT_WIDTH: f32 = WIDTH - PADDING * 2.0;
const HEADER_TOP: f32 = Y + PADDING;
const POWER_BUTTON_LEFT: f32 = CONTENT_LEFT + CONTENT_WIDTH - 44.0;
const SCROLL_TOP: f32 = Y + 192.0;
const SCROLL_BOTTOM: f32 = Y + HEIGHT - PADDING;
const TILE_HEIGHT: f32 = 80.0;
const TILE_GAP: f32 = 12.0;
const TILE_WIDTH: f32 = (CONTENT_WIDTH - TILE_GAP) * 0.5;
const SCROLL_CONTENT_HEIGHT: f32 = TILE_HEIGHT * 4.0 + TILE_GAP * 3.0;
const SCROLL_HEIGHT: f32 = SCROLL_BOTTOM - SCROLL_TOP;
const SCROLL_DISTANCE: f32 = SCROLL_CONTENT_HEIGHT - SCROLL_HEIGHT;
const SCROLLBAR_HEIGHT: f32 = SCROLL_HEIGHT * SCROLL_HEIGHT / SCROLL_CONTENT_HEIGHT;
const POWER_HEIGHT: f32 = 308.0;
const TRAVEL: f32 = 440.0;

fn icon(
    builder: &mut LayerBuilder<'_>,
    icon: Icon,
    parent: usize,
    position: [f32; 2],
) -> Result<(), BenchmarkError> {
    let scale = ICON_SIZE / 40.0;
    builder.add(
        icon.key,
        Some(parent),
        ShapeDrawOptions::new()
            .color(TEXT)
            .draw_mode(icon.draw_mode)
            .transform(
                TransformInstance::scale(scale, scale)
                    .then(&TransformInstance::translation(position[0], position[1])),
            ),
    )?;
    Ok(())
}

fn icon_chip(
    builder: &mut LayerBuilder<'_>,
    assets: &Assets,
    symbol: Icon,
    parent: usize,
    position: [f32; 2],
    color: Color,
) -> Result<(), BenchmarkError> {
    let chip = builder.shape(assets.chip, Some(parent), position, color)?;
    let inset = (CHIP_SIZE - ICON_SIZE) * 0.5;
    icon(
        builder,
        symbol,
        chip,
        [position[0] + inset, position[1] + inset],
    )
}

struct MainMenu {
    root: usize,
    controls: Layer,
    tiles: Layer,
    scrollbar: Layer,
    viewport: usize,
    offset: Option<f32>,
}

impl MainMenu {
    fn new(
        scene: &mut Scene,
        fonts: &mut FontSystem,
        assets: &Assets,
        parent: usize,
    ) -> Result<Self, BenchmarkError> {
        let root = scene.add_group(Some(parent), Default::default())?;
        let mut controls = Layer::default();
        let mut builder = LayerBuilder::new(scene, fonts, &mut controls);
        builder.label(
            "Quick settings",
            Some(root),
            [CONTENT_LEFT, font::centered_y(HEADER_TOP, 44.0, 23.0)],
            (23.0, TEXT),
        )?;
        icon_chip(
            &mut builder,
            assets,
            assets.icons[0],
            root,
            [
                POWER_BUTTON_LEFT + (44.0 - CHIP_SIZE) * 0.5,
                HEADER_TOP + (44.0 - CHIP_SIZE) * 0.5,
            ],
            Color::rgb(105, 43, 55),
        )?;
        for (index, label) in ["Volume", "Brightness"].iter().enumerate() {
            let y = HEADER_TOP + 52.0 + index as f32 * 54.0;
            builder.label(label, Some(root), [CONTENT_LEFT, y], (14.0, MUTED))?;
            for (key, [left, top], color) in [
                (assets.track, [0.0, 24.0], Color::rgb(64, 75, 92)),
                (assets.fill, [0.0, 24.0], Color::rgb(144, 201, 238)),
                (assets.thumb, [188.0, 21.0], TEXT),
            ] {
                builder.shape(key, Some(root), [CONTENT_LEFT + left, y + top], color)?;
            }
        }
        let viewport = scene.add_clipping_rect(
            [(X, SCROLL_TOP), (X + WIDTH, SCROLL_BOTTOM)],
            Some(root),
            None::<TransformInstance>,
            true,
        )?;
        let mut tiles = Layer::default();
        let mut builder = LayerBuilder::new(scene, fonts, &mut tiles);
        for (index, (label, symbol)) in [
            ("Wi-Fi", 4),
            ("Bluetooth", 5),
            ("Focus", 0),
            ("Night mode", 1),
            ("Appearance", 1),
            ("Keyboard", 3),
            ("Security", 0),
            ("Updates", 2),
        ]
        .iter()
        .enumerate()
        {
            let x = CONTENT_LEFT + (index % 2) as f32 * (TILE_WIDTH + TILE_GAP);
            let y = SCROLL_TOP + (index / 2) as f32 * (TILE_HEIGHT + TILE_GAP);
            let tile =
                builder.shape(assets.tile, Some(viewport), [x, y], Color::rgb(55, 64, 79))?;
            icon_chip(
                &mut builder,
                assets,
                assets.icons[*symbol],
                tile,
                [x + 12.0, y + (TILE_HEIGHT - CHIP_SIZE) * 0.5],
                Color::rgb(43, 51, 66),
            )?;
            builder.label(
                label,
                Some(tile),
                [
                    x + 12.0 + CHIP_SIZE + 8.0,
                    font::centered_y(y, TILE_HEIGHT, 13.0),
                ],
                (13.0, TEXT),
            )?;
        }
        let mut scrollbar = Layer::default();
        LayerBuilder::new(scene, fonts, &mut scrollbar).shape(
            assets.scrollbar,
            Some(viewport),
            [X + WIDTH - 12.0, SCROLL_TOP],
            Color::rgb(116, 126, 144),
        )?;
        Ok(Self {
            root,
            controls,
            tiles,
            scrollbar,
            viewport,
            offset: None,
        })
    }

    fn update(
        &mut self,
        scene: &mut Scene,
        fonts: &mut FontSystem,
        offset: f32,
        scroll: f32,
        is_switching: bool,
    ) -> Result<(), BenchmarkError> {
        if self.offset != Some(offset) {
            scene.replace_with_clipping_rect(
                self.viewport,
                [
                    (X + offset, SCROLL_TOP),
                    (X + WIDTH + offset, SCROLL_BOTTOM),
                ],
                None::<TransformInstance>,
                true,
            )?;
            self.offset = Some(offset);
        }
        self.controls.translate(scene, fonts, [offset, 0.0])?;
        self.tiles
            .translate(scene, fonts, [offset, -scroll * SCROLL_DISTANCE])?;
        self.scrollbar.translate(
            scene,
            fonts,
            [
                offset + if is_switching { WIDTH } else { 0.0 },
                scroll * (SCROLL_HEIGHT - SCROLLBAR_HEIGHT),
            ],
        )?;
        Ok(())
    }
}

struct PowerMenu {
    root: usize,
    layer: Layer,
}

impl PowerMenu {
    fn new(
        scene: &mut Scene,
        fonts: &mut FontSystem,
        assets: &Assets,
        parent: usize,
    ) -> Result<Self, BenchmarkError> {
        let root = scene.add_group(Some(parent), Default::default())?;
        let mut layer = Layer::default();
        let mut builder = LayerBuilder::new(scene, fonts, &mut layer);
        let back = builder.shape(
            assets.row,
            Some(root),
            [CONTENT_LEFT, HEADER_TOP],
            Color::rgb(66, 77, 94),
        )?;
        builder.label(
            "<   Back",
            Some(back),
            [
                CONTENT_LEFT + 18.0,
                font::centered_y(HEADER_TOP, 44.0, 18.0),
            ],
            (18.0, TEXT),
        )?;
        for (index, label) in ["Shut down", "Suspend", "Reboot", "Log out"]
            .iter()
            .enumerate()
        {
            let y = HEADER_TOP + 60.0 + index as f32 * 52.0;
            let row = builder.shape(
                assets.row,
                Some(root),
                [CONTENT_LEFT, y],
                if index == 0 {
                    Color::rgb(105, 43, 55)
                } else {
                    Color::rgb(55, 64, 79)
                },
            )?;
            icon(
                &mut builder,
                assets.icons[index],
                row,
                [
                    CONTENT_LEFT + 12.0 + (CHIP_SIZE - ICON_SIZE) * 0.5,
                    y + (44.0 - ICON_SIZE) * 0.5,
                ],
            )?;
            builder.label(
                label,
                Some(row),
                [
                    CONTENT_LEFT + 12.0 + CHIP_SIZE + 8.0,
                    font::centered_y(y, 44.0, 18.0),
                ],
                (18.0, TEXT),
            )?;
        }
        Ok(Self { root, layer })
    }
}

pub struct Settings {
    assets: Assets,
    panel: usize,
    viewport: usize,
    main: Option<MainMenu>,
    power: Option<PowerMenu>,
    pose: Option<Pose>,
}

impl Settings {
    pub fn new(
        scene: &mut Scene,
        artwork: &mut Artwork,
        fonts: &mut FontSystem,
    ) -> Result<Self, BenchmarkError> {
        let assets = Assets::new(scene, artwork)?;
        let interface_clip = scene.add_clipping_rect(
            [(0.0, 0.0), (interface::WIDTH, interface::HEIGHT)],
            None,
            None::<TransformInstance>,
            true,
        )?;
        let panel = scene.add_cached_shape(
            assets.panel,
            Some(interface_clip),
            interface::shape_options([X + TRAVEL, Y], Color::rgb(37, 45, 59)),
        )?;
        shadow::attach(scene, panel, WIDTH)?;
        let viewport = scene.add_clipping_rect(
            [(X, Y), (X + WIDTH, Y + HEIGHT)],
            Some(panel),
            None::<TransformInstance>,
            false,
        )?;
        let main = Some(MainMenu::new(scene, fonts, &assets, viewport)?);
        Ok(Self {
            assets,
            panel,
            viewport,
            main,
            power: None,
            pose: None,
        })
    }

    pub fn update(
        &mut self,
        scene: &mut Scene,
        fonts: &mut FontSystem,
        pose: Pose,
    ) -> Result<(), BenchmarkError> {
        let offset = (1.0 - pose.settings) * TRAVEL;
        let height = HEIGHT + (POWER_HEIGHT - HEIGHT) * pose.power;
        let is_switching = pose.power > 0.0 && pose.power < 1.0;
        if self
            .pose
            .is_none_or(|previous| previous.power != pose.power)
        {
            scene.load_shape(
                Shape::rounded_rect([(0.0, 0.0), (WIDTH, height)], BorderRadii::new(28.0)),
                self.assets.panel,
                None,
            )?;
        }
        if self.pose.is_none_or(|previous| {
            previous.settings != pose.settings || previous.power != pose.power
        }) {
            scene.replace_with_cached_shape(
                self.panel,
                self.assets.panel,
                interface::shape_options([X + offset, Y], Color::rgb(37, 45, 59)),
            )?;
            scene.replace_with_clipping_rect(
                self.viewport,
                [(X + offset, Y), (X + WIDTH + offset, Y + height)],
                None::<TransformInstance>,
                is_switching,
            )?;
        }
        // Keep both menus during a slide; remove the outgoing menu at the endpoint.
        if pose.power < 1.0 && self.main.is_none() {
            self.main = Some(MainMenu::new(scene, fonts, &self.assets, self.viewport)?);
        }
        if pose.power > 0.0 && self.power.is_none() {
            self.power = Some(PowerMenu::new(scene, fonts, &self.assets, self.viewport)?);
        }
        if let Some(main) = &mut self.main {
            main.update(
                scene,
                fonts,
                offset - pose.power * WIDTH,
                pose.scroll,
                is_switching,
            )?;
        }
        if let Some(power) = &mut self.power {
            power
                .layer
                .translate(scene, fonts, [offset + (1.0 - pose.power) * WIDTH, 0.0])?;
        }
        if pose.power == 1.0 {
            if let Some(main) = self.main.take() {
                scene.remove_subtrees([main.root], |_| {});
            }
        } else if pose.power == 0.0 {
            if let Some(power) = self.power.take() {
                scene.remove_subtrees([power.root], |_| {});
            }
        }
        self.pose = Some(pose);
        Ok(())
    }
}
