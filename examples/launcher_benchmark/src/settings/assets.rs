use super::{HEIGHT, HEIGHT_STEP, HEIGHT_STEPS, WIDTH};
use crate::artwork::Artwork;
use crate::BenchmarkError;
use grafo::{Scene, Shape, StrokeGeometryOptions};

pub(super) struct Assets {
    pub panels: Vec<u64>,
    pub tile: u64,
    pub chip: u64,
    pub track: u64,
    pub fill: u64,
    pub thumb: u64,
    pub row: u64,
    pub scrollbar: u64,
    pub icons: [u64; 6],
}

impl Assets {
    pub fn new(scene: &mut Scene, artwork: &mut Artwork) -> Result<Self, BenchmarkError> {
        // These outlines remain general paths; all rounded surfaces use the native constructor.
        let outlines = [
            Shape::builder()
                .begin((20.0, 5.0))
                .line_to((20.0, 19.0))
                .end()
                .begin((12.0, 10.0))
                .cubic_bezier_to((0.0, 19.0), (8.0, 35.0), (20.0, 35.0))
                .cubic_bezier_to((32.0, 35.0), (40.0, 19.0), (28.0, 10.0))
                .end()
                .build(),
            Shape::builder()
                .begin((25.0, 5.0))
                .cubic_bezier_to((0.0, 1.0), (0.0, 39.0), (29.0, 34.0))
                .cubic_bezier_to((10.0, 28.0), (13.0, 12.0), (25.0, 5.0))
                .close()
                .build(),
            Shape::builder()
                .begin((33.0, 16.0))
                .cubic_bezier_to((28.0, 0.0), (4.0, 3.0), (5.0, 21.0))
                .cubic_bezier_to((5.0, 37.0), (30.0, 40.0), (35.0, 24.0))
                .end()
                .begin((24.0, 15.0))
                .line_to((34.0, 17.0))
                .line_to((35.0, 7.0))
                .end()
                .build(),
            Shape::builder()
                .begin((18.0, 7.0))
                .line_to((5.0, 7.0))
                .line_to((5.0, 33.0))
                .line_to((18.0, 33.0))
                .end()
                .begin((15.0, 20.0))
                .line_to((35.0, 20.0))
                .end()
                .begin((27.0, 12.0))
                .line_to((35.0, 20.0))
                .line_to((27.0, 28.0))
                .end()
                .build(),
            Shape::builder()
                .begin((3.0, 13.0))
                .cubic_bezier_to((12.0, 3.0), (28.0, 3.0), (37.0, 13.0))
                .end()
                .begin((9.0, 21.0))
                .cubic_bezier_to((15.0, 14.0), (25.0, 14.0), (31.0, 21.0))
                .end()
                .begin((15.0, 29.0))
                .cubic_bezier_to((18.0, 25.0), (22.0, 25.0), (25.0, 29.0))
                .end()
                .begin((20.0, 33.0))
                .line_to((20.0, 35.0))
                .end()
                .build(),
            Shape::builder()
                .begin((12.0, 11.0))
                .line_to((28.0, 27.0))
                .line_to((20.0, 35.0))
                .line_to((20.0, 5.0))
                .line_to((28.0, 13.0))
                .line_to((12.0, 29.0))
                .end()
                .build(),
        ];
        let mut icons = [0; 6];
        for (key, shape) in icons.iter_mut().zip(outlines) {
            *key = artwork.load(scene, shape.with_stroke(StrokeGeometryOptions::new(2.4)))?;
        }
        // Half-pixel height steps preserve the radius and change mask bounds without per-draw tessellation.
        let mut panels = Vec::with_capacity(HEIGHT_STEPS + 1);
        for step in 0..=HEIGHT_STEPS {
            panels.push(artwork.rounded(
                scene,
                [WIDTH, HEIGHT - step as f32 * HEIGHT_STEP],
                28.0,
            )?);
        }
        Ok(Self {
            panels,
            tile: artwork.rounded(scene, [153.0, 80.0], 18.0)?,
            chip: artwork.rounded(scene, [44.0, 44.0], 22.0)?,
            track: artwork.rounded(scene, [318.0, 22.0], 11.0)?,
            fill: artwork.rounded(scene, [202.0, 22.0], 11.0)?,
            thumb: artwork.rounded(scene, [28.0, 28.0], 14.0)?,
            row: artwork.rounded(scene, [318.0, 44.0], 22.0)?,
            scrollbar: artwork.rounded(scene, [6.0, 160.0], 3.0)?,
            icons,
        })
    }
}
