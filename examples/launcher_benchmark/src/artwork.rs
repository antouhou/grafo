use crate::BenchmarkError;
use grafo::{BorderRadii, Scene, Shape, StrokeGeometryOptions};

pub const APP_NAMES: [&str; 54] = [
    "Cinema",
    "Music",
    "Photos",
    "Explorer",
    "Messages",
    "Calendar",
    "Weather",
    "Podcasts",
    "Maps",
    "Notes",
    "Camera",
    "Browser",
    "Gallery",
    "Radio",
    "Books",
    "Files",
    "Clock",
    "Settings",
    "Studio",
    "Compose",
    "Travel",
    "Reader",
    "Connect",
    "Tasks",
    "Forecast",
    "Listen",
    "Navigate",
    "Journal",
    "Capture",
    "Discover",
    "Library",
    "Playlists",
    "Stories",
    "Documents",
    "Timer",
    "System",
    "Movies",
    "Albums",
    "Memories",
    "Archive",
    "People",
    "Agenda",
    "Outdoors",
    "Episodes",
    "Places",
    "Sketches",
    "Record",
    "Search",
    "Favorites",
    "Stations",
    "Comics",
    "Downloads",
    "Focus",
    "Tools",
];

pub struct Artwork {
    next_key: u64,
    pub symbols: [u64; 6],
    pub circle: u64,
    pub ribbon: u64,
}

impl Artwork {
    pub fn new(scene: &mut Scene) -> Result<Self, BenchmarkError> {
        let mut artwork = Self {
            next_key: 1,
            symbols: [0; 6],
            circle: 0,
            ribbon: 0,
        };
        artwork.circle = artwork.rounded(scene, [80.0, 80.0], 40.0)?;
        artwork.ribbon = artwork.load(
            scene,
            Shape::builder()
                .begin((-20.0, 54.0))
                .line_to((80.0, -12.0))
                .line_to((106.0, 16.0))
                .line_to((2.0, 94.0))
                .close()
                .build(),
        )?;
        artwork.symbols[0] = artwork.load(
            scene,
            Shape::builder()
                .begin((27.0, 20.0))
                .line_to((60.0, 40.0))
                .line_to((27.0, 60.0))
                .close()
                .build(),
        )?;
        artwork.symbols[1] = artwork.load(
            scene,
            Shape::builder()
                .begin((33.0, 56.0))
                .line_to((33.0, 25.0))
                .line_to((58.0, 19.0))
                .line_to((58.0, 50.0))
                .end()
                .build()
                .with_stroke(StrokeGeometryOptions::new(5.0)),
        )?;
        artwork.symbols[2] = artwork.load(
            scene,
            Shape::builder()
                .begin((18.0, 58.0))
                .line_to((35.0, 34.0))
                .line_to((47.0, 48.0))
                .line_to((56.0, 38.0))
                .line_to((68.0, 58.0))
                .close()
                .build(),
        )?;
        artwork.symbols[3] = artwork.load(
            scene,
            Shape::builder()
                .begin((15.0, 30.0))
                .line_to((34.0, 30.0))
                .line_to((40.0, 24.0))
                .line_to((65.0, 24.0))
                .line_to((65.0, 59.0))
                .line_to((15.0, 59.0))
                .close()
                .build(),
        )?;
        artwork.symbols[4] = artwork.load(
            scene,
            Shape::builder()
                .begin((18.0, 27.0))
                .line_to((62.0, 27.0))
                .line_to((62.0, 55.0))
                .line_to((18.0, 55.0))
                .close()
                .begin((18.0, 28.0))
                .line_to((40.0, 44.0))
                .line_to((62.0, 28.0))
                .end()
                .build()
                .with_stroke(StrokeGeometryOptions::new(3.0)),
        )?;
        artwork.symbols[5] = artwork.load(
            scene,
            Shape::builder()
                .begin((20.0, 25.0))
                .line_to((60.0, 25.0))
                .line_to((60.0, 60.0))
                .line_to((20.0, 60.0))
                .close()
                .begin((20.0, 35.0))
                .line_to((60.0, 35.0))
                .end()
                .begin((30.0, 18.0))
                .line_to((30.0, 30.0))
                .end()
                .begin((50.0, 18.0))
                .line_to((50.0, 30.0))
                .end()
                .build()
                .with_stroke(StrokeGeometryOptions::new(3.0)),
        )?;
        Ok(artwork)
    }

    pub fn load(&mut self, scene: &mut Scene, shape: Shape) -> Result<u64, BenchmarkError> {
        let key = self.next_key;
        self.next_key += 1;
        // Both workload densities load identical geometry at each asset key.
        scene.load_shape(shape, key, Some(key))?;
        Ok(key)
    }

    pub fn rounded(
        &mut self,
        scene: &mut Scene,
        size: [f32; 2],
        radius: f32,
    ) -> Result<u64, BenchmarkError> {
        self.load(
            scene,
            Shape::rounded_rect([(0.0, 0.0), (size[0], size[1])], BorderRadii::new(radius)),
        )
    }
}
