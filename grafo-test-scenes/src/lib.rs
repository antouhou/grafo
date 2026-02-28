pub mod expectations;
pub mod scene;
#[allow(dead_code)]
pub mod shaders;

pub use expectations::{check_pixels, PixelExpectation};
pub use scene::{build_main_scene, CANVAS_HEIGHT, CANVAS_WIDTH};
