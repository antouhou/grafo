pub mod expectations;
pub mod scene;
pub mod shaders;

pub use expectations::{check_pixels, PixelExpectation};
pub use scene::{
    build_bounded_targets_scene, build_main_scene, build_nested_targets_scene, CANVAS_HEIGHT,
    CANVAS_WIDTH,
};
