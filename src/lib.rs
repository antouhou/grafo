pub use wgpu;

mod vertex;
mod wgpu_state;
mod pipeline;
mod debug_tools;
mod util;
mod stroke;

pub use wgpu_state::{Renderer, Color, Shape, TextLayout, TextAlignment};
pub use stroke::Stroke;

pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
