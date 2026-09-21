use super::copy_padded_readback_rows;
use crate::{Renderer, RendererCreationError};
use futures::executor::block_on;

#[test]
fn clearing_draw_queue_preserves_large_readback_storage() {
    let mut renderer = match block_on(Renderer::try_new_headless((16, 16), 1.0)) {
        Ok(renderer) => renderer,
        Err(RendererCreationError::AdapterNotAvailable(_)) => {
            println!("Skipping test: no suitable GPU adapter available.");
            return;
        }
        Err(error) => panic!("Failed to create headless renderer: {error}"),
    };
    let readback_size = 65 * 1024 * 1024;
    renderer
        .state
        .scratch
        .readback_bytes
        .resize(readback_size, 0);
    let storage_pointer = renderer.state.scratch.readback_bytes.as_ptr();
    let storage_capacity = renderer.state.scratch.readback_bytes.capacity();

    for _ in 0..3 {
        renderer.clear_draw_queue();
        renderer.begin_frame_scratch();
        assert_eq!(
            renderer.state.scratch.readback_bytes.capacity(),
            storage_capacity
        );
        assert_eq!(
            renderer.state.scratch.readback_bytes.as_ptr(),
            storage_pointer
        );
        renderer
            .state
            .scratch
            .readback_bytes
            .resize(readback_size, 0);
    }
}

#[test]
fn copy_padded_readback_rows_handles_unpadded_data() {
    let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let mut output = Vec::new();

    copy_padded_readback_rows(&data, 2, 4, 4, &mut output);
    assert_eq!(output, data);
}

#[test]
fn copy_padded_readback_rows_strips_padding() {
    let data = vec![1, 2, 3, 4, 9, 9, 9, 9, 5, 6, 7, 8, 8, 8, 8, 8];
    let mut output = Vec::new();

    copy_padded_readback_rows(&data, 2, 4, 8, &mut output);
    assert_eq!(output, vec![1, 2, 3, 4, 5, 6, 7, 8]);
}
