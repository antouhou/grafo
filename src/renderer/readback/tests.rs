use super::copy_padded_readback_rows;
use crate::renderer::types::RendererScratch;

#[test]
fn begin_frame_preserves_large_readback_storage() {
    let mut scratch = RendererScratch::new();
    // Exceed the 64 MiB trimming threshold to detect accidental storage trimming.
    let readback_size = 65 * 1024 * 1024;
    scratch.readback_bytes.resize(readback_size, 0);
    let storage_pointer = scratch.readback_bytes.as_ptr();
    let storage_capacity = scratch.readback_bytes.capacity();

    for _ in 0..3 {
        scratch.begin_frame();
        assert!(scratch.readback_bytes.is_empty());
        assert_eq!(scratch.readback_bytes.capacity(), storage_capacity);
        assert_eq!(scratch.readback_bytes.as_ptr(), storage_pointer);
        scratch.readback_bytes.resize(readback_size, 0);
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
