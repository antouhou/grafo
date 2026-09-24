use super::copy_padded_readback_rows;
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
