use grafo::Color;

#[test]
fn byte_colors_roundtrip_through_float_storage() {
    for channel in u8::MIN..=u8::MAX {
        let bytes = [channel, u8::MAX - channel, channel / 2, channel];
        let color = Color::rgba(bytes[0], bytes[1], bytes[2], bytes[3]);
        assert_eq!(color.to_array(), bytes);
    }
}

#[test]
fn extended_colors_decode_without_clipping_or_premultiplication() {
    let encoded = [-0.5, 0.04045, 1.5, 0.25];
    let color = Color::rgba_f32(encoded[0], encoded[1], encoded[2], encoded[3]);
    assert_eq!(color.to_array_f32(), encoded);

    // Reference values from the extended sRGB transfer function.
    let expected = [-0.214_041_14, 0.003_130_805, 2.537_155_4, 0.25];
    for (actual, expected) in color.normalize().into_iter().zip(expected) {
        assert!((actual - expected).abs() < 1e-6);
    }
    assert_eq!(color.to_array(), [0, 10, 255, 64]);
}
