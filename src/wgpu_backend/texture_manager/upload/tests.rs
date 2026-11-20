use super::TextureUpload;
use crate::render_backend::texture_manager::{
    AlphaMode, RgbEncoding, TexturePixelFormat, TexturePixels, TextureStorageFormat,
};
use crate::render_backend::PixelBufferError;
use half::f16;
use std::borrow::Cow;

#[test]
fn every_source_layout_uploads_to_both_storage_formats() {
    let rgba8 = [64u8, 128, 192, 17];
    let bgra8 = [192, 128, 64, 17];
    let argb = 0x114080C0_u32.to_ne_bytes();
    let rgba16 = rgba8.map(|channel| u16::from(channel) * 257);
    let rgba32 = rgba8.map(|channel| f32::from(channel) / 255.0);
    let rgba_half = rgba32.map(|channel| f16::from_f32(channel).to_bits());
    for (format, bytes, alpha) in [
        (TexturePixelFormat::Rgb8, &rgba8[..3], 255),
        (TexturePixelFormat::Rgba8, &rgba8[..], 17),
        (TexturePixelFormat::Bgra8, &bgra8[..], 17),
        (TexturePixelFormat::Argb32, &argb[..], 17),
        (
            TexturePixelFormat::Rgb16Unorm,
            bytemuck::cast_slice(&rgba16[..3]),
            255,
        ),
        (
            TexturePixelFormat::Rgba16Unorm,
            bytemuck::cast_slice(&rgba16),
            17,
        ),
        (
            TexturePixelFormat::Rgb16Float,
            bytemuck::cast_slice(&rgba_half[..3]),
            255,
        ),
        (
            TexturePixelFormat::Rgba16Float,
            bytemuck::cast_slice(&rgba_half),
            17,
        ),
        (
            TexturePixelFormat::Rgb32Float,
            bytemuck::cast_slice(&rgba32[..3]),
            255,
        ),
        (
            TexturePixelFormat::Rgba32Float,
            bytemuck::cast_slice(&rgba32),
            17,
        ),
    ] {
        let pixels = TexturePixels {
            bytes,
            size: (1, 1),
            bytes_per_row: bytes.len(),
            format,
            encoding: RgbEncoding::Srgb,
            alpha: AlphaMode::PremultipliedInLinearSpace,
        };
        let srgb = TextureUpload::new(TextureStorageFormat::Rgba8Srgb, pixels).unwrap();
        assert_eq!(srgb.bytes.as_ref(), [64, 128, 192, alpha], "{format:?}");

        let linear = TextureUpload::new(TextureStorageFormat::Rgba16Float, pixels).unwrap();
        assert_eq!(linear.bytes.len(), 8);
        let expected = [0.05127, 0.21586, 0.52712, f32::from(alpha) / 255.0];
        for (bytes, expected) in linear.bytes.as_chunks::<2>().0.iter().zip(expected) {
            let value = f16::from_le_bytes(*bytes).to_f32();
            assert!(
                (value - expected).abs() < 0.0006,
                "{format:?}: {value}, expected {expected}"
            );
        }
    }
}

#[test]
fn encoding_and_alpha_modes_produce_linear_premultiplied_colors() {
    for (alpha, encoded_source, linear_source) in [
        (AlphaMode::Opaque, [128, 64, 0, 255], [188, 137, 0, 255]),
        (AlphaMode::Straight, [93, 45, 0, 128], [137, 99, 0, 128]),
        (
            AlphaMode::PremultipliedInLinearSpace,
            [128, 64, 0, 128],
            [188, 137, 0, 128],
        ),
        (
            AlphaMode::PremultipliedInEncodedSpace,
            [188, 92, 0, 128],
            [188, 137, 0, 128],
        ),
    ] {
        for (encoding, expected) in [
            (RgbEncoding::Srgb, encoded_source),
            (RgbEncoding::LinearSrgb, linear_source),
        ] {
            let pixels = TexturePixels {
                encoding,
                ..TexturePixels::rgba8_srgb(&[128, 64, 0, 128], (1, 1), alpha)
            };
            let upload = TextureUpload::new(TextureStorageFormat::Rgba8Srgb, pixels).unwrap();
            assert_eq!(upload.bytes.as_ref(), expected, "{encoding:?} {alpha:?}");
        }
    }
}

#[test]
fn float_storage_preserves_hdr_and_negative_channels_after_premultiplication() {
    let samples = [[8.0_f32, -4.0, 1e6, 0.5], [1.0, 1.0, 1.0, 0.0]];
    let pixels = TexturePixels {
        bytes: bytemuck::cast_slice(&samples),
        size: (2, 1),
        bytes_per_row: 32,
        format: TexturePixelFormat::Rgba32Float,
        encoding: RgbEncoding::LinearSrgb,
        alpha: AlphaMode::Straight,
    };
    let upload = TextureUpload::new(TextureStorageFormat::Rgba16Float, pixels).unwrap();
    let expected = [4.0, -2.0, 65504.0, 0.5, 0.0, 0.0, 0.0, 0.0];
    assert_eq!(upload.bytes.len(), 16);
    for (bytes, expected) in upload.bytes.as_chunks::<2>().0.iter().zip(expected) {
        assert_eq!(f16::from_le_bytes(*bytes).to_f32(), expected);
    }
    let upload = TextureUpload::new(TextureStorageFormat::Rgba8Srgb, pixels).unwrap();
    assert_eq!(upload.bytes.as_ref(), [255, 0, 255, 128, 0, 0, 0, 0]);
}

#[test]
fn uploads_respect_byte_strides_and_end_at_the_final_pixel() {
    let bytes = [1, 2, 3, 255, 99, 4, 5, 6, 255, 77];
    let pixels = TexturePixels {
        bytes_per_row: 5,
        ..TexturePixels::rgba8_srgb(&bytes, (1, 2), AlphaMode::PremultipliedInLinearSpace)
    };
    let upload = TextureUpload::new(TextureStorageFormat::Rgba8Srgb, pixels).unwrap();
    assert!(matches!(upload.bytes, Cow::Borrowed(_)));
    assert_eq!(upload.bytes_per_row, 5);
    assert_eq!(upload.bytes.as_ref(), &bytes[..9]);

    let pixels = TexturePixels {
        format: TexturePixelFormat::Bgra8,
        ..pixels
    };
    let upload = TextureUpload::new(TextureStorageFormat::Rgba8Srgb, pixels).unwrap();
    assert_eq!(upload.bytes_per_row, 4);
    assert_eq!(upload.bytes.as_ref(), [3, 2, 1, 255, 6, 5, 4, 255]);
    let short = TexturePixels {
        bytes: &bytes[..8],
        ..pixels
    };
    assert!(matches!(
        TextureUpload::new(TextureStorageFormat::Rgba8Srgb, short),
        Err(PixelBufferError::StorageTooSmall {
            required: 9,
            provided: 8
        })
    ));
}
