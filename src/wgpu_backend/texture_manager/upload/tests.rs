use super::{TextureUpload, TEXTURE_UPLOAD_BATCH_BYTE_LIMIT};
use crate::render_backend::texture_manager::{
    AlphaMode, RgbEncoding, TexturePixelFormat, TexturePixelLayout, TexturePixels,
    TextureStorageFormat,
};
use crate::render_backend::PixelBufferError;
use half::f16;

fn storage_pixel_bytes(storage: TextureStorageFormat) -> usize {
    match storage {
        TextureStorageFormat::Rgba8Srgb => 4,
        TextureStorageFormat::Rgba16Float => 8,
    }
}

fn converted_bytes(
    storage: TextureStorageFormat,
    pixels: TexturePixels<'_>,
    chunk_size: usize,
) -> Vec<u8> {
    let mut upload = TextureUpload::new(storage, pixels.layout).unwrap();
    let mut source = pixels.bytes;
    let mut converted = Vec::new();
    while !upload.is_complete() {
        assert!(!source.is_empty(), "{:?}", upload.short_input_error());
        let consumed = upload.convert_into_batch(&source[..source.len().min(chunk_size)]);
        assert_ne!(consumed, 0);
        source = &source[consumed..];
        if upload.is_batch_ready() {
            let batch = upload.batch();
            let row_bytes = batch.size.0 as usize * storage_pixel_bytes(storage);
            for row in batch.bytes.chunks_exact(batch.bytes_per_row as usize) {
                converted.extend_from_slice(&row[..row_bytes]);
            }
            upload.clear_batch();
        }
    }
    converted
}

#[test]
fn every_source_layout_uploads_to_both_storage_formats_with_fragmented_pixels() {
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
        let pixels = TexturePixelLayout {
            size: (1, 1),
            bytes_per_row: bytes.len(),
            format,
            encoding: RgbEncoding::Srgb,
            alpha: AlphaMode::PremultipliedInLinearSpace,
        }
        .pixels(bytes);
        for chunk_size in [1, 2, 3, 5, usize::MAX] {
            let srgb = converted_bytes(TextureStorageFormat::Rgba8Srgb, pixels, chunk_size);
            assert_eq!(srgb, [64, 128, 192, alpha], "{format:?} {chunk_size}");

            let linear = converted_bytes(TextureStorageFormat::Rgba16Float, pixels, chunk_size);
            assert_eq!(linear.len(), 8);
            let expected = [0.05127, 0.21586, 0.52712, f32::from(alpha) / 255.0];
            for (bytes, expected) in linear.as_chunks::<2>().0.iter().zip(expected) {
                let value = f16::from_le_bytes(*bytes).to_f32();
                assert!(
                    (value - expected).abs() < 0.0006,
                    "{format:?} {chunk_size}: {value}, expected {expected}"
                );
            }
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
            let pixels = TexturePixelLayout {
                encoding,
                ..TexturePixelLayout::rgba8_srgb((1, 1), alpha)
            }
            .pixels(&[128, 64, 0, 128]);
            for chunk_size in [1, usize::MAX] {
                let upload = converted_bytes(TextureStorageFormat::Rgba8Srgb, pixels, chunk_size);
                assert_eq!(upload, expected, "{encoding:?} {alpha:?}");
            }
        }
    }
}

#[test]
fn float_storage_preserves_hdr_and_negative_channels_after_premultiplication() {
    let samples = [[8.0_f32, -4.0, 1e6, 0.5], [1.0, 1.0, 1.0, 0.0]];
    let pixels = TexturePixelLayout {
        size: (2, 1),
        bytes_per_row: 32,
        format: TexturePixelFormat::Rgba32Float,
        encoding: RgbEncoding::LinearSrgb,
        alpha: AlphaMode::Straight,
    }
    .pixels(bytemuck::cast_slice(&samples));
    for chunk_size in [1, 3, 7, usize::MAX] {
        let upload = converted_bytes(TextureStorageFormat::Rgba16Float, pixels, chunk_size);
        let expected = [4.0, -2.0, 65504.0, 0.5, 0.0, 0.0, 0.0, 0.0];
        assert_eq!(upload.len(), 16);
        for (bytes, expected) in upload.as_chunks::<2>().0.iter().zip(expected) {
            assert_eq!(f16::from_le_bytes(*bytes).to_f32(), expected);
        }
        let upload = converted_bytes(TextureStorageFormat::Rgba8Srgb, pixels, chunk_size);
        assert_eq!(upload, [255, 0, 255, 128, 0, 0, 0, 0]);
    }
}

#[test]
fn uploads_respect_byte_strides_and_end_at_the_final_pixel() {
    let bytes = [1, 2, 3, 255, 99, 4, 5, 6, 255, 77];
    let pixels = TexturePixelLayout {
        bytes_per_row: 5,
        ..TexturePixelLayout::rgba8_srgb((1, 2), AlphaMode::PremultipliedInLinearSpace)
    }
    .pixels(&bytes);
    for (format, expected) in [
        (TexturePixelFormat::Rgba8, [1, 2, 3, 255, 4, 5, 6, 255]),
        (TexturePixelFormat::Bgra8, [3, 2, 1, 255, 6, 5, 4, 255]),
    ] {
        let pixels = TexturePixelLayout {
            format,
            ..pixels.layout
        }
        .pixels(pixels.bytes);
        for chunk_size in [1, 2, 5, usize::MAX] {
            assert_eq!(
                converted_bytes(TextureStorageFormat::Rgba8Srgb, pixels, chunk_size),
                expected
            );
        }

        let mut upload =
            TextureUpload::new(TextureStorageFormat::Rgba8Srgb, pixels.layout).unwrap();
        assert_eq!(upload.convert_into_batch(&bytes), 9);
        assert!(upload.is_complete());
        assert_eq!(upload.convert_into_batch(&bytes[9..]), 0);
        let batch = upload.batch();
        assert_eq!(batch.bytes_per_row, 256);
        assert_eq!(&batch.bytes[..4], &expected[..4]);
        assert_eq!(&batch.bytes[256..260], &expected[4..]);
        assert!(batch.bytes[4..256].iter().all(|&byte| byte == 0));
        assert!(batch.bytes[260..].iter().all(|&byte| byte == 0));

        let mut short = TextureUpload::new(TextureStorageFormat::Rgba8Srgb, pixels.layout).unwrap();
        assert_eq!(short.convert_into_batch(&bytes[..8]), 8);
        assert!(!short.is_complete());
        assert!(!short.is_batch_ready());
        assert_eq!(
            short.short_input_error(),
            PixelBufferError::StorageTooSmall {
                required: 9,
                provided: 8,
            }
        );
    }
}

#[test]
fn narrow_uploads_bound_padded_staging_and_preserve_rows_across_batches() {
    let height = 9000;
    let mut bytes = Vec::new();
    let mut expected = Vec::new();
    for row in 0..height {
        let pixel = [row as u8, (row >> 8) as u8, 13, 255];
        bytes.extend_from_slice(&pixel);
        expected.extend_from_slice(&pixel);
        if row + 1 < height {
            bytes.extend_from_slice(&[91, 92, 93]);
        }
    }
    let pixels = TexturePixelLayout {
        bytes_per_row: 7,
        ..TexturePixelLayout::rgba8_srgb((1, height), AlphaMode::PremultipliedInLinearSpace)
    }
    .pixels(&bytes);
    let mut upload = TextureUpload::new(TextureStorageFormat::Rgba8Srgb, pixels.layout).unwrap();
    let capacity = upload.bytes.capacity();
    let mut source = pixels.bytes;
    let mut batches = Vec::new();
    let mut actual = Vec::new();
    while !upload.is_complete() {
        let consumed = upload.convert_into_batch(source);
        assert_ne!(consumed, 0);
        source = &source[consumed..];
        let batch = upload.batch();
        assert!(batch.bytes.len() <= TEXTURE_UPLOAD_BATCH_BYTE_LIMIT);
        batches.push((batch.origin, batch.size));
        for row in batch.bytes.chunks_exact(batch.bytes_per_row as usize) {
            actual.extend_from_slice(&row[..4]);
            assert!(row[4..].iter().all(|&byte| byte == 0));
        }
        upload.clear_batch();
        assert_eq!(upload.bytes.capacity(), capacity);
    }
    assert!(source.is_empty());
    assert_eq!(actual, expected);
    assert_eq!(
        batches,
        [
            ((0, 0), (1, 4096)),
            ((0, 4096), (1, 4096)),
            ((0, 8192), (1, 808))
        ]
    );
    assert!(!upload.is_batch_ready());
    assert_eq!(
        converted_bytes(TextureStorageFormat::Rgba8Srgb, pixels, 3),
        expected
    );
}

#[test]
fn input_padding_never_requires_a_source_row_buffer() {
    let stride = 64 * 1024 * 1024;
    let layout = TexturePixelLayout {
        size: (1, 2),
        bytes_per_row: stride,
        format: TexturePixelFormat::Rgb32Float,
        encoding: RgbEncoding::LinearSrgb,
        alpha: AlphaMode::Straight,
    };
    let pixel = [1.0_f32, 0.5, 0.0];
    let pixel_bytes = bytemuck::cast_slice(&pixel);
    let mut upload = TextureUpload::new(TextureStorageFormat::Rgba16Float, layout).unwrap();
    assert_eq!(upload.convert_into_batch(pixel_bytes), pixel_bytes.len());
    let padding = [0; 4096];
    let mut remaining_padding = stride - pixel_bytes.len();
    while remaining_padding != 0 {
        let count = remaining_padding.min(padding.len());
        assert_eq!(upload.convert_into_batch(&padding[..count]), count);
        remaining_padding -= count;
    }
    assert_eq!(upload.bytes.len(), 256);
    assert_eq!(upload.bytes.capacity(), 512);
    assert_eq!(upload.pending_pixel_len, 0);
    assert_eq!(upload.convert_into_batch(pixel_bytes), pixel_bytes.len());
    assert!(upload.is_complete());
    assert_eq!(upload.source_bytes_consumed, stride + pixel_bytes.len());
    assert_eq!(upload.batch().bytes.len(), 512);
}

#[test]
fn a_row_larger_than_the_batch_limit_still_uploads_as_one_row() {
    let width = TEXTURE_UPLOAD_BATCH_BYTE_LIMIT as u32 / 4 + 1;
    let bytes = vec![255; width as usize * 4 * 2];
    let pixels =
        TexturePixels::rgba8_srgb(&bytes, (width, 2), AlphaMode::PremultipliedInLinearSpace);
    let mut upload = TextureUpload::new(TextureStorageFormat::Rgba8Srgb, pixels.layout).unwrap();
    let consumed = upload.convert_into_batch(&bytes);
    assert_eq!(consumed, width as usize * 4);
    assert_eq!(upload.batch().size, (width, 1));
    assert!(upload.batch().bytes.len() > TEXTURE_UPLOAD_BATCH_BYTE_LIMIT);
    assert_eq!(upload.batch().bytes.len(), upload.bytes.capacity());
    upload.clear_batch();
    assert_eq!(upload.convert_into_batch(&bytes[consumed..]), consumed);
    assert!(upload.is_complete());
    assert_eq!(upload.batch().origin, (0, 1));
}

#[test]
fn invalid_layouts_are_rejected_before_consuming_input() {
    let layout = TexturePixelLayout::rgba8_srgb((1, 1), AlphaMode::Straight);
    for invalid in [
        TexturePixelLayout {
            size: (0, 1),
            ..layout
        },
        TexturePixelLayout {
            size: (1, 0),
            ..layout
        },
        TexturePixelLayout {
            size: (1, 2),
            bytes_per_row: usize::MAX,
            ..layout
        },
        TexturePixelLayout {
            size: (u32::MAX, 1),
            bytes_per_row: usize::MAX,
            ..layout
        },
    ] {
        assert!(matches!(
            TextureUpload::new(TextureStorageFormat::Rgba16Float, invalid),
            Err(PixelBufferError::InvalidSize)
        ));
    }
    assert!(matches!(
        TextureUpload::new(
            TextureStorageFormat::Rgba8Srgb,
            TexturePixelLayout {
                bytes_per_row: 3,
                ..layout
            },
        ),
        Err(PixelBufferError::InvalidStride {
            required: 4,
            provided: 3
        })
    ));
}
