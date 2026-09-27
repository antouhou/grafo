use super::copy_readback_rows;
use crate::render_backend::render_target::{PixelFormat, PixelLayout, PixmapMut};
use wgpu::TextureFormat;

#[test]
fn readback_converts_formats_and_preserves_destination_padding() {
    for source_format in [TextureFormat::Bgra8UnormSrgb, TextureFormat::Rgba8UnormSrgb] {
        let pixel = if source_format == TextureFormat::Bgra8UnormSrgb {
            [30, 20, 10, 40]
        } else {
            [10, 20, 30, 40]
        };
        let mut source = vec![71; 24];
        source[..4].copy_from_slice(&pixel);
        source[12..16].copy_from_slice(&pixel);
        for format in [PixelFormat::Bgra8, PixelFormat::Rgba8, PixelFormat::Argb32] {
            let layout = PixelLayout::new((1, 2), format, 8).unwrap();
            let mut destination = [99; 20];
            let mut surface = PixmapMut::new(&mut destination, layout).unwrap();
            copy_readback_rows(&source, 12, source_format, &mut surface).unwrap();
            let expected = match format {
                PixelFormat::Bgra8 => [30, 20, 10, 40],
                PixelFormat::Rgba8 => [10, 20, 30, 40],
                PixelFormat::Argb32 => 0x280a141e_u32.to_ne_bytes(),
            };
            assert_eq!(destination[..4], expected);
            assert_eq!(destination[8..12], expected);
            assert_eq!(destination[4..8], [99; 4]);
            assert_eq!(destination[12..], [99; 8]);
        }
    }
}

#[test]
fn readback_handles_tight_rows_and_native_argb_words() {
    let source = [153, 102, 51, 255, 0, 0, 0, 0];
    let mut destination = [0_u32; 3];
    let mut surface = PixmapMut::argb32(&mut destination, (2, 1)).unwrap();
    copy_readback_rows(&source, 8, TextureFormat::Bgra8UnormSrgb, &mut surface).unwrap();
    assert_eq!(destination, [0xff336699, 0, 0]);
}

#[test]
fn linear_readback_is_encoded_as_srgb_without_changing_alpha() {
    let mut destination = [0; 4];
    let layout = PixelLayout::tightly_packed((1, 1), PixelFormat::Rgba8).unwrap();
    let mut surface = PixmapMut::new(&mut destination, layout).unwrap();
    copy_readback_rows(
        &[0, 128, 255, 64],
        4,
        TextureFormat::Rgba8Unorm,
        &mut surface,
    )
    .unwrap();
    assert_eq!(destination, [0, 188, 255, 64]);
}

#[test]
fn unsupported_source_format_leaves_output_unchanged() {
    let mut destination = [99; 4];
    let mut surface = PixmapMut::bgra8(&mut destination, (1, 1)).unwrap();
    assert!(copy_readback_rows(&[0; 8], 8, TextureFormat::Rgba16Float, &mut surface).is_err());
    assert_eq!(destination, [99; 4]);
}
