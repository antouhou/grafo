use crate::core::color;
use crate::render_backend::texture_manager::{
    AlphaMode, RgbEncoding, TexturePixelFormat, TexturePixels, TextureStorageFormat,
};
use crate::render_backend::PixelBufferError;
use half::f16;
use std::borrow::Cow;
use wgpu::{Extent3d, Origin3d, Queue, TexelCopyBufferLayout, Texture};

fn read_channels<const N: usize>(pixel: &[u8], decode: impl Fn([u8; N]) -> f32) -> [f32; 4] {
    let mut rgba = [0.0, 0.0, 0.0, 1.0];
    for (channel, sample) in rgba.iter_mut().zip(pixel.as_chunks::<N>().0) {
        *channel = decode(*sample);
    }
    rgba
}

fn decode_pixel(format: TexturePixelFormat, pixel: &[u8]) -> [f32; 4] {
    let unorm8 = |[byte]: [u8; 1]| f32::from(byte) / 255.0;
    match format {
        TexturePixelFormat::Rgb8 | TexturePixelFormat::Rgba8 => read_channels(pixel, unorm8),
        TexturePixelFormat::Bgra8 => {
            let [blue, green, red, alpha] = read_channels(pixel, unorm8);
            [red, green, blue, alpha]
        }
        TexturePixelFormat::Argb32 => {
            let word = u32::from_ne_bytes([pixel[0], pixel[1], pixel[2], pixel[3]]);
            let [alpha, red, green, blue] = word.to_be_bytes();
            [red, green, blue, alpha].map(|byte| unorm8([byte]))
        }
        TexturePixelFormat::Rgb16Unorm | TexturePixelFormat::Rgba16Unorm => {
            read_channels(pixel, |bytes| {
                f32::from(u16::from_ne_bytes(bytes)) / 65535.0
            })
        }
        TexturePixelFormat::Rgb16Float | TexturePixelFormat::Rgba16Float => {
            read_channels(pixel, |bytes| f16::from_ne_bytes(bytes).to_f32())
        }
        TexturePixelFormat::Rgb32Float | TexturePixelFormat::Rgba32Float => {
            read_channels(pixel, f32::from_ne_bytes)
        }
    }
}

fn encode_premultiplied_channels(
    [red, green, blue, source_alpha]: [f32; 4],
    encoding: RgbEncoding,
    alpha_mode: AlphaMode,
    storage: TextureStorageFormat,
) -> [f32; 4] {
    let rgb = [red, green, blue];
    let alpha = if alpha_mode == AlphaMode::Opaque {
        1.0
    } else {
        source_alpha
    };
    // Already-encoded premultiplied RGB can pass through a channel reorder unchanged.
    if encoding == RgbEncoding::Srgb
        && storage == TextureStorageFormat::Rgba8Srgb
        && (alpha == 1.0 || alpha_mode == AlphaMode::PremultipliedInLinearSpace)
    {
        return [red, green, blue, alpha];
    }
    let linear = match (encoding, alpha_mode) {
        (RgbEncoding::LinearSrgb, AlphaMode::Straight) => rgb.map(|channel| channel * alpha),
        (RgbEncoding::LinearSrgb, _) => rgb,
        (RgbEncoding::Srgb, AlphaMode::Straight | AlphaMode::PremultipliedInEncodedSpace) => {
            if alpha == 0.0 {
                return [0.0; 4];
            }
            let alpha = f64::from(alpha);
            rgb.map(|channel| {
                let mut encoded = f64::from(channel);
                if alpha_mode == AlphaMode::PremultipliedInEncodedSpace {
                    encoded /= alpha;
                }
                (color::srgb_to_linear_f64(encoded) * alpha) as f32
            })
        }
        (RgbEncoding::Srgb, _) => rgb.map(color::srgb_to_linear),
    };
    let [red, green, blue] = match storage {
        TextureStorageFormat::Rgba8Srgb => {
            linear.map(|channel| color::linear_to_srgb(channel.clamp(0.0, 1.0)))
        }
        TextureStorageFormat::Rgba16Float => linear,
    };
    [red, green, blue, alpha]
}

fn can_copy_source_bytes(storage: TextureStorageFormat, pixels: TexturePixels<'_>) -> bool {
    match storage {
        TextureStorageFormat::Rgba8Srgb => {
            pixels.format == TexturePixelFormat::Rgba8
                && pixels.encoding == RgbEncoding::Srgb
                && pixels.alpha == AlphaMode::PremultipliedInLinearSpace
        }
        TextureStorageFormat::Rgba16Float => {
            cfg!(target_endian = "little")
                && pixels.format == TexturePixelFormat::Rgba16Float
                && pixels.encoding == RgbEncoding::LinearSrgb
                && matches!(
                    pixels.alpha,
                    AlphaMode::PremultipliedInEncodedSpace | AlphaMode::PremultipliedInLinearSpace
                )
        }
    }
}

fn append_pixel(bytes: &mut Vec<u8>, storage: TextureStorageFormat, rgba: [f32; 4]) {
    match storage {
        TextureStorageFormat::Rgba8Srgb => {
            bytes.extend_from_slice(&rgba.map(color::normalized_channel_to_u8));
        }
        TextureStorageFormat::Rgba16Float => {
            for channel in rgba {
                let value = channel.clamp(f16::MIN.to_f32(), f16::MAX.to_f32());
                bytes.extend_from_slice(&f16::from_f32(value).to_le_bytes());
            }
        }
    }
}

pub(super) struct TextureUpload<'a> {
    bytes: Cow<'a, [u8]>,
    bytes_per_row: u32,
    size: (u32, u32),
}

impl<'a> TextureUpload<'a> {
    pub(super) fn new(
        storage: TextureStorageFormat,
        pixels: TexturePixels<'a>,
    ) -> Result<Self, PixelBufferError> {
        let source = pixels.validated_bytes()?;
        let (width, height) = pixels.size;
        let source_pixel_bytes = pixels.format.bytes_per_pixel();
        let source_row_bytes = width as usize * source_pixel_bytes;
        let stride = if height == 1 {
            source_row_bytes
        } else {
            pixels.bytes_per_row
        };
        let can_copy = can_copy_source_bytes(storage, pixels);
        if can_copy {
            if let Ok(bytes_per_row) = u32::try_from(stride) {
                return Ok(Self {
                    bytes: Cow::Borrowed(source),
                    bytes_per_row,
                    size: pixels.size,
                });
            }
        }

        let storage_pixel_bytes = match storage {
            TextureStorageFormat::Rgba8Srgb => 4,
            TextureStorageFormat::Rgba16Float => 8,
        };
        let bytes_per_row = width
            .checked_mul(storage_pixel_bytes)
            .ok_or(PixelBufferError::InvalidSize)?;
        let byte_len = (bytes_per_row as usize)
            .checked_mul(height as usize)
            .filter(|&len| len <= isize::MAX as usize)
            .ok_or(PixelBufferError::InvalidSize)?;
        let mut bytes = Vec::with_capacity(byte_len);
        for row in source.chunks(stride) {
            let row = &row[..source_row_bytes];
            if can_copy {
                bytes.extend_from_slice(row);
                continue;
            }
            for pixel in row.chunks_exact(source_pixel_bytes) {
                let rgba = decode_pixel(pixels.format, pixel);
                append_pixel(
                    &mut bytes,
                    storage,
                    encode_premultiplied_channels(rgba, pixels.encoding, pixels.alpha, storage),
                );
            }
        }
        Ok(Self {
            bytes: Cow::Owned(bytes),
            bytes_per_row,
            size: pixels.size,
        })
    }

    pub(super) fn write(&self, queue: &Queue, texture: &Texture, origin: (u32, u32)) {
        let mut destination = texture.as_image_copy();
        destination.origin = Origin3d {
            x: origin.0,
            y: origin.1,
            z: 0,
        };
        queue.write_texture(
            destination,
            &self.bytes,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(self.bytes_per_row),
                rows_per_image: None,
            },
            Extent3d {
                width: self.size.0,
                height: self.size.1,
                depth_or_array_layers: 1,
            },
        );
    }
}

#[cfg(test)]
mod tests;
