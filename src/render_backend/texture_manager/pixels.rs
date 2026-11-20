use crate::render_backend::{pixel_buffer, PixelBufferError};

/// Channel order and sample type in an interleaved pixel buffer.
/// Multibyte samples use native byte order. Integer channels are unsigned and
/// normalized to [0, 1]. Floating-point channels use IEEE binary16 or binary32.
/// Floating-point RGB may be negative or greater than one. Supply finite channels
/// and alpha in [0, 1].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TexturePixelFormat {
    Rgb8,
    Rgba8,
    Bgra8,
    /// Native-endian words with numeric value `0xAARRGGBB`.
    Argb32,
    Rgb16Unorm,
    Rgba16Unorm,
    Rgb16Float,
    Rgba16Float,
    Rgb32Float,
    Rgba32Float,
}

impl TexturePixelFormat {
    /// Number of bytes occupied by one pixel.
    pub const fn bytes_per_pixel(self) -> usize {
        match self {
            Self::Rgb8 => 3,
            Self::Rgba8 | Self::Bgra8 | Self::Argb32 => 4,
            Self::Rgb16Unorm | Self::Rgb16Float => 6,
            Self::Rgba16Unorm | Self::Rgba16Float => 8,
            Self::Rgb32Float => 12,
            Self::Rgba32Float => 16,
        }
    }
}

/// Transfer function for RGB channels with sRGB primaries. Alpha is linear.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RgbEncoding {
    Srgb,
    LinearSrgb,
}

/// How alpha is represented in the source pixels. RGB-only formats are opaque.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AlphaMode {
    /// Every pixel has full opacity. Ignore any stored alpha channel.
    Opaque,
    /// RGB is independent of alpha.
    Straight,
    /// RGB stores the encoding of `linear_rgb * alpha`.
    PremultipliedInLinearSpace,
    /// RGB stores `encoded_rgb * alpha`.
    PremultipliedInEncodedSpace,
}

/// Borrowed rows of decoded pixels, starting at the top-left pixel.
/// The buffer may end immediately after the last pixel of the final row.
#[derive(Clone, Copy, Debug)]
pub struct TexturePixels<'a> {
    pub bytes: &'a [u8],
    /// Width and height in pixels.
    pub size: (u32, u32),
    /// Distance between the starts of consecutive rows in bytes.
    pub bytes_per_row: usize,
    pub format: TexturePixelFormat,
    pub encoding: RgbEncoding,
    pub alpha: AlphaMode,
}

impl<'a> TexturePixels<'a> {
    /// Describes tightly packed RGBA8 pixels with sRGB-encoded color channels.
    pub const fn rgba8_srgb(bytes: &'a [u8], size: (u32, u32), alpha: AlphaMode) -> Self {
        Self {
            bytes,
            size,
            bytes_per_row: (size.0 as usize)
                .saturating_mul(TexturePixelFormat::Rgba8.bytes_per_pixel()),
            format: TexturePixelFormat::Rgba8,
            encoding: RgbEncoding::Srgb,
            alpha,
        }
    }

    /// Checks dimensions, stride, and buffer length. Returns the bytes through the final pixel.
    pub fn validated_bytes(&self) -> Result<&'a [u8], PixelBufferError> {
        let required = pixel_buffer::required_byte_len(
            self.size,
            self.format.bytes_per_pixel(),
            self.bytes_per_row,
        )?;
        pixel_buffer::validate_storage(required, self.bytes.len())?;
        Ok(&self.bytes[..required])
    }
}

/// Color storage requested when allocating a texture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextureStorageFormat {
    /// Four 8-bit normalized channels with sRGB-encoded RGB and linear alpha.
    Rgba8Srgb,
    /// Four linear half-float channels in RGBA order.
    Rgba16Float,
}
