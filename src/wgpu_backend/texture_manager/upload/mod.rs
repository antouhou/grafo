use crate::core::color;
use crate::render_backend::texture_manager::{
    AlphaMode, RgbEncoding, TexturePixelFormat, TexturePixelLayout, TextureStorageFormat,
};
use crate::render_backend::PixelBufferError;
use half::f16;
use wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;

pub(super) mod stream;
pub(super) mod transfer;

pub(super) const TEXTURE_UPLOAD_BATCH_BYTE_LIMIT: usize = 1024 * 1024;

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

fn can_copy_source_bytes(storage: TextureStorageFormat, layout: TexturePixelLayout) -> bool {
    match storage {
        TextureStorageFormat::Rgba8Srgb => {
            layout.format == TexturePixelFormat::Rgba8
                && layout.encoding == RgbEncoding::Srgb
                && layout.alpha == AlphaMode::PremultipliedInLinearSpace
        }
        TextureStorageFormat::Rgba16Float => {
            cfg!(target_endian = "little")
                && layout.format == TexturePixelFormat::Rgba16Float
                && layout.encoding == RgbEncoding::LinearSrgb
                && matches!(
                    layout.alpha,
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

pub(super) struct TextureUploadBatch<'a> {
    pub(super) bytes: &'a [u8],
    pub(super) origin: (u32, u32),
    pub(super) size: (u32, u32),
    pub(super) bytes_per_row: u32,
}

pub(super) struct TextureUpload {
    storage: TextureStorageFormat,
    layout: TexturePixelLayout,
    can_copy_source_bytes: bool,
    source_row_bytes: usize,
    source_row_remaining: usize,
    source_padding_remaining: usize,
    source_bytes_required: usize,
    source_bytes_consumed: usize,
    pending_pixel: [u8; 16],
    pending_pixel_len: usize,
    bytes: Vec<u8>,
    bytes_per_row: u32,
    batch_row_limit: u32,
    batch_start_row: u32,
    batch_rows: u32,
    completed_rows: u32,
}

impl TextureUpload {
    pub(super) fn new(
        storage: TextureStorageFormat,
        layout: TexturePixelLayout,
    ) -> Result<Self, PixelBufferError> {
        let source_bytes_required = layout.required_byte_len()?;
        let (width, height) = layout.size;
        let source_row_bytes = width as usize * layout.format.bytes_per_pixel();
        let storage_pixel_bytes = match storage {
            TextureStorageFormat::Rgba8Srgb => 4,
            TextureStorageFormat::Rgba16Float => 8,
        };
        let bytes_per_row = width
            .checked_mul(storage_pixel_bytes)
            .and_then(|bytes| bytes.checked_add(COPY_BYTES_PER_ROW_ALIGNMENT - 1))
            .map(|bytes| bytes / COPY_BYTES_PER_ROW_ALIGNMENT * COPY_BYTES_PER_ROW_ALIGNMENT)
            .ok_or(PixelBufferError::InvalidSize)?;
        let batch_row_limit = (TEXTURE_UPLOAD_BATCH_BYTE_LIMIT / bytes_per_row as usize)
            .max(1)
            .min(height as usize) as u32;
        let batch_byte_capacity = (bytes_per_row as usize)
            .checked_mul(batch_row_limit as usize)
            .filter(|&length| length <= isize::MAX as usize)
            .ok_or(PixelBufferError::InvalidSize)?;
        Ok(Self {
            storage,
            layout,
            can_copy_source_bytes: can_copy_source_bytes(storage, layout),
            source_row_bytes,
            source_row_remaining: source_row_bytes,
            source_padding_remaining: 0,
            source_bytes_required,
            source_bytes_consumed: 0,
            pending_pixel: [0; 16],
            pending_pixel_len: 0,
            bytes: Vec::with_capacity(batch_byte_capacity),
            bytes_per_row,
            batch_row_limit,
            batch_start_row: 0,
            batch_rows: 0,
            completed_rows: 0,
        })
    }

    pub(super) fn is_complete(&self) -> bool {
        self.completed_rows == self.layout.size.1
    }

    pub(super) fn is_batch_ready(&self) -> bool {
        self.batch_rows != 0 && (self.batch_rows == self.batch_row_limit || self.is_complete())
    }

    fn append_converted_pixel(&mut self, rgba: [f32; 4]) {
        append_pixel(
            &mut self.bytes,
            self.storage,
            encode_premultiplied_channels(
                rgba,
                self.layout.encoding,
                self.layout.alpha,
                self.storage,
            ),
        );
    }

    fn convert_source_pixels(&mut self, mut source: &[u8]) {
        let source_pixel_bytes = self.layout.format.bytes_per_pixel();
        if self.pending_pixel_len != 0 {
            let count = (source_pixel_bytes - self.pending_pixel_len).min(source.len());
            self.pending_pixel[self.pending_pixel_len..self.pending_pixel_len + count]
                .copy_from_slice(&source[..count]);
            self.pending_pixel_len += count;
            source = &source[count..];
            if self.pending_pixel_len == source_pixel_bytes {
                let rgba = decode_pixel(
                    self.layout.format,
                    &self.pending_pixel[..source_pixel_bytes],
                );
                self.append_converted_pixel(rgba);
                self.pending_pixel_len = 0;
            }
        }

        let mut pixels = source.chunks_exact(source_pixel_bytes);
        for pixel in &mut pixels {
            let rgba = decode_pixel(self.layout.format, pixel);
            self.append_converted_pixel(rgba);
        }
        let remainder = pixels.remainder();
        self.pending_pixel[..remainder.len()].copy_from_slice(remainder);
        self.pending_pixel_len += remainder.len();
    }

    fn finish_row(&mut self) {
        debug_assert_eq!(self.pending_pixel_len, 0);
        self.batch_rows += 1;
        self.completed_rows += 1;
        self.bytes
            .resize(self.batch_rows as usize * self.bytes_per_row as usize, 0);
        self.source_row_remaining = self.source_row_bytes;
        if !self.is_complete() {
            self.source_padding_remaining = self.layout.bytes_per_row - self.source_row_bytes;
        }
    }

    /// Consumes input through the next complete batch or the final source pixel.
    pub(super) fn convert_into_batch(&mut self, input: &[u8]) -> usize {
        let mut consumed = 0;
        while consumed < input.len() && !self.is_batch_ready() && !self.is_complete() {
            let remaining = &input[consumed..];
            if self.source_padding_remaining != 0 {
                let count = self.source_padding_remaining.min(remaining.len());
                self.source_padding_remaining -= count;
                consumed += count;
                continue;
            }

            let count = self.source_row_remaining.min(remaining.len());
            let source = &remaining[..count];
            if self.can_copy_source_bytes {
                self.bytes.extend_from_slice(source);
            } else {
                self.convert_source_pixels(source);
            }
            self.source_row_remaining -= count;
            consumed += count;
            if self.source_row_remaining == 0 {
                self.finish_row();
            }
        }
        self.source_bytes_consumed += consumed;
        consumed
    }

    pub(super) fn batch(&self) -> TextureUploadBatch<'_> {
        debug_assert!(self.is_batch_ready());
        TextureUploadBatch {
            bytes: &self.bytes,
            origin: (0, self.batch_start_row),
            size: (self.layout.size.0, self.batch_rows),
            bytes_per_row: self.bytes_per_row,
        }
    }

    pub(super) fn clear_batch(&mut self) {
        debug_assert!(self.is_batch_ready());
        self.bytes.clear();
        self.batch_start_row = self.completed_rows;
        self.batch_rows = 0;
    }

    pub(super) fn short_input_error(&self) -> PixelBufferError {
        PixelBufferError::StorageTooSmall {
            required: self.source_bytes_required,
            provided: self.source_bytes_consumed,
        }
    }
}

#[cfg(test)]
mod tests;
