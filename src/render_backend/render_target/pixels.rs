use super::RenderTargetError;

/// RGB is premultiplied in linear space, then encoded as sRGB. Alpha is linear.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
    /// Four bytes per pixel in blue, green, red, alpha order.
    Bgra8,
    /// Four bytes per pixel in red, green, blue, alpha order.
    Rgba8,
    /// Native-endian `u32` words whose numeric value is `0xAARRGGBB`.
    Argb32,
}

/// Pixel dimensions, format, and row stride in bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PixelLayout {
    size: (u32, u32),
    format: PixelFormat,
    stride: usize,
    byte_len: usize,
}

impl PixelLayout {
    pub fn new(
        size: (u32, u32),
        format: PixelFormat,
        stride: usize,
    ) -> Result<Self, RenderTargetError> {
        if size.0 == 0 || size.1 == 0 {
            return Err(RenderTargetError::InvalidSize);
        }
        let row_bytes = (size.0 as usize)
            .checked_mul(4)
            .ok_or(RenderTargetError::InvalidSize)?;
        if stride < row_bytes {
            return Err(RenderTargetError::InvalidStride {
                required: row_bytes,
                provided: stride,
            });
        }
        let byte_len = stride
            .checked_mul(size.1 as usize - 1)
            .and_then(|offset| offset.checked_add(row_bytes))
            .filter(|length| *length <= isize::MAX as usize)
            .ok_or(RenderTargetError::InvalidSize)?;
        Ok(Self {
            size,
            format,
            stride,
            byte_len,
        })
    }

    pub fn tightly_packed(
        size: (u32, u32),
        format: PixelFormat,
    ) -> Result<Self, RenderTargetError> {
        let stride = (size.0 as usize)
            .checked_mul(4)
            .ok_or(RenderTargetError::InvalidSize)?;
        Self::new(size, format, stride)
    }

    pub fn size(&self) -> (u32, u32) {
        self.size
    }
    pub fn format(&self) -> PixelFormat {
        self.format
    }
    pub fn stride(&self) -> usize {
        self.stride
    }
    /// Minimum storage size; padding after the final row is optional.
    pub fn byte_len(&self) -> usize {
        self.byte_len
    }

    fn validate_storage(&self, length: usize) -> Result<(), RenderTargetError> {
        if length < self.byte_len {
            return Err(RenderTargetError::StorageTooSmall {
                required: self.byte_len,
                provided: length,
            });
        }
        Ok(())
    }
}

/// A mutable view of a pixel buffer. Rendering preserves row padding and trailing bytes.
pub struct PixmapMut<'pixels> {
    pixels: &'pixels mut [u8],
    layout: PixelLayout,
}

impl<'pixels> PixmapMut<'pixels> {
    pub fn new(pixels: &'pixels mut [u8], layout: PixelLayout) -> Result<Self, RenderTargetError> {
        layout.validate_storage(pixels.len())?;
        Ok(Self { pixels, layout })
    }

    pub fn bgra8(pixels: &'pixels mut [u8], size: (u32, u32)) -> Result<Self, RenderTargetError> {
        Self::new(
            pixels,
            PixelLayout::tightly_packed(size, PixelFormat::Bgra8)?,
        )
    }

    pub fn argb32(pixels: &'pixels mut [u32], size: (u32, u32)) -> Result<Self, RenderTargetError> {
        Self::new(
            bytemuck::cast_slice_mut(pixels),
            PixelLayout::tightly_packed(size, PixelFormat::Argb32)?,
        )
    }

    pub fn layout(&self) -> PixelLayout {
        self.layout
    }
    pub fn pixels(&self) -> &[u8] {
        self.pixels
    }
    pub fn pixels_mut(&mut self) -> &mut [u8] {
        self.pixels
    }
    pub fn as_mut(&mut self) -> PixmapMut<'_> {
        PixmapMut {
            pixels: self.pixels,
            layout: self.layout,
        }
    }
}

/// A pixel buffer with owned, resizable storage.
pub struct Pixmap {
    pixels: Vec<u8>,
    layout: PixelLayout,
}

impl Pixmap {
    pub fn new(size: (u32, u32), format: PixelFormat) -> Result<Self, RenderTargetError> {
        let layout = PixelLayout::tightly_packed(size, format)?;
        Ok(Self {
            pixels: vec![0; layout.byte_len()],
            layout,
        })
    }

    /// Takes ownership of the buffer, including any row padding or trailing bytes.
    pub fn from_vec(pixels: Vec<u8>, layout: PixelLayout) -> Result<Self, RenderTargetError> {
        layout.validate_storage(pixels.len())?;
        Ok(Self { pixels, layout })
    }

    /// Resizes the buffer to tightly packed rows, reusing capacity when possible.
    /// Truncates excess bytes or fills new bytes with zero. Does not resample the image.
    pub fn resize(&mut self, size: (u32, u32)) -> Result<(), RenderTargetError> {
        let layout = PixelLayout::tightly_packed(size, self.layout.format())?;
        self.pixels.resize(layout.byte_len(), 0);
        self.layout = layout;
        Ok(())
    }

    pub fn layout(&self) -> PixelLayout {
        self.layout
    }
    pub fn pixels(&self) -> &[u8] {
        &self.pixels
    }
    pub fn pixels_mut(&mut self) -> &mut [u8] {
        &mut self.pixels
    }
    pub fn into_pixels(self) -> Vec<u8> {
        self.pixels
    }
    pub fn as_mut(&mut self) -> PixmapMut<'_> {
        PixmapMut {
            pixels: &mut self.pixels,
            layout: self.layout,
        }
    }
}
