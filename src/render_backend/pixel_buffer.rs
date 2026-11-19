use thiserror::Error;

/// Errors in the dimensions, byte stride, or storage of a CPU pixel buffer.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum PixelBufferError {
    #[error("Pixel dimensions must be nonzero and fit in addressable memory")]
    InvalidSize,
    #[error("Row stride {provided} is smaller than the required {required} bytes")]
    InvalidStride { required: usize, provided: usize },
    #[error("Pixel storage needs {required} bytes, but has {provided}")]
    StorageTooSmall { required: usize, provided: usize },
}

/// Minimum length through the final pixel. The last row needs no trailing padding.
pub(super) fn required_byte_len(
    size: (u32, u32),
    bytes_per_pixel: usize,
    bytes_per_row: usize,
) -> Result<usize, PixelBufferError> {
    let (width, height) = size;
    if width == 0 || height == 0 {
        return Err(PixelBufferError::InvalidSize);
    }
    let row_bytes = (width as usize)
        .checked_mul(bytes_per_pixel)
        .ok_or(PixelBufferError::InvalidSize)?;
    if bytes_per_row < row_bytes {
        return Err(PixelBufferError::InvalidStride {
            required: row_bytes,
            provided: bytes_per_row,
        });
    }
    bytes_per_row
        .checked_mul(height as usize - 1)
        .and_then(|offset| offset.checked_add(row_bytes))
        .filter(|length| *length <= isize::MAX as usize)
        .ok_or(PixelBufferError::InvalidSize)
}

pub(super) fn validate_storage(required: usize, provided: usize) -> Result<(), PixelBufferError> {
    if provided < required {
        return Err(PixelBufferError::StorageTooSmall { required, provided });
    }
    Ok(())
}
