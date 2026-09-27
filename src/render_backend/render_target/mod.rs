//! Surface and pixmap render targets.

pub use self::pixels::{PixelFormat, PixelLayout, Pixmap, PixmapMut};
use thiserror::Error;

mod pixels;

/// Errors from validating a render target.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum RenderTargetError {
    #[error("Render target dimensions must be nonzero and fit in addressable memory")]
    InvalidSize,
    #[error("Row stride {provided} is smaller than the required {required} bytes")]
    InvalidStride { required: usize, provided: usize },
    #[error("Pixel storage needs {required} bytes, but has {provided}")]
    StorageTooSmall { required: usize, provided: usize },
    #[error("Render target dimensions {size:?} exceed the backend limit of {maximum}")]
    SizeLimit { size: (u32, u32), maximum: u32 },
    #[error("The surface belongs to an incompatible backend device")]
    IncompatibleSurface,
}

/// A platform surface with its size and vsync settings.
/// `Resource` holds the backend's native surface resources.
pub struct Surface<Resource> {
    resource: Resource,
    size: (u32, u32),
    vsync: bool,
    needs_configuration: bool,
}

impl<Resource> Surface<Resource> {
    /// Creates a surface from backend resources. It needs configuration before rendering.
    pub fn from_resource(resource: Resource, size: (u32, u32), vsync: bool) -> Self {
        Self {
            resource,
            size,
            vsync,
            needs_configuration: true,
        }
    }

    pub fn size(&self) -> (u32, u32) {
        self.size
    }

    /// Applies the new size when this surface is next rendered.
    pub fn resize(&mut self, size: (u32, u32)) {
        if self.size != size {
            self.size = size;
            self.invalidate();
        }
    }

    pub fn vsync(&self) -> bool {
        self.vsync
    }

    pub fn set_vsync(&mut self, vsync: bool) {
        if self.vsync != vsync {
            self.vsync = vsync;
            self.invalidate();
        }
    }

    /// Requests configuration on the next render, for example after surface loss.
    pub fn invalidate(&mut self) {
        self.needs_configuration = true;
    }

    /// Whether the backend needs to configure the surface before rendering.
    pub fn needs_configuration(&self) -> bool {
        self.needs_configuration
    }

    /// Marks the surface as configured. Call after the backend applies its size and vsync settings.
    pub fn finish_configuration(&mut self) {
        self.needs_configuration = false;
    }

    pub fn resource(&self) -> &Resource {
        &self.resource
    }

    pub fn resource_mut(&mut self) -> &mut Resource {
        &mut self.resource
    }
}

/// A surface or pixel buffer to render into.
pub enum RenderTarget<'target, Resource> {
    Surface(&'target mut Surface<Resource>),
    Pixmap(PixmapMut<'target>),
}

impl<Resource> RenderTarget<'_, Resource> {
    pub fn size(&self) -> (u32, u32) {
        match self {
            Self::Surface(surface) => surface.size(),
            Self::Pixmap(pixels) => pixels.layout().size(),
        }
    }

    /// Returns the dimensions if both are nonzero and at most `maximum`.
    pub fn validate_size(&self, maximum: u32) -> Result<(u32, u32), RenderTargetError> {
        let size = self.size();
        if size.0 == 0 || size.1 == 0 {
            return Err(RenderTargetError::InvalidSize);
        }
        if size.0 > maximum || size.1 > maximum {
            return Err(RenderTargetError::SizeLimit { size, maximum });
        }
        Ok(size)
    }
}

impl<'target, Resource> From<&'target mut Surface<Resource>> for RenderTarget<'target, Resource> {
    fn from(surface: &'target mut Surface<Resource>) -> Self {
        Self::Surface(surface)
    }
}

impl<'target, Resource> From<PixmapMut<'target>> for RenderTarget<'target, Resource> {
    fn from(pixels: PixmapMut<'target>) -> Self {
        Self::Pixmap(pixels)
    }
}

impl<'target, Resource> From<&'target mut Pixmap> for RenderTarget<'target, Resource> {
    fn from(pixels: &'target mut Pixmap) -> Self {
        Self::Pixmap(pixels.as_mut())
    }
}

impl<'target, Resource> From<&'target mut PixmapMut<'_>> for RenderTarget<'target, Resource> {
    fn from(pixels: &'target mut PixmapMut<'_>) -> Self {
        Self::Pixmap(pixels.as_mut())
    }
}

#[cfg(test)]
mod tests;
