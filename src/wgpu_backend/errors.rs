use super::readback::ReadbackError;
use super::texture_manager::{TextureManagerError, TextureTransferError};
use super::types::GeometryBufferError;
use crate::render_backend::render_target::RenderTargetError;
use crate::render_backend::texture_manager::TextureStreamError;
use crate::render_backend::PixelBufferError;
use naga::front::wgsl::ParseError;
use naga::valid::ValidationError;
use naga::WithSpan;

/// Why an effect shader was rejected.
#[derive(Debug, Clone, thiserror::Error)]
pub enum EffectShaderError {
    #[error("WGSL parsing failed: {0}")]
    Parse(#[source] Box<ParseError>),
    #[error("WGSL validation failed: {0}")]
    Validation(#[source] Box<WithSpan<ValidationError>>),
    #[error("Missing @fragment entry point effect_main")]
    MissingFragmentEntryPoint,
    #[error(
        "Unsupported resource at @group({group}) @binding({binding}); effect parameters must be a uniform at @group(1) @binding(0)"
    )]
    UnsupportedBinding { group: u32, binding: u32 },
    #[error("Only one effect parameter uniform may be declared")]
    DuplicateParameterBinding,
}

/// Errors from loading or attaching effects and updating their parameters.
#[derive(Debug, Clone, thiserror::Error)]
pub enum EffectResourceError {
    /// WGSL or the effect interface is invalid for the zero-based pass index.
    #[error("Invalid shader in effect pass {pass_index}: {reason}")]
    InvalidShader {
        pass_index: usize,
        #[source]
        reason: EffectShaderError,
    },
    /// The referenced effect_id has not been loaded.
    #[error("Effect {0} has not been loaded")]
    EffectNotLoaded(u64),
    /// Invalid effect parameters or configuration.
    #[error("Invalid effect parameters: {0}")]
    InvalidParams(String),
}

/// Why the window surface could not provide a texture for drawing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum SurfaceError {
    #[error("Surface acquisition timed out")]
    Timeout,
    #[error("Window is occluded")]
    Occluded,
    #[error("Surface configuration is outdated")]
    Outdated,
    #[error("Surface was lost and must be recreated")]
    Lost,
    #[error("Surface acquisition failed validation")]
    Validation,
}

/// Errors returned by WGPU backend operations.
#[derive(Debug, thiserror::Error)]
pub enum WgpuBackendError {
    /// The shared GPU device is unusable. Create a new context and rebuild its scenes and assets.
    #[error("GPU device lost")]
    DeviceLost,
    #[error(transparent)]
    Output(#[from] RenderTargetError),
    #[error(transparent)]
    Surface(#[from] SurfaceError),
    #[error(transparent)]
    Upload(#[from] GeometryBufferError),
    #[error(transparent)]
    Texture(#[from] TextureManagerError),
    #[error(transparent)]
    Effect(#[from] EffectResourceError),
    #[error(transparent)]
    Readback(#[from] ReadbackError),
}

impl From<TextureTransferError> for WgpuBackendError {
    fn from(error: TextureTransferError) -> Self {
        match error {
            TextureTransferError::DeviceLost => Self::DeviceLost,
            error => Self::Texture(TextureManagerError::Transfer(error)),
        }
    }
}

impl<SourceError> From<TextureManagerError> for TextureStreamError<WgpuBackendError, SourceError> {
    fn from(error: TextureManagerError) -> Self {
        Self::Backend(error.into())
    }
}

impl<SourceError> From<PixelBufferError> for TextureStreamError<WgpuBackendError, SourceError> {
    fn from(error: PixelBufferError) -> Self {
        TextureManagerError::from(error).into()
    }
}
