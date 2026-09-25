use super::readback::ReadbackError;
use super::types::GeometryBufferError;
use naga::front::wgsl::ParseError;
use naga::valid::ValidationError;
use naga::WithSpan;
use wgpu::SurfaceError;

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

/// Errors returned by WGPU backend operations.
#[derive(Debug, thiserror::Error)]
pub enum WgpuBackendError {
    #[error(transparent)]
    Surface(#[from] SurfaceError),
    #[error(transparent)]
    Upload(#[from] GeometryBufferError),
    #[error(transparent)]
    Effect(#[from] EffectResourceError),
    #[error(transparent)]
    Readback(#[from] ReadbackError),
}
