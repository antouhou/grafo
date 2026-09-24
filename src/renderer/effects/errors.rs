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
pub enum EffectError {
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
    /// The referenced node_id does not exist in the draw tree.
    #[error("Node {0} not found in draw tree")]
    NodeNotFound(usize),
    /// Parameter data does not match the existing uniform buffer size.
    #[error(
        "Effect {effect_id} expects {expected_size} parameter bytes but {actual_size} were provided"
    )]
    ParameterSizeMismatch {
        effect_id: u64,
        expected_size: u64,
        actual_size: u64,
    },
    /// Invalid effect parameters or configuration.
    #[error("Invalid effect parameters: {0}")]
    InvalidParams(String),
}
