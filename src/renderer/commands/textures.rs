/// Identifies an execution-owned texture, separately from registered source texture IDs.
/// Cached textures retain their IDs; transient references end after submission.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct IntermediateTextureId(pub(crate) u64);
