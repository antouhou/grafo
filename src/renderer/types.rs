use crate::backend::WgpuBackendError;
use crate::scene::SceneError;

/// Scene insertion or backend resource preparation failed.
#[derive(Debug, thiserror::Error)]
pub enum DrawCommandError<E = WgpuBackendError> {
    #[error(transparent)]
    Scene(#[from] SceneError),
    #[error("Backend upload failed: {0}")]
    Backend(#[source] E),
}

/// Scene attachment or backend effect validation failed.
#[derive(Debug, thiserror::Error)]
pub enum EffectError<E = WgpuBackendError> {
    #[error(transparent)]
    Scene(#[from] SceneError),
    #[error("Backend effect failed: {0}")]
    Backend(#[source] E),
}
