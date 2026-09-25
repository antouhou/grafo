#[derive(Debug, thiserror::Error)]
pub enum SceneError {
    #[error("Invalid shape ID: {0}")]
    InvalidShapeId(usize),
    #[error("Shape {0} is not loaded")]
    ShapeNotLoaded(u64),
    #[error("Clip rect node only supports axis-aligned transforms")]
    UnsupportedClipRectTransform,
    #[error("Clip rect node {0} does not support {1}")]
    UnsupportedClipRectOperation(usize, &'static str),
    #[error("Node {0} was not found")]
    NodeNotFound(usize),
    #[error("Effect {effect_id} expects {expected_size} parameter bytes, got {actual_size}")]
    ParameterSizeMismatch {
        effect_id: u64,
        expected_size: u64,
        actual_size: u64,
    },
    #[error("Invalid effect parameters: {0}")]
    InvalidParams(String),
}
