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
    #[error("Could not calculate shape effect bounds for node {0}")]
    InvalidShapeEffectBounds(usize),
    #[error("Invalid effect parameters: {0}")]
    InvalidParams(String),
}
