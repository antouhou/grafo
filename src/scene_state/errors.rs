use crate::core::shape::{ShapeTessellationError, StrokeGeometryError};
use lyon::tessellation::TessellationError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SceneError {
    #[error("Path tessellation failed: {0}")]
    Tessellation(#[from] TessellationError),
    #[error("Invalid stroke geometry: {0}")]
    InvalidStrokeGeometry(#[from] StrokeGeometryError),
    #[error("Invalid shape ID: {0}")]
    InvalidShapeId(usize),
    #[error("Stroke draw {0} can't have any nodes attached to it")]
    StrokeDrawMustBeLeaf(usize),
    #[error("Shape {0} is not loaded")]
    ShapeNotLoaded(u64),
    #[error("Clip rect node only supports axis-aligned transforms")]
    UnsupportedClipRectTransform,
    #[error("Group gradient paint transform must have a finite, invertible projection")]
    InvalidPaintTransform,
    #[error("Node {0} does not support {1}")]
    UnsupportedNodeOperation(usize, &'static str),
    #[error("Node {0} was not found")]
    NodeNotFound(usize),
    #[error("Could not calculate shape mask effect bounds for node {0}")]
    InvalidShapeMaskEffectBounds(usize),
    #[error("Invalid effect parameters: {0}")]
    InvalidParams(String),
}

impl From<ShapeTessellationError> for SceneError {
    fn from(error: ShapeTessellationError) -> Self {
        match error {
            ShapeTessellationError::Tessellation(error) => Self::Tessellation(error),
            ShapeTessellationError::StrokeGeometry(error) => Self::InvalidStrokeGeometry(error),
        }
    }
}
