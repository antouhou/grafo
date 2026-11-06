use crate::core::shape::ShapeTessellationError;
use crate::core::StrokeOptionsError;
use lyon::tessellation::TessellationError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SceneError {
    #[error("Path tessellation failed: {0}")]
    Tessellation(#[from] TessellationError),
    #[error("Invalid stroke options: {0}")]
    InvalidStrokeOptions(#[from] StrokeOptionsError),
    #[error("Stroke produces {triangle_count} triangles; the limit is {maximum_triangles}")]
    StrokeTooComplex {
        triangle_count: usize,
        maximum_triangles: usize,
    },
    #[error("Invalid shape ID: {0}")]
    InvalidShapeId(usize),
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
            ShapeTessellationError::StrokeTooComplex {
                triangle_count,
                maximum_triangles,
            } => Self::StrokeTooComplex {
                triangle_count,
                maximum_triangles,
            },
        }
    }
}
