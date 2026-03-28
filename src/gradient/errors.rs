use thiserror::Error;

#[derive(Debug, Error)]
pub enum GradientError {
    #[error("gradient has no stops")]
    EmptyStops,

    #[error("reversed double-position stop at index {stop_index}: first={first}, second={second}")]
    ReversedDoublePositionStop {
        stop_index: usize,
        first: f32,
        second: f32,
    },

    #[error("stop offset at index {stop_index} uses wrong scalar kind for this gradient type")]
    InvalidStopOffsetKind { stop_index: usize },

    #[error("hint offset at index {stop_index} uses wrong scalar kind for this gradient type")]
    InvalidHintOffsetKind { stop_index: usize },

    #[error("invalid radial gradient definition (shape/size mismatch)")]
    InvalidRadialDefinition,

    #[error("invalid conic gradient definition")]
    InvalidConicDefinition,

    #[error("non-finite stop offset at index {stop_index}")]
    NonFiniteStopOffset { stop_index: usize },

    #[error("non-finite hint at index {stop_index}")]
    NonFiniteHint { stop_index: usize },

    #[error("non-finite color component '{component}' at stop index {stop_index}")]
    NonFiniteColorComponent {
        stop_index: usize,
        component: &'static str,
    },

    #[error("non-finite angle parameter '{field}'")]
    NonFiniteAngle { field: &'static str },

    #[error("non-finite geometry parameter '{field}'")]
    NonFiniteGeometryParameter { field: &'static str },
}
