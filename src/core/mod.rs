//! CPU geometry, colors, fills, effects, and transforms.

pub use blend::BlendMode;
pub use cache::CachedTessellation;
pub use color::Color;
pub use effect::{
    BackdropCaptureArea, BackdropEffectConfig, ContentEffectConfig, ShapeMaskEffectConfig,
};
pub use geometry::{
    MathRect, PhysicalRect, Size, UnsignedPhysicalPoint, UnsignedPhysicalRect, Viewport,
};
pub use gradient::errors::GradientError;
pub use gradient::types::{
    ColorInterpolation, ConicGradientDesc, Fill, Gradient, GradientCommonDesc, GradientDesc,
    GradientStop, GradientStopOffset, GradientStopPositions, GradientUnits, HueInterpolationMethod,
    LinearGradientDesc, LinearGradientLine, RadialGradientDesc, RadialGradientSize, SpreadMode,
};
pub use group::GroupDrawCommandOptions;
pub use shape::*;
pub use text::*;
pub use vertex::InstanceTransform as TransformInstance;
pub use vertex::{CustomVertex, TextureUvTransform};

mod blend;
pub(crate) mod cache;
pub mod color;
pub(crate) mod effect;
pub(crate) mod geometry;
pub mod gradient;
mod group;
pub(crate) mod math;
pub(crate) mod shape;
pub(crate) mod slot_pool;
mod text;
pub(crate) mod util;
pub(crate) mod vertex;
