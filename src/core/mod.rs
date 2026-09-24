//! CPU geometry, colors, fills, effects, and transforms.

pub(crate) mod cache;
mod color;
pub(crate) mod effect;
mod geometry;
pub mod gradient;
pub(crate) mod shape;
mod stroke;
pub(crate) mod util;
pub(crate) mod vertex;

pub use color::{premultiply_rgba8_srgb_inplace, Color};
pub use effect::{BackdropCaptureArea, BackdropEffectConfig, ShapeEffectConfig};
pub use geometry::{MathRect, PhysicalRect, Size, UnsignedPhysicalPoint, UnsignedPhysicalRect};
pub use gradient::errors::GradientError;
pub use gradient::types::{
    ColorInterpolation, ConicGradientDesc, Fill, Gradient, GradientColor, GradientCommonDesc,
    GradientDesc, GradientStop, GradientStopOffset, GradientStopPositions, GradientUnits,
    HueComponent, HueInterpolationMethod, LinearGradientDesc, LinearGradientLine,
    RadialGradientDesc, RadialGradientSize, SpreadMode,
};
pub use shape::*;
pub use stroke::Stroke;
pub use vertex::InstanceTransform as TransformInstance;
