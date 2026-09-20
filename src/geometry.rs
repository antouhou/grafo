//! Shared geometry types backed by Lyon and Euclid.

use lyon::geom::{self, Box2D, Point};
use lyon::math;

/// Rectangle with floating-point coordinates.
pub type MathRect = math::Box2D;

/// Physical pixel bounds that can extend into negative offscreen coordinates.
pub type PhysicalRect = Box2D<i32>;

/// Physical pixel bounds with nonnegative coordinates.
pub type UnsignedPhysicalRect = Box2D<u32>;

/// Position in physical pixels with nonnegative coordinates.
pub type UnsignedPhysicalPoint = Point<u32>;

/// Width and height in physical pixels.
pub type Size = geom::Size<u32>;
