/// The shape of each end of an open path.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum LineCap {
    /// Ends at the path endpoint without extending beyond it.
    #[default]
    Butt,
    /// Adds a semicircle with a radius of half the stroke width.
    Round,
    /// Extends beyond the endpoint by half the stroke width.
    Square,
}

/// How a stroke connects two path segments at a corner.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum LineJoin {
    /// Extends the edges to meet, using a bevel when the miter limit is exceeded.
    #[default]
    Miter,
    /// Connects the outer edges with a straight line.
    Bevel,
    /// Connects the outer edges with a circular arc.
    Round,
}

/// Width, ends, and corners of a centered stroke.
///
/// Width is measured in the shape's local logical coordinates and scales with its transform.
/// Set the stroke's color, gradient, or textures through [`super::ShapeDrawCommandOptions`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StrokeOptions {
    /// Stroke width. Must be finite and nonnegative. Defaults to `1.0`.
    pub width: f32,
    /// Style for both ends of each open path. Defaults to [`LineCap::Butt`].
    /// Closed paths join at their starting point and do not use caps.
    pub line_cap: LineCap,
    /// Corner style. Defaults to [`LineJoin::Miter`].
    pub line_join: LineJoin,
    /// Maximum ratio of the distance from a corner to its outer tip to half the stroke width.
    /// Longer miter joins use a bevel. Must be finite and at least `1.0`; defaults to `4.0`.
    /// Does not affect bevel or round joins.
    pub miter_limit: f32,
}

impl Default for StrokeOptions {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl StrokeOptions {
    /// Uses `width`, butt caps, miter joins, and a miter limit of `4.0`.
    pub fn new(width: f32) -> Self {
        Self {
            width,
            line_cap: LineCap::Butt,
            line_join: LineJoin::Miter,
            miter_limit: 4.0,
        }
    }

    /// Sets the style for both ends of each open path.
    pub fn line_cap(mut self, line_cap: LineCap) -> Self {
        self.line_cap = line_cap;
        self
    }

    /// Sets how the stroke connects segments at corners.
    pub fn line_join(mut self, line_join: LineJoin) -> Self {
        self.line_join = line_join;
        self
    }

    /// Sets the limit beyond which miter joins use a bevel.
    pub fn miter_limit(mut self, miter_limit: f32) -> Self {
        self.miter_limit = miter_limit;
        self
    }
}
