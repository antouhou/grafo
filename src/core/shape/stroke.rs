use super::compute_vertex_bounds;
use crate::core::cache::CachedTessellation;
use crate::core::util::ShapeResources;
use crate::core::vertex::CustomVertex;
use lyon::math::Point;
use lyon::path::polygon::Polygon;
use lyon::path::{LineCap as LyonLineCap, LineJoin as LyonLineJoin, PathEvent};
use lyon::tessellation::{
    BuffersBuilder, GeometryBuilderError, StrokeOptions as LyonStrokeOptions, StrokeVertex,
    StrokeVertexConstructor, TessellationError,
};
use std::sync::Arc;
use thiserror::Error;

/// The shape of each end of an open path
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

/// Why stroke settings cannot be used to draw a shape.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum StrokeOptionsError {
    /// The width is negative or not finite.
    #[error("width must be finite and nonnegative")]
    InvalidWidth,
    /// The miter limit is below `1.0` or not finite.
    #[error("miter limit must be finite and at least 1")]
    InvalidMiterLimit,
}

/// Width, ends, and corners of a centered stroke.
///
/// Width is measured in the shape's local logical coordinates and scales with its transform.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StrokeOptions {
    /// Stroke width. Must be finite and nonnegative. Zero draws nothing. Defaults to `1.0`.
    pub width: f32,
    /// Style for both ends of each open path. Defaults to [`LineCap::Butt`].
    /// Closed paths join at their starting point and ignore caps, unless they collapse
    /// to a point or a single segment.
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

    /// Sets the stroke's cap style.
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

    pub(crate) fn validate(self) -> Result<(), StrokeOptionsError> {
        if !self.width.is_finite() || self.width < 0.0 {
            return Err(StrokeOptionsError::InvalidWidth);
        }
        if !self.miter_limit.is_finite() || self.miter_limit < 1.0 {
            return Err(StrokeOptionsError::InvalidMiterLimit);
        }
        Ok(())
    }

    fn lyon_options(self) -> LyonStrokeOptions {
        let line_cap = match self.line_cap {
            LineCap::Butt => LyonLineCap::Butt,
            LineCap::Round => LyonLineCap::Round,
            LineCap::Square => LyonLineCap::Square,
        };
        let line_join = match self.line_join {
            LineJoin::Miter => LyonLineJoin::Miter,
            LineJoin::Bevel => LyonLineJoin::Bevel,
            LineJoin::Round => LyonLineJoin::Round,
        };
        let mut options = LyonStrokeOptions::default()
            .with_line_width(self.width)
            .with_line_cap(line_cap)
            .with_line_join(line_join);
        // Lyon measures the tip distance against the full width.
        options.miter_limit = self.miter_limit * 0.5;
        if self.line_cap == LineCap::Round || self.line_join == LineJoin::Round {
            // Keep the tolerance representable when Lyon subtracts it from an arc's radius.
            options.tolerance = options.tolerance.max(self.width * f32::EPSILON);
        }
        options
    }
}

fn has_finite_positions(event: PathEvent) -> bool {
    let is_finite = |point: Point| point.x.is_finite() && point.y.is_finite();
    match event {
        PathEvent::Begin { at } => is_finite(at),
        PathEvent::Line { from, to }
        | PathEvent::End {
            last: from,
            first: to,
            ..
        } => is_finite(from) && is_finite(to),
        PathEvent::Quadratic { from, ctrl, to } => {
            is_finite(from) && is_finite(ctrl) && is_finite(to)
        }
        PathEvent::Cubic {
            from,
            ctrl1,
            ctrl2,
            to,
        } => is_finite(from) && is_finite(ctrl1) && is_finite(ctrl2) && is_finite(to),
    }
}

pub(super) fn validate_positions(
    mut events: impl Iterator<Item = PathEvent>,
) -> Result<(), TessellationError> {
    if events.all(has_finite_positions) {
        Ok(())
    } else {
        Err(GeometryBuilderError::InvalidVertex.into())
    }
}

struct StrokeVertexConverter;

impl StrokeVertexConstructor<CustomVertex> for StrokeVertexConverter {
    fn new_vertex(&mut self, vertex: StrokeVertex) -> CustomVertex {
        CustomVertex {
            position: vertex.position().to_array(),
            tex_coords: [0.0; 2],
        }
    }
}

pub(super) fn tessellate(
    events: impl Iterator<Item = PathEvent> + Clone,
    options: StrokeOptions,
    resources: &mut ShapeResources,
) -> Result<Arc<CachedTessellation>, TessellationError> {
    validate_positions(events.clone())?;

    let mut tessellation = resources.tessellation_cache.acquire_tessellation();
    if options.width != 0.0 {
        if let Err(error) = resources.stroke_tessellator.tessellate(
            events,
            &options.lyon_options(),
            &mut BuffersBuilder::new(&mut tessellation.vertex_buffers, StrokeVertexConverter),
        ) {
            resources
                .tessellation_cache
                .recycle_tessellation(tessellation);
            return Err(error);
        }
    }
    if !tessellation
        .vertex_buffers
        .vertices
        .iter()
        .all(|vertex| vertex.position.into_iter().all(f32::is_finite))
    {
        resources
            .tessellation_cache
            .recycle_tessellation(tessellation);
        return Err(GeometryBuilderError::InvalidVertex.into());
    }
    tessellation.local_bounds = compute_vertex_bounds(&tessellation.vertex_buffers.vertices);
    Ok(Arc::new(tessellation))
}

pub(super) fn tessellate_rect(
    bounds: [(f32, f32); 2],
    options: StrokeOptions,
    resources: &mut ShapeResources,
) -> Result<Arc<CachedTessellation>, TessellationError> {
    let points: [Point; 4] = [
        bounds[0].into(),
        (bounds[1].0, bounds[0].1).into(),
        bounds[1].into(),
        (bounds[0].0, bounds[1].1).into(),
    ];
    let rectangle = Polygon {
        points: &points,
        closed: true,
    };
    tessellate(rectangle.path_events(), options, resources)
}
