use super::{finish_path_tessellation, ShapeTessellationError, VertexConverter};
use crate::core::cache::CachedTessellation;
use crate::core::util::ShapeResources;
use crate::core::vertex::CustomVertex;
use lyon::math::Point;
use lyon::path::polygon::Polygon;
use lyon::path::{LineCap as LyonLineCap, LineJoin as LyonLineJoin, PathEvent};
use lyon::tessellation::{
    BuffersBuilder, FillOptions, GeometryBuilderError, StrokeOptions as LyonStrokeOptions,
    StrokeVertex, StrokeVertexConstructor, TessellationError,
};
use std::sync::Arc;
use thiserror::Error;

// Each input triangle can add one to Lyon's i16 winding count.
const MAX_STROKE_TRIANGLES: usize = i16::MAX as usize;

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
/// Set the stroke's color, gradient, or textures through [`super::ShapeDrawCommandOptions`].
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

struct StrokeVertexConverter;

impl StrokeVertexConstructor<CustomVertex> for StrokeVertexConverter {
    fn new_vertex(&mut self, vertex: StrokeVertex) -> CustomVertex {
        CustomVertex {
            position: vertex.position().to_array(),
            tex_coords: [0.0; 2],
        }
    }
}

fn normalize_stroke_vertices(
    vertices: &mut [CustomVertex],
    width: f32,
) -> Result<f64, TessellationError> {
    if !vertices
        .iter()
        .all(|vertex| vertex.position.into_iter().all(f32::is_finite))
    {
        return Err(GeometryBuilderError::InvalidVertex.into());
    }
    if width >= 1.0 || vertices.is_empty() {
        return Ok(1.0);
    }

    let maximum_coordinate = vertices
        .iter()
        .flat_map(|vertex| vertex.position)
        .map(|position| (position as f64).abs())
        .fold(0.0, f64::max);
    // Lyon also snaps nearby intersections using a fixed distance, regardless of tolerance.
    let requested_exponent = -(width as f64).log2().floor();
    let maximum_exponent = ((f32::MAX as f64 * 0.25) / maximum_coordinate)
        .log2()
        .floor();
    let scale = 2.0_f64.powf(requested_exponent.min(maximum_exponent).max(0.0));
    for vertex in vertices {
        for position in &mut vertex.position {
            *position = (*position as f64 * scale) as f32;
        }
    }
    Ok(scale)
}

pub(super) fn tessellate(
    events: impl Iterator<Item = PathEvent> + Clone,
    options: StrokeOptions,
    resources: &mut ShapeResources,
    cache_key: Option<u64>,
) -> Result<Arc<CachedTessellation>, ShapeTessellationError> {
    if let Some(cache_key) = cache_key {
        if let Some(tessellation) = resources.tessellation_cache.get_tessellation(&cache_key) {
            return Ok(tessellation);
        }
    }
    if !events.clone().all(has_finite_positions) {
        return Err(TessellationError::GeometryBuilder(GeometryBuilderError::InvalidVertex).into());
    }

    let stroke_buffers = &mut resources.stroke_vertex_buffers;
    stroke_buffers.clear();
    if options.width != 0.0 {
        resources.stroke_tessellator.tessellate(
            events,
            &options.lyon_options(),
            &mut BuffersBuilder::new(stroke_buffers, StrokeVertexConverter),
        )?;
    }
    let triangle_count = stroke_buffers.indices.len() / 3;
    if triangle_count > MAX_STROKE_TRIANGLES {
        return Err(ShapeTessellationError::StrokeTooComplex {
            triangle_count,
            maximum_triangles: MAX_STROKE_TRIANGLES,
        });
    }
    let fill_scale = normalize_stroke_vertices(&mut stroke_buffers.vertices, options.width)?;
    let mut tessellation = resources.tessellation_cache.acquire_tessellation();
    if triangle_count != 0 {
        // Straight boundaries need no flattening or merging of nearby edges.
        let fill_options = FillOptions::non_zero().with_tolerance(f32::MIN_POSITIVE);
        if let Err(error) = resources.fill_tessellator.tessellate(
            resources
                .path_boundary_builder
                .path_events(&stroke_buffers.vertices, &stroke_buffers.indices),
            &fill_options,
            &mut BuffersBuilder::new(&mut tessellation.vertex_buffers, VertexConverter),
        ) {
            resources
                .tessellation_cache
                .recycle_tessellation(tessellation);
            return Err(error.into());
        }
    }
    if fill_scale != 1.0 {
        for vertex in &mut tessellation.vertex_buffers.vertices {
            for position in &mut vertex.position {
                *position = (*position as f64 / fill_scale) as f32;
            }
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
        return Err(TessellationError::GeometryBuilder(GeometryBuilderError::InvalidVertex).into());
    }
    Ok(finish_path_tessellation(
        tessellation,
        false,
        resources,
        cache_key,
    ))
}

pub(super) fn tessellate_rect(
    bounds: [(f32, f32); 2],
    options: StrokeOptions,
    resources: &mut ShapeResources,
    cache_key: Option<u64>,
) -> Result<Arc<CachedTessellation>, ShapeTessellationError> {
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
    tessellate(rectangle.path_events(), options, resources, cache_key)
}
