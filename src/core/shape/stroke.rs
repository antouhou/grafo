use super::{
    finalize_path_tessellation, Shape, ShapeGeometry, ShapeResources, ShapeTessellationError,
    StrokeGeometryOptions,
};
use crate::core::cache::CachedTessellation;
use crate::core::vertex::CustomVertex;
use lyon::math::{point, Box2D, Point};
use lyon::path::{LineCap, LineJoin as LyonLineJoin, PathEvent, Polygon};
use lyon::tessellation::{
    BuffersBuilder, StrokeOptions, StrokeVertex, StrokeVertexConstructor, VertexBuffers,
};
use std::sync::Arc;
use thiserror::Error;

/// How the outer stroke edges connect at a corner.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LineJoin {
    /// Extends the edges until they meet. Falls back to a bevel beyond the miter limit.
    Miter,
    /// Connects the edges with a circular arc.
    Round,
    /// Connects the edges with a straight line.
    Bevel,
}

/// Invalid input to stroke tessellation.
#[derive(Debug, Error)]
pub enum StrokeGeometryError {
    #[error("stroke width must be finite and nonnegative")]
    InvalidWidth,
    #[error("stroke miter limit must be finite and at least 1")]
    InvalidMiterLimit,
    #[error("stroke path must have finite coordinates")]
    NonFinitePosition,
    #[error("stroke geometry exceeds the tessellator's numeric range")]
    UnrepresentableMesh,
}

impl StrokeGeometryOptions {
    fn tessellation_options(&self) -> Result<Option<StrokeOptions>, StrokeGeometryError> {
        if !self.width.is_finite() || self.width < 0.0 {
            return Err(StrokeGeometryError::InvalidWidth);
        }
        if !self.miter_limit.is_finite() || self.miter_limit < StrokeOptions::MINIMUM_MITER_LIMIT {
            return Err(StrokeGeometryError::InvalidMiterLimit);
        }

        let radius = self.width * 0.5;
        if radius == 0.0 {
            return Ok(None);
        }

        let line_join = match self.line_join {
            LineJoin::Miter => LyonLineJoin::Miter,
            LineJoin::Round => LyonLineJoin::Round,
            LineJoin::Bevel => LyonLineJoin::Bevel,
        };
        Ok(Some(
            StrokeOptions::default()
                .with_line_width(self.width)
                .with_start_cap(self.start_cap)
                .with_end_cap(self.end_cap)
                .with_line_join(line_join)
                .with_miter_limit(self.miter_limit),
        ))
    }
}

fn validate_position_span(positions: &[Point]) -> Result<(), StrokeGeometryError> {
    if !positions.iter().all(|position| position.is_finite()) {
        return Err(StrokeGeometryError::NonFinitePosition);
    }
    let bounds = Box2D::from_points(positions);
    // Lyon normalizes the segment directions using squared lengths in f32.
    if !(bounds.max - bounds.min).square_length().is_finite() {
        return Err(StrokeGeometryError::UnrepresentableMesh);
    }
    Ok(())
}

fn validate_path_event(event: PathEvent) -> Result<(), StrokeGeometryError> {
    match event {
        PathEvent::Begin { at } => validate_position_span(&[at]),
        PathEvent::Line { from, to } => validate_position_span(&[from, to]),
        PathEvent::Quadratic { from, ctrl, to } => validate_position_span(&[from, ctrl, to]),
        PathEvent::Cubic {
            from,
            ctrl1,
            ctrl2,
            to,
        } => validate_position_span(&[from, ctrl1, ctrl2, to]),
        PathEvent::End {
            last,
            first,
            close: true,
        } => validate_position_span(&[last, first]),
        PathEvent::End { close: false, .. } => Ok(()),
    }
}

fn stroke_options_for_path(
    mut options: StrokeOptions,
    events: impl Iterator<Item = PathEvent>,
) -> Result<StrokeOptions, StrokeGeometryError> {
    let radius = options.line_width * 0.5;
    let has_round_cap = options.start_cap == LineCap::Round || options.end_cap == LineCap::Round;
    if has_round_cap {
        // Lyon requires a minimum tolerance for flattening Bezier curves.
        options.tolerance = options.tolerance.min((radius / 8.0).max(1e-7));
    }

    let mut shortest_edge_squared = f32::INFINITY;
    for event in events {
        validate_path_event(event)?;
        if matches!(
            event,
            PathEvent::Line { .. } | PathEvent::End { close: true, .. }
        ) {
            let edge_squared = (event.to() - event.from()).square_length();
            // Lyon always merges points closer than 1e-4 local units.
            if edge_squared >= 1e-8 {
                shortest_edge_squared = shortest_edge_squared.min(edge_squared);
            }
        }
    }

    let mut tolerance = options.tolerance.min(shortest_edge_squared.sqrt() * 0.5);
    if (has_round_cap || options.line_join == LyonLineJoin::Round) && radius - tolerance == radius {
        // Keep enough precision for round arcs when an edge requires a smaller tolerance.
        tolerance = options.tolerance;
        if radius - tolerance == radius {
            return Err(StrokeGeometryError::UnrepresentableMesh);
        }
    }
    Ok(options.with_tolerance(tolerance))
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

impl Shape {
    pub(super) fn tessellate_stroke(
        &self,
        shape_resources: &mut ShapeResources,
    ) -> Result<Option<Arc<CachedTessellation>>, ShapeTessellationError> {
        let Some(stroke) = self.stroke() else {
            return Ok(None);
        };
        let Some(options) = stroke.tessellation_options()? else {
            return Ok(None);
        };

        let mut buffers = VertexBuffers::new();
        let mut builder = BuffersBuilder::new(&mut buffers, StrokeVertexConverter);
        let tessellator = &mut shape_resources.stroke_tessellator;
        match &self.geometry {
            ShapeGeometry::Path(shape) => {
                let options = stroke_options_for_path(options, shape.path.iter())?;
                tessellator.tessellate(shape.path.iter(), &options, &mut builder)?;
            }
            ShapeGeometry::Rect(shape) => {
                let [top_left, bottom_right] = shape.rect.map(Point::from);
                let rectangle = Polygon {
                    points: &[
                        top_left,
                        point(bottom_right.x, top_left.y),
                        bottom_right,
                        point(top_left.x, bottom_right.y),
                    ],
                    closed: true,
                };
                let options = stroke_options_for_path(options, rectangle.path_events())?;
                tessellator.tessellate_polygon(rectangle, &options, &mut builder)?;
            }
        }

        if buffers.indices.is_empty() {
            return Ok(None);
        }
        if buffers
            .vertices
            .iter()
            .any(|vertex| !vertex.position.into_iter().all(f32::is_finite))
        {
            return Err(StrokeGeometryError::UnrepresentableMesh.into());
        }

        let tessellation = finalize_path_tessellation(buffers, false, shape_resources);
        if !tessellation
            .texture_mapping_size
            .into_iter()
            .all(f32::is_finite)
        {
            return Err(StrokeGeometryError::UnrepresentableMesh.into());
        }
        Ok(Some(tessellation))
    }
}
