use super::{
    finalize_path_tessellation, Shape, ShapeGeometry, ShapeTessellationError, StrokeGeometryOptions,
};
use crate::core::cache::CachedTessellation;
use crate::core::util::ShapeResources;
use crate::core::vertex::CustomVertex;
use lyon::math::{point, Box2D, Point};
use lyon::path::{LineCap, LineJoin, PathEvent, Polygon};
use lyon::tessellation::{
    BuffersBuilder, StrokeOptions, StrokeVertex, StrokeVertexConstructor, VertexBuffers,
};
use std::sync::Arc;
use thiserror::Error;

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
        let radius = self.width * 0.5;
        if radius == 0.0 {
            return Ok(None);
        }

        let mut options = StrokeOptions::default()
            .with_line_width(self.width)
            .with_start_cap(self.start_cap)
            .with_end_cap(self.end_cap)
            .with_line_join(self.line_join);
        let has_round_edges = self.line_join == LineJoin::Round
            || self.start_cap == LineCap::Round
            || self.end_cap == LineCap::Round;
        // Lyon's arc subdivision cannot advance if this subtraction rounds back to the radius.
        if has_round_edges && radius - options.tolerance == radius {
            return Err(StrokeGeometryError::UnrepresentableMesh);
        }
        if matches!(self.line_join, LineJoin::Miter | LineJoin::MiterClip) {
            if !self.miter_limit.is_finite()
                || self.miter_limit < StrokeOptions::MINIMUM_MITER_LIMIT
            {
                return Err(StrokeGeometryError::InvalidMiterLimit);
            }
            options = options.with_miter_limit(self.miter_limit);
        }
        Ok(Some(options))
    }
}

fn validate_segment(positions: &[Point]) -> Result<(), StrokeGeometryError> {
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
        PathEvent::Begin { at } => validate_segment(&[at]),
        PathEvent::Line { from, to } => validate_segment(&[from, to]),
        PathEvent::Quadratic { from, ctrl, to } => validate_segment(&[from, ctrl, to]),
        PathEvent::Cubic {
            from,
            ctrl1,
            ctrl2,
            to,
        } => validate_segment(&[from, ctrl1, ctrl2, to]),
        PathEvent::End {
            last,
            first,
            close: true,
        } => validate_segment(&[last, first]),
        PathEvent::End { close: false, .. } => Ok(()),
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
                shape.path.iter().try_for_each(validate_path_event)?;
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
                rectangle.path_events().try_for_each(validate_path_event)?;
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
