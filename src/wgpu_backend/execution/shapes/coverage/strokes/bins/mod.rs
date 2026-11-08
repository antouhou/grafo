use super::ATLAS_SIDE;
use crate::core::vertex::{CustomVertex, InstanceTransform};
use crate::core::{geometry, UnsignedPhysicalRect};
use lyon::geom::Point;
use smallvec::SmallVec;
use std::mem;

#[cfg(test)]
mod tests;

/// Whether GPU f32 projection can lose subpixel precision before viewport clipping.
pub(super) fn requires_precise_projection(
    transform: Option<InstanceTransform>,
    bounds: [(f32, f32); 2],
    scale: f64,
) -> bool {
    let transform = transform.unwrap_or_else(InstanceTransform::identity);
    let largest_x = f64::from(bounds[0].0.abs().max(bounds[1].0.abs()));
    let largest_y = f64::from(bounds[0].1.abs().max(bounds[1].1.abs()));
    let magnitude = |axis: usize| {
        f64::from(transform.col0[axis]).abs() * largest_x
            + f64::from(transform.col1[axis]).abs() * largest_y
            + f64::from(transform.col3[axis]).abs()
    };
    let range = |coefficient: f32, minimum: f32, maximum: f32| {
        let endpoints = [minimum, maximum].map(|value| f64::from(coefficient) * f64::from(value));
        [
            endpoints[0].min(endpoints[1]),
            endpoints[0].max(endpoints[1]),
        ]
    };
    let x_range = range(transform.col0[3], bounds[0].0, bounds[1].0);
    let y_range = range(transform.col1[3], bounds[0].1, bounds[1].1);
    let minimum_w = x_range[0] + y_range[0] + f64::from(transform.col3[3]);
    let maximum_w = x_range[1] + y_range[1] + f64::from(transform.col3[3]);
    let divisor = minimum_w.max(-maximum_w).max(1e-6);
    let relative_error = 4.0 * f64::from(f32::EPSILON);
    let position_error = scale * magnitude(0).max(magnitude(1)) / divisor
        * relative_error
        * (1.0 + magnitude(3) / divisor);
    position_error > 1.0 / 32.0
}

#[derive(Default)]
pub(super) struct TriangleBin {
    pub(super) bounds: Option<UnsignedPhysicalRect>,
    pub(super) indices: Vec<u32>,
}

impl TriangleBin {
    fn push(&mut self, bounds: UnsignedPhysicalRect, triangle: &[u32; 3]) {
        self.bounds = Some(
            self.bounds
                .map_or(bounds, |previous| previous.union(&bounds)),
        );
        self.indices.extend_from_slice(triangle);
    }
}

#[derive(Default)]
pub(super) struct TriangleBins {
    pub(super) projected_vertices: Vec<CustomVertex>,
    bins: Vec<TriangleBin>,
    active_bin_count: usize,
}

fn pixel_bounds(points: &[[f64; 2]]) -> UnsignedPhysicalRect {
    let minimum = points.iter().fold([f64::INFINITY; 2], |minimum, point| {
        [minimum[0].min(point[0]), minimum[1].min(point[1])]
    });
    let maximum = points
        .iter()
        .fold([f64::NEG_INFINITY; 2], |maximum, point| {
            [maximum[0].max(point[0]), maximum[1].max(point[1])]
        });
    UnsignedPhysicalRect::new(
        minimum.map(|value| value.floor() as u32).into(),
        maximum.map(|value| value.ceil() as u32).into(),
    )
}

fn clipped_triangle_bounds(
    triangle: &[[f64; 2]; 3],
    tile: UnsignedPhysicalRect,
) -> Option<UnsignedPhysicalRect> {
    // A triangle clipped by a rectangle has at most seven vertices.
    let mut polygon = SmallVec::<[[f64; 2]; 7]>::from_slice(triangle);
    let mut clipped = SmallVec::<[[f64; 2]; 7]>::new();
    for (axis, edge, side) in [
        (0, tile.min.x, 1.0),
        (0, tile.max.x, -1.0),
        (1, tile.min.y, 1.0),
        (1, tile.max.y, -1.0),
    ] {
        let edge = f64::from(edge);
        let mut previous = *polygon.last()?;
        for &point in &polygon {
            let previous_distance = (previous[axis] - edge) * side;
            let distance = (point[axis] - edge) * side;
            if previous_distance * distance < 0.0 {
                let ratio = (edge - previous[axis]) / (point[axis] - previous[axis]);
                let other = 1 - axis;
                let mut intersection = [edge; 2];
                intersection[other] = previous[other] + (point[other] - previous[other]) * ratio;
                clipped.push(intersection);
            }
            if distance >= 0.0 {
                clipped.push(point);
            }
            previous = point;
        }
        mem::swap(&mut polygon, &mut clipped);
        clipped.clear();
    }
    (polygon.len() >= 3)
        .then(|| pixel_bounds(&polygon))?
        .intersection(&tile)
}

impl TriangleBins {
    pub(super) fn prepare(
        &mut self,
        vertices: &[CustomVertex],
        indices: &[u32],
        transform: InstanceTransform,
        scale: f64,
        bounds: UnsignedPhysicalRect,
    ) {
        let projection = geometry::project_transform(transform);
        self.projected_vertices.clear();
        self.projected_vertices
            .extend(vertices.iter().map(|vertex| {
                let projected = projection.transform_point2d_homogeneous(Point::new(
                    f64::from(vertex.position[0]),
                    f64::from(vertex.position[1]),
                ));
                let factor = scale / projected.w.abs().max(1e-6);
                CustomVertex {
                    position: [(projected.x * factor) as f32, (projected.y * factor) as f32],
                    tex_coords: vertex.position,
                }
            }));

        let columns = bounds.width().div_ceil(ATLAS_SIDE);
        let rows = bounds.height().div_ceil(ATLAS_SIDE);
        self.active_bin_count = (columns * rows) as usize;
        if self.bins.len() < self.active_bin_count {
            self.bins
                .resize_with(self.active_bin_count, TriangleBin::default);
        }
        for bin in &mut self.bins[..self.active_bin_count] {
            bin.bounds = None;
            bin.indices.clear();
        }
        for triangle in indices.as_chunks::<3>().0 {
            let points = triangle.map(|index| {
                self.projected_vertices[index as usize]
                    .position
                    .map(f64::from)
            });
            if !points.iter().flatten().all(|value| value.is_finite()) {
                continue;
            }
            let Some(visible) = pixel_bounds(&points).intersection(&bounds) else {
                continue;
            };
            let first_column = (visible.min.x - bounds.min.x) / ATLAS_SIDE;
            let last_column = (visible.max.x - 1 - bounds.min.x) / ATLAS_SIDE;
            let first_row = (visible.min.y - bounds.min.y) / ATLAS_SIDE;
            let last_row = (visible.max.y - 1 - bounds.min.y) / ATLAS_SIDE;
            if first_column == last_column && first_row == last_row {
                self.bins[(first_row * columns + first_column) as usize].push(visible, triangle);
                continue;
            }
            for row in first_row..=last_row {
                for column in first_column..=last_column {
                    let x = bounds.min.x + column * ATLAS_SIDE;
                    let y = bounds.min.y + row * ATLAS_SIDE;
                    let tile = UnsignedPhysicalRect::new(
                        (x, y).into(),
                        (
                            (x + ATLAS_SIDE).min(bounds.max.x),
                            (y + ATLAS_SIDE).min(bounds.max.y),
                        )
                            .into(),
                    );
                    if let Some(touched) = clipped_triangle_bounds(&points, tile) {
                        self.bins[(row * columns + column) as usize].push(touched, triangle);
                    }
                }
            }
        }
    }

    pub(super) fn occupied(&self) -> impl Iterator<Item = (UnsignedPhysicalRect, &[u32])> {
        self.bins[..self.active_bin_count]
            .iter()
            .filter_map(|bin| Some((bin.bounds?, bin.indices.as_slice())))
    }
}
