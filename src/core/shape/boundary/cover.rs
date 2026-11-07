use super::{accumulate_triangle_edges, area_products, signed_double_area};
use super::{PathBoundaryBuilder, PathBoundaryEdge};
use crate::core::vertex::CustomVertex;
use std::mem;

fn is_area_reliable([positive, negative]: [f64; 2]) -> bool {
    // Allow for rounding in the coordinate differences, products and subtraction.
    (positive - negative).abs() > (positive.abs() + negative.abs()) * (8.0 * f64::EPSILON)
}

fn turns_left(points: [[f32; 2]; 3]) -> bool {
    let products = area_products(points);
    products[0] > products[1] && is_area_reliable(products)
}

struct CoveringQuad {
    corners: [[f32; 2]; 4],
    horizontal_clearance: f64,
}

impl CoveringQuad {
    fn new(first: [[f32; 2]; 3], second: [[f32; 2]; 3]) -> Option<Self> {
        let first_area = area_products(first);
        if !is_area_reliable(first_area) || !is_area_reliable(area_products(second)) {
            return None;
        }
        let [first_corner, mut second_corner, mut third_corner] = first;
        if first_area[0] < first_area[1] {
            mem::swap(&mut second_corner, &mut third_corner);
        }
        let [start, end, opposite] = [
            [first_corner, second_corner, third_corner],
            [second_corner, third_corner, first_corner],
            [third_corner, first_corner, second_corner],
        ]
        .into_iter()
        .find(|[start, end, _]| second.contains(start) && second.contains(end))?;
        let other = second
            .into_iter()
            .find(|point| *point != start && *point != end)?;
        let corners = [start, other, end, opposite];
        if !(0..4).all(|index| {
            turns_left([
                corners[index],
                corners[(index + 1) % 4],
                corners[(index + 2) % 4],
            ])
        }) {
            return None;
        }
        let maximum_x = corners
            .iter()
            .map(|corner| f64::from(corner[0]).abs())
            .fold(0.0, f64::max);
        // Row clipping interpolates in f64. Distinct f32 y values bound the
        // ratio of the starting y to the edge height by 2^24.
        let horizontal_clearance = maximum_x * (4.0 * f64::from(f32::EPSILON));
        Some(Self {
            corners,
            horizontal_clearance,
        })
    }

    fn contains(&self, triangle: [[f32; 2]; 3]) -> bool {
        (0..4).all(|index| {
            let start = self.corners[index];
            let end = self.corners[(index + 1) % 4];
            let minimum_area =
                self.horizontal_clearance * (f64::from(end[1]) - f64::from(start[1])).abs();
            triangle.into_iter().all(|point| {
                let products = area_products([start, end, point]);
                products[0] - products[1] > minimum_area && is_area_reliable(products)
            })
        })
    }
}

fn find_cover(
    triangles: &[[u32; 3]],
    local_position: impl Fn(u32) -> [f32; 2],
    project: impl Fn([f32; 2]) -> [f32; 2],
) -> Option<CoveringQuad> {
    // Stroke quads are usually adjacent triangle pairs. Consider only the neighbors
    // of the largest triangle, regardless of where it occurs in the stroke.
    let (largest, _) = triangles
        .iter()
        .enumerate()
        .map(|(index, triangle)| {
            (
                index,
                signed_double_area(triangle.map(&local_position)).abs(),
            )
        })
        .max_by(|first, second| first.1.total_cmp(&second.1))?;
    let projected_position = |index| project(local_position(index));
    let first = triangles[largest].map(projected_position);
    triangles[..largest]
        .last()
        .into_iter()
        .chain(triangles.get(largest + 1))
        .find_map(|triangle| CoveringQuad::new(first, triangle.map(projected_position)))
}

fn preserves_winding(
    projected: [[f32; 2]; 3],
    local: impl FnOnce() -> [[f32; 2]; 3],
    orientation: Option<i32>,
) -> bool {
    let products = area_products(projected);
    let area = products[0] - products[1];
    if area != 0.0 && !is_area_reliable(products) {
        return false;
    }
    orientation.is_none_or(|orientation| {
        let local_area = signed_double_area(local());
        local_area.partial_cmp(&0.0) == (area * f64::from(orientation)).partial_cmp(&0.0)
    })
}

impl PathBoundaryBuilder {
    #[cold]
    pub(crate) fn cull_covered_triangles(
        &mut self,
        vertices: &[CustomVertex],
        indices: &[u32],
        project: impl Fn([f32; 2]) -> [f32; 2],
        orientation: Option<i32>,
    ) -> Option<impl Iterator<Item = PathBoundaryEdge> + Clone + '_> {
        let triangles = indices.as_chunks::<3>().0;
        let local_position = |index: u32| vertices[index as usize].position;
        let cover = find_cover(triangles, local_position, &project)?;
        self.positions.clear();
        self.positions
            .extend(vertices.iter().map(|vertex| project(vertex.position)));
        if !self
            .positions
            .iter()
            .flatten()
            .all(|value| value.is_finite())
        {
            return None;
        }
        let position = |index: u32| self.positions[index as usize];
        let is_covered = |triangle: [u32; 3]| cover.contains(triangle.map(position));
        if !triangles.iter().copied().any(is_covered) {
            return None;
        }
        // Rounded projections can reverse triangles. Keep the original sweep if
        // rebuilding the boundary would change any triangle's winding.
        if !triangles.iter().copied().all(|triangle| {
            preserves_winding(
                triangle.map(position),
                || triangle.map(local_position),
                orientation,
            )
        }) {
            return None;
        }
        self.edges.clear();
        accumulate_triangle_edges(
            &mut self.edges,
            triangles
                .iter()
                .copied()
                .filter(|&triangle| !is_covered(triangle)),
            position,
        );
        Some(self.boundary_edges())
    }
}
