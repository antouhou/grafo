use super::projection::Projection;
use crate::core::vertex::{CustomVertex, InstanceTransform};
use ahash::HashMap;
use bytemuck::{Pod, Zeroable};
use std::{mem, ops::Range};

pub(super) const TILE_SIZE: u32 = 8;
pub(super) const SOLID_COVERAGE: u32 = u32::MAX;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct CoverageVertex {
    pub position: [f32; 2],
    pub texture_position: [f32; 3],
    pub model_position: [f32; 3],
    pub row_start: u32,
    pub tile_y: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(super) struct CoverageRow {
    pub baseline: f32,
    pub edge_start: u32,
    pub edge_count: u32,
    pub tile_x: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(super) struct CoverageEdge {
    pub x0: f32,
    pub x1: f32,
    pub height: f32,
    pub inverse_width: f32,
}

#[derive(Clone, Copy)]
struct BoundaryEdge {
    start: [f32; 2],
    end: [f32; 2],
}

type EdgeKey = ([u32; 2], [u32; 2]);

fn position_key(position: [f32; 2]) -> [u32; 2] {
    position.map(|value| if value == 0.0 { 0 } else { value.to_bits() })
}

#[derive(Default)]
pub(crate) struct CoverageBuilder {
    pub vertices: Vec<CoverageVertex>,
    pub indices: Vec<u32>,
    pub(super) rows: Vec<CoverageRow>,
    pub(super) edges: Vec<CoverageEdge>,
    boundary_edges: HashMap<EdgeKey, (i32, BoundaryEdge)>,
    row_edges: Vec<Vec<CoverageEdge>>,
    populated_rows: Range<usize>,
    projected_positions: Vec<[f32; 2]>,
}

impl CoverageBuilder {
    fn collect_boundary(
        &mut self,
        vertices: &[CustomVertex],
        indices: &[u16],
        projection: &Projection,
    ) {
        self.boundary_edges.clear();
        self.projected_positions.clear();
        self.projected_positions.extend(
            vertices
                .iter()
                .map(|vertex| projection.project(vertex.position)),
        );
        for triangle in indices.as_chunks::<3>().0 {
            if triangle
                .iter()
                .any(|&index| vertices[index as usize].coverage != 1.0)
            {
                continue;
            }
            let [a, mut b, mut c] = [triangle[0], triangle[1], triangle[2]]
                .map(|index| self.projected_positions[index as usize]);
            let area = (b[0] as f64 - a[0] as f64) * (c[1] as f64 - a[1] as f64)
                - (b[1] as f64 - a[1] as f64) * (c[0] as f64 - a[0] as f64);
            if !area.is_finite() || area == 0.0 {
                continue;
            }
            if area < 0.0 {
                mem::swap(&mut b, &mut c);
            }
            for (start, end) in [(a, b), (b, c), (c, a)] {
                let start_key = position_key(start);
                let end_key = position_key(end);
                let (key, direction, edge) = if start_key < end_key {
                    ((start_key, end_key), 1, BoundaryEdge { start, end })
                } else {
                    (
                        (end_key, start_key),
                        -1,
                        BoundaryEdge {
                            start: end,
                            end: start,
                        },
                    )
                };
                self.boundary_edges
                    .entry(key)
                    .and_modify(|entry| entry.0 += direction)
                    .or_insert((direction, edge));
            }
        }
    }

    fn collect_rows(&mut self, size: (u32, u32)) -> Option<[u32; 4]> {
        self.row_edges.resize_with(size.1 as usize, Vec::new);
        let previous_end = self.populated_rows.end.min(self.row_edges.len());
        let previous_start = self.populated_rows.start.min(previous_end);
        for row in &mut self.row_edges[previous_start..previous_end] {
            row.clear();
        }
        self.populated_rows = self.row_edges.len()..0;
        let mut bounds = [
            f32::INFINITY,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
        ];
        for &(count, edge) in self.boundary_edges.values() {
            if count == 0 {
                continue;
            }
            for position in [edge.start, edge.end] {
                bounds[0] = bounds[0].min(position[0]);
                bounds[1] = bounds[1].min(position[1]);
                bounds[2] = bounds[2].max(position[0]);
                bounds[3] = bounds[3].max(position[1]);
            }
            let height = edge.end[1] - edge.start[1];
            if height == 0.0 {
                continue;
            }
            let first_row = edge.start[1].min(edge.end[1]).floor().max(0.0) as u32;
            let last_row = edge.start[1]
                .max(edge.end[1])
                .ceil()
                .clamp(0.0, size.1 as f32) as u32;
            if first_row < last_row {
                self.populated_rows.start = self.populated_rows.start.min(first_row as usize);
                self.populated_rows.end = self.populated_rows.end.max(last_row as usize);
            }
            for row in first_row..last_row {
                let low = edge.start[1].min(edge.end[1]).max(row as f32);
                let high = edge.start[1].max(edge.end[1]).min(row as f32 + 1.0);
                let slope = (edge.end[0] as f64 - edge.start[0] as f64) / height as f64;
                let x0 =
                    (edge.start[0] as f64 + (low as f64 - edge.start[1] as f64) * slope) as f32;
                let x1 =
                    (edge.start[0] as f64 + (high as f64 - edge.start[1] as f64) * slope) as f32;
                self.row_edges[row as usize].push(CoverageEdge {
                    x0,
                    x1,
                    height: (high - low).copysign(height) * count as f32,
                    inverse_width: if x0 == x1 { 0.0 } else { 1.0 / (x1 - x0) },
                });
            }
        }
        let bounds = [
            bounds[0].floor().clamp(0.0, size.0 as f32) as u32,
            bounds[1].floor().clamp(0.0, size.1 as f32) as u32,
            bounds[2].ceil().clamp(0.0, size.0 as f32) as u32,
            bounds[3].ceil().clamp(0.0, size.1 as f32) as u32,
        ];
        (bounds[0] < bounds[2] && bounds[1] < bounds[3]).then_some(bounds)
    }

    fn append_tile_rows(&mut self, bounds: [u32; 4]) -> (u32, bool, bool) {
        let [left, top, right, bottom] = bounds;
        let row_start = self.rows.len() as u32;
        let mut is_solid = true;
        let mut is_empty = true;
        for y in top..bottom {
            let edge_start = self.edges.len() as u32;
            let mut baseline = 0.0_f64;
            for &edge in &self.row_edges[y as usize] {
                if edge.x0.max(edge.x1) <= left as f32 {
                    continue;
                }
                if edge.x0.min(edge.x1) >= right as f32 {
                    baseline += edge.height as f64;
                } else {
                    self.edges.push(CoverageEdge {
                        x0: edge.x0 - left as f32,
                        x1: edge.x1 - left as f32,
                        ..edge
                    });
                }
            }
            let edge_count = self.edges.len() as u32 - edge_start;
            is_solid &= edge_count == 0 && (baseline - 1.0).abs() < 1e-6;
            is_empty &= edge_count == 0 && baseline.abs() < 1e-6;
            self.rows.push(CoverageRow {
                baseline: baseline as f32,
                edge_start,
                edge_count,
                tile_x: left,
            });
        }
        (row_start, is_solid, is_empty)
    }

    fn append_quad(
        &mut self,
        bounds: [u32; 4],
        row_start: u32,
        projection: &Projection,
        local_bounds: [(f32, f32); 2],
    ) {
        let [left, top, right, bottom] = bounds;
        let vertex_start = self.vertices.len() as u32;
        let width = (local_bounds[1].0 - local_bounds[0].0).max(1e-6);
        let height = (local_bounds[1].1 - local_bounds[0].1).max(1e-6);
        for position in [[left, top], [right, top], [right, bottom], [left, bottom]] {
            let position = position.map(|value| value as f32);
            let model_position = projection.unproject(position);
            self.vertices.push(CoverageVertex {
                position,
                model_position,
                texture_position: [
                    (model_position[0] - local_bounds[0].0 * model_position[2]) / width,
                    (model_position[1] - local_bounds[0].1 * model_position[2]) / height,
                    model_position[2],
                ],
                row_start,
                tile_y: top,
            });
        }
        self.indices
            .extend([0, 1, 2, 0, 2, 3].map(|index| index + vertex_start));
    }

    pub(crate) fn build(
        &mut self,
        vertices: &[CustomVertex],
        indices: &[u16],
        transform: Option<InstanceTransform>,
        scale: f64,
        size: (u32, u32),
        local_bounds: [(f32, f32); 2],
    ) {
        self.vertices.clear();
        self.indices.clear();
        self.rows.clear();
        self.edges.clear();
        let Some(projection) = Projection::new(transform, scale, local_bounds) else {
            return;
        };
        self.collect_boundary(vertices, indices, &projection);
        let Some([left, top, right, bottom]) = self.collect_rows(size) else {
            return;
        };
        for y in (top..bottom).step_by(TILE_SIZE as usize) {
            let mut solid_start = None;
            for x in (left..right).step_by(TILE_SIZE as usize) {
                let bounds = [
                    x,
                    y,
                    (x + TILE_SIZE).min(right),
                    (y + TILE_SIZE).min(bottom),
                ];
                let first_edge = self.edges.len();
                let (row_start, is_solid, is_empty) = self.append_tile_rows(bounds);
                if !is_solid {
                    if let Some(start) = solid_start.take() {
                        self.append_quad(
                            [start, y, x, bounds[3]],
                            SOLID_COVERAGE,
                            &projection,
                            local_bounds,
                        );
                    }
                }
                if is_solid || is_empty {
                    self.rows.truncate(row_start as usize);
                    self.edges.truncate(first_edge);
                    if is_solid {
                        solid_start.get_or_insert(x);
                    }
                } else {
                    self.append_quad(bounds, row_start, &projection, local_bounds);
                }
            }
            if let Some(start) = solid_start {
                self.append_quad(
                    [start, y, right, (y + TILE_SIZE).min(bottom)],
                    SOLID_COVERAGE,
                    &projection,
                    local_bounds,
                );
            }
        }
    }
}
