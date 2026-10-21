use super::projection::Projection;
use super::rectangles;
use super::rounded_rect::RoundedRectSweep;
use super::scanline::{RowEdge, Scanline};
use super::tiles::{CoverageDrawMetadata, CoverageDrawRange, CoverageGeometry, CoverageTile};
use crate::core::cache::CachedTessellation;
use crate::core::shape::boundary::{PathBoundaryBuilder, PathBoundaryEdge};
use crate::core::vertex::InstanceTransform;
use crate::core::UnsignedPhysicalRect;
use std::ops::Range;

pub(super) const TILE_SIZE: u32 = 8;
pub(super) const SOLID_COVERAGE: u32 = u32::MAX;

enum TileCoverage {
    Empty,
    Solid,
    Partial,
}

#[derive(Default)]
pub(crate) struct CoverageBuilder {
    boundary_tiles: Vec<CoverageTile>,
    boundary_edges: Vec<PathBoundaryEdge>,
    boundary_builder: PathBoundaryBuilder,
    tile_intervals: Vec<Range<u32>>,
    row_edges: Vec<Vec<RowEdge>>,
    scanlines: [Scanline; TILE_SIZE as usize],
    tile_values: [[f32; TILE_SIZE as usize]; TILE_SIZE as usize],
    populated_rows: Range<usize>,
    rounded_rect: RoundedRectSweep,
    has_identical_rows: bool,
}

impl CoverageBuilder {
    fn collect_boundary(&mut self, tessellation: &CachedTessellation, projection: &Projection) {
        self.boundary_edges.clear();
        if let Some(orientation) = projection.boundary_orientation {
            self.boundary_edges
                .extend(
                    tessellation
                        .boundary_edges
                        .iter()
                        .map(|edge| PathBoundaryEdge {
                            start: projection.project(edge.start),
                            end: projection.project(edge.end),
                            winding: edge.winding * orientation,
                        }),
                );
            if self
                .boundary_edges
                .iter()
                .all(|edge| edge.start.into_iter().chain(edge.end).all(f32::is_finite))
            {
                return;
            }
        }
        self.boundary_edges.clear();
        let geometry = &tessellation.vertex_buffers;
        self.boundary_edges.extend(self.boundary_builder.build(
            &geometry.vertices,
            &geometry.indices,
            |position| projection.project(position),
        ));
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
        for edge in &self.boundary_edges {
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
                self.row_edges[row as usize].push(RowEdge {
                    x0,
                    x1,
                    signed_winding_height: (high as f64 - low as f64).copysign(height as f64)
                        * edge.winding as f64,
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

    fn append_tile(
        &mut self,
        geometry: &mut CoverageGeometry,
        bounds: [u32; 4],
        coverage_start: u32,
    ) {
        let [left, top, right, bottom] = bounds;
        let tile = CoverageTile {
            min: [left, top],
            max: [right, bottom],
            coverage_start,
            projection_index: geometry.projections.len() as u32 - 1,
        };
        if coverage_start == SOLID_COVERAGE {
            geometry.tiles.push(tile);
        } else {
            self.boundary_tiles.push(tile);
        }
    }

    fn collect_tile_intervals(&mut self, bounds: [u32; 4]) {
        let [left, top, right, bottom] = bounds;
        self.tile_intervals.clear();
        for row in &self.row_edges[top as usize..bottom as usize] {
            for edge in row {
                let minimum = edge.x0.min(edge.x1);
                let maximum = edge.x0.max(edge.x1);
                if maximum <= left as f32 || minimum >= right as f32 {
                    continue;
                }
                let first = (minimum.max(left as f32) as u32 - left) / TILE_SIZE;
                // The coverage change after the last crossed pixel must be swept too.
                let last = (maximum.ceil().min(right as f32) as u32 - left) / TILE_SIZE;
                self.tile_intervals
                    .push(left + first * TILE_SIZE..(left + (last + 1) * TILE_SIZE).min(right));
            }
        }
        self.tile_intervals
            .sort_unstable_by_key(|interval| interval.start);
        let mut count = 0;
        for index in 0..self.tile_intervals.len() {
            let interval = self.tile_intervals[index].clone();
            if count > 0 && self.tile_intervals[count - 1].end >= interval.start {
                self.tile_intervals[count - 1].end =
                    self.tile_intervals[count - 1].end.max(interval.end);
            } else {
                self.tile_intervals[count] = interval;
                count += 1;
            }
        }
        self.tile_intervals.truncate(count);
    }

    fn classify_constant_rows(&self, height: usize) -> Option<TileCoverage> {
        let height = if self.has_identical_rows { 1 } else { height };
        if self.scanlines[..height]
            .iter()
            .all(|row| row.coverage() == 1.0)
        {
            Some(TileCoverage::Solid)
        } else if self.scanlines[..height]
            .iter()
            .all(|row| row.coverage() == 0.0)
        {
            Some(TileCoverage::Empty)
        } else {
            None
        }
    }

    fn tile_coverage(&mut self, bounds: [u32; 4], has_edges: bool) -> TileCoverage {
        let [left, top, right, bottom] = bounds;
        self.tile_values.fill([0.0; TILE_SIZE as usize]);
        let width = (right - left) as usize;
        let height = (bottom - top) as usize;
        let row_count = if self.has_identical_rows { 1 } else { height };
        for (row_index, row) in self.scanlines[..row_count].iter_mut().enumerate() {
            let values = &mut self.tile_values[row_index][..width];
            if has_edges {
                row.fill_coverage(left, values);
            } else {
                values.fill(row.coverage());
            }
        }
        if self.has_identical_rows {
            let row = self.tile_values[0];
            self.tile_values[1..height].fill(row);
        }
        let values = self.tile_values[..height]
            .iter()
            .flat_map(|row| &row[..width]);
        if values.clone().all(|&value| value == 1.0) {
            TileCoverage::Solid
        } else if values.clone().all(|&value| value == 0.0) {
            TileCoverage::Empty
        } else {
            TileCoverage::Partial
        }
    }

    fn append_span(
        &mut self,
        bounds: [u32; 4],
        coverage: TileCoverage,
        geometry: &mut CoverageGeometry,
        solid_start: &mut Option<u32>,
    ) {
        let [left, top, _, bottom] = bounds;
        if !matches!(coverage, TileCoverage::Solid) {
            if let Some(start) = solid_start.take() {
                self.append_tile(geometry, [start, top, left, bottom], SOLID_COVERAGE);
            }
        }
        match coverage {
            TileCoverage::Solid => {
                solid_start.get_or_insert(left);
            }
            TileCoverage::Empty => {}
            TileCoverage::Partial => {
                let coverage_start = geometry.coverage.len() as u32;
                for values in &self.tile_values[..(bottom - top) as usize] {
                    geometry.coverage.extend_from_slice(values);
                }
                self.append_tile(geometry, bounds, coverage_start);
            }
        }
    }

    fn append_strip(
        &mut self,
        bounds: [u32; 4],
        has_edges: bool,
        geometry: &mut CoverageGeometry,
        solid_start: &mut Option<u32>,
    ) {
        let [left, top, right, bottom] = bounds;
        if !has_edges {
            if let Some(coverage) = self.classify_constant_rows((bottom - top) as usize) {
                self.append_span(bounds, coverage, geometry, solid_start);
                return;
            }
        }
        for left in (left..right).step_by(TILE_SIZE as usize) {
            let bounds = [left, top, (left + TILE_SIZE).min(right), bottom];
            let coverage = self.tile_coverage(bounds, has_edges);
            self.append_span(bounds, coverage, geometry, solid_start);
        }
    }

    pub(crate) fn build(
        &mut self,
        tessellation: &CachedTessellation,
        transform: Option<InstanceTransform>,
        scale: f64,
        size: (u32, u32),
        geometry: &mut CoverageGeometry,
    ) -> CoverageDrawMetadata {
        let mut coverage_draw = CoverageDrawMetadata {
            range: CoverageDrawRange {
                tile_start: geometry.tiles.len() as u32,
                ..Default::default()
            },
            bounds: UnsignedPhysicalRect::zero(),
        };
        let local_bounds = tessellation.local_bounds;
        self.boundary_tiles.clear();
        let Some(projection) = Projection::new(transform, scale, local_bounds) else {
            return coverage_draw;
        };
        if tessellation.is_rect {
            if let Some(rectangle) = rectangles::build(local_bounds, &projection, size, geometry) {
                return rectangle;
            }
        }
        let rounded_bounds = self.rounded_rect.prepare(tessellation, &projection, size);
        let bounds = rounded_bounds.or_else(|| {
            self.collect_boundary(tessellation, &projection);
            self.collect_rows(size)
        });
        let Some([left, top, right, bottom]) = bounds else {
            return coverage_draw;
        };
        if left >= right || top >= bottom {
            return coverage_draw;
        }
        coverage_draw.bounds =
            UnsignedPhysicalRect::new((left, top).into(), (right, bottom).into());
        geometry
            .projections
            .push(projection.mapping([left, top], local_bounds));
        for top in (top..bottom).step_by(TILE_SIZE as usize) {
            let bottom = (top + TILE_SIZE).min(bottom);
            self.has_identical_rows = false;
            if rounded_bounds.is_some() {
                self.has_identical_rows = self.rounded_rect.prepare_band(
                    [left, top, right, bottom],
                    &mut self.tile_intervals,
                    &mut self.scanlines[..(bottom - top) as usize],
                );
            } else {
                self.collect_tile_intervals([left, top, right, bottom]);
                for (index, row) in self.scanlines[..(bottom - top) as usize]
                    .iter_mut()
                    .enumerate()
                {
                    row.prepare(
                        self.row_edges[top as usize + index].iter().copied(),
                        left,
                        right,
                        &self.tile_intervals,
                    );
                }
            }
            let mut solid_start = None;
            let mut cursor = left;
            for index in 0..self.tile_intervals.len() {
                let interval = self.tile_intervals[index].clone();
                // Coverage between boundary intervals is constant across each pixel row.
                if cursor < interval.start {
                    self.append_strip(
                        [cursor, top, interval.start, bottom],
                        false,
                        geometry,
                        &mut solid_start,
                    );
                }
                self.append_strip(
                    [interval.start, top, interval.end, bottom],
                    true,
                    geometry,
                    &mut solid_start,
                );
                cursor = interval.end;
            }
            if cursor < right {
                self.append_strip(
                    [cursor, top, right, bottom],
                    false,
                    geometry,
                    &mut solid_start,
                );
            }
            if let Some(start) = solid_start {
                self.append_tile(geometry, [start, top, right, bottom], SOLID_COVERAGE);
            }
        }
        coverage_draw.range.solid_tile_count =
            geometry.tiles.len() as u32 - coverage_draw.range.tile_start;
        geometry.tiles.extend_from_slice(&self.boundary_tiles);
        coverage_draw.range.tile_count =
            geometry.tiles.len() as u32 - coverage_draw.range.tile_start;
        coverage_draw
    }
}
