use super::path_sweep::PathSweep;
use super::projection::Projection;
use super::rectangles;
use super::rounded_rect::RoundedRectSweep;
use super::scanline::Scanline;
use super::tiles::{
    CoverageDrawMetadata, CoverageDrawRange, CoverageGeometry, CoverageTile, TILE_SIZE,
};
use crate::core::cache::CachedTessellation;
use crate::core::shape::boundary::{PathBoundaryBuilder, PathBoundaryEdge};
use crate::core::vertex::InstanceTransform;
use crate::core::UnsignedPhysicalRect;
use std::ops::Range;

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
    path_sweep: PathSweep,
    scanlines: [Scanline; TILE_SIZE as usize],
    tile_values: [[f32; TILE_SIZE as usize]; TILE_SIZE as usize],
    rounded_rect: RoundedRectSweep,
    has_identical_rows: bool,
}

impl CoverageBuilder {
    pub(super) fn trim_row_buckets(&mut self, height: u32) {
        self.path_sweep.trim_row_buckets(height);
    }

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
        let values = self.tile_values[..row_count]
            .iter()
            .flat_map(|row| &row[..width]);
        if values.clone().all(|&value| value == 1.0) {
            TileCoverage::Solid
        } else if values.clone().all(|&value| value == 0.0) {
            TileCoverage::Empty
        } else {
            if width < TILE_SIZE as usize {
                for row in &mut self.tile_values[..row_count] {
                    row[width..].fill(0.0);
                }
            }
            if self.has_identical_rows {
                let row = self.tile_values[0];
                self.tile_values[1..height].fill(row);
            }
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

    fn append_band(&mut self, bounds: [u32; 4], geometry: &mut CoverageGeometry, repeat_end: u32) {
        let [left, top, right, bottom] = bounds;
        let solid_tile_start = geometry.tiles.len();
        let boundary_tile_start = self.boundary_tiles.len();
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

        if repeat_end <= bottom {
            return;
        }
        for tile in &mut geometry.tiles[solid_tile_start..] {
            tile.max[1] = repeat_end;
        }
        let boundary_tiles = boundary_tile_start..self.boundary_tiles.len();
        if boundary_tiles.is_empty() {
            return;
        }
        // Repeated boundary tiles can reference the same rows of samples.
        for top in (bottom..repeat_end).step_by(TILE_SIZE as usize) {
            let bottom = top.saturating_add(TILE_SIZE).min(repeat_end);
            for index in boundary_tiles.clone() {
                let mut tile = self.boundary_tiles[index];
                tile.min[1] = top;
                tile.max[1] = bottom;
                self.boundary_tiles.push(tile);
            }
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
        let Some(projection) = Projection::new(transform, scale, local_bounds) else {
            return coverage_draw;
        };
        if tessellation.is_rect {
            if let Some(rectangle) = rectangles::build(local_bounds, &projection, size, geometry) {
                return rectangle;
            }
        }
        self.boundary_tiles.clear();
        let rounded_bounds = self.rounded_rect.prepare(tessellation, &projection, size);
        let bounds = rounded_bounds.or_else(|| {
            self.collect_boundary(tessellation, &projection);
            self.path_sweep.prepare(&self.boundary_edges, size)
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
        let mut band_top = top;
        while band_top < bottom {
            let band_bottom = band_top.saturating_add(TILE_SIZE).min(bottom);
            let band = [left, band_top, right, band_bottom];
            let scanlines = &mut self.scanlines[..(band_bottom - band_top) as usize];
            let repeat_end = if rounded_bounds.is_some() {
                self.has_identical_rows =
                    self.rounded_rect
                        .prepare_band(band, &mut self.tile_intervals, scanlines);
                band_bottom
            } else {
                let repeat_end =
                    self.path_sweep
                        .prepare_band(band, bottom, &mut self.tile_intervals, scanlines);
                self.has_identical_rows = repeat_end.is_some();
                repeat_end.unwrap_or(band_bottom)
            };
            self.append_band(band, geometry, repeat_end);
            band_top = repeat_end;
        }
        if rounded_bounds.is_none() {
            self.path_sweep.finish([left, top, right, bottom]);
        }
        coverage_draw.range.solid_tile_count =
            geometry.tiles.len() as u32 - coverage_draw.range.tile_start;
        geometry.tiles.extend_from_slice(&self.boundary_tiles);
        coverage_draw.range.tile_count =
            geometry.tiles.len() as u32 - coverage_draw.range.tile_start;
        coverage_draw
    }
}
