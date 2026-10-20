use super::geometry::TILE_SIZE;
use super::projection::Projection;
use super::scanline::{RowEdge, Scanline};
use crate::core::cache::CachedTessellation;
use crate::core::shape::boundary::PathBoundaryEdge;
use std::ops::Range;

struct ChainEdge {
    start: [f32; 2],
    top: f32,
    bottom: f32,
    slope: f64,
    winding: f64,
}

impl ChainEdge {
    fn new(edge: &PathBoundaryEdge, projection: &Projection) -> Option<Self> {
        let start = projection.project(edge.start);
        let end = projection.project(edge.end);
        let height = end[1] - start[1];
        if height == 0.0 {
            return None;
        }
        Some(Self {
            start,
            top: start[1].min(end[1]),
            bottom: start[1].max(end[1]),
            slope: (end[0] as f64 - start[0] as f64) / height as f64,
            winding: (edge.winding * projection.boundary_orientation?) as f64
                * (height as f64).signum(),
        })
    }

    fn clip(&self, top: f32, bottom: f32) -> RowEdge {
        let low = self.top.max(top) as f64;
        let high = self.bottom.min(bottom) as f64;
        let x = |y| (self.start[0] as f64 + (y - self.start[1] as f64) * self.slope) as f32;
        RowEdge {
            x0: x(low),
            x1: x(high),
            signed_winding_height: (high - low) * self.winding,
        }
    }
}

#[derive(Default)]
struct BoundaryChain {
    edges: Vec<ChainEdge>,
    cursor: usize,
}

impl BoundaryChain {
    fn prepare(&mut self, edges: &[PathBoundaryEdge], projection: &Projection) {
        self.edges.clear();
        self.edges.extend(
            edges
                .iter()
                .filter_map(|edge| ChainEdge::new(edge, projection)),
        );
        if let (Some(first), Some(last)) = (self.edges.first(), self.edges.last()) {
            if first.top > last.top {
                self.edges.reverse();
            }
        }
        self.cursor = 0;
    }

    fn advance(&mut self, top: f32) {
        while self.cursor < self.edges.len() && self.edges[self.cursor].bottom <= top {
            self.cursor += 1;
        }
    }

    fn intersecting(&self, bottom: f32) -> impl Iterator<Item = &ChainEdge> {
        self.edges[self.cursor..]
            .iter()
            .take_while(move |edge| edge.top < bottom)
    }

    fn interval(&mut self, bounds: [u32; 4]) -> Option<Range<u32>> {
        let [left, top, right, bottom] = bounds;
        self.advance(top as f32);
        let mut minimum = f32::INFINITY;
        let mut maximum = f32::NEG_INFINITY;
        for edge in self.intersecting(bottom as f32) {
            let row = edge.clip(top as f32, bottom as f32);
            minimum = minimum.min(row.x0).min(row.x1);
            maximum = maximum.max(row.x0).max(row.x1);
        }
        if maximum <= left as f32 || minimum >= right as f32 {
            return None;
        }
        let first = (minimum.max(left as f32) as u32 - left) / TILE_SIZE;
        let last = (maximum.ceil().min(right as f32) as u32 - left) / TILE_SIZE;
        Some(left + first * TILE_SIZE..(left + (last + 1) * TILE_SIZE).min(right))
    }

    fn row(&mut self, top: u32) -> impl Iterator<Item = RowEdge> + '_ {
        let top = top as f32;
        self.advance(top);
        self.intersecting(top + 1.0)
            .map(move |edge| edge.clip(top, top + 1.0))
    }

    fn is_vertical_through(&self, top: u32, bottom: u32) -> bool {
        self.edges.get(self.cursor).is_some_and(|edge| {
            edge.slope == 0.0 && edge.top <= top as f32 && edge.bottom >= bottom as f32
        })
    }
}

#[derive(Default)]
pub(super) struct RoundedRectSweep {
    chains: [BoundaryChain; 2],
}

impl RoundedRectSweep {
    pub(super) fn prepare(
        &mut self,
        tessellation: &CachedTessellation,
        projection: &Projection,
        size: (u32, u32),
    ) -> Option<[u32; 4]> {
        let boundary = tessellation.rounded_rect_boundary.as_ref()?;
        if !projection.preserves_axis_alignment() {
            return None;
        }
        let corners = tessellation
            .local_bounds
            .map(|point| projection.project([point.0, point.1]));
        if !corners.into_iter().flatten().all(f32::is_finite) {
            return None;
        }
        for (chain, range) in self
            .chains
            .iter_mut()
            .zip([boundary.left.clone(), boundary.right.clone()])
        {
            chain.prepare(&tessellation.boundary_edges[range], projection);
        }
        Some([
            corners[0][0]
                .min(corners[1][0])
                .floor()
                .clamp(0.0, size.0 as f32) as u32,
            corners[0][1]
                .min(corners[1][1])
                .floor()
                .clamp(0.0, size.1 as f32) as u32,
            corners[0][0]
                .max(corners[1][0])
                .ceil()
                .clamp(0.0, size.0 as f32) as u32,
            corners[0][1]
                .max(corners[1][1])
                .ceil()
                .clamp(0.0, size.1 as f32) as u32,
        ])
    }

    pub(super) fn prepare_band(
        &mut self,
        bounds: [u32; 4],
        intervals: &mut Vec<Range<u32>>,
        scanlines: &mut [Scanline],
    ) -> bool {
        intervals.clear();
        intervals.extend(
            self.chains
                .iter_mut()
                .filter_map(|chain| chain.interval(bounds)),
        );
        if intervals.len() == 2 {
            if intervals[0].start > intervals[1].start {
                intervals.swap(0, 1);
            }
            if intervals[0].end >= intervals[1].start {
                intervals[0].end = intervals[0].end.max(intervals[1].end);
                intervals.pop();
            }
        }
        let [left, top, right, bottom] = bounds;
        let has_identical_rows = self
            .chains
            .iter()
            .all(|chain| chain.is_vertical_through(top, bottom));
        let row_count = if has_identical_rows {
            1
        } else {
            scanlines.len()
        };
        let [first, second] = &mut self.chains;
        for (index, scanline) in scanlines[..row_count].iter_mut().enumerate() {
            let row = top + index as u32;
            scanline.prepare(
                first.row(row).chain(second.row(row)),
                left,
                right,
                intervals,
            );
        }
        has_identical_rows
    }
}
