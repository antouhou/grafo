use super::projection::Projection;
use super::scanline::{RowEdge, ScanEdge, Scanline};
use super::tiles::tile_aligned_columns;
use crate::core::cache::CachedTessellation;
use crate::core::shape::boundary::PathBoundaryEdge;
use std::ops::Range;

#[derive(Default)]
struct BoundaryChain {
    edges: Vec<ScanEdge>,
    cursor: usize,
}

impl BoundaryChain {
    fn prepare(&mut self, edges: &[PathBoundaryEdge], projection: &Projection) {
        self.edges.clear();
        self.edges.extend(edges.iter().filter_map(|edge| {
            ScanEdge::new(
                projection.project(edge.start),
                projection.project(edge.end),
                edge.winding * projection.boundary_orientation?,
            )
        }));
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

    fn intersecting(&self, bottom: f32) -> impl Iterator<Item = &ScanEdge> {
        self.edges[self.cursor..]
            .iter()
            .take_while(move |edge| edge.top < bottom)
    }

    fn interval(&mut self, bounds: [u32; 4]) -> Option<Range<u32>> {
        let [left, top, right, bottom] = bounds;
        self.advance(top as f32);
        let mut minimum_x = f32::INFINITY;
        let mut maximum_x = f32::NEG_INFINITY;
        for edge in self.intersecting(bottom as f32) {
            let row = edge.clip(top as f32, bottom as f32);
            minimum_x = minimum_x.min(row.minimum_x);
            maximum_x = maximum_x.max(row.maximum_x);
        }
        tile_aligned_columns(minimum_x, maximum_x, left, right)
    }

    fn row(&mut self, top: u32) -> impl Iterator<Item = RowEdge> + '_ {
        let top = top as f32;
        self.advance(top);
        self.intersecting(top + 1.0)
            .map(move |edge| edge.clip(top, top + 1.0))
    }

    fn is_vertical_through(&self, top: u32, bottom: u32) -> bool {
        self.edges.get(self.cursor).is_some_and(|edge| {
            edge.is_vertical() && edge.top <= top as f32 && edge.bottom >= bottom as f32
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
