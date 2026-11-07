use super::scanline::RowEdge;
use super::winding_order::WindingOrder;
use crate::core::shape::boundary::PathBoundaryEdge;
use std::{cmp::Ordering, mem, ops::Range};

#[derive(Clone, Copy)]
pub(super) struct RowSegment {
    start_y: f64,
    pub(super) end_y: f64,
    start_x: f64,
    end_x: f64,
    slope: f64,
    pub(super) winding: i64,
}

impl RowSegment {
    pub(super) fn slope(self) -> f64 {
        self.slope
    }

    pub(super) fn x_at(self, y: f64) -> f64 {
        if y == self.end_y {
            self.end_x
        } else {
            self.slope().mul_add(y - self.start_y, self.start_x)
        }
    }
}

fn append_clipped_segment(
    row: &mut Vec<RowSegment>,
    start: [f64; 2],
    end: [f64; 2],
    winding: i64,
    width: f64,
) {
    let mut points = [[0.0; 2]; 4];
    points[0] = [start[0].clamp(0.0, width), start[1]];
    let mut point_count = 1;
    let boundaries = if start[0] < end[0] {
        [0.0, width]
    } else {
        [width, 0.0]
    };
    for x in boundaries {
        if x > start[0].min(end[0]) && x < start[0].max(end[0]) {
            let fraction = (x - start[0]) / (end[0] - start[0]);
            let y = (end[1] - start[1]).mul_add(fraction, start[1]);
            points[point_count] = [x, y];
            point_count += 1;
        }
    }
    points[point_count] = [end[0].clamp(0.0, width), end[1]];
    point_count += 1;
    for endpoints in points[..point_count].windows(2) {
        let [start, end] = [endpoints[0], endpoints[1]];
        if start[1] < end[1] {
            row.push(RowSegment {
                start_y: start[1],
                end_y: end[1],
                start_x: start[0],
                end_x: end[0],
                slope: (end[0] - start[0]) / (end[1] - start[1]),
                winding,
            });
        }
    }
}

#[derive(Clone, Copy)]
struct Endpoint {
    y: f64,
    x: f64,
    segment: usize,
    is_start: bool,
}

#[derive(Clone, Copy, Default)]
struct VisibleEdge {
    sign: i8,
    next_sign: i8,
    since: f64,
}

#[derive(Default)]
struct BoundedRowSweep {
    active: Vec<usize>,
    visible: Vec<VisibleEdge>,
}

impl BoundedRowSweep {
    fn build(&mut self, segments: &[RowSegment], output: &mut Vec<RowEdge>) -> bool {
        self.active.clear();
        self.visible.resize(segments.len(), VisibleEdge::default());
        self.visible.fill(VisibleEdge::default());
        let output_start = output.len();
        let mut y = segments
            .iter()
            .map(|segment| segment.start_y)
            .fold(1.0, f64::min);
        let last_y = segments
            .iter()
            .map(|segment| segment.end_y)
            .fold(0.0, f64::max);
        // A fixed number of full passes bounds setup cost before the event tree takes over.
        for _ in 0..16 {
            self.active.clear();
            for edge in &mut self.visible {
                edge.next_sign = 0;
            }
            let mut next_y = last_y;
            for (index, segment) in segments.iter().enumerate() {
                if segment.start_y > y {
                    next_y = next_y.min(segment.start_y);
                }
                if segment.start_y <= y && segment.end_y > y {
                    self.active.push(index);
                    next_y = next_y.min(segment.end_y);
                }
            }
            let active = &mut self.active;
            active.sort_unstable_by(|&first, &second| {
                let first_x = segments[first].x_at(y);
                let second_x = segments[second].x_at(y);
                (if first_x == second_x {
                    Ordering::Equal
                } else {
                    first_x.total_cmp(&second_x)
                })
                .then_with(|| segments[first].slope.total_cmp(&segments[second].slope))
                .then_with(|| first.cmp(&second))
            });
            for pair in active.windows(2) {
                let left = segments[pair[0]];
                let right = segments[pair[1]];
                let relative_slope = left.slope - right.slope;
                if relative_slope > 0.0 {
                    let crossing_y = y + (right.x_at(y) - left.x_at(y)) / relative_slope;
                    if crossing_y >= y {
                        next_y = next_y.min(crossing_y.max(y.next_up()));
                    }
                }
            }
            let mut winding = 0_i64;
            for &index in active.iter() {
                let before = winding;
                winding += segments[index].winding;
                self.visible[index].next_sign = if before == 0 && winding != 0 {
                    -1
                } else if before != 0 && winding == 0 {
                    1
                } else {
                    0
                };
            }
            for (index, segment) in segments.iter().enumerate() {
                let edge = &mut self.visible[index];
                if edge.next_sign == edge.sign {
                    continue;
                }
                if edge.sign != 0 && y > edge.since {
                    output.push(RowEdge {
                        x0: segment.x_at(edge.since) as f32,
                        x1: segment.x_at(y) as f32,
                        signed_winding_height: (y - edge.since) * f64::from(edge.sign),
                    });
                }
                edge.sign = edge.next_sign;
                edge.since = y;
            }
            if y == last_y {
                return true;
            }
            y = next_y;
        }
        output.truncate(output_start);
        false
    }
}

#[derive(Default)]
struct RowSweep {
    bounded: BoundedRowSweep,
    endpoints: Vec<Endpoint>,
    order: WindingOrder,
}

impl RowSweep {
    fn append_unchanging_row(segments: &mut [RowSegment], output: &mut Vec<RowEdge>) -> bool {
        let Some(first) = segments.first().copied() else {
            return true;
        };
        if !segments
            .iter()
            .all(|segment| segment.start_y == first.start_y && segment.end_y == first.end_y)
        {
            return false;
        }
        segments.sort_unstable_by(|first, second| {
            first
                .start_x
                .total_cmp(&second.start_x)
                .then_with(|| first.slope.total_cmp(&second.slope))
        });
        if segments
            .windows(2)
            .any(|pair| pair[0].end_x > pair[1].end_x)
        {
            return false;
        }
        let mut winding = 0_i64;
        for segment in segments {
            let was_inside = winding != 0;
            winding += segment.winding;
            let is_inside = winding != 0;
            if was_inside != is_inside {
                output.push(RowEdge {
                    x0: segment.start_x as f32,
                    x1: segment.end_x as f32,
                    signed_winding_height: (segment.end_y - segment.start_y)
                        * if is_inside { -1.0 } else { 1.0 },
                });
            }
        }
        true
    }

    #[inline(always)]
    fn try_build(&mut self, segments: &mut [RowSegment], output: &mut Vec<RowEdge>) -> bool {
        Self::append_unchanging_row(segments, output) || self.bounded.build(segments, output)
    }

    fn start(&mut self, segments: &[RowSegment], output: &mut Vec<RowEdge>) -> usize {
        self.endpoints.clear();
        self.order.clear(segments.len());
        for (index, segment) in segments.iter().enumerate() {
            self.endpoints.push(Endpoint {
                y: segment.start_y,
                x: segment.start_x,
                segment: index,
                is_start: true,
            });
            self.endpoints.push(Endpoint {
                y: segment.end_y,
                x: segment.end_x,
                segment: index,
                is_start: false,
            });
        }
        self.endpoints.sort_unstable_by(|first, second| {
            first
                .y
                .total_cmp(&second.y)
                .then_with(|| {
                    if first.x == second.x {
                        Ordering::Equal
                    } else {
                        first.x.total_cmp(&second.x)
                    }
                })
                .then_with(|| first.is_start.cmp(&second.is_start))
                .then_with(|| {
                    segments[first.segment]
                        .slope
                        .total_cmp(&segments[second.segment].slope)
                })
                .then_with(|| first.segment.cmp(&second.segment))
        });
        let first_y = self.endpoints[0].y;
        let cursor = self
            .endpoints
            .partition_point(|endpoint| endpoint.y == first_y);
        self.order.initialize(
            self.endpoints[..cursor]
                .iter()
                .map(|endpoint| endpoint.segment),
            first_y,
            segments,
            output,
        );
        cursor
    }

    fn sweep<const PAUSE_AT_CROSSING: bool>(
        &mut self,
        mut cursor: usize,
        segments: &[RowSegment],
        output: &mut Vec<RowEdge>,
    ) -> Option<usize> {
        while cursor < self.endpoints.len() {
            let y = self.endpoints[cursor].y;
            if let Some(crossing_y) = self.order.next_crossing() {
                if crossing_y <= y {
                    if PAUSE_AT_CROSSING {
                        return Some(cursor);
                    }
                    // Only the crossing pair changes its prefix winding.
                    self.order.cross(crossing_y, segments, output);
                    continue;
                }
            }
            if y == self.endpoints[self.endpoints.len() - 1].y {
                self.order.finish(y, segments, output);
                break;
            }
            let mut winding_delta = 0_i64;
            let mut previous_x = self.endpoints[cursor].x;
            while cursor < self.endpoints.len() && self.endpoints[cursor].y == y {
                let x = self.endpoints[cursor].x;
                // Horizontal contour sides change winding between different endpoint x values.
                if winding_delta != 0 {
                    self.order.refresh([previous_x, x], y, segments, output);
                }
                let mut affected_bounds = [x, x];
                while cursor < self.endpoints.len()
                    && self.endpoints[cursor].y == y
                    && self.endpoints[cursor].x == x
                {
                    let endpoint = self.endpoints[cursor];
                    let neighbors = if endpoint.is_start {
                        winding_delta += segments[endpoint.segment].winding;
                        self.order.insert(endpoint.segment, y, segments, output)
                    } else {
                        winding_delta -= segments[endpoint.segment].winding;
                        self.order.remove(endpoint.segment, y, segments, output)
                    };
                    affected_bounds[0] = affected_bounds[0].min(neighbors[0]);
                    affected_bounds[1] = affected_bounds[1].max(neighbors[1]);
                    cursor += 1;
                }
                // Interpolated crossings can lie just beside an exactly shared endpoint.
                self.order.refresh(affected_bounds, y, segments, output);
                previous_x = x;
            }
            if winding_delta != 0 {
                self.order
                    .refresh([previous_x, f64::INFINITY], y, segments, output);
            }
        }
        None
    }
}

struct ProjectedEdge {
    start: [f64; 2],
    end: [f64; 2],
    slope: f64,
    intercept: f64,
    winding: i64,
    first_row: usize,
    last_row: usize,
}

impl ProjectedEdge {
    fn x_at(&self, y: f64) -> f64 {
        if y == self.start[1] {
            self.start[0]
        } else if y == self.end[1] {
            self.end[0]
        } else {
            self.slope.mul_add(y, self.intercept)
        }
    }
}

/// Converts overlapping contours into nonzero-filled row spans before integrating coverage.
#[derive(Default)]
pub(super) struct WindingCoverage {
    edges: Vec<ProjectedEdge>,
    active_edges: Vec<usize>,
    segments: Vec<RowSegment>,
    populated_rows: Range<usize>,
    sweep: RowSweep,
}

impl WindingCoverage {
    pub(super) fn populated_rows(&self) -> Range<usize> {
        self.populated_rows.clone()
    }

    #[inline(always)]
    fn collect_edges(&mut self, edges: &[PathBoundaryEdge], size: (u32, u32)) -> Option<[u32; 4]> {
        self.edges.clear();
        self.active_edges.clear();
        self.populated_rows = size.1 as usize..0;
        let mut bounds = [
            f32::INFINITY,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
        ];
        for edge in edges {
            for point in [edge.start, edge.end] {
                bounds[0] = bounds[0].min(point[0]);
                bounds[1] = bounds[1].min(point[1]);
                bounds[2] = bounds[2].max(point[0]);
                bounds[3] = bounds[3].max(point[1]);
            }
            let [mut start, mut end] = [edge.start.map(f64::from), edge.end.map(f64::from)];
            if start[1] == end[1] {
                continue;
            }
            let winding = if start[1] < end[1] {
                i64::from(edge.winding)
            } else {
                mem::swap(&mut start, &mut end);
                -i64::from(edge.winding)
            };
            let first_row = start[1].floor().clamp(0.0, f64::from(size.1)) as usize;
            let last_row = end[1].ceil().clamp(0.0, f64::from(size.1)) as usize;
            if first_row >= last_row {
                continue;
            }
            self.populated_rows.start = self.populated_rows.start.min(first_row);
            self.populated_rows.end = self.populated_rows.end.max(last_row);
            let slope = (end[0] - start[0]) / (end[1] - start[1]);
            self.edges.push(ProjectedEdge {
                start,
                end,
                slope,
                intercept: (-slope).mul_add(start[1], start[0]),
                winding,
                first_row,
                last_row,
            });
        }
        self.edges.sort_unstable_by_key(|edge| edge.first_row);
        let bounds = [
            bounds[0].floor().clamp(0.0, size.0 as f32) as u32,
            bounds[1].floor().clamp(0.0, size.1 as f32) as u32,
            bounds[2].ceil().clamp(0.0, size.0 as f32) as u32,
            bounds[3].ceil().clamp(0.0, size.1 as f32) as u32,
        ];
        (bounds[0] < bounds[2] && bounds[1] < bounds[3]).then_some(bounds)
    }

    #[inline(always)]
    fn collect_segments(&mut self, row: usize, width: u32, next_edge: &mut usize) {
        self.active_edges
            .retain(|&index| self.edges[index].last_row > row);
        while *next_edge < self.edges.len() && self.edges[*next_edge].first_row <= row {
            self.active_edges.push(*next_edge);
            *next_edge += 1;
        }
        self.segments.clear();
        for &index in &self.active_edges {
            let edge = &self.edges[index];
            let low = edge.start[1].max(row as f64);
            let high = edge.end[1].min(row as f64 + 1.0);
            append_clipped_segment(
                &mut self.segments,
                [edge.x_at(low), low - row as f64],
                [edge.x_at(high), high - row as f64],
                edge.winding,
                f64::from(width),
            );
        }
    }

    #[cold]
    fn restart_row(&mut self, edges: &[PathBoundaryEdge], size: (u32, u32), row: usize) -> usize {
        let populated_rows = self.populated_rows.clone();
        self.collect_edges(edges, size);
        self.populated_rows = populated_rows;
        let mut next_edge = self.edges.partition_point(|edge| edge.first_row <= row);
        self.active_edges
            .extend((0..next_edge).filter(|&index| self.edges[index].last_row > row));
        self.collect_segments(row, size.0, &mut next_edge);
        next_edge
    }

    pub(super) fn build(
        &mut self,
        edges: &mut Vec<PathBoundaryEdge>,
        size: (u32, u32),
        output: &mut [Vec<RowEdge>],
        cull_covered_triangles: impl FnOnce(&mut Vec<PathBoundaryEdge>, usize) -> bool,
    ) -> Option<[u32; 4]> {
        let bounds = self.collect_edges(edges, size);
        let mut cull_covered_triangles = Some(cull_covered_triangles);
        let mut next_edge = 0;
        for row in self.populated_rows.clone() {
            self.collect_segments(row, size.0, &mut next_edge);
            let row_edges = &mut output[row];
            if self.sweep.try_build(&mut self.segments, row_edges) {
                continue;
            }
            let row_start = row_edges.len();
            let mut cursor = self.sweep.start(&self.segments, row_edges);
            if self.segments.len() >= 64 && cull_covered_triangles.is_some() {
                let Some(crossing_cursor) =
                    self.sweep.sweep::<true>(cursor, &self.segments, row_edges)
                else {
                    continue;
                };
                cursor = crossing_cursor;
                if cull_covered_triangles
                    .take()
                    .is_some_and(|cull| cull(edges, self.segments.len()))
                {
                    // The proof preserves coverage, including rows already swept.
                    row_edges.truncate(row_start);
                    next_edge = self.restart_row(edges, size, row);
                    if self.sweep.try_build(&mut self.segments, row_edges) {
                        continue;
                    }
                    cursor = self.sweep.start(&self.segments, row_edges);
                }
            }
            self.sweep.sweep::<false>(cursor, &self.segments, row_edges);
        }
        bounds
    }
}
