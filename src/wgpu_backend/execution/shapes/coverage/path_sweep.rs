use super::scanline::{ClippedScanEdge, ScanEdge, Scanline};
use super::tiles::{tile_aligned_columns, TILE_SIZE};
use crate::core::math::CompensatedSum;
use crate::core::shape::boundary::PathBoundaryEdge;
use std::ops::Range;

#[derive(Default)]
struct ActiveAreaSum {
    signed_area: CompensatedSum,
    contribution_count: usize,
}

impl ActiveAreaSum {
    fn update(&mut self, signed_area: f64, is_start: bool) {
        if is_start {
            self.contribution_count += 1;
            self.signed_area.add(signed_area);
        } else {
            self.contribution_count -= 1;
            if self.contribution_count == 0 {
                self.signed_area = CompensatedSum::default();
            } else {
                self.signed_area.add(-signed_area);
            }
        }
    }
}

#[derive(Default)]
struct VerticalColumn {
    change: ActiveAreaSum,
    active_index: usize,
}

#[derive(Default)]
struct VerticalEdgeContributions {
    initial_area: ActiveAreaSum,
    columns: Vec<VerticalColumn>,
    active_columns: Vec<usize>,
}

impl VerticalEdgeContributions {
    fn update_column(&mut self, offset: usize, signed_area: f64, is_start: bool) {
        if offset >= self.columns.len() {
            self.columns
                .resize_with(offset + 1, VerticalColumn::default);
        }
        let column = &mut self.columns[offset];
        let was_active = column.change.signed_area.total() != 0.0;
        column.change.update(signed_area, is_start);
        match (was_active, column.change.signed_area.total() != 0.0) {
            (false, true) => {
                column.active_index = self.active_columns.len();
                self.active_columns.push(offset);
            }
            (true, false) => {
                let index = column.active_index;
                self.active_columns.swap_remove(index);
                if let Some(&moved) = self.active_columns.get(index) {
                    self.columns[moved].active_index = index;
                }
            }
            _ => {}
        }
    }

    fn update(
        &mut self,
        x: f32,
        signed_winding_height: f64,
        is_start: bool,
        left: u32,
        right: u32,
    ) {
        if x <= left as f32 {
            return;
        }
        self.initial_area.update(signed_winding_height, is_start);
        if x >= right as f32 {
            return;
        }
        let column = x as u32;
        let fraction = x as f64 - column as f64;
        let offset = (column - left) as usize;
        self.update_column(offset, (fraction - 1.0) * signed_winding_height, is_start);
        if fraction != 0.0 && column + 1 < right {
            self.update_column(offset + 1, -fraction * signed_winding_height, is_start);
        }
    }

    fn add_to(&self, scanline: &mut Scanline) {
        scanline.add_vertical_changes(
            self.initial_area.signed_area.total(),
            self.active_columns
                .iter()
                .map(|&offset| (offset, self.columns[offset].change.signed_area.total())),
        );
    }
}

#[derive(Default)]
struct RowEdgeBuckets {
    rows: Vec<Vec<ClippedScanEdge>>,
    populated_rows: Range<u32>,
}

impl RowEdgeBuckets {
    fn prepare(&mut self, height: u32) {
        // Bands consume their row buckets as they advance.
        self.populated_rows = height..0;
    }

    fn add(&mut self, edge: &ScanEdge, rows: Range<u32>) {
        if self.rows.len() < rows.end as usize {
            self.rows.resize_with(rows.end as usize, Vec::new);
        }
        self.populated_rows.start = self.populated_rows.start.min(rows.start);
        self.populated_rows.end = self.populated_rows.end.max(rows.end);
        for row in rows {
            self.rows[row as usize].push(edge.clip_to_rows(row as f32, (row + 1) as f32));
        }
    }

    fn clear(&mut self) {
        let rows = self.populated_rows.start as usize..self.populated_rows.end as usize;
        if let Some(rows) = self.rows.get_mut(rows) {
            for edges in rows {
                edges.clear();
            }
        }
    }

    fn rows_in(&self, rows: Range<u32>) -> &[Vec<ClippedScanEdge>] {
        let end = (rows.end as usize).min(self.rows.len());
        self.rows.get(rows.start as usize..end).unwrap_or_default()
    }

    fn first_populated_row(&self, rows: Range<u32>) -> Option<u32> {
        let start = rows.start.max(self.populated_rows.start);
        let end = rows.end.min(self.populated_rows.end);
        self.rows_in(start..end)
            .iter()
            .position(|edges| !edges.is_empty())
            .map(|offset| start + offset as u32)
    }
}

#[derive(Clone, Copy)]
enum EdgeUpdate {
    Vertical {
        x: f32,
        signed_winding_height: f64,
        is_start: bool,
    },
    StartSlopedEdge(usize),
    EndSlopedEdge(usize),
}

struct RowEvent {
    row: u32,
    update: EdgeUpdate,
}

struct SlopedEdge {
    edge: ScanEdge,
    active_index: usize,
}

#[derive(Default)]
pub(super) struct PathSweep {
    row_edge_buckets: RowEdgeBuckets,
    events: Vec<RowEvent>,
    next_event_index: usize,
    vertical_contributions: VerticalEdgeContributions,
    sloped_edges: Vec<SlopedEdge>,
    active_sloped_edges: Vec<usize>,
}

impl PathSweep {
    pub(super) fn trim_row_buckets(&mut self, height: u32) {
        self.row_edge_buckets.rows.truncate(height as usize);
    }

    fn add_vertical_spans(&mut self, edge: &ScanEdge, x: f32, rows: Range<u32>) {
        let first_full_row = (edge.top.ceil() as u32).clamp(rows.start, rows.end);
        let end_full_row = (edge.bottom.floor() as u32).clamp(first_full_row, rows.end);
        // The first and last rows can have a different winding height from the full rows.
        for span in [
            rows.start..first_full_row,
            first_full_row..end_full_row,
            end_full_row..rows.end,
        ] {
            if span.is_empty() {
                continue;
            }
            let signed_winding_height = edge
                .clip_to_rows(span.start as f32, (span.start + 1) as f32)
                .signed_winding_height;
            for (row, is_start) in [(span.start, true), (span.end, false)] {
                self.events.push(RowEvent {
                    row,
                    update: EdgeUpdate::Vertical {
                        x,
                        signed_winding_height,
                        is_start,
                    },
                });
            }
        }
    }

    /// If this returns `Some([left, top, right, bottom])`, cover `top..bottom`
    /// with `TILE_SIZE`-row bands, clipping the last band at `bottom`. Pass
    /// `bottom` as `path_bottom` to `prepare_band`. Start the next band at
    /// the returned repeat limit if present, or at the current band's bottom.
    /// The next `prepare` relies on these calls clearing populated row buckets.
    ///
    /// Use the returned `left` and `right` for every band. Call `finish` with
    /// the full bounds to close remaining edge events before the next sweep.
    /// Fixed horizontal bounds let vertical edge start and end events update
    /// the same columns.
    ///
    /// `None` leaves the sweep ready for reuse.
    pub(super) fn prepare(
        &mut self,
        edges: &[PathBoundaryEdge],
        size: (u32, u32),
    ) -> Option<[u32; 4]> {
        self.events.clear();
        self.sloped_edges.clear();
        self.row_edge_buckets.prepare(size.1);
        self.next_event_index = 0;
        let mut minimum = [f32::INFINITY; 2];
        let mut maximum = [f32::NEG_INFINITY; 2];
        for edge in edges {
            for [x, y] in [edge.start, edge.end] {
                minimum = [minimum[0].min(x), minimum[1].min(y)];
                maximum = [maximum[0].max(x), maximum[1].max(y)];
            }
            let Some(scan_edge) = ScanEdge::new(edge.start, edge.end, edge.winding) else {
                continue;
            };
            let first_row = scan_edge.top.floor() as u32;
            let end_row = scan_edge.bottom.ceil().min(size.1 as f32) as u32;
            if first_row >= end_row {
                continue;
            }
            let is_vertical = scan_edge.is_vertical();
            let maximum_bucketed_rows = if is_vertical {
                TILE_SIZE
            } else {
                4 * TILE_SIZE
            };
            // Bucketing short spans avoids sorting a pair of events for each edge.
            if end_row - first_row <= maximum_bucketed_rows {
                self.row_edge_buckets.add(&scan_edge, first_row..end_row);
                continue;
            }
            if is_vertical {
                if edge.start[0] > 0.0 {
                    self.add_vertical_spans(&scan_edge, edge.start[0], first_row..end_row);
                }
                continue;
            }
            // Outside the viewport the visible contribution is constant.
            let edge_in_visible_rows = scan_edge.clip_to_rows(first_row as f32, end_row as f32);
            if edge_in_visible_rows.maximum_x <= 0.0 {
                continue;
            }
            if edge_in_visible_rows.minimum_x == edge_in_visible_rows.maximum_x
                || edge_in_visible_rows.minimum_x >= size.0 as f32
            {
                self.add_vertical_spans(
                    &scan_edge,
                    edge_in_visible_rows.minimum_x,
                    first_row..end_row,
                );
                continue;
            }
            let index = self.sloped_edges.len();
            self.sloped_edges.push(SlopedEdge {
                edge: scan_edge,
                active_index: 0,
            });
            self.events.extend([
                RowEvent {
                    row: first_row,
                    update: EdgeUpdate::StartSlopedEdge(index),
                },
                RowEvent {
                    row: end_row,
                    update: EdgeUpdate::EndSlopedEdge(index),
                },
            ]);
        }
        let left = minimum[0].floor().clamp(0.0, size.0 as f32) as u32;
        let top = minimum[1].floor().clamp(0.0, size.1 as f32) as u32;
        let right = maximum[0].ceil().clamp(0.0, size.0 as f32) as u32;
        let bottom = maximum[1].ceil().clamp(0.0, size.1 as f32) as u32;
        if left >= right || top >= bottom {
            self.row_edge_buckets.clear();
            return None;
        }
        self.events.sort_unstable_by_key(|event| event.row);
        Some([left, top, right, bottom])
    }

    fn advance(&mut self, row: u32, left: u32, right: u32) -> Option<u32> {
        while let Some(event) = self.events.get(self.next_event_index) {
            if event.row > row {
                return Some(event.row);
            }
            match event.update {
                EdgeUpdate::Vertical {
                    x,
                    signed_winding_height,
                    is_start,
                } => self.vertical_contributions.update(
                    x,
                    signed_winding_height,
                    is_start,
                    left,
                    right,
                ),
                EdgeUpdate::StartSlopedEdge(index) => {
                    self.sloped_edges[index].active_index = self.active_sloped_edges.len();
                    self.active_sloped_edges.push(index);
                }
                EdgeUpdate::EndSlopedEdge(index) => {
                    let active_index = self.sloped_edges[index].active_index;
                    self.active_sloped_edges.swap_remove(active_index);
                    if let Some(&moved) = self.active_sloped_edges.get(active_index) {
                        self.sloped_edges[moved].active_index = active_index;
                    }
                }
            }
            self.next_event_index += 1;
        }
        None
    }

    fn collect_tile_intervals(&self, bounds: [u32; 4], intervals: &mut Vec<Range<u32>>) {
        let [left, top, right, bottom] = bounds;
        intervals.clear();
        if right - left <= TILE_SIZE {
            intervals.push(left..right);
            return;
        }
        let sloped_interval = |index: usize| {
            let edge = self.sloped_edges[index]
                .edge
                .clip_to_rows(top as f32, bottom as f32);
            tile_aligned_columns(edge.minimum_x, edge.maximum_x, left, right)
        };
        for edges in self.row_edge_buckets.rows_in(top..bottom) {
            for edge in edges {
                let Some(interval) =
                    tile_aligned_columns(edge.minimum_x, edge.maximum_x, left, right)
                else {
                    continue;
                };
                if interval == (left..right) {
                    intervals.clear();
                    intervals.push(interval);
                    return;
                }
                intervals.push(interval);
            }
        }
        intervals.extend(
            self.vertical_contributions
                .active_columns
                .iter()
                .map(|&offset| {
                    let start = left + (offset as u32 / TILE_SIZE) * TILE_SIZE;
                    start..start.saturating_add(TILE_SIZE).min(right)
                }),
        );
        intervals.extend(
            self.active_sloped_edges
                .iter()
                .filter_map(|&index| sloped_interval(index)),
        );
        intervals.extend(
            self.events[self.next_event_index..]
                .iter()
                .take_while(|event| event.row < bottom)
                .filter_map(|event| match event.update {
                    // A later event can uncover a previously cancelled change.
                    EdgeUpdate::Vertical { x, .. } => tile_aligned_columns(x, x, left, right),
                    EdgeUpdate::StartSlopedEdge(index) => sloped_interval(index),
                    EdgeUpdate::EndSlopedEdge(_) => None,
                }),
        );
        intervals.sort_unstable_by_key(|interval| interval.start);
        let mut count = 0;
        for index in 0..intervals.len() {
            let interval = intervals[index].clone();
            if count > 0 && intervals[count - 1].end >= interval.start {
                intervals[count - 1].end = intervals[count - 1].end.max(interval.end);
            } else {
                intervals[count] = interval;
                count += 1;
            }
        }
        intervals.truncate(count);
    }

    /// Returns `Some(end)` if the first scanline can be reused from the band's top
    /// up to the exclusive row `end`.
    pub(super) fn prepare_band(
        &mut self,
        bounds: [u32; 4],
        path_bottom: u32,
        intervals: &mut Vec<Range<u32>>,
        scanlines: &mut [Scanline],
    ) -> Option<u32> {
        let [left, top, right, bottom] = bounds;
        let mut next_event_row = self
            .events
            .get(self.next_event_index)
            .map_or(path_bottom, |event| event.row);
        if next_event_row <= top {
            next_event_row = self.advance(top, left, right).unwrap_or(path_bottom);
        }
        let mut repeat_limit_row = next_event_row;
        if self.active_sloped_edges.is_empty() && repeat_limit_row >= bottom {
            if let Some(populated_row) = self
                .row_edge_buckets
                .first_populated_row(top..repeat_limit_row)
            {
                repeat_limit_row = populated_row;
            }
        }
        let has_identical_rows = self.active_sloped_edges.is_empty() && repeat_limit_row >= bottom;
        let row_count = if has_identical_rows {
            1
        } else {
            scanlines.len()
        };
        self.collect_tile_intervals(bounds, intervals);
        for (index, scanline) in scanlines[..row_count].iter_mut().enumerate() {
            let row = top + index as u32;
            if row >= next_event_row {
                next_event_row = self.advance(row, left, right).unwrap_or(path_bottom);
            }
            let row_bucket = self.row_edge_buckets.rows.get_mut(row as usize);
            let row_edges = row_bucket.as_deref().map_or(&[][..], Vec::as_slice);
            if self.active_sloped_edges.is_empty() {
                scanline.prepare(row_edges.iter().copied(), left, right, intervals);
            } else {
                let sloped_edges = self.active_sloped_edges.iter().map(|&index| {
                    self.sloped_edges[index]
                        .edge
                        .clip_to_rows(row as f32, (row + 1) as f32)
                });
                scanline.prepare(
                    row_edges.iter().copied().chain(sloped_edges),
                    left,
                    right,
                    intervals,
                );
            }
            if let Some(edges) = row_bucket {
                edges.clear();
            }
            if self.vertical_contributions.initial_area.contribution_count != 0 {
                self.vertical_contributions.add_to(scanline);
            }
        }
        has_identical_rows.then(|| {
            // Keep mixed bands on the tile grid so repeats do not add boundary tiles.
            if repeat_limit_row == path_bottom {
                repeat_limit_row
            } else {
                top + (repeat_limit_row - top) / TILE_SIZE * TILE_SIZE
            }
        })
    }

    pub(super) fn finish(&mut self, [left, _, right, bottom]: [u32; 4]) {
        // Close the remaining contributions so column storage can be reused.
        self.advance(bottom, left, right);
    }
}
