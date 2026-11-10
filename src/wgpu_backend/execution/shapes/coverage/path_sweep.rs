use super::scanline::{CoverageSum, RowEdge, ScanEdge, Scanline};
use super::tiles::{tile_interval, TILE_SIZE};
use crate::core::shape::boundary::PathBoundaryEdge;
use std::ops::Range;

#[derive(Default)]
struct VerticalChange {
    value: CoverageSum,
    contribution_count: usize,
}

impl VerticalChange {
    fn update(&mut self, value: f64, is_start: bool) {
        if is_start {
            self.contribution_count += 1;
            self.value.add(value);
        } else {
            self.contribution_count -= 1;
            if self.contribution_count == 0 {
                self.value = CoverageSum::default();
            } else {
                self.value.add(-value);
            }
        }
    }
}

#[derive(Default)]
struct VerticalColumn {
    change: VerticalChange,
    active_index: usize,
}

#[derive(Default)]
struct VerticalScanline {
    initial_area: VerticalChange,
    columns: Vec<VerticalColumn>,
    active_columns: Vec<usize>,
}

impl VerticalScanline {
    fn update_column(&mut self, offset: usize, value: f64, is_start: bool) {
        if offset >= self.columns.len() {
            self.columns
                .resize_with(offset + 1, VerticalColumn::default);
        }
        let column = &mut self.columns[offset];
        let was_active = column.change.value.total() != 0.0;
        column.change.update(value, is_start);
        match (was_active, column.change.value.total() != 0.0) {
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

    fn update(&mut self, x: f32, height: f64, is_start: bool, left: u32, right: u32) {
        if x <= left as f32 {
            return;
        }
        self.initial_area.update(height, is_start);
        if x >= right as f32 {
            return;
        }
        let column = x as u32;
        let fraction = x as f64 - column as f64;
        let offset = (column - left) as usize;
        self.update_column(offset, (fraction - 1.0) * height, is_start);
        if fraction != 0.0 && column + 1 < right {
            self.update_column(offset + 1, -fraction * height, is_start);
        }
    }

    fn add_to(&self, scanline: &mut Scanline) {
        scanline.add_vertical_changes(
            self.initial_area.value.total(),
            self.active_columns
                .iter()
                .map(|&offset| (offset, self.columns[offset].change.value.total())),
        );
    }
}

#[derive(Default)]
struct ShortEdges {
    rows: Vec<Vec<RowEdge>>,
    populated_rows: Range<u32>,
}

impl ShortEdges {
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
            self.rows[row as usize].push(edge.clip(row as f32, (row + 1) as f32));
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

    fn rows_in(&self, rows: Range<u32>) -> &[Vec<RowEdge>] {
        let end = (rows.end as usize).min(self.rows.len());
        self.rows.get(rows.start as usize..end).unwrap_or_default()
    }

    fn first_row(&self, rows: Range<u32>) -> Option<u32> {
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
        signed_height: f64,
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
    short_edges: ShortEdges,
    events: Vec<RowEvent>,
    next_event: usize,
    vertical: VerticalScanline,
    sloped_edges: Vec<SlopedEdge>,
    active_sloped_edges: Vec<usize>,
}

impl PathSweep {
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
            let signed_height = edge
                .clip(span.start as f32, (span.start + 1) as f32)
                .signed_winding_height;
            for (row, is_start) in [(span.start, true), (span.end, false)] {
                self.events.push(RowEvent {
                    row,
                    update: EdgeUpdate::Vertical {
                        x,
                        signed_height,
                        is_start,
                    },
                });
            }
        }
    }

    pub(super) fn prepare(
        &mut self,
        edges: &[PathBoundaryEdge],
        size: (u32, u32),
    ) -> Option<[u32; 4]> {
        self.events.clear();
        self.sloped_edges.clear();
        self.short_edges.prepare(size.1);
        self.next_event = 0;
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
            let direct_limit = if is_vertical {
                TILE_SIZE
            } else {
                4 * TILE_SIZE
            };
            // Bucketing short spans avoids sorting a pair of events for each edge.
            if end_row - first_row <= direct_limit {
                self.short_edges.add(&scan_edge, first_row..end_row);
                continue;
            }
            if is_vertical {
                if edge.start[0] > 0.0 {
                    self.add_vertical_spans(&scan_edge, edge.start[0], first_row..end_row);
                }
                continue;
            }
            // Outside the viewport the visible contribution is constant.
            let visible = scan_edge.clip(first_row as f32, end_row as f32);
            if visible.maximum <= 0.0 {
                continue;
            }
            if visible.minimum == visible.maximum || visible.minimum >= size.0 as f32 {
                self.add_vertical_spans(&scan_edge, visible.minimum, first_row..end_row);
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
            self.short_edges.clear();
            return None;
        }
        self.events.sort_unstable_by_key(|event| event.row);
        Some([left, top, right, bottom])
    }

    fn advance(&mut self, row: u32, left: u32, right: u32) -> Option<u32> {
        while let Some(event) = self.events.get(self.next_event) {
            if event.row > row {
                return Some(event.row);
            }
            match event.update {
                EdgeUpdate::Vertical {
                    x,
                    signed_height,
                    is_start,
                } => self
                    .vertical
                    .update(x, signed_height, is_start, left, right),
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
            self.next_event += 1;
        }
        None
    }

    fn collect_intervals(&self, bounds: [u32; 4], intervals: &mut Vec<Range<u32>>) {
        let [left, top, right, bottom] = bounds;
        intervals.clear();
        if right - left <= TILE_SIZE {
            intervals.push(left..right);
            return;
        }
        let sloped_interval = |index: usize| {
            let edge = self.sloped_edges[index]
                .edge
                .clip(top as f32, bottom as f32);
            tile_interval(edge.minimum, edge.maximum, left, right)
        };
        for edges in self.short_edges.rows_in(top..bottom) {
            for edge in edges {
                let Some(interval) = tile_interval(edge.minimum, edge.maximum, left, right) else {
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
        intervals.extend(self.vertical.active_columns.iter().map(|&offset| {
            let start = left + (offset as u32 / TILE_SIZE) * TILE_SIZE;
            start..start.saturating_add(TILE_SIZE).min(right)
        }));
        intervals.extend(
            self.active_sloped_edges
                .iter()
                .filter_map(|&index| sloped_interval(index)),
        );
        intervals.extend(
            self.events[self.next_event..]
                .iter()
                .take_while(|event| event.row < bottom)
                .filter_map(|event| match event.update {
                    // A later event can uncover a previously cancelled change.
                    EdgeUpdate::Vertical { x, .. } => tile_interval(x, x, left, right),
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
            .get(self.next_event)
            .map_or(path_bottom, |event| event.row);
        if next_event_row <= top {
            next_event_row = self.advance(top, left, right).unwrap_or(path_bottom);
        }
        let mut next_row = next_event_row;
        if self.active_sloped_edges.is_empty() && next_row >= bottom {
            if let Some(short_row) = self.short_edges.first_row(top..next_row) {
                next_row = short_row;
            }
        }
        let has_identical_rows = self.active_sloped_edges.is_empty() && next_row >= bottom;
        let row_count = if has_identical_rows {
            1
        } else {
            scanlines.len()
        };
        self.collect_intervals(bounds, intervals);
        for (index, scanline) in scanlines[..row_count].iter_mut().enumerate() {
            let row = top + index as u32;
            if row >= next_event_row {
                next_event_row = self.advance(row, left, right).unwrap_or(path_bottom);
            }
            let short_edges = self.short_edges.rows.get_mut(row as usize);
            let row_edges = short_edges.as_deref().map_or(&[][..], Vec::as_slice);
            if self.active_sloped_edges.is_empty() {
                scanline.prepare(row_edges.iter().copied(), left, right, intervals);
            } else {
                let sloped_edges = self.active_sloped_edges.iter().map(|&index| {
                    self.sloped_edges[index]
                        .edge
                        .clip(row as f32, (row + 1) as f32)
                });
                scanline.prepare(
                    row_edges.iter().copied().chain(sloped_edges),
                    left,
                    right,
                    intervals,
                );
            }
            if let Some(edges) = short_edges {
                edges.clear();
            }
            if self.vertical.initial_area.contribution_count != 0 {
                self.vertical.add_to(scanline);
            }
        }
        has_identical_rows.then(|| {
            // Keep mixed bands on the tile grid so repeats do not add boundary tiles.
            if next_row == path_bottom {
                next_row
            } else {
                top + (next_row - top) / TILE_SIZE * TILE_SIZE
            }
        })
    }

    pub(super) fn finish(&mut self, [left, _, right, bottom]: [u32; 4]) {
        // Close the remaining contributions so column storage can be reused.
        self.advance(bottom, left, right);
    }
}
