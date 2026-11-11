use super::tiles::TILE_SIZE;
use crate::core::math::CompensatedSum;
use std::ops::Range;

const MINIMUM_COLUMNS_FOR_SLOPE_EVENTS: u32 = 16;
const MINIMUM_COLUMNS_FOR_CONSTANT_SLOPE: u32 = 6;

fn clamp_coverage(area: f64) -> f32 {
    // NaN and roundoff near zero leave the stencil empty.
    if area > 1e-12 {
        area.min(1.0) as f32
    } else {
        0.0
    }
}

fn fill_coverage_from_column_deltas(
    column_deltas: &[f64],
    coverage_values: &mut [f32],
    mut area: f64,
) -> f64 {
    for (&delta, coverage) in column_deltas.iter().zip(coverage_values) {
        area += delta;
        *coverage = clamp_coverage(area);
    }
    area
}

#[derive(Clone, Copy)]
pub(super) struct ClippedScanEdge {
    pub minimum_x: f32,
    pub maximum_x: f32,
    pub signed_winding_height: f64,
}

impl ClippedScanEdge {
    fn coverage(&self, pixel: f64, inverse_width: f64) -> f64 {
        let minimum_x_offset = self.minimum_x as f64 - pixel;
        let maximum_x_offset = self.maximum_x as f64 - pixel;
        let area = if minimum_x_offset >= 1.0 {
            1.0
        } else if maximum_x_offset <= 0.0 {
            0.0
        } else if minimum_x_offset >= 0.0 && maximum_x_offset <= 1.0 {
            (minimum_x_offset + maximum_x_offset) * 0.5
        } else if minimum_x_offset < 0.0 {
            if maximum_x_offset <= 1.0 {
                maximum_x_offset * maximum_x_offset * 0.5 * inverse_width
            } else {
                (maximum_x_offset - 0.5) * inverse_width
            }
        } else {
            1.0 - (1.0 - minimum_x_offset) * (1.0 - minimum_x_offset) * 0.5 * inverse_width
        };
        area * self.signed_winding_height
    }
}

pub(super) struct ScanEdge {
    start: [f32; 2],
    pub top: f32,
    pub bottom: f32,
    x_per_y: f64,
    signed_winding: f64,
}

impl ScanEdge {
    pub(super) fn new(start: [f32; 2], end: [f32; 2], winding: i32) -> Option<Self> {
        let height = end[1] - start[1];
        if height == 0.0 {
            return None;
        }
        Some(Self {
            start,
            top: start[1].min(end[1]),
            bottom: start[1].max(end[1]),
            x_per_y: (end[0] as f64 - start[0] as f64) / height as f64,
            signed_winding: (height as f64).signum() * winding as f64,
        })
    }

    pub(super) fn is_vertical(&self) -> bool {
        self.x_per_y == 0.0
    }

    pub(super) fn clip_to_rows(&self, top: f32, bottom: f32) -> ClippedScanEdge {
        let low = self.top.max(top) as f64;
        let high = self.bottom.min(bottom) as f64;
        let x = |y| (self.start[0] as f64 + (y - self.start[1] as f64) * self.x_per_y) as f32;
        let first = x(low);
        let second = x(high);
        ClippedScanEdge {
            minimum_x: first.min(second),
            maximum_x: first.max(second),
            signed_winding_height: (high - low) * self.signed_winding,
        }
    }
}

#[derive(Default)]
struct ScanlineSlopeChanges {
    slope_changes: Vec<CompensatedSum>,
    // A bit per column marks the slope changes to read in each tile.
    slope_event_masks: Vec<u8>,
    slope: CompensatedSum,
    has_slope_events: bool,
}

impl ScanlineSlopeChanges {
    fn prepare(&mut self, left: u32, right: u32, intervals: &[Range<u32>]) {
        let width = (right - left) as usize;
        if self.slope_changes.len() < width {
            self.slope_changes.resize(width, CompensatedSum::default());
        }
        let tile_count = width.div_ceil(TILE_SIZE as usize);
        if self.slope_event_masks.len() < tile_count {
            self.slope_event_masks.resize(tile_count, 0);
        }
        for interval in intervals {
            let first = (interval.start - left) as usize / TILE_SIZE as usize;
            let end = ((interval.end - left) as usize).div_ceil(TILE_SIZE as usize);
            self.slope_event_masks[first..end].fill(0);
        }
        self.slope = CompensatedSum::default();
        self.has_slope_events = true;
    }

    fn add_event(&mut self, offset: usize, slope: f64) {
        let tile = &mut self.slope_event_masks[offset / TILE_SIZE as usize];
        let mask = 1 << (offset % TILE_SIZE as usize);
        let change = &mut self.slope_changes[offset];
        if *tile & mask == 0 {
            *tile |= mask;
            *change = CompensatedSum::default();
        }
        change.add(slope);
    }

    #[inline(never)]
    fn fill_tile_row(
        &mut self,
        offset: usize,
        area_deltas: &[f64],
        values: &mut [f32],
        mut area: f64,
    ) -> f64 {
        debug_assert_eq!(offset % TILE_SIZE as usize, 0);
        let event_mask = self.slope_event_masks[offset / TILE_SIZE as usize];
        let mut slope = self.slope.total();
        let mut position = 0.0;
        if event_mask == 0 {
            if slope == 0.0 {
                return fill_coverage_from_column_deltas(area_deltas, values, area);
            }
            for (&area_delta, value) in area_deltas.iter().zip(values) {
                area += area_delta;
                position += 1.0;
                *value = clamp_coverage(area + position * slope);
            }
            return area + position * slope;
        }
        let mut slope_start_column = 0.0;
        // Anchor the linear term at the tile and at each change in slope.
        for (column, (&area_delta, value)) in area_deltas.iter().zip(values).enumerate() {
            position += 1.0;
            if event_mask & (1 << column) != 0 {
                area += (position - slope_start_column) * slope;
                slope_start_column = position;
                self.slope.add_sum(self.slope_changes[offset + column]);
                slope = self.slope.total();
            }
            area += area_delta;
            *value = clamp_coverage(area + (position - slope_start_column) * slope);
        }
        area + (position - slope_start_column) * slope
    }
}

#[derive(Default)]
pub(super) struct Scanline {
    area_deltas: Vec<f64>,
    left: u32,
    area: f64,
    slope_changes: ScanlineSlopeChanges,
}

impl Scanline {
    #[inline]
    fn add_boundary_changes(
        &mut self,
        edge: ClippedScanEdge,
        inverse_width: f64,
        columns: Range<u32>,
        mut previous: f64,
    ) -> f64 {
        let first = (columns.start - self.left) as usize;
        let end = (columns.end - self.left) as usize;
        let mut pixel = columns.start as f64;
        for area_delta in &mut self.area_deltas[first..end] {
            let next = edge.coverage(pixel, inverse_width);
            *area_delta += next - previous;
            previous = next;
            pixel += 1.0;
        }
        previous
    }

    fn add_crossed_columns(
        &mut self,
        edge: ClippedScanEdge,
        inverse_width: f64,
        columns: Range<u32>,
        right: u32,
        previous: f64,
        intervals: &[Range<u32>],
    ) -> f64 {
        let first_full_column = (edge.minimum_x.ceil() as u32).max(self.left);
        let last_full_column = (edge.maximum_x.floor() as u32).min(right) - 1;
        let leading = self.add_boundary_changes(
            edge,
            inverse_width,
            columns.start..first_full_column + 1,
            previous,
        );
        // Between fully crossed pixels the coverage difference is constant.
        let slope = -edge.signed_winding_height * inverse_width;
        let first = (first_full_column - self.left) as usize;
        let last = (last_full_column - self.left) as usize;
        if columns.end - columns.start >= MINIMUM_COLUMNS_FOR_SLOPE_EVENTS {
            if !self.slope_changes.has_slope_events {
                self.slope_changes.prepare(self.left, right, intervals);
            }
            self.slope_changes.add_event(first, slope);
            self.slope_changes.add_event(last, -slope);
        } else {
            // Small runs fit in the point buffer without slope events.
            for area_delta in &mut self.area_deltas[first + 1..=last] {
                *area_delta += slope;
            }
        }

        let trailing = edge.coverage(last_full_column as f64, inverse_width);
        // Close the range at the analytic endpoint to keep rounding local.
        self.area_deltas[last] += (trailing - leading) - (last - first) as f64 * slope;
        self.add_boundary_changes(
            edge,
            inverse_width,
            last_full_column + 1..columns.end,
            trailing,
        )
    }

    pub(super) fn prepare(
        &mut self,
        edges: impl IntoIterator<Item = ClippedScanEdge>,
        left: u32,
        right: u32,
        intervals: &[Range<u32>],
    ) {
        if self.area_deltas.len() < (right - left) as usize {
            self.area_deltas.resize((right - left) as usize, 0.0);
        }
        self.left = left;
        for interval in intervals {
            self.area_deltas[(interval.start - left) as usize..(interval.end - left) as usize]
                .fill(0.0);
        }
        self.area = 0.0;
        self.slope_changes.has_slope_events = false;
        for edge in edges {
            let minimum_x = edge.minimum_x;
            let maximum_x = edge.maximum_x;
            if maximum_x <= left as f32 {
                continue;
            }
            if minimum_x >= right as f32 {
                self.area += edge.signed_winding_height;
                continue;
            }
            if minimum_x == maximum_x {
                let column = minimum_x as u32;
                let offset = (column - left) as usize;
                let partial_area = (minimum_x as f64 - column as f64) * edge.signed_winding_height;
                self.area += edge.signed_winding_height;
                self.area_deltas[offset] += partial_area - edge.signed_winding_height;
                if column + 1 < right && partial_area != 0.0 {
                    self.area_deltas[offset + 1] -= partial_area;
                }
                continue;
            }
            let width = maximum_x as f64 - minimum_x as f64;
            let inverse_width = 1.0 / width;
            let first = minimum_x.floor().max(left as f32) as u32;
            let end = maximum_x.ceil().min(right as f32) as u32;
            let previous = if minimum_x >= left as f32 {
                edge.signed_winding_height
            } else {
                edge.coverage(left as f64 - 1.0, inverse_width)
            };
            self.area += previous;
            let previous = if end - first >= MINIMUM_COLUMNS_FOR_CONSTANT_SLOPE {
                self.add_crossed_columns(
                    edge,
                    inverse_width,
                    first..end,
                    right,
                    previous,
                    intervals,
                )
            } else {
                self.add_boundary_changes(edge, inverse_width, first..end, previous)
            };
            if end < right {
                self.area_deltas[(end - left) as usize] -= previous;
            }
        }
    }

    pub(super) fn add_vertical_changes(
        &mut self,
        area: f64,
        area_deltas: impl Iterator<Item = (usize, f64)>,
    ) {
        self.area += area;
        for (offset, area_delta) in area_deltas {
            self.area_deltas[offset] += area_delta;
        }
    }

    #[inline]
    pub(super) fn fill_coverage(&mut self, left: u32, values: &mut [f32]) {
        let offset = (left - self.left) as usize;
        let area_deltas = &self.area_deltas[offset..offset + values.len()];
        self.area = if self.slope_changes.has_slope_events {
            self.slope_changes
                .fill_tile_row(offset, area_deltas, values, self.area)
        } else {
            fill_coverage_from_column_deltas(area_deltas, values, self.area)
        };
    }

    pub(super) fn coverage(&self) -> f32 {
        clamp_coverage(self.area)
    }
}
