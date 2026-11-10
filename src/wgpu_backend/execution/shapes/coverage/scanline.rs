use super::tiles::TILE_SIZE;
use std::ops::Range;

const MIN_RANGE_UPDATE_COLUMNS: u32 = 16;
const MIN_CONSTANT_DELTA_COLUMNS: u32 = 6;

fn clamp_coverage(area: f64) -> f32 {
    // NaN and roundoff near zero leave the stencil empty.
    if area > 1e-12 {
        area.min(1.0) as f32
    } else {
        0.0
    }
}

fn fill_point_changes(changes: &[f64], values: &mut [f32], mut area: f64) -> f64 {
    for (&change, value) in changes.iter().zip(values) {
        area += change;
        *value = clamp_coverage(area);
    }
    area
}

/// Retains the low bits when signed contributions nearly cancel.
#[derive(Clone, Copy, Default)]
pub(super) struct CoverageSum {
    sum: f64,
    error: f64,
}

impl CoverageSum {
    pub(super) fn add(&mut self, value: f64) {
        let sum = self.sum + value;
        let rounded_value = sum - self.sum;
        self.error += (self.sum - (sum - rounded_value)) + (value - rounded_value);
        self.sum = sum;
    }

    fn add_sum(&mut self, other: Self) {
        self.add(other.sum);
        self.add(other.error);
    }

    pub(super) fn total(&self) -> f64 {
        self.sum + self.error
    }
}

#[derive(Clone, Copy)]
pub(super) struct RowEdge {
    pub minimum: f32,
    pub maximum: f32,
    pub signed_winding_height: f64,
}

impl RowEdge {
    fn coverage(&self, pixel: f64, inverse_width: f64) -> f64 {
        let a = self.minimum as f64 - pixel;
        let b = self.maximum as f64 - pixel;
        let area = if a >= 1.0 {
            1.0
        } else if b <= 0.0 {
            0.0
        } else if a >= 0.0 && b <= 1.0 {
            (a + b) * 0.5
        } else if a < 0.0 {
            if b <= 1.0 {
                b * b * 0.5 * inverse_width
            } else {
                (b - 0.5) * inverse_width
            }
        } else {
            1.0 - (1.0 - a) * (1.0 - a) * 0.5 * inverse_width
        };
        area * self.signed_winding_height
    }
}

pub(super) struct ScanEdge {
    start: [f32; 2],
    pub top: f32,
    pub bottom: f32,
    slope: f64,
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
            slope: (end[0] as f64 - start[0] as f64) / height as f64,
            signed_winding: (height as f64).signum() * winding as f64,
        })
    }

    pub(super) fn is_vertical(&self) -> bool {
        self.slope == 0.0
    }

    pub(super) fn clip(&self, top: f32, bottom: f32) -> RowEdge {
        let low = self.top.max(top) as f64;
        let high = self.bottom.min(bottom) as f64;
        let x = |y| (self.start[0] as f64 + (y - self.start[1] as f64) * self.slope) as f32;
        let first = x(low);
        let second = x(high);
        RowEdge {
            minimum: first.min(second),
            maximum: first.max(second),
            signed_winding_height: (high - low) * self.signed_winding,
        }
    }
}

#[derive(Default)]
struct LinearChanges {
    slope_changes: Vec<CoverageSum>,
    // A bit per column marks the slope changes to read in each tile.
    slope_event_masks: Vec<u8>,
    slope: CoverageSum,
    is_active: bool,
}

impl LinearChanges {
    fn prepare(&mut self, left: u32, right: u32, intervals: &[Range<u32>]) {
        let width = (right - left) as usize;
        if self.slope_changes.len() < width {
            self.slope_changes.resize(width, CoverageSum::default());
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
        self.slope = CoverageSum::default();
        self.is_active = true;
    }

    fn add_event(&mut self, offset: usize, slope: f64) {
        let tile = &mut self.slope_event_masks[offset / TILE_SIZE as usize];
        let mask = 1 << (offset % TILE_SIZE as usize);
        let change = &mut self.slope_changes[offset];
        if *tile & mask == 0 {
            *tile |= mask;
            *change = CoverageSum::default();
        }
        change.add(slope);
    }

    #[inline(never)]
    fn fill_tile(
        &mut self,
        offset: usize,
        changes: &[f64],
        values: &mut [f32],
        mut area: f64,
    ) -> f64 {
        debug_assert_eq!(offset % TILE_SIZE as usize, 0);
        let event_mask = self.slope_event_masks[offset / TILE_SIZE as usize];
        let mut slope = self.slope.total();
        let mut position = 0.0;
        if event_mask == 0 {
            if slope == 0.0 {
                return fill_point_changes(changes, values, area);
            }
            for (&change, value) in changes.iter().zip(values) {
                area += change;
                position += 1.0;
                *value = clamp_coverage(area + position * slope);
            }
            return area + position * slope;
        }
        let mut slope_start_column = 0.0;
        // Anchor the linear term at the tile and at each change in slope.
        for (column, (&change, value)) in changes.iter().zip(values).enumerate() {
            position += 1.0;
            if event_mask & (1 << column) != 0 {
                area += (position - slope_start_column) * slope;
                slope_start_column = position;
                self.slope.add_sum(self.slope_changes[offset + column]);
                slope = self.slope.total();
            }
            area += change;
            *value = clamp_coverage(area + (position - slope_start_column) * slope);
        }
        area + (position - slope_start_column) * slope
    }
}

#[derive(Default)]
pub(super) struct Scanline {
    changes: Vec<f64>,
    left: u32,
    area: f64,
    linear: LinearChanges,
}

impl Scanline {
    pub(super) fn prepare(
        &mut self,
        edges: impl IntoIterator<Item = RowEdge>,
        left: u32,
        right: u32,
        intervals: &[Range<u32>],
    ) {
        if self.changes.len() < (right - left) as usize {
            self.changes.resize((right - left) as usize, 0.0);
        }
        self.left = left;
        for interval in intervals {
            self.changes[(interval.start - left) as usize..(interval.end - left) as usize]
                .fill(0.0);
        }
        self.area = 0.0;
        self.linear.is_active = false;
        for edge in edges {
            let minimum = edge.minimum;
            let maximum = edge.maximum;
            if maximum <= left as f32 {
                continue;
            }
            if minimum >= right as f32 {
                self.area += edge.signed_winding_height;
                continue;
            }
            if minimum == maximum {
                let column = minimum as u32;
                let offset = (column - left) as usize;
                let partial_area = (minimum as f64 - column as f64) * edge.signed_winding_height;
                self.area += edge.signed_winding_height;
                self.changes[offset] += partial_area - edge.signed_winding_height;
                if column + 1 < right && partial_area != 0.0 {
                    self.changes[offset + 1] -= partial_area;
                }
                continue;
            }
            let width = maximum as f64 - minimum as f64;
            let inverse_width = 1.0 / width;
            let first = minimum.floor().max(left as f32) as u32;
            let end = maximum.ceil().min(right as f32) as u32;
            let previous = if minimum >= left as f32 {
                edge.signed_winding_height
            } else {
                edge.coverage(left as f64 - 1.0, inverse_width)
            };
            self.area += previous;
            let previous = if end - first >= MIN_CONSTANT_DELTA_COLUMNS {
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
                self.changes[(end - left) as usize] -= previous;
            }
        }
    }

    #[inline]
    fn add_boundary_changes(
        &mut self,
        edge: RowEdge,
        inverse_width: f64,
        columns: Range<u32>,
        mut previous: f64,
    ) -> f64 {
        let first = (columns.start - self.left) as usize;
        let end = (columns.end - self.left) as usize;
        let mut pixel = columns.start as f64;
        for change in &mut self.changes[first..end] {
            let next = edge.coverage(pixel, inverse_width);
            *change += next - previous;
            previous = next;
            pixel += 1.0;
        }
        previous
    }

    fn add_crossed_columns(
        &mut self,
        edge: RowEdge,
        inverse_width: f64,
        columns: Range<u32>,
        right: u32,
        previous: f64,
        intervals: &[Range<u32>],
    ) -> f64 {
        let first_full_column = (edge.minimum.ceil() as u32).max(self.left);
        let last_full_column = (edge.maximum.floor() as u32).min(right) - 1;
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
        if columns.end - columns.start >= MIN_RANGE_UPDATE_COLUMNS {
            if !self.linear.is_active {
                self.linear.prepare(self.left, right, intervals);
            }
            self.linear.add_event(first, slope);
            self.linear.add_event(last, -slope);
        } else {
            // Small runs fit in the point buffer without slope events.
            for change in &mut self.changes[first + 1..=last] {
                *change += slope;
            }
        }

        let trailing = edge.coverage(last_full_column as f64, inverse_width);
        // Close the range at the analytic endpoint to keep rounding local.
        self.changes[last] += (trailing - leading) - (last - first) as f64 * slope;
        self.add_boundary_changes(
            edge,
            inverse_width,
            last_full_column + 1..columns.end,
            trailing,
        )
    }

    pub(super) fn add_vertical_changes(
        &mut self,
        area: f64,
        changes: impl Iterator<Item = (usize, f64)>,
    ) {
        self.area += area;
        for (offset, change) in changes {
            self.changes[offset] += change;
        }
    }

    #[inline]
    pub(super) fn fill_coverage(&mut self, left: u32, values: &mut [f32]) {
        let offset = (left - self.left) as usize;
        let changes = &self.changes[offset..offset + values.len()];
        self.area = if self.linear.is_active {
            self.linear.fill_tile(offset, changes, values, self.area)
        } else {
            fill_point_changes(changes, values, self.area)
        };
    }

    pub(super) fn coverage(&self) -> f32 {
        clamp_coverage(self.area)
    }
}
