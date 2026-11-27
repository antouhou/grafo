use super::geometry::SOLID_COVERAGE;
use super::tiles::CoverageTile;
use crate::core::{PhysicalRect, UnsignedPhysicalRect};
use std::{array, sync::Arc};

pub(in crate::wgpu_backend) mod cache;
mod compiled;

const EMPTY_BLOCK: u32 = 0;
const FULL_BLOCK: u32 = 1;

#[derive(Clone, Copy, Debug)]
struct BorderBlock {
    origin: [i32; 2],
    values: [f32; 64],
}

#[derive(Debug, Default)]
pub(in crate::wgpu_backend) struct ClipCoverage {
    offscreen_parent: Option<Arc<Self>>,
    rectangle: Option<[f64; 4]>,
    is_convex: bool,
    pub bounds: PhysicalRect,
    origin: [i32; 2],
    columns: usize,
    cells: Vec<u32>,
    borders: Vec<BorderBlock>,
}

impl ClipCoverage {
    /// Shares fractional edges while keeping outside content available to effects.
    pub(super) fn for_offscreen_target(
        parent: &Arc<Self>,
        viewport: UnsignedPhysicalRect,
        storage: &mut Vec<Arc<Self>>,
    ) -> Arc<Self> {
        if parent.offscreen_parent.is_some() {
            return Arc::clone(parent);
        }
        let mut output = Self::new(viewport.to_i32(), storage);
        let result = Arc::get_mut(&mut output).expect("recycled coverage has one owner");
        result.offscreen_parent = Some(Arc::clone(parent));
        result.cells.fill(FULL_BLOCK);
        for block in &parent.borders {
            let mut values = block.values;
            for value in &mut values {
                if *value == 0.0 {
                    *value = 1.0;
                }
            }
            if values == [1.0; 64] {
                continue;
            }
            let Some(cell) = result.cell_index(block.origin[0], block.origin[1]) else {
                continue;
            };
            result.set_border(
                cell,
                BorderBlock {
                    origin: block.origin,
                    values,
                },
            );
        }
        output
    }

    fn new(bounds: PhysicalRect, storage: &mut Vec<Arc<Self>>) -> Arc<Self> {
        let mut output = storage.pop().unwrap_or_default();
        let result = Arc::get_mut(&mut output).expect("recycled coverage has one owner");
        result.offscreen_parent = None;
        result.rectangle = None;
        result.is_convex = false;
        result.bounds = bounds;
        result.origin = [
            bounds.min.x.div_euclid(8) * 8,
            bounds.min.y.div_euclid(8) * 8,
        ];
        result.columns = ((bounds.max.x - result.origin[0]).max(0) as usize).div_ceil(8);
        let rows = ((bounds.max.y - result.origin[1]).max(0) as usize).div_ceil(8);
        result.cells.clear();
        result.cells.resize(
            if bounds.is_empty() {
                0
            } else {
                result.columns * rows
            },
            EMPTY_BLOCK,
        );
        result.borders.clear();
        output
    }

    fn cell_index(&self, x: i32, y: i32) -> Option<usize> {
        if self.bounds.is_empty()
            || x < self.origin[0]
            || y < self.origin[1]
            || x >= self.bounds.max.x
            || y >= self.bounds.max.y
        {
            return None;
        }
        Some(
            ((y - self.origin[1]) / 8) as usize * self.columns
                + ((x - self.origin[0]) / 8) as usize,
        )
    }

    fn block(&self, x: i32, y: i32) -> u32 {
        match self.cell_index(x, y) {
            Some(index) => self.cells[index],
            None => EMPTY_BLOCK,
        }
    }

    pub fn value(&self, x: i32, y: i32) -> f32 {
        if x < self.bounds.min.x || y < self.bounds.min.y {
            return 0.0;
        }
        match self.block(x, y) {
            EMPTY_BLOCK => 0.0,
            FULL_BLOCK => 1.0,
            index => {
                let block = &self.borders[(index - 2) as usize];
                block.values[((y - block.origin[1]) * 8 + x - block.origin[0]) as usize]
            }
        }
    }

    pub(in crate::wgpu_backend) fn covers_rect(&self, bounds: PhysicalRect) -> bool {
        if bounds.is_empty() {
            return true;
        }
        if self.bounds.min.x > bounds.min.x
            || self.bounds.min.y > bounds.min.y
            || self.bounds.max.x < bounds.max.x
            || self.bounds.max.y < bounds.max.y
        {
            return false;
        }
        // A convex clip contains the entire pixel rectangle when its corner pixels are full.
        if self.is_convex {
            let covered = [bounds.min.x, bounds.max.x - 1].into_iter().all(|x| {
                [bounds.min.y, bounds.max.y - 1]
                    .into_iter()
                    .all(|y| self.value(x, y) == 1.0)
            });
            return covered;
        }
        for top in (bounds.min.y.div_euclid(8) * 8..bounds.max.y).step_by(8) {
            for left in (bounds.min.x.div_euclid(8) * 8..bounds.max.x).step_by(8) {
                if self.block(left, top) == FULL_BLOCK {
                    continue;
                }
                for y in top.max(bounds.min.y)..(top + 8).min(bounds.max.y) {
                    for x in left.max(bounds.min.x)..(left + 8).min(bounds.max.x) {
                        if self.value(x, y) != 1.0 {
                            return false;
                        }
                    }
                }
            }
        }
        true
    }

    fn set_border(&mut self, cell: usize, block: BorderBlock) {
        self.cells[cell] = if block.values == [0.0; 64] {
            EMPTY_BLOCK
        } else if block.values == [1.0; 64] {
            FULL_BLOCK
        } else {
            let index = self.borders.len() as u32 + 2;
            self.borders.push(block);
            index
        };
    }

    pub fn rectangle(
        bounds: [f64; 4],
        viewport: UnsignedPhysicalRect,
        storage: &mut Vec<Arc<Self>>,
    ) -> Arc<Self> {
        let clipped = [
            bounds[0].max(f64::from(viewport.min.x)),
            bounds[1].max(f64::from(viewport.min.y)),
            bounds[2].min(f64::from(viewport.max.x)),
            bounds[3].min(f64::from(viewport.max.y)),
        ];
        if clipped[0] >= clipped[2] || clipped[1] >= clipped[3] {
            return Self::new(PhysicalRect::zero(), storage);
        }
        let mut output = Self::new(
            PhysicalRect::new(
                [clipped[0].floor() as i32, clipped[1].floor() as i32].into(),
                [clipped[2].ceil() as i32, clipped[3].ceil() as i32].into(),
            ),
            storage,
        );
        let result = Arc::get_mut(&mut output).expect("recycled coverage has one owner");
        result.rectangle = Some(clipped);
        result.is_convex = true;
        let interior = [
            clipped[0].ceil() as i32,
            clipped[1].ceil() as i32,
            clipped[2].floor() as i32,
            clipped[3].floor() as i32,
        ];
        for row in 0..result.cells.len() / result.columns.max(1) {
            for column in 0..result.columns {
                let index = row * result.columns + column;
                let origin = [
                    result.origin[0] + column as i32 * 8,
                    result.origin[1] + row as i32 * 8,
                ];
                if origin[0] >= interior[0]
                    && origin[1] >= interior[1]
                    && origin[0] + 8 <= interior[2]
                    && origin[1] + 8 <= interior[3]
                {
                    result.cells[index] = FULL_BLOCK;
                    continue;
                }
                let mut values = [0.0; 64];
                let widths: [f64; 8] = array::from_fn(|x| {
                    let left = f64::from(origin[0] + x as i32);
                    (clipped[2].min(left + 1.0) - clipped[0].max(left)).max(0.0)
                });
                for y in 0..8 {
                    let top = f64::from(origin[1] + y as i32);
                    let height = (clipped[3].min(top + 1.0) - clipped[1].max(top)).max(0.0);
                    for (x, width) in widths.iter().enumerate() {
                        values[y * 8 + x] = (width * height) as f32;
                    }
                }
                result.set_border(index, BorderBlock { origin, values });
            }
        }
        output
    }

    fn from_raster(
        tiles: &[CoverageTile],
        coverage: &[f32],
        bounds: UnsignedPhysicalRect,
        translation: [f32; 2],
        storage: &mut Vec<Arc<Self>>,
        solid_samples: &mut Vec<u64>,
    ) -> Arc<Self> {
        let translation = [translation[0] as i32, translation[1] as i32];
        let world_bounds = bounds.to_i32().translate(translation.into());
        let mut output = Self::new(world_bounds, storage);
        let result = Arc::get_mut(&mut output).expect("recycled coverage has one owner");
        solid_samples.clear();
        solid_samples.resize(result.cells.len(), 0);
        for tile in tiles {
            let minimum = [
                tile.min[0] as i32 + translation[0],
                tile.min[1] as i32 + translation[1],
            ];
            let maximum = [
                tile.max[0] as i32 + translation[0],
                tile.max[1] as i32 + translation[1],
            ];
            let first = [minimum[0].div_euclid(8) * 8, minimum[1].div_euclid(8) * 8];
            for top in (first[1]..maximum[1]).step_by(8) {
                for left in (first[0]..maximum[0]).step_by(8) {
                    let Some(cell) = result.cell_index(left, top) else {
                        continue;
                    };
                    let start = [left.max(minimum[0]), top.max(minimum[1])];
                    let end = [(left + 8).min(maximum[0]), (top + 8).min(maximum[1])];
                    if tile.coverage_start == SOLID_COVERAGE {
                        let columns = ((1_u64 << (end[0] - start[0])) - 1) << (start[0] - left);
                        let rows =
                            (u64::MAX >> ((8 - end[1] + start[1]) * 8)) << ((start[1] - top) * 8);
                        solid_samples[cell] |= (columns * 0x0101_0101_0101_0101) & rows;
                        if solid_samples[cell] == u64::MAX {
                            result.cells[cell] = FULL_BLOCK;
                        }
                        continue;
                    }
                    let block = result.mutable_border(cell, [left, top]);
                    let width = (end[0] - start[0]) as usize;
                    for y in start[1]..end[1] {
                        let destination = ((y - top) * 8 + start[0] - left) as usize;
                        let source = tile.coverage_start as usize
                            + ((y - minimum[1]) * 8 + start[0] - minimum[0]) as usize;
                        block.values[destination..destination + width]
                            .copy_from_slice(&coverage[source..source + width]);
                    }
                }
            }
        }
        for (cell, &samples) in solid_samples.iter().enumerate() {
            if samples == 0 || result.cells[cell] == FULL_BLOCK {
                continue;
            }
            let origin = [
                result.origin[0] + (cell % result.columns) as i32 * 8,
                result.origin[1] + (cell / result.columns) as i32 * 8,
            ];
            let block = result.mutable_border(cell, origin);
            let mut remaining = samples;
            while remaining != 0 {
                block.values[remaining.trailing_zeros() as usize] = 1.0;
                remaining &= remaining - 1;
            }
        }
        result.compact_borders();
        output
    }

    fn mutable_border(&mut self, cell: usize, origin: [i32; 2]) -> &mut BorderBlock {
        if self.cells[cell] < 2 {
            let initial = if self.cells[cell] == FULL_BLOCK {
                1.0
            } else {
                0.0
            };
            self.cells[cell] = self.borders.len() as u32 + 2;
            self.borders.push(BorderBlock {
                origin,
                values: [initial; 64],
            });
        }
        &mut self.borders[(self.cells[cell] - 2) as usize]
    }

    fn compact_borders(&mut self) {
        let mut border_index = 2;
        self.borders.retain(|block| {
            let cell = ((block.origin[1] - self.origin[1]) / 8) as usize * self.columns
                + ((block.origin[0] - self.origin[0]) / 8) as usize;
            let value = if block.values == [0.0; 64] {
                EMPTY_BLOCK
            } else if block.values == [1.0; 64] {
                FULL_BLOCK
            } else {
                border_index += 1;
                border_index - 1
            };
            self.cells[cell] = value;
            value >= 2
        });
    }

    fn edge_direction(&self, x: i32, y: i32) -> [f32; 2] {
        let source = self.offscreen_parent.as_deref().unwrap_or(self);
        [
            source.value(x + 1, y) - source.value(x - 1, y),
            source.value(x, y + 1) - source.value(x, y - 1),
        ]
    }

    fn matching_border(&self, other: &Self, first: &BorderBlock, second: &BorderBlock) -> bool {
        if first.values != second.values {
            return false;
        }
        let [left, top] = first.origin;
        for offset in 0..8 {
            for (x, y) in [
                (left - 1, top + offset),
                (left + 8, top + offset),
                (left + offset, top - 1),
                (left + offset, top + 8),
            ] {
                if self.value(x, y) != other.value(x, y) {
                    return false;
                }
            }
        }
        true
    }

    fn intersect_border(&self, other: &Self, left: u32, right: u32) -> BorderBlock {
        let first = &self.borders[(left - 2) as usize];
        let second = &other.borders[(right - 2) as usize];
        if self.matching_border(other, first, second) {
            return *first;
        }
        let mut values = [0.0; 64];
        for (index, value) in values.iter_mut().enumerate() {
            let left = first.values[index];
            let right = second.values[index];
            if left == 0.0 || right == 0.0 || left == 1.0 || right == 1.0 {
                *value = left * right;
                continue;
            }
            let x = first.origin[0] + (index % 8) as i32;
            let y = first.origin[1] + (index / 8) as i32;
            let first_direction = self.edge_direction(x, y);
            let second_direction = other.edge_direction(x, y);
            let length_product = ((first_direction[0] * first_direction[0]
                + first_direction[1] * first_direction[1])
                * (second_direction[0] * second_direction[0]
                    + second_direction[1] * second_direction[1]))
                .sqrt();
            let correlation = if length_product > 0.0 {
                ((first_direction[0] * second_direction[0]
                    + first_direction[1] * second_direction[1])
                    / length_product)
                    .clamp(-1.0, 1.0)
            } else {
                // Retain thin features when neighboring pixels cannot resolve an edge direction.
                1.0
            };
            // Parallel edges share coverage; opposite edges cover disjoint parts.
            // Interpolate through independent coverage for crossing boundaries.
            let independent = left * right;
            let correlated = if correlation >= 0.0 {
                left.min(right)
            } else {
                (left + right - 1.0).max(0.0)
            };
            *value = independent + correlation.abs() * (correlated - independent);
        }
        BorderBlock {
            origin: first.origin,
            values,
        }
    }

    pub fn intersection(
        first: &Arc<Self>,
        second: &Arc<Self>,
        storage: &mut Vec<Arc<Self>>,
    ) -> Arc<Self> {
        if Arc::ptr_eq(first, second) {
            return Arc::clone(first);
        }
        if first.covers_rect(second.bounds) {
            return Arc::clone(second);
        }
        if second.covers_rect(first.bounds) {
            return Arc::clone(first);
        }
        let bounds = match first.bounds.intersection(&second.bounds) {
            Some(bounds) => bounds,
            None => PhysicalRect::zero(),
        };
        if let (Some(left), Some(right)) = (first.rectangle, second.rectangle) {
            return Self::rectangle(
                [
                    left[0].max(right[0]),
                    left[1].max(right[1]),
                    left[2].min(right[2]),
                    left[3].min(right[3]),
                ],
                bounds.to_u32(),
                storage,
            );
        }
        let mut output = Self::new(bounds, storage);
        let result = Arc::get_mut(&mut output).expect("recycled coverage has one owner");
        result.is_convex = first.is_convex && second.is_convex;
        for row in 0..result.cells.len() / result.columns.max(1) {
            let top = result.origin[1] + row as i32 * 8;
            let first_start = ((top - first.origin[1]) / 8) as usize * first.columns
                + ((result.origin[0] - first.origin[0]) / 8) as usize;
            let second_start = ((top - second.origin[1]) / 8) as usize * second.columns
                + ((result.origin[0] - second.origin[0]) / 8) as usize;
            let first_row = &first.cells[first_start..first_start + result.columns];
            let second_row = &second.cells[second_start..second_start + result.columns];
            for (column, (&left, &right)) in first_row.iter().zip(second_row).enumerate() {
                let index = row * result.columns + column;
                match (left, right) {
                    (EMPTY_BLOCK, _) | (_, EMPTY_BLOCK) => {}
                    (FULL_BLOCK, FULL_BLOCK) => result.cells[index] = FULL_BLOCK,
                    (FULL_BLOCK, border) => {
                        result.set_border(index, second.borders[(border - 2) as usize])
                    }
                    (border, FULL_BLOCK) => {
                        result.set_border(index, first.borders[(border - 2) as usize])
                    }
                    (left, right) => {
                        result.set_border(index, first.intersect_border(second, left, right))
                    }
                }
            }
        }
        output
    }
}

#[cfg(test)]
mod tests;
