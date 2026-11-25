use super::{ClipCoverage, EMPTY_BLOCK, FULL_BLOCK, SOLID_COVERAGE};
use crate::wgpu_backend::execution::shapes::coverage::tiles::{
    CoverageGeometry, CoverageProjection, CoverageTile,
};
use bytemuck::Zeroable;

impl ClipCoverage {
    fn block_values(&self, x: i32, y: i32) -> &[f32; 64] {
        match self.block(x, y) {
            EMPTY_BLOCK => &[0.0; 64],
            FULL_BLOCK => &[1.0; 64],
            index => &self.borders[(index - 2) as usize].values,
        }
    }

    fn compile_border(
        &self,
        denominator: Option<&Self>,
        minimum: [i32; 2],
        maximum: [i32; 2],
        output: &mut CoverageGeometry,
    ) -> Option<u32> {
        let origin = [minimum[0].div_euclid(8) * 8, minimum[1].div_euclid(8) * 8];
        let values = self.block_values(origin[0], origin[1]);
        let parent_values = denominator.and_then(|parent| {
            (parent.block(origin[0], origin[1]) != FULL_BLOCK)
                .then(|| parent.block_values(origin[0], origin[1]))
        });
        let start = output.coverage.len();
        output.coverage.resize(start + 64, 0.0);
        let mut is_solid = true;
        let mut is_empty = true;
        for y in minimum[1]..maximum[1] {
            for x in minimum[0]..maximum[0] {
                let index = ((y - origin[1]) * 8 + x - origin[0]) as usize;
                let value = values[index];
                let value = if let Some(parent) = parent_values {
                    let parent = parent[index];
                    if parent > 0.0 {
                        (value / parent).min(1.0)
                    } else {
                        0.0
                    }
                } else {
                    value
                };
                output.coverage[start + ((y - minimum[1]) * 8 + x - minimum[0]) as usize] = value;
                is_solid &= value == 1.0;
                is_empty &= value == 0.0;
            }
        }
        if is_solid || is_empty {
            output.coverage.truncate(start);
        }
        if is_empty {
            None
        } else if is_solid {
            Some(SOLID_COVERAGE)
        } else {
            Some(start as u32)
        }
    }

    /// Compiles coverage relative to an optional containing mask.
    pub(super) fn compile(
        &self,
        denominator: Option<&Self>,
        translation: [i32; 2],
        projection: Option<CoverageProjection>,
        output: &mut CoverageGeometry,
    ) -> u32 {
        output.clear();
        output.projections.push(match projection {
            Some(projection) => projection,
            None => CoverageProjection::zeroed(),
        });
        for (row_index, row) in self.cells.chunks(self.columns.max(1)).enumerate() {
            let top = self.origin[1] + row_index as i32 * 8;
            let mut column = 0;
            while column < row.len() {
                let cell = row[column];
                let left = self.origin[0] + column as i32 * 8;
                column += 1;
                if cell == EMPTY_BLOCK {
                    continue;
                }
                if cell == FULL_BLOCK {
                    column += row[column..]
                        .iter()
                        .position(|&cell| cell != FULL_BLOCK)
                        .unwrap_or(row.len() - column);
                }
                let minimum = [left.max(self.bounds.min.x), top.max(self.bounds.min.y)];
                let maximum = [
                    (self.origin[0] + column as i32 * 8).min(self.bounds.max.x),
                    (top + 8).min(self.bounds.max.y),
                ];
                let coverage_start = if cell == FULL_BLOCK {
                    SOLID_COVERAGE
                } else {
                    let Some(start) = self.compile_border(denominator, minimum, maximum, output)
                    else {
                        continue;
                    };
                    start
                };
                let is_solid = coverage_start == SOLID_COVERAGE;
                let tile = CoverageTile {
                    min: [
                        (minimum[0] - translation[0]) as u32,
                        (minimum[1] - translation[1]) as u32,
                    ],
                    max: [
                        (maximum[0] - translation[0]) as u32,
                        (maximum[1] - translation[1]) as u32,
                    ],
                    coverage_start,
                    projection_index: 0,
                };
                let mut merged = false;
                if let Some(previous) = output.tiles.last_mut() {
                    if is_solid
                        && previous.coverage_start == SOLID_COVERAGE
                        && previous.min[1] == tile.min[1]
                        && previous.max[1] == tile.max[1]
                        && previous.max[0] == tile.min[0]
                    {
                        previous.max[0] = tile.max[0];
                        merged = true;
                    }
                }
                if !merged {
                    output.tiles.push(tile);
                }
            }
        }
        let mut solid_count = 0;
        for index in 0..output.tiles.len() {
            if output.tiles[index].coverage_start == SOLID_COVERAGE {
                output.tiles.swap(index, solid_count);
                solid_count += 1;
            }
        }
        solid_count as u32
    }
}
