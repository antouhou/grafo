use crate::core::UnsignedPhysicalRect;
use bytemuck::{Pod, Zeroable};
use std::mem;
use std::ops::Range;

pub(super) const TILE_SIZE: u32 = 8;

pub(super) fn tile_aligned_columns(
    minimum_x: f32,
    maximum_x: f32,
    left: u32,
    right: u32,
) -> Option<Range<u32>> {
    if maximum_x <= left as f32 || minimum_x >= right as f32 {
        return None;
    }
    let first = (minimum_x.max(left as f32) as u32 - left) / TILE_SIZE * TILE_SIZE;
    // Include the change after the last crossed pixel.
    let last = (maximum_x.ceil().min(right as f32) as u32 - left) / TILE_SIZE * TILE_SIZE;
    Some(left + first..left + last.saturating_add(TILE_SIZE).min(right - left))
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct CoverageTile {
    pub min: [u32; 2],
    pub max: [u32; 2],
    pub coverage_start: u32,
    pub projection_index: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct CoverageProjection {
    pub model_origin: [f32; 3],
    pub screen_origin_x: f32,
    pub model_dx: [f32; 3],
    pub screen_origin_y: f32,
    pub model_dy: [f32; 3],
    pub padding: f32,
    pub local_origin: [f32; 2],
    pub local_size: [f32; 2],
}

#[derive(Clone, Copy, Debug, Default)]
pub(in crate::wgpu_backend) struct CoverageDrawRange {
    pub tile_start: u32,
    pub tile_count: u32,
    pub solid_tile_count: u32,
}

pub(in crate::wgpu_backend) struct CoverageDrawMetadata {
    pub range: CoverageDrawRange,
    pub bounds: UnsignedPhysicalRect,
}

#[derive(Default)]
pub(crate) struct CoverageGeometry {
    pub tiles: Vec<CoverageTile>,
    pub projections: Vec<CoverageProjection>,
    pub coverage: Vec<f32>,
}

impl CoverageGeometry {
    pub(crate) fn clear(&mut self) {
        self.tiles.clear();
        self.projections.clear();
        self.coverage.clear();
    }

    pub(super) fn byte_len(&self) -> usize {
        mem::size_of_val(self.tiles.as_slice())
            + mem::size_of_val(self.projections.as_slice())
            + mem::size_of_val(self.coverage.as_slice())
    }
}
