use crate::core::UnsignedPhysicalRect;
use bytemuck::{Pod, Zeroable};
use std::mem;

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
