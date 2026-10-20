use super::CoverageAllocation;
use crate::core::UnsignedPhysicalRect;
use smallvec::SmallVec;
use std::sync::Arc;
pub(in crate::wgpu_backend) use uploads::{CoverageBatchUploads, CoverageTileInstance};

mod uploads;

pub(super) struct CoverageBatchDraw {
    allocation: Arc<CoverageAllocation>,
    instance_index: u32,
}

pub(in crate::wgpu_backend) struct CoverageBatch {
    pub allocation: Arc<CoverageAllocation>,
    draws: SmallVec<[CoverageBatchDraw; 4]>,
    bounds: UnsignedPhysicalRect,
    first_instance_index: u32,
    instance_count: u32,
    tile_count: usize,
}

impl CoverageBatch {
    pub(in crate::wgpu_backend) fn new(
        allocation: &Arc<CoverageAllocation>,
        bounds: UnsignedPhysicalRect,
        instance_index: u32,
    ) -> Self {
        Self {
            allocation: Arc::clone(allocation),
            draws: SmallVec::new(),
            bounds,
            first_instance_index: instance_index,
            instance_count: 1,
            tile_count: allocation.range.tile_count as usize,
        }
    }

    pub(in crate::wgpu_backend) fn can_draw_instanced(&self) -> bool {
        self.draws.is_empty()
    }

    pub(in crate::wgpu_backend) fn try_append(&mut self, next: &Self) -> bool {
        // Compare only the accumulated bounds. Holes in that rectangle end the batch too.
        if next.instance_count != 1
            || !self.allocation.shares_buffers(&next.allocation)
            || self.bounds.intersects(&next.bounds)
        {
            return false;
        }
        let can_draw_instanced = self.can_draw_instanced()
            && Arc::ptr_eq(&self.allocation, &next.allocation)
            && next.first_instance_index == self.first_instance_index + self.instance_count;
        if !can_draw_instanced {
            if self.can_draw_instanced() && self.instance_count > 1 {
                return false;
            }
            if self.draws.is_empty() {
                self.draws.push(CoverageBatchDraw {
                    allocation: Arc::clone(&self.allocation),
                    instance_index: self.first_instance_index,
                });
            }
            self.draws.push(CoverageBatchDraw {
                allocation: Arc::clone(&next.allocation),
                instance_index: next.first_instance_index,
            });
            self.tile_count += next.tile_count;
        }
        self.bounds = self.bounds.union(&next.bounds);
        self.instance_count += 1;
        true
    }
}

#[cfg(test)]
mod tests;
