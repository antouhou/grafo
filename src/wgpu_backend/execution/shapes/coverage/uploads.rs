use super::geometry::SOLID_COVERAGE;
use super::tiles::{CoverageDrawRange, CoverageGeometry};
use super::{CoverageBuffers, CoverageBuilder};
use crate::core::cache::CachedTessellation;
use crate::core::vertex::InstanceTransform;
use crate::core::{UnsignedPhysicalRect, Viewport};
use crate::wgpu_backend::execution::render_pass::RenderPass;
use crate::wgpu_backend::resources::ShapePipelines;
use std::sync::{Arc, OnceLock};
use std::{mem, ops::Range};
use wgpu::{Device, Queue, RenderPipeline};

// Limit how much unused geometry one retained allocation can keep alive.
const SHARED_BUFFER_BYTES: usize = 256 * 1024;

#[cfg(test)]
mod tests;

#[derive(Clone, Copy)]
pub(in crate::wgpu_backend) struct CoverageUpload<'a> {
    pub geometry: &'a CoverageGeometry,
    pub solid_tile_count: u32,
}

impl CoverageDrawRange {
    pub(super) fn draw<'a>(
        self,
        pass: &mut RenderPass<'_>,
        mut pipeline: impl FnMut(bool) -> &'a RenderPipeline,
        instances: Range<u32>,
    ) {
        let boundary_start = self.tile_start + self.solid_tile_count;
        for (index, tiles) in [
            self.tile_start..boundary_start,
            boundary_start..self.tile_start + self.tile_count,
        ]
        .into_iter()
        .enumerate()
        {
            if !tiles.is_empty() {
                pass.set_pipeline(pipeline(index == 1));
                pass.draw_indexed(tiles.start * 6..tiles.end * 6, 0, instances.clone());
            }
        }
    }
}

#[derive(Debug, Default)]
pub(super) struct CoverageStorage {
    pub buffers: OnceLock<CoverageBuffers>,
    pub geometry: OnceLock<CoverageGeometry>,
}

#[derive(Debug)]
pub(in crate::wgpu_backend) struct CoverageAllocation {
    pub(super) storage: Arc<CoverageStorage>,
    pub(super) range: CoverageDrawRange,
    pub bounds: UnsignedPhysicalRect,
}

impl CoverageAllocation {
    pub(in crate::wgpu_backend) fn draw_with_pipeline(
        &self,
        pass: &mut RenderPass<'_>,
        pipeline: &RenderPipeline,
        instances: Range<u32>,
    ) {
        if self.range.tile_count != 0 {
            pass.set_pipeline(pipeline);
            let start = self.range.tile_start;
            pass.draw_indexed(start * 6..(start + self.range.tile_count) * 6, 0, instances);
        }
    }

    pub(in crate::wgpu_backend) fn buffers(&self) -> &CoverageBuffers {
        self.storage
            .buffers
            .get()
            .expect("coverage uploads finish before drawing")
    }

    pub(in crate::wgpu_backend) fn draw<'a>(
        &self,
        pass: &mut RenderPass<'_>,
        pipeline: impl FnMut(bool) -> &'a RenderPipeline,
        instances: Range<u32>,
    ) {
        self.range.draw(pass, pipeline, instances);
    }
}

struct PendingCoverageUpload {
    geometry: CoverageGeometry,
    storage: Arc<CoverageStorage>,
}

#[derive(Default)]
pub(super) struct CoverageUploads {
    pending: Vec<PendingCoverageUpload>,
    storage: Vec<CoverageGeometry>,
    indices: Vec<u32>,
}

impl CoverageUploads {
    pub(super) fn clear_caches(&mut self) {
        // Live draws can still reference uploads queued before the first render.
        self.pending
            .retain(|pending| Arc::strong_count(&pending.storage) > 1);
        for pending in &mut self.pending {
            pending.geometry.clear_caches();
        }
        self.pending.shrink_to_fit();
        self.storage = Vec::new();
        self.indices = Vec::new();
    }

    pub(super) fn prepare(
        &mut self,
        builder: &mut CoverageBuilder,
        tessellation: &CachedTessellation,
        transform: Option<InstanceTransform>,
        viewport: Viewport,
    ) -> Arc<CoverageAllocation> {
        let should_start_group = self.pending.last().is_none_or(|pending| {
            pending.geometry.byte_len() >= SHARED_BUFFER_BYTES
                || pending.storage.geometry.get().is_some()
        });
        if should_start_group {
            self.pending.push(PendingCoverageUpload {
                geometry: self.storage.pop().unwrap_or_default(),
                storage: Arc::default(),
            });
        }
        let pending = self.pending.last_mut().expect("coverage upload was added");
        let projection_start = pending.geometry.projections.len();
        let coverage_start = pending.geometry.coverage.len();
        let mut coverage_draw = builder.build(
            tessellation,
            transform,
            viewport.scale_factor,
            viewport.physical_size,
            &mut pending.geometry,
        );
        if !should_start_group && pending.geometry.byte_len() > SHARED_BUFFER_BYTES {
            let mut geometry = self.storage.pop().unwrap_or_default();
            geometry
                .projections
                .extend(pending.geometry.projections.drain(projection_start..));
            geometry
                .coverage
                .extend(pending.geometry.coverage.drain(coverage_start..));
            geometry.tiles.extend(
                pending
                    .geometry
                    .tiles
                    .drain(coverage_draw.range.tile_start as usize..)
                    .map(|mut tile| {
                        tile.projection_index -= projection_start as u32;
                        if tile.coverage_start != SOLID_COVERAGE {
                            tile.coverage_start -= coverage_start as u32;
                        }
                        tile
                    }),
            );
            coverage_draw.range.tile_start = 0;
            self.pending.push(PendingCoverageUpload {
                geometry,
                storage: Arc::default(),
            });
        }
        let pending = self.pending.last().expect("coverage upload was added");
        Arc::new(CoverageAllocation {
            storage: Arc::clone(&pending.storage),
            range: coverage_draw.range,
            bounds: coverage_draw.bounds,
        })
    }

    pub(super) fn seal_pending(&mut self) {
        for pending in &mut self.pending {
            if pending.storage.geometry.get().is_none() {
                pending
                    .storage
                    .geometry
                    .set(mem::take(&mut pending.geometry))
                    .expect("pending coverage is sealed once");
            }
        }
    }

    pub(super) fn recycle_geometry(&mut self, mut geometry: CoverageGeometry) {
        geometry.clear();
        self.storage.push(geometry);
    }

    pub(super) fn upload(
        &mut self,
        device: &Device,
        queue: &Queue,
        pipelines: &ShapePipelines,
        reusable: &mut Vec<CoverageBuffers>,
    ) {
        for pending in self.pending.drain(..) {
            let mesh = CoverageUpload {
                geometry: pending.storage.geometry.get().expect("coverage was sealed"),
                solid_tile_count: 0,
            };
            let buffers = match reusable.pop() {
                Some(mut buffers) => {
                    buffers.update(device, queue, mesh, pipelines, &mut self.indices);
                    buffers
                }
                None => CoverageBuffers::new(device, queue, mesh, pipelines, &mut self.indices),
            };
            pending
                .storage
                .buffers
                .set(buffers)
                .expect("coverage buffers are uploaded once");
        }
    }

    pub(super) fn clear(&mut self) {
        for mut pending in self.pending.drain(..) {
            pending.geometry.clear();
            self.storage.push(pending.geometry);
        }
    }
}
