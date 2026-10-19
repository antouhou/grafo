use super::geometry::SOLID_COVERAGE;
use super::{take_reusable_buffers, CoverageBuffers, CoverageBuilder, CoverageVertex};
use crate::core::UnsignedPhysicalRect;
use crate::wgpu_backend::resources::ShapePipelines;
use std::mem;
use std::ops::Range;
use std::sync::{Arc, OnceLock};
use wgpu::{Device, Queue, RenderPass, RenderPipeline};

// Limit how much unused geometry one retained allocation can keep alive.
const SHARED_BUFFER_BYTES: usize = 256 * 1024;

#[derive(Clone, Copy)]
pub(in crate::wgpu_backend) struct CoverageUpload<'a> {
    pub vertices: &'a [CoverageVertex],
    pub indices: &'a [u32],
    pub coverage: &'a [f32],
    pub solid_index_count: u32,
}

impl<'a> From<&'a CoverageBuilder> for CoverageUpload<'a> {
    fn from(builder: &'a CoverageBuilder) -> Self {
        Self {
            vertices: &builder.vertices,
            indices: &builder.indices,
            coverage: &builder.coverage,
            solid_index_count: builder.solid_index_count,
        }
    }
}

impl CoverageUpload<'_> {
    fn byte_len(self) -> usize {
        mem::size_of_val(self.vertices)
            + mem::size_of_val(self.indices)
            + mem::size_of_val(self.coverage)
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct CoverageDrawRange {
    pub index_start: u32,
    pub index_count: u32,
    pub solid_index_count: u32,
}

impl CoverageDrawRange {
    pub(super) fn draw<'a>(
        self,
        pass: &mut RenderPass<'_>,
        mut pipeline: impl FnMut(bool) -> &'a RenderPipeline,
        instances: Range<u32>,
    ) -> u32 {
        let boundary_start = self.index_start + self.solid_index_count;
        let mut draw_count = 0;
        for (index, indices) in [
            self.index_start..boundary_start,
            boundary_start..self.index_start + self.index_count,
        ]
        .into_iter()
        .enumerate()
        {
            if !indices.is_empty() {
                pass.set_pipeline(pipeline(index == 1));
                pass.draw_indexed(indices, 0, instances.clone());
                draw_count += 1;
            }
        }
        draw_count
    }
}

#[derive(Debug)]
pub(in crate::wgpu_backend) struct CoverageAllocation {
    pub(super) buffers: Arc<OnceLock<CoverageBuffers>>,
    pub(super) range: CoverageDrawRange,
    pub bounds: UnsignedPhysicalRect,
}

impl CoverageAllocation {
    pub(in crate::wgpu_backend) fn buffers(&self) -> &CoverageBuffers {
        self.buffers
            .get()
            .expect("coverage uploads finish before drawing")
    }

    pub(in crate::wgpu_backend) fn draw<'a>(
        &self,
        pass: &mut RenderPass<'_>,
        pipeline: impl FnMut(bool) -> &'a RenderPipeline,
        instances: Range<u32>,
    ) -> u32 {
        self.range.draw(pass, pipeline, instances)
    }
}

#[derive(Default)]
struct CoverageUploadStorage {
    vertices: Vec<CoverageVertex>,
    indices: Vec<u32>,
    coverage: Vec<f32>,
}

impl CoverageUploadStorage {
    fn as_upload(&self) -> CoverageUpload<'_> {
        CoverageUpload {
            vertices: &self.vertices,
            indices: &self.indices,
            coverage: &self.coverage,
            solid_index_count: 0,
        }
    }

    fn append(&mut self, mesh: CoverageUpload<'_>) -> CoverageDrawRange {
        let range = CoverageDrawRange {
            index_start: self.indices.len() as u32,
            index_count: mesh.indices.len() as u32,
            solid_index_count: mesh.solid_index_count,
        };
        let vertex_start = self.vertices.len() as u32;
        let coverage_start = self.coverage.len() as u32;
        self.vertices.extend(mesh.vertices.iter().map(|vertex| {
            let mut vertex = *vertex;
            if vertex.coverage_start != SOLID_COVERAGE {
                vertex.coverage_start += coverage_start;
            }
            vertex
        }));
        // Rebase indices so shared meshes do not require base-vertex support.
        self.indices
            .extend(mesh.indices.iter().map(|index| index + vertex_start));
        self.coverage.extend_from_slice(mesh.coverage);
        range
    }

    fn clear(&mut self) {
        self.vertices.clear();
        self.indices.clear();
        self.coverage.clear();
    }
}

struct PendingCoverageUpload {
    storage: CoverageUploadStorage,
    buffers: Arc<OnceLock<CoverageBuffers>>,
}

#[derive(Default)]
pub(super) struct CoverageUploads {
    pending: Vec<PendingCoverageUpload>,
    storage: Vec<CoverageUploadStorage>,
}

impl CoverageUploads {
    pub(super) fn append(&mut self, builder: &CoverageBuilder) -> Arc<CoverageAllocation> {
        let mesh = CoverageUpload::from(builder);
        if self.pending.last().is_none_or(|pending| {
            pending.storage.as_upload().byte_len() + mesh.byte_len() > SHARED_BUFFER_BYTES
        }) {
            self.pending.push(PendingCoverageUpload {
                storage: self.storage.pop().unwrap_or_default(),
                buffers: Arc::new(OnceLock::new()),
            });
        }
        let pending = self.pending.last_mut().expect("coverage upload was added");
        Arc::new(CoverageAllocation {
            buffers: Arc::clone(&pending.buffers),
            range: pending.storage.append(mesh),
            bounds: builder.bounds,
        })
    }

    pub(super) fn upload(
        &mut self,
        device: &Device,
        queue: &Queue,
        pipelines: &ShapePipelines,
        reusable: &mut Vec<CoverageBuffers>,
    ) {
        for mut pending in self.pending.drain(..) {
            let mesh = pending.storage.as_upload();
            let buffers = match take_reusable_buffers(reusable, mesh) {
                Some(mut buffers) => {
                    buffers.update(device, queue, mesh, pipelines);
                    buffers
                }
                None => CoverageBuffers::new(device, queue, mesh, pipelines),
            };
            pending
                .buffers
                .set(buffers)
                .expect("coverage buffers are uploaded once");
            pending.storage.clear();
            self.storage.push(pending.storage);
        }
    }

    pub(super) fn clear(&mut self) {
        for mut pending in self.pending.drain(..) {
            pending.storage.clear();
            self.storage.push(pending.storage);
        }
    }
}
