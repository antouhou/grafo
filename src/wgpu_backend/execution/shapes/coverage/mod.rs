use super::buffers::upsert_gpu_buffer;
use crate::core::cache::CachedTessellation;
use crate::core::vertex::InstanceTransform;
use crate::core::{geometry as core_geometry, MathRect};
use crate::wgpu_backend::execution::render_pass::RenderPass;
use crate::wgpu_backend::resources::ShapePipelines;
pub(in crate::wgpu_backend) use cache::CoverageCache;
use cache::CoverageKey;
pub(crate) use geometry::CoverageBuilder;
use std::{mem, ops::Range, sync::Arc};
use tiles::CoverageDrawRange;
pub(crate) use tiles::CoverageGeometry;
pub(in crate::wgpu_backend) use uploads::{CoverageAllocation, CoverageUpload};
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, Buffer, BufferUsages, Device,
    IndexFormat, Queue, RenderPipeline,
};

mod cache;
mod geometry;
mod placement;
mod projection;
mod scanline;
mod tiles;
mod uploads;

pub(in crate::wgpu_backend) fn supports_projection(
    transform: Option<InstanceTransform>,
    bounds: [(f32, f32); 2],
) -> bool {
    core_geometry::transformed_bounds_to_logical_screen_rect(
        MathRect::new(bounds[0].into(), bounds[1].into()),
        transform,
    )
    .is_finite()
}

#[derive(Clone, Debug)]
pub(in crate::wgpu_backend) struct CoverageBuffers {
    tiles: Option<Buffer>,
    projections: Option<Buffer>,
    indices: Option<Buffer>,
    coverage: Option<Buffer>,
    pub bind_group: BindGroup,
    pub decrement_bind_group: BindGroup,
    range: CoverageDrawRange,
}

pub(in crate::wgpu_backend) fn create_bind_group(
    device: &Device,
    layout: &BindGroupLayout,
    uniform: &Buffer,
    storage: [&Buffer; 3],
) -> BindGroup {
    let [coverage, tiles, projections] = storage;
    device.create_bind_group(&BindGroupDescriptor {
        label: Some("path_coverage"),
        layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: uniform.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: coverage.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: tiles.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: projections.as_entire_binding(),
            },
        ],
    })
}

fn upload_buffers(
    device: &Device,
    queue: &Queue,
    mesh: CoverageUpload<'_>,
    slots: [&mut Option<Buffer>; 3],
) -> bool {
    let [tiles, projections, coverage] = slots;
    let mut needs_rebind = false;
    for (buffer, label, bytes) in [
        (
            tiles,
            "path_coverage_tiles",
            bytemuck::cast_slice(mesh.geometry.tiles.as_slice()),
        ),
        (
            projections,
            "path_coverage_projections",
            bytemuck::cast_slice(mesh.geometry.projections.as_slice()),
        ),
        (
            coverage,
            "path_coverage_values",
            bytemuck::cast_slice(mesh.geometry.coverage.as_slice()),
        ),
    ] {
        let empty = [0_u8; 64];
        let bytes = if bytes.is_empty() { &empty } else { bytes };
        needs_rebind |= buffer
            .as_ref()
            .is_none_or(|buffer| buffer.size() < bytes.len() as u64);
        upsert_gpu_buffer(
            device,
            queue,
            buffer,
            label,
            bytes,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        );
    }
    needs_rebind
}

fn prepare_indices(
    device: &Device,
    queue: &Queue,
    buffer: &mut Option<Buffer>,
    index_storage: &mut Vec<u32>,
    tile_count: usize,
) {
    let count = tile_count.max(1) as u32;
    if buffer
        .as_ref()
        .is_some_and(|buffer| buffer.size() >= u64::from(count) * 24)
    {
        return;
    }
    let first_tile = index_storage.len() as u32 / 6;
    index_storage.extend(
        (first_tile..count).flat_map(|tile| [0, 1, 2, 0, 2, 3].map(|index| tile * 4 + index)),
    );
    upsert_gpu_buffer(
        device,
        queue,
        buffer,
        "path_coverage_quad_indices",
        bytemuck::cast_slice(&index_storage[..count as usize * 6]),
        BufferUsages::INDEX | BufferUsages::COPY_DST,
    );
}

fn take_reusable_buffers(
    pool: &mut Vec<CoverageBuffers>,
    mesh: CoverageUpload<'_>,
) -> Option<CoverageBuffers> {
    let required = [
        mem::size_of_val(mesh.geometry.tiles.as_slice()),
        mem::size_of_val(mesh.geometry.projections.as_slice()),
        mem::size_of_val(mesh.geometry.coverage.as_slice()),
    ]
    .map(|size| if size == 0 { 64 } else { size as u64 });
    let index = pool
        .iter()
        .enumerate()
        .min_by_key(|(_, buffers)| {
            let capacities =
                [&buffers.tiles, &buffers.projections, &buffers.coverage].map(|buffer| {
                    buffer
                        .as_ref()
                        .expect("coverage buffers were uploaded")
                        .size()
                });
            let fits = capacities
                .into_iter()
                .zip(required)
                .all(|(capacity, required)| capacity >= required);
            let total: u64 = capacities.iter().sum();
            (!fits, if fits { total } else { u64::MAX - total })
        })
        .map(|(index, _)| index)?;
    Some(pool.swap_remove(index))
}

impl CoverageBuffers {
    fn storage(&self) -> [&Buffer; 3] {
        [&self.coverage, &self.tiles, &self.projections]
            .map(|buffer| buffer.as_ref().expect("coverage buffers were uploaded"))
    }

    pub(in crate::wgpu_backend) fn bind(&self, pass: &mut RenderPass<'_>) {
        pass.set_index_buffer(
            self.indices
                .as_ref()
                .expect("coverage indices were uploaded")
                .slice(..),
            IndexFormat::Uint32,
        );
    }

    pub(in crate::wgpu_backend) fn draw(
        &self,
        pass: &mut RenderPass<'_>,
        pipelines: [&RenderPipeline; 2],
        instances: Range<u32>,
    ) {
        self.range
            .draw(pass, |boundary| pipelines[boundary as usize], instances);
    }

    pub(in crate::wgpu_backend) fn create_mask_bind_group(
        &self,
        device: &Device,
        layout: &BindGroupLayout,
        uniform: &Buffer,
    ) -> BindGroup {
        create_bind_group(device, layout, uniform, self.storage())
    }

    pub(in crate::wgpu_backend) fn new(
        device: &Device,
        queue: &Queue,
        mesh: CoverageUpload<'_>,
        pipelines: &ShapePipelines,
        index_storage: &mut Vec<u32>,
    ) -> Self {
        let mut tiles = None;
        let mut projections = None;
        let mut indices = None;
        let mut coverage = None;
        upload_buffers(
            device,
            queue,
            mesh,
            [&mut tiles, &mut projections, &mut coverage],
        );
        prepare_indices(
            device,
            queue,
            &mut indices,
            index_storage,
            mesh.geometry.tiles.len(),
        );
        let storage = [&coverage, &tiles, &projections]
            .map(|buffer| buffer.as_ref().expect("coverage buffers were uploaded"));
        let bind_group = create_bind_group(
            device,
            &pipelines.layouts.uniform,
            &pipelines.and_uniform_buffer,
            storage,
        );
        let decrement_bind_group = create_bind_group(
            device,
            &pipelines.layouts.uniform,
            &pipelines.decrementing_uniform_buffer,
            storage,
        );
        Self {
            tiles,
            projections,
            indices,
            coverage,
            bind_group,
            decrement_bind_group,
            range: CoverageDrawRange {
                tile_start: 0,
                tile_count: mesh.geometry.tiles.len() as u32,
                solid_tile_count: mesh.solid_tile_count,
            },
        }
    }

    pub(in crate::wgpu_backend) fn update(
        &mut self,
        device: &Device,
        queue: &Queue,
        mesh: CoverageUpload<'_>,
        pipelines: &ShapePipelines,
        index_storage: &mut Vec<u32>,
    ) {
        if upload_buffers(
            device,
            queue,
            mesh,
            [&mut self.tiles, &mut self.projections, &mut self.coverage],
        ) {
            self.rebind(device, pipelines);
        }
        prepare_indices(
            device,
            queue,
            &mut self.indices,
            index_storage,
            mesh.geometry.tiles.len(),
        );
        self.range.tile_count = mesh.geometry.tiles.len() as u32;
        self.range.solid_tile_count = mesh.solid_tile_count;
    }

    pub(in crate::wgpu_backend) fn rebind(&mut self, device: &Device, pipelines: &ShapePipelines) {
        self.bind_group = create_bind_group(
            device,
            &pipelines.layouts.uniform,
            &pipelines.and_uniform_buffer,
            self.storage(),
        );
        self.decrement_bind_group = create_bind_group(
            device,
            &pipelines.layouts.uniform,
            &pipelines.decrementing_uniform_buffer,
            self.storage(),
        );
    }
}

#[derive(Clone, Debug)]
pub(in crate::wgpu_backend) struct PathCoverage {
    pub allocation: Arc<CoverageAllocation>,
    pub translation: [f32; 2],
    key: CoverageKey,
    pub tessellation: Arc<CachedTessellation>,
    pub transform: Option<InstanceTransform>,
}

#[cfg(test)]
mod tests;
