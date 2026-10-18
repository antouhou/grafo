use super::buffers::upsert_gpu_buffer;
use crate::core::cache::CachedTessellation;
use crate::core::vertex::InstanceTransform;
use crate::core::UnsignedPhysicalRect;
use crate::wgpu_backend::resources::ShapePipelines;
use std::{mem, ops::Range, sync::Arc};
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, Buffer, BufferUsages, Device,
    IndexFormat, Queue, RenderPass, RenderPipeline,
};

mod cache;
pub(in crate::wgpu_backend) use cache::CoverageCache;
mod geometry;
pub(in crate::wgpu_backend) mod pipelines;
mod placement;
mod projection;
mod scanline;
use cache::CoverageKey;
pub(crate) use geometry::{CoverageBuilder, CoverageVertex};

#[derive(Clone, Debug)]
pub(in crate::wgpu_backend) struct CoverageBuffers {
    pub vertices: Option<Buffer>,
    pub indices: Option<Buffer>,
    coverage: Option<Buffer>,
    pub bind_group: BindGroup,
    pub decrement_bind_group: BindGroup,
    pub index_count: u32,
    pub bounds: UnsignedPhysicalRect,
    solid_index_count: u32,
}

pub(in crate::wgpu_backend) fn create_bind_group(
    device: &Device,
    layout: &BindGroupLayout,
    uniform: &Buffer,
    coverage: &Buffer,
) -> BindGroup {
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
        ],
    })
}

fn upload_buffers(
    device: &Device,
    queue: &Queue,
    mesh: &CoverageBuilder,
    slots: [&mut Option<Buffer>; 3],
) -> bool {
    let [vertices, indices, coverage] = slots;
    let mut needs_rebind = false;
    for (buffer, label, bytes, usage) in [
        (
            vertices,
            "path_coverage_vertices",
            bytemuck::cast_slice(&mesh.vertices),
            BufferUsages::VERTEX,
        ),
        (
            indices,
            "path_coverage_indices",
            bytemuck::cast_slice(&mesh.indices),
            BufferUsages::INDEX,
        ),
        (
            coverage,
            "path_coverage_values",
            bytemuck::cast_slice(&mesh.coverage),
            BufferUsages::STORAGE,
        ),
    ] {
        let empty = [0_u8; 16];
        needs_rebind |= usage == BufferUsages::STORAGE
            && buffer
                .as_ref()
                .is_none_or(|buffer| buffer.size() < bytes.len().max(16) as u64);
        upsert_gpu_buffer(
            device,
            queue,
            buffer,
            label,
            if bytes.is_empty() { &empty } else { bytes },
            usage | BufferUsages::COPY_DST,
        );
    }
    needs_rebind
}

fn take_reusable_buffers(
    pool: &mut Vec<CoverageBuffers>,
    mesh: &CoverageBuilder,
) -> Option<CoverageBuffers> {
    let required = [
        mem::size_of_val(mesh.vertices.as_slice()),
        mem::size_of_val(mesh.indices.as_slice()),
        mem::size_of_val(mesh.coverage.as_slice()),
    ]
    .map(|size| size.max(16) as u64);
    let index = pool
        .iter()
        .enumerate()
        .min_by_key(|(_, buffers)| {
            let capacities =
                [&buffers.vertices, &buffers.indices, &buffers.coverage].map(|buffer| {
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
    pub(in crate::wgpu_backend) fn bind(&self, pass: &mut RenderPass<'_>) {
        pass.set_vertex_buffer(
            0,
            self.vertices
                .as_ref()
                .expect("coverage vertices were uploaded")
                .slice(..),
        );
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
    ) -> u32 {
        let mut draw_count = 0;
        for (indices, pipeline) in [
            0..self.solid_index_count,
            self.solid_index_count..self.index_count,
        ]
        .into_iter()
        .zip(pipelines)
        {
            if !indices.is_empty() {
                pass.set_pipeline(pipeline);
                pass.draw_indexed(indices, 0, instances.clone());
                draw_count += 1;
            }
        }
        draw_count
    }

    pub(in crate::wgpu_backend) fn mask_binding(
        &self,
        device: &Device,
        layout: &BindGroupLayout,
        uniform: &Buffer,
    ) -> BindGroup {
        create_bind_group(
            device,
            layout,
            uniform,
            self.coverage
                .as_ref()
                .expect("coverage values were uploaded"),
        )
    }
    pub(in crate::wgpu_backend) fn new(
        device: &Device,
        queue: &Queue,
        mesh: &CoverageBuilder,
        pipelines: &ShapePipelines,
    ) -> Self {
        let mut vertices = None;
        let mut indices = None;
        let mut coverage = None;
        upload_buffers(
            device,
            queue,
            mesh,
            [&mut vertices, &mut indices, &mut coverage],
        );
        let coverage_buffer = coverage.as_ref().expect("coverage values were uploaded");
        let bind_group = create_bind_group(
            device,
            &pipelines.and_pipeline.get_bind_group_layout(0),
            &pipelines.and_uniform_buffer,
            coverage_buffer,
        );
        let decrement_bind_group = create_bind_group(
            device,
            &pipelines.decrementing_pipeline.get_bind_group_layout(0),
            &pipelines.decrementing_uniform_buffer,
            coverage_buffer,
        );
        Self {
            vertices,
            indices,
            coverage,
            bind_group,
            decrement_bind_group,
            index_count: mesh.indices.len() as u32,
            bounds: mesh.bounds,
            solid_index_count: mesh.solid_index_count,
        }
    }

    pub(in crate::wgpu_backend) fn update(
        &mut self,
        device: &Device,
        queue: &Queue,
        mesh: &CoverageBuilder,
        pipelines: &ShapePipelines,
    ) {
        if upload_buffers(
            device,
            queue,
            mesh,
            [&mut self.vertices, &mut self.indices, &mut self.coverage],
        ) {
            self.rebind(device, pipelines);
        }
        self.index_count = mesh.indices.len() as u32;
        self.bounds = mesh.bounds;
        self.solid_index_count = mesh.solid_index_count;
    }

    pub(in crate::wgpu_backend) fn rebind(&mut self, device: &Device, pipelines: &ShapePipelines) {
        let coverage = self
            .coverage
            .as_ref()
            .expect("coverage values were uploaded");
        self.bind_group = create_bind_group(
            device,
            &pipelines.and_pipeline.get_bind_group_layout(0),
            &pipelines.and_uniform_buffer,
            coverage,
        );
        self.decrement_bind_group = create_bind_group(
            device,
            &pipelines.decrementing_pipeline.get_bind_group_layout(0),
            &pipelines.decrementing_uniform_buffer,
            coverage,
        );
    }
}

#[derive(Clone, Debug)]
pub(in crate::wgpu_backend) struct PathCoverage {
    pub buffers: Arc<CoverageBuffers>,
    pub translation: [f32; 2],
    key: CoverageKey,
    pub tessellation: Arc<CachedTessellation>,
    pub transform: Option<InstanceTransform>,
}

#[cfg(test)]
mod tests;
