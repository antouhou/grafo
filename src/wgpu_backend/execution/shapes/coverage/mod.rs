use super::buffers::upsert_gpu_buffer;
use crate::core::cache::CachedTessellation;
use crate::core::vertex::InstanceTransform;
use crate::core::{ShapeInstance, Viewport};
use crate::wgpu_backend::resources::ShapePipelines;
use std::{mem, sync::Arc};
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, Buffer, BufferUsages, Device,
    Queue,
};

mod geometry;
pub(in crate::wgpu_backend) mod pipelines;
mod projection;
pub(crate) use geometry::{CoverageBuilder, CoverageVertex};

#[derive(Debug)]
pub(in crate::wgpu_backend) struct CoverageBuffers {
    pub vertices: Option<Buffer>,
    pub indices: Option<Buffer>,
    rows: Option<Buffer>,
    edges: Option<Buffer>,
    pub bind_group: BindGroup,
    pub decrement_bind_group: BindGroup,
    pub index_count: u32,
}

pub(in crate::wgpu_backend) fn create_bind_group(
    device: &Device,
    layout: &BindGroupLayout,
    uniform: &Buffer,
    rows: &Buffer,
    edges: &Buffer,
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
                resource: rows.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: edges.as_entire_binding(),
            },
        ],
    })
}

fn upload_buffers(
    device: &Device,
    queue: &Queue,
    mesh: &CoverageBuilder,
    slots: [&mut Option<Buffer>; 4],
) -> bool {
    let [vertices, indices, rows, edges] = slots;
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
            rows,
            "path_coverage_rows",
            bytemuck::cast_slice(&mesh.rows),
            BufferUsages::STORAGE,
        ),
        (
            edges,
            "path_coverage_edges",
            bytemuck::cast_slice(&mesh.edges),
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
        mem::size_of_val(mesh.rows.as_slice()),
        mem::size_of_val(mesh.edges.as_slice()),
    ]
    .map(|size| size.max(16) as u64);
    let index = pool
        .iter()
        .enumerate()
        .min_by_key(|(_, buffers)| {
            let capacities = [
                &buffers.vertices,
                &buffers.indices,
                &buffers.rows,
                &buffers.edges,
            ]
            .map(|buffer| {
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
            self.rows.as_ref().expect("coverage rows were uploaded"),
            self.edges.as_ref().expect("coverage edges were uploaded"),
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
        let mut rows = None;
        let mut edges = None;
        upload_buffers(
            device,
            queue,
            mesh,
            [&mut vertices, &mut indices, &mut rows, &mut edges],
        );
        let rows_buffer = rows.as_ref().expect("coverage rows were uploaded");
        let edges_buffer = edges.as_ref().expect("coverage edges were uploaded");
        let bind_group = create_bind_group(
            device,
            &pipelines.and_pipeline.get_bind_group_layout(0),
            &pipelines.and_uniform_buffer,
            rows_buffer,
            edges_buffer,
        );
        let decrement_bind_group = create_bind_group(
            device,
            &pipelines.decrementing_pipeline.get_bind_group_layout(0),
            &pipelines.decrementing_uniform_buffer,
            rows_buffer,
            edges_buffer,
        );
        Self {
            vertices,
            indices,
            rows,
            edges,
            bind_group,
            decrement_bind_group,
            index_count: mesh.indices.len() as u32,
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
            [
                &mut self.vertices,
                &mut self.indices,
                &mut self.rows,
                &mut self.edges,
            ],
        ) {
            self.rebind(device, pipelines);
        }
        self.index_count = mesh.indices.len() as u32;
    }

    pub(in crate::wgpu_backend) fn rebind(&mut self, device: &Device, pipelines: &ShapePipelines) {
        let rows = self.rows.as_ref().expect("coverage rows were uploaded");
        let edges = self.edges.as_ref().expect("coverage edges were uploaded");
        self.bind_group = create_bind_group(
            device,
            &pipelines.and_pipeline.get_bind_group_layout(0),
            &pipelines.and_uniform_buffer,
            rows,
            edges,
        );
        self.decrement_bind_group = create_bind_group(
            device,
            &pipelines.decrementing_pipeline.get_bind_group_layout(0),
            &pipelines.decrementing_uniform_buffer,
            rows,
            edges,
        );
    }
}

#[derive(Debug)]
pub(in crate::wgpu_backend) struct PathCoverage {
    pub buffers: CoverageBuffers,
    pub tessellation: Arc<CachedTessellation>,
    pub transform: Option<InstanceTransform>,
}

impl PathCoverage {
    pub(in crate::wgpu_backend) fn new(
        shape: &ShapeInstance,
        viewport: Viewport,
        builder: &mut CoverageBuilder,
        device: &Device,
        queue: &Queue,
        pipelines: &ShapePipelines,
        buffer_pool: &mut Vec<CoverageBuffers>,
    ) -> Self {
        let tessellation = Arc::clone(&shape.cached_shape.tessellation);
        let geometry = &tessellation.vertex_buffers;
        builder.build(
            &geometry.vertices,
            &geometry.indices,
            shape.transform,
            viewport.scale_factor,
            viewport.physical_size,
            tessellation.local_bounds,
        );
        let buffers = match take_reusable_buffers(buffer_pool, builder) {
            Some(mut buffers) => {
                buffers.update(device, queue, builder, pipelines);
                buffers
            }
            None => CoverageBuffers::new(device, queue, builder, pipelines),
        };
        Self {
            buffers,
            tessellation,
            transform: shape.transform,
        }
    }

    pub(in crate::wgpu_backend) fn resize(
        &mut self,
        viewport: Viewport,
        builder: &mut CoverageBuilder,
        device: &Device,
        queue: &Queue,
        pipelines: &ShapePipelines,
    ) {
        let geometry = &self.tessellation.vertex_buffers;
        builder.build(
            &geometry.vertices,
            &geometry.indices,
            self.transform,
            viewport.scale_factor,
            viewport.physical_size,
            self.tessellation.local_bounds,
        );
        self.buffers.update(device, queue, builder, pipelines);
    }
}

#[cfg(test)]
mod tests;
