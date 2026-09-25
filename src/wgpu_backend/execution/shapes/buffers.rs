use super::ShapeExecutionResources;
use crate::wgpu_backend::resources::Buffers;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{Buffer, BufferDescriptor, BufferUsages, Device, Queue, COPY_BUFFER_ALIGNMENT};

fn upsert_gpu_buffer(
    device: &Device,
    queue: &Queue,
    buffer: &mut Option<Buffer>,
    label: &'static str,
    bytes: &[u8],
    usage: BufferUsages,
) {
    match buffer.as_ref() {
        Some(existing_buffer) if existing_buffer.size() >= bytes.len() as u64 => {
            queue.write_buffer(existing_buffer, 0, bytes);
        }
        None if bytes.is_empty() => {
            // Keep empty scenes bindable until geometry fills these buffers.
            *buffer = Some(device.create_buffer(&BufferDescriptor {
                label: Some(label),
                size: COPY_BUFFER_ALIGNMENT,
                usage,
                mapped_at_creation: false,
            }));
        }
        _ => {
            *buffer = Some(device.create_buffer_init(&BufferInitDescriptor {
                label: Some(label),
                contents: bytes,
                usage,
            }))
        }
    }
}

impl ShapeExecutionResources {
    pub(in crate::wgpu_backend) fn upload(
        &self,
        device: &Device,
        queue: &Queue,
        buffers: &mut Buffers,
    ) {
        if !self.vertices.is_empty() || buffers.aggregated_vertex_buffer.is_none() {
            upsert_gpu_buffer(
                device,
                queue,
                &mut buffers.aggregated_vertex_buffer,
                "Aggregated Vertex Buffer",
                bytemuck::cast_slice(&self.vertices),
                BufferUsages::VERTEX | BufferUsages::COPY_DST,
            );
        }

        if !self.indices.is_empty() || buffers.aggregated_index_buffer.is_none() {
            upsert_gpu_buffer(
                device,
                queue,
                &mut buffers.aggregated_index_buffer,
                "Aggregated Index Buffer",
                bytemuck::cast_slice(&self.indices),
                BufferUsages::INDEX | BufferUsages::COPY_DST,
            );
        }

        if !self.instance_transforms.is_empty() {
            upsert_gpu_buffer(
                device,
                queue,
                &mut buffers.aggregated_instance_transform_buffer,
                "Aggregated Instance Transform Buffer",
                bytemuck::cast_slice(&self.instance_transforms),
                BufferUsages::VERTEX | BufferUsages::COPY_DST,
            );
        }

        if !self.instance_colors.is_empty() {
            upsert_gpu_buffer(
                device,
                queue,
                &mut buffers.aggregated_instance_color_buffer,
                "Aggregated Instance Color Buffer",
                bytemuck::cast_slice(&self.instance_colors),
                BufferUsages::VERTEX | BufferUsages::COPY_DST,
            );
        }

        if !self.instance_metadata.is_empty() {
            upsert_gpu_buffer(
                device,
                queue,
                &mut buffers.aggregated_instance_metadata_buffer,
                "Aggregated Instance Metadata Buffer",
                bytemuck::cast_slice(&self.instance_metadata),
                BufferUsages::VERTEX | BufferUsages::COPY_DST,
            );
        }
    }
}
