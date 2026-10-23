use crate::core::slot_pool::SlotPool;
use crate::wgpu_backend::pipeline::Uniforms;
use crate::wgpu_backend::resources::ShapePipelines;
use std::mem;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{Buffer, BufferUsages, CommandEncoder, Device, Queue};

/// Target snapshots remain distinct until submission, including suspended parents.
#[derive(Default)]
pub(super) struct TargetUniformPool {
    buffers: SlotPool<Option<Buffer>>,
}

impl TargetUniformPool {
    pub(super) fn begin_render(&mut self) {
        self.buffers.reset();
    }

    pub(super) fn finish_render(&mut self) {
        self.buffers.truncate_unused();
    }

    pub(super) fn prepare(&mut self, device: &Device, queue: &Queue, uniforms: Uniforms) -> Buffer {
        let slot = self.buffers.acquire(|| None);
        let contents = bytemuck::bytes_of(&uniforms);
        let buffer = match slot {
            Some(buffer) => {
                queue.write_buffer(buffer, 0, contents);
                buffer
            }
            None => slot.insert(device.create_buffer_init(&BufferInitDescriptor {
                label: Some("target_uniform_snapshot"),
                contents,
                usage: BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            })),
        };
        buffer.clone()
    }
}

/// Encoder copies order target changes between draws that share shape bindings.
pub(super) fn activate(encoder: &mut CommandEncoder, source: &Buffer, pipelines: &ShapePipelines) {
    for destination in [
        &pipelines.and_uniform_buffer,
        &pipelines.decrementing_uniform_buffer,
    ] {
        encoder.copy_buffer_to_buffer(source, 0, destination, 0, mem::size_of::<Uniforms>() as u64);
    }
}
