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

    pub(super) fn prepare(&mut self, device: &Device, queue: &Queue, uniforms: Uniforms) -> usize {
        let (index, slot) = self.buffers.acquire_indexed(|| None);
        let contents = bytemuck::bytes_of(&uniforms);
        match slot {
            Some(buffer) => queue.write_buffer(buffer, 0, contents),
            None => {
                *slot = Some(device.create_buffer_init(&BufferInitDescriptor {
                    label: Some("target_uniform_snapshot"),
                    contents,
                    usage: BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                }));
            }
        }
        index
    }

    pub(super) fn buffer(&self, index: usize) -> &Buffer {
        self.buffers
            .slot(index)
            .as_ref()
            .expect("prepared target uniforms have a buffer")
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
