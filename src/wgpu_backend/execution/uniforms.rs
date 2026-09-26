use bytemuck::NoUninit;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{Buffer, BufferUsages, Device, Queue};

/// Each buffer slot keeps the same uniform layout across updates.
pub(super) fn prepare_buffer<'a>(
    buffer: &'a mut Option<Buffer>,
    device: &Device,
    queue: &Queue,
    params: &impl NoUninit,
    label: &'static str,
) -> &'a Buffer {
    let contents = bytemuck::bytes_of(params);
    match buffer {
        Some(buffer) => {
            queue.write_buffer(buffer, 0, contents);
            buffer
        }
        None => buffer.insert(device.create_buffer_init(&BufferInitDescriptor {
            label: Some(label),
            contents,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        })),
    }
}
