use super::pipelines::ClipPipelines;
use crate::core::slot_pool::SlotPool;
use crate::wgpu_backend::execution::shapes::coverage::{CoverageAllocation, CoverageBuffers};
use crate::wgpu_backend::resources::ShapePipelines;
use std::sync::Arc;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindingResource, Buffer,
    BufferUsages, Device, Queue, TextureView,
};

struct CoverageBinding {
    buffer: Buffer,
    parameters: [[f32; 4]; 2],
    storage: CoverageBuffers,
    layout: BindGroupLayout,
    binding: BindGroup,
}

struct ImageBinding {
    views: [TextureView; 3],
    layout: BindGroupLayout,
    binding: BindGroup,
}

#[derive(Default)]
pub(in crate::wgpu_backend) struct ClipBindings {
    coverage: SlotPool<Option<CoverageBinding>>,
    images: SlotPool<Option<ImageBinding>>,
}

impl ClipBindings {
    pub fn begin_render(&mut self) {
        self.coverage.reset();
        self.images.reset();
    }
    pub fn finish_render(&mut self) {
        self.coverage.truncate_unused();
        self.images.truncate_unused();
    }

    pub fn coverage(
        &mut self,
        device: &Device,
        queue: &Queue,
        pipelines: &ShapePipelines,
        allocation: &Arc<CoverageAllocation>,
        parameters: [[f32; 4]; 2],
    ) -> BindGroup {
        let slot = self.coverage.acquire(|| None);
        let layout = &pipelines.layouts.uniform;
        if slot.is_none() {
            let buffer = device.create_buffer_init(&BufferInitDescriptor {
                label: Some("clip_parameters"),
                contents: bytemuck::cast_slice(&parameters),
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            });
            let binding = allocation
                .buffers()
                .create_mask_bind_group(device, layout, &buffer);
            *slot = Some(CoverageBinding {
                buffer,
                parameters,
                storage: allocation.buffers().clone(),
                layout: layout.clone(),
                binding,
            });
        }
        let binding = slot.as_mut().expect("clip coverage binding was prepared");
        if binding.parameters != parameters {
            queue.write_buffer(&binding.buffer, 0, bytemuck::cast_slice(&parameters));
            binding.parameters = parameters;
        }
        if !binding.storage.shares_storage_with(allocation.buffers()) || binding.layout != *layout {
            binding.binding =
                allocation
                    .buffers()
                    .create_mask_bind_group(device, layout, &binding.buffer);
            binding.storage = allocation.buffers().clone();
            binding.layout = layout.clone();
        }
        binding.binding.clone()
    }

    pub fn images(
        &mut self,
        device: &Device,
        pipelines: &ClipPipelines,
        views: [&TextureView; 3],
    ) -> BindGroup {
        self.images_with_layout(device, &pipelines.images, views)
    }

    pub fn images_with_layout(
        &mut self,
        device: &Device,
        layout: &BindGroupLayout,
        views: [&TextureView; 3],
    ) -> BindGroup {
        let slot = self.images.acquire(|| None);
        let needs_binding = match slot {
            Some(binding) => {
                binding.views[0] != *views[0]
                    || binding.views[1] != *views[1]
                    || binding.views[2] != *views[2]
                    || binding.layout != *layout
            }
            None => true,
        };
        if needs_binding {
            let binding = device.create_bind_group(&BindGroupDescriptor {
                label: Some("clip_images"),
                layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::TextureView(views[0]),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::TextureView(views[1]),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: BindingResource::TextureView(views[2]),
                    },
                ],
            });
            *slot = Some(ImageBinding {
                views: [views[0].clone(), views[1].clone(), views[2].clone()],
                layout: layout.clone(),
                binding,
            });
        }
        slot.as_ref()
            .expect("clip image binding was prepared")
            .binding
            .clone()
    }
}
