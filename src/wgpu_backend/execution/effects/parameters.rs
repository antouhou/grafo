use super::bindings::{create_backdrop_layer_composite_bind_group, create_params_bind_group};
use crate::core::slot_pool::SlotPool;
use crate::wgpu_backend::execution::shapes::coverage::{
    CoverageBuffers, CoverageBuilder, CoverageGeometry,
};
use crate::wgpu_backend::execution::uniforms;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{BindGroup, BindGroupLayout, Buffer, BufferUsages, Device, Queue, TextureView};

struct ParameterBinding {
    layout: BindGroupLayout,
    bind_group: BindGroup,
}

#[derive(Default)]
struct ParameterSlot {
    buffer: Option<Buffer>,
    binding: Option<ParameterBinding>,
}

/// Each execution before submission gets a distinct writable uniform buffer.
#[derive(Default)]
pub(crate) struct EffectParameterPool {
    slots: SlotPool<ParameterSlot>,
}

impl EffectParameterPool {
    fn begin_render(&mut self) {
        self.slots.reset();
    }

    fn finish_render(&mut self) {
        self.slots.truncate_unused();
    }

    fn prepare_slot(
        &mut self,
        device: &Device,
        queue: &Queue,
        params: &[u8],
    ) -> &mut ParameterSlot {
        let slot = self.slots.acquire(ParameterSlot::default);
        if slot
            .buffer
            .as_ref()
            .is_none_or(|buffer| buffer.size() < params.len() as u64)
        {
            slot.buffer = Some(device.create_buffer_init(&BufferInitDescriptor {
                label: Some("effect_params_buffer"),
                contents: params,
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            }));
            slot.binding = None;
        } else {
            queue.write_buffer(
                slot.buffer
                    .as_ref()
                    .expect("parameter buffer was allocated"),
                0,
                params,
            );
        }
        slot
    }

    pub(in crate::wgpu_backend::execution) fn prepare_buffer(
        &mut self,
        device: &Device,
        queue: &Queue,
        params: &[u8],
    ) -> &Buffer {
        self.prepare_slot(device, queue, params)
            .buffer
            .as_ref()
            .expect("parameter buffer was prepared")
    }

    pub(in crate::wgpu_backend::execution) fn prepare(
        &mut self,
        device: &Device,
        queue: &Queue,
        layout: &BindGroupLayout,
        params: &[u8],
    ) -> &BindGroup {
        let slot = self.prepare_slot(device, queue, params);
        if slot
            .binding
            .as_ref()
            .is_none_or(|binding| binding.layout != *layout)
        {
            slot.binding = Some(ParameterBinding {
                layout: layout.clone(),
                bind_group: create_params_bind_group(
                    device,
                    layout,
                    slot.buffer.as_ref().expect("parameter buffer was prepared"),
                ),
            });
        }
        &slot
            .binding
            .as_ref()
            .expect("parameter binding was prepared")
            .bind_group
    }
}

struct BackdropCompositeBinding {
    texture_view: TextureView,
    layout: BindGroupLayout,
    bind_group: BindGroup,
}

#[derive(Default)]
struct BackdropCompositeSlot {
    buffer: Option<Buffer>,
    binding: Option<BackdropCompositeBinding>,
}

/// Capture transforms cannot share writable storage until their commands are submitted.
#[derive(Default)]
pub(crate) struct BackdropCompositePool {
    slots: SlotPool<BackdropCompositeSlot>,
}

impl BackdropCompositePool {
    pub(crate) fn invalidate_bindings(&mut self) {
        for slot in self.slots.iter_mut() {
            slot.binding = None;
        }
    }

    pub(crate) fn prepare(
        &mut self,
        device: &Device,
        queue: &Queue,
        layout: &BindGroupLayout,
        foreground_view: &TextureView,
        params: [i32; 4],
    ) -> &BindGroup {
        let slot = self.slots.acquire(BackdropCompositeSlot::default);
        let buffer = uniforms::prepare_buffer(
            &mut slot.buffer,
            device,
            queue,
            &params,
            "backdrop_layer_params_buffer",
        );
        if slot.binding.as_ref().is_none_or(|binding| {
            binding.texture_view != *foreground_view || binding.layout != *layout
        }) {
            slot.binding = Some(BackdropCompositeBinding {
                texture_view: foreground_view.clone(),
                layout: layout.clone(),
                bind_group: create_backdrop_layer_composite_bind_group(
                    device,
                    layout,
                    foreground_view,
                    buffer,
                ),
            });
        }
        &slot
            .binding
            .as_ref()
            .expect("capture binding was prepared")
            .bind_group
    }
}

/// Persistent GPU allocations, independent of draw queue attachment lifetimes.
#[derive(Default)]
pub(crate) struct EffectExecutionResources {
    pub(crate) coverage_builder: CoverageBuilder,
    pub(crate) coverage_geometry: CoverageGeometry,
    pub(crate) coverage_indices: Vec<u32>,
    pub(in crate::wgpu_backend::execution) mask_coverage_buffers: SlotPool<Option<CoverageBuffers>>,
    pub(crate) parameters: EffectParameterPool,
    pub(crate) backdrop_composites: BackdropCompositePool,
}

impl EffectExecutionResources {
    pub(crate) fn begin_render(&mut self) {
        self.parameters.begin_render();
        self.backdrop_composites.slots.reset();
        self.mask_coverage_buffers.reset();
    }

    pub(crate) fn finish_render(&mut self) {
        self.parameters.finish_render();
        self.backdrop_composites.slots.truncate_unused();
        self.mask_coverage_buffers.truncate_unused();
    }
}
