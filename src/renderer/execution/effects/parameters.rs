use super::bindings::create_params_bind_group;
use crate::effect::EffectError;
use ahash::{HashMap, HashMapExt};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{BindGroup, BindGroupLayout, Buffer, BufferUsages, Device, Queue};

/// Uploaded parameters. The attachment owns the only retained CPU copy.
pub(in crate::renderer) struct EffectParameterResources {
    buffer: Buffer,
    pub(super) bind_group: BindGroup,
}

impl EffectParameterResources {
    pub(super) fn new(device: &Device, layout: &BindGroupLayout, params: &[u8]) -> Self {
        let buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("effect_params_buffer"),
            contents: params,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });
        let bind_group = create_params_bind_group(device, layout, &buffer);
        Self { buffer, bind_group }
    }

    pub(in crate::renderer) fn update(
        &self,
        queue: &Queue,
        effect_id: u64,
        params: &[u8],
    ) -> Result<(), EffectError> {
        let expected_size = self.buffer.size();
        let actual_size = params.len() as u64;
        if actual_size != expected_size {
            return Err(EffectError::ParameterSizeMismatch {
                effect_id,
                expected_size,
                actual_size,
            });
        }
        queue.write_buffer(&self.buffer, 0, params);
        Ok(())
    }

    pub(super) fn rebind(&mut self, device: &Device, layout: &BindGroupLayout) {
        self.bind_group = create_params_bind_group(device, layout, &self.buffer);
    }
}

/// Capture bindings and uploaded parameters for one backdrop attachment.
#[derive(Default)]
pub(in crate::renderer) struct BackdropEffectResources {
    pub(in crate::renderer) parameters: Option<EffectParameterResources>,
    pub(in crate::renderer) backdrop_material_params_buffer: Option<Buffer>,
    pub(in crate::renderer) backdrop_layer_params_buffer: Option<Buffer>,
    pub(in crate::renderer) backdrop_texture_bind_group: Option<BindGroup>,
    pub(in crate::renderer) backdrop_texture_id: Option<u64>,
}

impl BackdropEffectResources {
    pub(in crate::renderer) fn invalidate_capture_binding(&mut self) {
        self.backdrop_texture_bind_group = None;
        self.backdrop_texture_id = None;
    }
}

/// Resources follow attachment mutations; rendering never scans for parameter changes.
pub(in crate::renderer) struct EffectExecutionResources {
    pub(in crate::renderer) group_parameters: HashMap<usize, EffectParameterResources>,
    pub(in crate::renderer) backdrops: HashMap<usize, BackdropEffectResources>,
}

impl EffectExecutionResources {
    pub(in crate::renderer) fn new() -> Self {
        Self {
            group_parameters: HashMap::new(),
            backdrops: HashMap::new(),
        }
    }

    pub(in crate::renderer) fn clear(&mut self) {
        self.group_parameters.clear();
        self.backdrops.clear();
    }
}
