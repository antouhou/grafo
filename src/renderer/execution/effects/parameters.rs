use super::bindings::create_params_bind_group;
use crate::effect::EffectError;
use ahash::{HashMap, HashMapExt};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{BindGroup, BindGroupLayout, Buffer, BufferUsages, Device, Queue, TextureView};

/// Uploaded parameters. The attachment owns the only retained CPU copy.
pub(crate) struct EffectParameterResources {
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

    pub(crate) fn update(
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
}

/// The view handle identifies the resource already retained by the bind group.
pub(crate) struct BackdropTextureBinding {
    pub(crate) texture_view: TextureView,
    pub(crate) bind_group: BindGroup,
}

/// Capture bindings and uploaded parameters for one backdrop attachment.
#[derive(Default)]
pub(crate) struct BackdropEffectResources {
    pub(crate) parameters: Option<EffectParameterResources>,
    pub(crate) backdrop_layer_params_buffer: Option<Buffer>,
    pub(crate) layer_composite_binding: Option<BackdropTextureBinding>,
    pub(crate) downsample_binding: Option<BackdropTextureBinding>,
}

impl BackdropEffectResources {
    /// Pipeline recreation replaces layouts; parameter buffers remain attachment-owned.
    pub(crate) fn invalidate_bindings(&mut self) {
        self.layer_composite_binding = None;
        self.downsample_binding = None;
    }
}

/// Resources follow attachment mutations; rendering never scans for parameter changes.
pub(crate) struct EffectExecutionResources {
    pub(crate) group_parameters: HashMap<usize, EffectParameterResources>,
    pub(crate) backdrops: HashMap<usize, BackdropEffectResources>,
}

impl EffectExecutionResources {
    pub(crate) fn new() -> Self {
        Self {
            group_parameters: HashMap::new(),
            backdrops: HashMap::new(),
        }
    }

    pub(crate) fn clear(&mut self) {
        self.group_parameters.clear();
        self.backdrops.clear();
    }
}
