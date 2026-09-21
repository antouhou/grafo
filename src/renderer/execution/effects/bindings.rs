use crate::gradient::gpu::GpuMaterialParams;
use crate::pipeline::BackdropSamplingUniform;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, Buffer, BufferBindingType, BufferUsages,
    Device, Queue, Sampler, SamplerBindingType, ShaderStages, TextureSampleType, TextureView,
    TextureViewDimension,
};
pub(in crate::renderer) fn backdrop_layer_params(
    capture_origin: (i32, i32),
    source_size: (u32, u32),
) -> [i32; 4] {
    [
        capture_origin.0,
        capture_origin.1,
        i32::try_from(source_size.0).unwrap_or(i32::MAX),
        i32::try_from(source_size.1).unwrap_or(i32::MAX),
    ]
}

/// Creates the layout for the input texture and sampler at group 0.
pub(in crate::renderer) fn create_effect_input_bind_group_layout(
    device: &Device,
) -> BindGroupLayout {
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("effect_input_bgl"),
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    multisampled: false,
                    view_dimension: TextureViewDimension::D2,
                    sample_type: TextureSampleType::Float { filterable: true },
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Sampler(SamplerBindingType::Filtering),
                count: None,
            },
        ],
    })
}

/// Creates the user parameter uniform layout at group 1, binding 0.
pub(in crate::renderer) fn create_effect_params_bind_group_layout(
    device: &Device,
) -> BindGroupLayout {
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("effect_params_bgl"),
        entries: &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::FRAGMENT,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    })
}

pub(in crate::renderer) fn create_backdrop_layer_composite_bind_group(
    device: &Device,
    layout: &BindGroupLayout,
    foreground_view: &TextureView,
    params_buffer: &Buffer,
) -> BindGroup {
    device.create_bind_group(&BindGroupDescriptor {
        label: Some("backdrop_layer_composite_bind_group"),
        layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(foreground_view),
            },
            BindGroupEntry {
                binding: 1,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    })
}

/// Creates a texture binding for effect input or compositing.
pub(in crate::renderer) fn create_texture_sample_bind_group(
    device: &Device,
    layout: &BindGroupLayout,
    texture_view: &TextureView,
    sampler: &Sampler,
    label: Option<&str>,
) -> BindGroup {
    device.create_bind_group(&BindGroupDescriptor {
        label,
        layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(texture_view),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::Sampler(sampler),
            },
        ],
    })
}

pub(in crate::renderer) fn create_backdrop_texture_sample_bind_group(
    device: &Device,
    layout: &BindGroupLayout,
    material_params_buffer: &Buffer,
    texture_view: &TextureView,
    sampler: &Sampler,
    label: Option<&str>,
) -> BindGroup {
    device.create_bind_group(&BindGroupDescriptor {
        label,
        layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: material_params_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: BindingResource::TextureView(texture_view),
            },
            BindGroupEntry {
                binding: 4,
                resource: BindingResource::Sampler(sampler),
            },
        ],
    })
}

pub(in crate::renderer) fn prepare_solid_backdrop_material_params_buffer(
    device: &Device,
    queue: &Queue,
    backdrop_material_params_buffer: &mut Option<Buffer>,
    sampling_uniform: BackdropSamplingUniform,
) -> Buffer {
    let material_params = GpuMaterialParams::for_backdrop_sampling(sampling_uniform);

    if let Some(existing_buffer) = backdrop_material_params_buffer.as_ref() {
        queue.write_buffer(existing_buffer, 0, bytemuck::bytes_of(&material_params));
    } else {
        *backdrop_material_params_buffer = Some(device.create_buffer_init(&BufferInitDescriptor {
            label: Some("solid_backdrop_material_params_buffer"),
            contents: bytemuck::bytes_of(&material_params),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        }));
    }

    backdrop_material_params_buffer
        .as_ref()
        .expect("backdrop material params buffer should be initialized")
        .clone()
}

pub(in crate::renderer) fn prepare_backdrop_layer_params_buffer(
    device: &Device,
    queue: &Queue,
    backdrop_layer_params_buffer: &mut Option<Buffer>,
    layer_params: [i32; 4],
) -> Buffer {
    if let Some(existing_buffer) = backdrop_layer_params_buffer.as_ref() {
        queue.write_buffer(existing_buffer, 0, bytemuck::bytes_of(&layer_params));
    } else {
        *backdrop_layer_params_buffer = Some(device.create_buffer_init(&BufferInitDescriptor {
            label: Some("backdrop_layer_params_buffer"),
            contents: bytemuck::bytes_of(&layer_params),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        }));
    }

    backdrop_layer_params_buffer
        .as_ref()
        .expect("backdrop layer params buffer should be initialized")
        .clone()
}

/// Create a bind group for effect parameter uniforms.
pub(in crate::renderer) fn create_params_bind_group(
    device: &Device,
    layout: &BindGroupLayout,
    buffer: &Buffer,
) -> BindGroup {
    device.create_bind_group(&BindGroupDescriptor {
        label: Some("effect_params_bg"),
        layout,
        entries: &[BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    })
}
