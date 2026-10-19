//! WGPU pipelines, stencil states, and buffer helpers.
use crate::core::vertex::CustomVertex;
use crate::wgpu_backend::vertex::GeometryBufferRange;
use std::ops::Range;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroup, BindGroupLayout, BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType,
    BufferDescriptor, BufferUsages, ComputePipeline, Device, RenderPass, ShaderStages, Texture,
};

pub(crate) const SHAPE_SHADER: &str = concat!(
    include_str!("../shaders/path_coverage.wgsl"),
    "\n",
    include_str!("../shaders/shader.wgsl"),
    "\n",
    include_str!("../shaders/path_coverage_vertex.wgsl"),
    "\n",
    include_str!("../shaders/solid.wgsl"),
);

/// Viewport dimensions and antialiasing settings used by the vertex shader.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Uniforms {
    pub canvas_size: [f32; 2],
    /// Physical pixels per logical pixel, used to convert the configured fringe width.
    pub scale_factor: f32,
    /// Outward AA fringe width in physical pixels. Zero disables the fringe.
    pub fringe_width: f32,
}

impl Uniforms {
    pub fn new(width: f32, height: f32, scale_factor: f32, fringe_width: f32) -> Self {
        Self {
            canvas_size: [width, height],
            scale_factor,
            fringe_width,
        }
    }
}

fn create_equal_increment_stencil_state() -> wgpu::StencilState {
    // Increment matching stencil values; leave other values unchanged.
    let face_state = wgpu::StencilFaceState {
        compare: wgpu::CompareFunction::Equal,
        fail_op: wgpu::StencilOperation::Keep,
        depth_fail_op: wgpu::StencilOperation::Keep,
        pass_op: wgpu::StencilOperation::IncrementClamp,
    };

    wgpu::StencilState {
        front: face_state,
        back: face_state,
        read_mask: 0xff,
        write_mask: 0xff,
    }
}

fn create_equal_decrement_stencil_state() -> wgpu::StencilState {
    // Decrement matching stencil values; leave other values unchanged.
    let face_state = wgpu::StencilFaceState {
        compare: wgpu::CompareFunction::Equal,
        fail_op: wgpu::StencilOperation::Keep,
        depth_fail_op: wgpu::StencilOperation::Keep,
        pass_op: wgpu::StencilOperation::DecrementClamp,
    };

    wgpu::StencilState {
        front: face_state,
        back: face_state,
        read_mask: 0xff,
        write_mask: 0xff,
    }
}

fn storage_binding(binding: u32) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::FRAGMENT,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Binds viewport uniforms at vertex shader group 0, binding 0.
pub fn create_uniform_bind_group_layout(device: &Device) -> BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            storage_binding(1),
        ],
    })
}

pub fn create_equal_increment_depth_state() -> wgpu::DepthStencilState {
    wgpu::DepthStencilState {
        format: wgpu::TextureFormat::Depth24PlusStencil8,
        depth_write_enabled: false,
        depth_compare: wgpu::CompareFunction::Always,
        stencil: create_equal_increment_stencil_state(),
        bias: wgpu::DepthBiasState::default(),
    }
}

pub fn create_equal_decrement_depth_state() -> wgpu::DepthStencilState {
    wgpu::DepthStencilState {
        format: wgpu::TextureFormat::Depth24PlusStencil8,
        depth_write_enabled: false,
        depth_compare: wgpu::CompareFunction::Always,
        stencil: create_equal_decrement_stencil_state(),
        bias: wgpu::DepthBiasState::default(),
    }
}

pub(in crate::wgpu_backend) fn create_equal_keep_depth_state() -> wgpu::DepthStencilState {
    let stencil_face = wgpu::StencilFaceState {
        compare: wgpu::CompareFunction::Equal,
        fail_op: wgpu::StencilOperation::Keep,
        depth_fail_op: wgpu::StencilOperation::Keep,
        pass_op: wgpu::StencilOperation::Keep,
    };
    wgpu::DepthStencilState {
        format: wgpu::TextureFormat::Depth24PlusStencil8,
        depth_write_enabled: false,
        depth_compare: wgpu::CompareFunction::Always,
        stencil: wgpu::StencilState {
            front: stencil_face,
            back: stencil_face,
            read_mask: 0xff,
            write_mask: 0,
        },
        bias: wgpu::DepthBiasState::default(),
    }
}

pub(in crate::wgpu_backend) fn create_shape_texture_bind_group_layout(
    device: &Device,
    label: &str,
) -> BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
        label: Some(label),
    })
}

pub fn create_gradient_bind_group_layout(device: &Device) -> BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX_FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D1,
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
        label: Some("gradient_bind_group_layout"),
    })
}

pub fn create_texture_material_bind_group_layout(device: &Device) -> BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 3,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 4,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
        label: Some("texture_material_bind_group_layout"),
    })
}

pub fn create_gradient_texture_material_bind_group_layout(device: &Device) -> BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX_FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D1,
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 3,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 4,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
        label: Some("gradient_texture_material_bind_group_layout"),
    })
}

pub fn create_and_depth_texture(device: &Device, size: (u32, u32), sample_count: u32) -> Texture {
    let size = wgpu::Extent3d {
        width: size.0,
        height: size.1,
        depth_or_array_layers: 1,
    };

    device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size,
        mip_level_count: 1,
        sample_count,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth24PlusStencil8,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

/// Draws local indices using a base vertex or an offset vertex binding.
pub(crate) fn draw_indexed_geometry(
    render_pass: &mut RenderPass<'_>,
    geometry_range: GeometryBufferRange,
    vertex_buffer: &Buffer,
    supports_base_vertex: bool,
    instances: Range<u32>,
) {
    let base_vertex = if supports_base_vertex {
        geometry_range.vertex_start
    } else {
        let vertex_offset = geometry_range.vertex_start as u64 * CustomVertex::STRIDE;
        render_pass.set_vertex_buffer(0, vertex_buffer.slice(vertex_offset..));
        0
    };
    render_pass.draw_indexed(geometry_range.indices(), base_vertex, instances);
}

/// Creates the GPU pipeline that removes row padding from ARGB readback data.
pub fn create_argb_row_packing_pipeline(device: &Device) -> (BindGroupLayout, ComputePipeline) {
    let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("argb_row_packing_cs"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/argb_row_packing.wgsl").into()),
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("argb_row_packing_bgl"),
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("argb_row_packing_pl"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("argb_row_packing_pipeline"),
        layout: Some(&pipeline_layout),
        module: &cs_module,
        entry_point: Some("cs_main"),
        compilation_options: Default::default(),
        cache: None,
    });

    (bgl, pipeline)
}

/// Binds the ARGB input, output, and parameter buffers.
pub fn create_argb_row_packing_bind_group(
    device: &Device,
    bgl: &BindGroupLayout,
    input_bytes: &wgpu::Buffer,
    output_words: &wgpu::Buffer,
    params: &wgpu::Buffer,
) -> BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("argb_row_packing_bg"),
        layout: bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_bytes.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_words.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params.as_entire_binding(),
            },
        ],
    })
}

/// Compute unpadded and padded bytes-per-row given a width and bytes-per-pixel.
/// Rounds the padded value up to a multiple of `wgpu::COPY_BYTES_PER_ROW_ALIGNMENT`.
pub fn compute_padded_bytes_per_row(width: u32, bytes_per_pixel: u32) -> (u32, u32) {
    let unpadded = width * bytes_per_pixel;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded = unpadded.div_ceil(align) * align;
    (unpadded, padded)
}

/// Encode a copy from a texture to a buffer with the provided padded bytes-per-row.
pub fn encode_copy_texture_to_buffer(
    encoder: &mut wgpu::CommandEncoder,
    texture: &wgpu::Texture,
    buffer: &wgpu::Buffer,
    width: u32,
    height: u32,
    padded_bytes_per_row: u32,
) {
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
}

/// Create a CPU readback buffer of given size.
pub fn create_readback_buffer(device: &Device, label: Option<&str>, size: u64) -> wgpu::Buffer {
    device.create_buffer(&BufferDescriptor {
        label,
        size,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    })
}

/// Texture dimensions and source row stride for GPU readback packing.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ArgbRowPackingParams {
    pub width: u32,
    pub height: u32,
    pub padded_bytes_per_row: u32,
    pub _pad: u32,
}

/// Creates a uniform buffer initialized with `ArgbRowPackingParams`.
pub fn create_argb_row_packing_params_buffer(
    device: &Device,
    params: &ArgbRowPackingParams,
) -> wgpu::Buffer {
    device.create_buffer_init(&BufferInitDescriptor {
        label: Some("argb_params"),
        contents: bytemuck::bytes_of(params),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    })
}

/// Creates a multisampled color texture for MSAA rendering.
/// Call only when `sample_count > 1`.
pub fn create_msaa_color_texture(
    device: &Device,
    size: (u32, u32),
    format: wgpu::TextureFormat,
    sample_count: u32,
) -> Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("msaa_color_texture"),
        size: wgpu::Extent3d {
            width: size.0,
            height: size.1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    })
}
