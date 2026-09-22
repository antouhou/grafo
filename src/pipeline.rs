//! WGPU pipelines, stencil states, and buffer helpers.
use crate::vertex::{
    CustomVertex, GeometryBufferRange, InstanceColor, InstanceMetadata, InstanceTransform,
};
use std::ops::Range;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroup, BindGroupLayout, Buffer, BufferDescriptor, BufferUsages, ComputePipeline, Device,
    RenderPass, RenderPipeline, StoreOp, Texture, TextureView,
};

struct ShapePipelineDescriptor<'a> {
    label: Option<&'a str>,
    bind_group_layouts: &'a [&'a BindGroupLayout],
    vertex_entry_point: &'a str,
    fragment_entry_point: &'a str,
    color_target: wgpu::ColorTargetState,
    depth_stencil: wgpu::DepthStencilState,
    sample_count: u32,
}

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

/// Binds viewport uniforms at vertex shader group 0, binding 0.
pub fn create_uniform_bind_group_layout(device: &Device) -> BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    })
}

pub fn create_equal_increment_depth_state() -> wgpu::DepthStencilState {
    wgpu::DepthStencilState {
        format: wgpu::TextureFormat::Depth24PlusStencil8,
        depth_write_enabled: true,
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

pub enum PipelineType {
    /// Increments stencil values equal to the reference, leaving other values unchanged.
    EqualIncrementStencil,
    /// Decrements the stencil value where the stencil is equal to the reference value.
    EqualDecrementStencil,
}

fn create_shape_texture_bind_group_layout(device: &Device, label: &str) -> BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
        label: Some(label),
    })
}

fn create_shape_pipeline(
    device: &Device,
    descriptor: ShapePipelineDescriptor<'_>,
) -> RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: descriptor.label,
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/shader.wgsl").into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: descriptor.label,
        bind_group_layouts: descriptor.bind_group_layouts,
        push_constant_ranges: &[],
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: descriptor.label,
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some(descriptor.vertex_entry_point),
            compilation_options: Default::default(),
            buffers: &[
                CustomVertex::desc(),
                InstanceTransform::desc(),
                InstanceColor::desc(),
                InstanceMetadata::desc(),
            ],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some(descriptor.fragment_entry_point),
            compilation_options: Default::default(),
            targets: &[Some(descriptor.color_target)],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: Some(descriptor.depth_stencil),
        multisample: wgpu::MultisampleState {
            count: descriptor.sample_count,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview: None,
        cache: None,
    })
}

pub fn create_gradient_bind_group_layout(device: &Device) -> BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D1,
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
        label: Some("gradient_bind_group_layout"),
    })
}

pub fn create_texture_material_bind_group_layout(device: &Device) -> BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
        label: Some("texture_material_bind_group_layout"),
    })
}

pub fn create_gradient_texture_material_bind_group_layout(device: &Device) -> BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D1,
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
        label: Some("gradient_texture_material_bind_group_layout"),
    })
}

pub fn create_pipeline(
    canvas_logical_size: (f32, f32),
    scale_factor: f64,
    fringe_width: f32,
    device: &Device,
    config: &wgpu::SurfaceConfiguration,
    pipeline_type: PipelineType,
    sample_count: u32,
) -> (
    Uniforms,
    wgpu::Buffer,
    BindGroup,
    BindGroupLayout,
    BindGroupLayout,
    RenderPipeline,
) {
    let (depth_stencil, color_writes, fragment_entry_point) = match pipeline_type {
        PipelineType::EqualIncrementStencil => (
            create_equal_increment_depth_state(),
            wgpu::ColorWrites::ALL,
            "fs_passthrough",
        ),
        PipelineType::EqualDecrementStencil => (
            create_equal_decrement_depth_state(),
            wgpu::ColorWrites::empty(),
            "fs_stencil_only",
        ),
    };
    let uniforms = Uniforms::new(
        canvas_logical_size.0,
        canvas_logical_size.1,
        scale_factor as f32,
        fringe_width,
    );

    let uniform_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&[uniforms]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let bind_group_layout = create_uniform_bind_group_layout(device);
    let texture_bind_group_layout_layer0 =
        create_shape_texture_bind_group_layout(device, "shape_texture_bind_group_layout_layer0");
    let texture_bind_group_layout_layer1 =
        create_shape_texture_bind_group_layout(device, "shape_texture_bind_group_layout_layer1");
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform_buffer.as_entire_binding(),
        }],
        label: None,
    });

    let render_pipeline = create_shape_pipeline(
        device,
        ShapePipelineDescriptor {
            label: None,
            bind_group_layouts: &[
                &bind_group_layout,
                &texture_bind_group_layout_layer0,
                &texture_bind_group_layout_layer1,
            ],
            vertex_entry_point: "vs_main",
            fragment_entry_point,
            color_target: wgpu::ColorTargetState {
                format: config.format,
                blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                write_mask: color_writes,
            },
            depth_stencil,
            sample_count,
        },
    );

    (
        uniforms,
        uniform_buffer,
        bind_group,
        texture_bind_group_layout_layer0,
        texture_bind_group_layout_layer1,
        render_pipeline,
    )
}

pub fn create_gradient_increment_pipeline(
    device: &Device,
    format: wgpu::TextureFormat,
    sample_count: u32,
    uniform_layout: &BindGroupLayout,
    background_texture_layout: &BindGroupLayout,
    foreground_texture_layout: &BindGroupLayout,
    gradient_layout: &BindGroupLayout,
) -> RenderPipeline {
    create_shape_pipeline(
        device,
        ShapePipelineDescriptor {
            label: Some("gradient_increment_pipeline"),
            bind_group_layouts: &[
                uniform_layout,
                background_texture_layout,
                foreground_texture_layout,
                gradient_layout,
            ],
            vertex_entry_point: "vs_main_gradient",
            fragment_entry_point: "fs_passthrough_gradient",
            color_target: wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            },
            depth_stencil: create_equal_increment_depth_state(),
            sample_count,
        },
    )
}

pub struct RenderPassLoadOperations {
    pub color_load_op: wgpu::LoadOp<wgpu::Color>,
    pub depth_load_op: wgpu::LoadOp<f32>,
    pub stencil_load_op: wgpu::LoadOp<u32>,
}

pub fn begin_render_pass_with_load_ops<'a, 'b: 'a>(
    encoder: &'a mut wgpu::CommandEncoder,
    label: Option<&'a str>,
    color_texture_view: &'b TextureView,
    resolve_target: Option<&'b TextureView>,
    depth_texture_view: &'b TextureView,
    load_operations: RenderPassLoadOperations,
) -> RenderPass<'a> {
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label,
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: color_texture_view,
            resolve_target,
            ops: wgpu::Operations {
                load: load_operations.color_load_op,
                store: StoreOp::Store,
            },
        })],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: depth_texture_view,
            depth_ops: Some(wgpu::Operations {
                load: load_operations.depth_load_op,
                store: StoreOp::Store,
            }),
            stencil_ops: Some(wgpu::Operations {
                load: load_operations.stencil_load_op,
                store: StoreOp::Store,
            }),
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
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

/// Creates an offscreen color texture for rendering and copying.
pub fn create_offscreen_color_texture(
    device: &Device,
    size: (u32, u32),
    format: wgpu::TextureFormat,
) -> Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("offscreen_render_texture"),
        size: wgpu::Extent3d {
            width: size.0,
            height: size.1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

/// Creates the GPU pipeline that removes row padding from ARGB readback data.
pub fn create_argb_row_packing_pipeline(device: &Device) -> (BindGroupLayout, ComputePipeline) {
    let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("argb_row_packing_cs"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/argb_row_packing.wgsl").into()),
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("argb_row_packing_bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
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

/// Creates a pipeline that increments stencil without writing color.
pub fn create_stencil_only_pipeline(
    device: &Device,
    format: wgpu::TextureFormat,
    sample_count: u32,
    uniform_layout: &BindGroupLayout,
    background_texture_layout: &BindGroupLayout,
    foreground_texture_layout: &BindGroupLayout,
) -> RenderPipeline {
    create_shape_pipeline(
        device,
        ShapePipelineDescriptor {
            label: Some("stencil_only_pipeline"),
            bind_group_layouts: &[
                uniform_layout,
                background_texture_layout,
                foreground_texture_layout,
            ],
            vertex_entry_point: "vs_main",
            fragment_entry_point: "fs_stencil_only",
            color_target: wgpu::ColorTargetState {
                format,
                blend: None,
                write_mask: wgpu::ColorWrites::empty(),
            },
            depth_stencil: create_equal_increment_depth_state(),
            sample_count,
        },
    )
}

fn create_color_pipeline_with_stencil_keep(
    device: &Device,
    format: wgpu::TextureFormat,
    sample_count: u32,
    bind_group_layouts: &[&wgpu::BindGroupLayout],
    vertex_entry_point: &str,
    fragment_entry_point: &str,
    label: &str,
) -> RenderPipeline {
    let stencil_face = wgpu::StencilFaceState {
        compare: wgpu::CompareFunction::Equal,
        fail_op: wgpu::StencilOperation::Keep,
        depth_fail_op: wgpu::StencilOperation::Keep,
        pass_op: wgpu::StencilOperation::Keep,
    };

    create_shape_pipeline(
        device,
        ShapePipelineDescriptor {
            label: Some(label),
            bind_group_layouts,
            vertex_entry_point,
            fragment_entry_point,
            color_target: wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            },
            depth_stencil: wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth24PlusStencil8,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: wgpu::StencilState {
                    front: stencil_face,
                    back: stencil_face,
                    read_mask: 0xff,
                    write_mask: 0x00,
                },
                bias: wgpu::DepthBiasState::default(),
            },
            sample_count,
        },
    )
}

/// Creates a pipeline that draws shape color without changing the stencil value.
pub fn create_stencil_keep_color_pipeline(
    device: &Device,
    format: wgpu::TextureFormat,
    sample_count: u32,
    uniform_bgl: &wgpu::BindGroupLayout,
    texture_bgl_layer0: &wgpu::BindGroupLayout,
    texture_bgl_layer1: &wgpu::BindGroupLayout,
) -> RenderPipeline {
    create_color_pipeline_with_stencil_keep(
        device,
        format,
        sample_count,
        &[uniform_bgl, texture_bgl_layer0, texture_bgl_layer1],
        "vs_main",
        "fs_main",
        "stencil_keep_color_pipeline",
    )
}

pub fn create_gradient_stencil_keep_color_pipeline(
    device: &Device,
    format: wgpu::TextureFormat,
    sample_count: u32,
    uniform_bgl: &wgpu::BindGroupLayout,
    texture_bgl_layer0: &wgpu::BindGroupLayout,
    texture_bgl_layer1: &wgpu::BindGroupLayout,
    gradient_bgl: &wgpu::BindGroupLayout,
) -> RenderPipeline {
    create_color_pipeline_with_stencil_keep(
        device,
        format,
        sample_count,
        &[
            uniform_bgl,
            texture_bgl_layer0,
            texture_bgl_layer1,
            gradient_bgl,
        ],
        "vs_main_gradient",
        "fs_main_gradient",
        "gradient_stencil_keep_color_pipeline",
    )
}

/// Compiles a shape material with a texture below its fill.
pub(crate) fn create_texture_material_pipeline(
    device: &Device,
    format: wgpu::TextureFormat,
    sample_count: u32,
    layouts: &[&BindGroupLayout],
    uses_gradient: bool,
    increments_stencil: bool,
) -> RenderPipeline {
    let depth_stencil = if increments_stencil {
        create_equal_increment_depth_state()
    } else {
        let stencil_face = wgpu::StencilFaceState {
            compare: wgpu::CompareFunction::Equal,
            fail_op: wgpu::StencilOperation::Keep,
            depth_fail_op: wgpu::StencilOperation::Keep,
            pass_op: wgpu::StencilOperation::Keep,
        };
        wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth24PlusStencil8,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Always,
            stencil: wgpu::StencilState {
                front: stencil_face,
                back: stencil_face,
                read_mask: 0xff,
                write_mask: 0,
            },
            bias: wgpu::DepthBiasState::default(),
        }
    };
    create_shape_pipeline(
        device,
        ShapePipelineDescriptor {
            label: Some("shape_texture_material_pipeline"),
            bind_group_layouts: layouts,
            vertex_entry_point: if uses_gradient {
                "vs_main_gradient"
            } else {
                "vs_main"
            },
            fragment_entry_point: if uses_gradient {
                "fs_texture_material_gradient"
            } else {
                "fs_texture_material"
            },
            color_target: wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            },
            depth_stencil,
            sample_count,
        },
    )
}
