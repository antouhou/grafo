//! WGPU pipelines, stencil states, and buffer helpers.
use crate::core::vertex::{CustomVertex, InstanceTransform};
use crate::wgpu_backend::execution::shapes::coverage::{self, CoverageVertex};
use crate::wgpu_backend::vertex::{GeometryBufferRange, InstanceColor, InstanceMetadata};
use std::ops::Range;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroup, BindGroupLayout, BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType,
    BufferDescriptor, BufferUsages, ComputePipeline, Device, PipelineCompilationOptions,
    RenderPass, RenderPipeline, ShaderStages, Texture, TextureFormat,
};

pub(crate) const SHAPE_SHADER: &str = concat!(
    include_str!("../shaders/path_coverage.wgsl"),
    "\n",
    include_str!("../shaders/shader.wgsl"),
    "\n",
    include_str!("../shaders/path_coverage_vertex.wgsl"),
);

pub(in crate::wgpu_backend) struct ShapePipelineDescriptor<'a> {
    pub label: Option<&'a str>,
    pub bind_group_layouts: &'a [&'a BindGroupLayout],
    pub vertex_entry_point: &'a str,
    pub fragment_entry_point: &'a str,
    pub color_target: wgpu::ColorTargetState,
    pub depth_stencil: wgpu::DepthStencilState,
    pub sample_count: u32,
    pub uses_coverage: bool,
    pub evaluates_path_coverage: bool,
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

pub enum PipelineType {
    /// Increments stencil values equal to the reference, leaving other values unchanged.
    EqualIncrementStencil,
    /// Decrements the stencil value where the stencil is equal to the reference value.
    EqualDecrementStencil,
}

fn create_shape_texture_bind_group_layout(device: &Device, label: &str) -> BindGroupLayout {
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

pub(in crate::wgpu_backend) fn create_shape_pipeline(
    device: &Device,
    descriptor: ShapePipelineDescriptor<'_>,
) -> RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: descriptor.label,
        source: wgpu::ShaderSource::Wgsl(SHAPE_SHADER.into()),
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
                if descriptor.uses_coverage {
                    CoverageVertex::desc()
                } else {
                    CustomVertex::desc()
                },
                InstanceTransform::desc(),
                InstanceColor::desc(),
                InstanceMetadata::desc(),
            ],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some(descriptor.fragment_entry_point),
            compilation_options: PipelineCompilationOptions {
                constants: &[(
                    "evaluates_path_coverage",
                    f64::from(descriptor.evaluates_path_coverage),
                )],
                ..Default::default()
            },
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

pub fn create_pipeline(
    canvas_logical_size: (f32, f32),
    scale_factor: f64,
    fringe_width: f32,
    device: &Device,
    format: TextureFormat,
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
    let empty_coverage = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("empty_coverage"),
        contents: &[0; 16],
        usage: BufferUsages::STORAGE,
    });
    let bind_group =
        coverage::create_bind_group(device, &bind_group_layout, &uniform_buffer, &empty_coverage);

    let render_pipeline = create_shape_pipeline(
        device,
        ShapePipelineDescriptor {
            uses_coverage: false,
            evaluates_path_coverage: false,
            label: None,
            bind_group_layouts: &[
                &bind_group_layout,
                &texture_bind_group_layout_layer0,
                &texture_bind_group_layout_layer1,
            ],
            vertex_entry_point: "vs_main",
            fragment_entry_point,
            color_target: wgpu::ColorTargetState {
                format,
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
            uses_coverage: false,
            evaluates_path_coverage: false,
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

fn create_color_pipeline_with_stencil_keep(
    device: &Device,
    format: wgpu::TextureFormat,
    sample_count: u32,
    bind_group_layouts: &[&wgpu::BindGroupLayout],
    vertex_entry_point: &str,
    fragment_entry_point: &str,
    label: &str,
) -> RenderPipeline {
    create_shape_pipeline(
        device,
        ShapePipelineDescriptor {
            uses_coverage: false,
            evaluates_path_coverage: false,
            label: Some(label),
            bind_group_layouts,
            vertex_entry_point,
            fragment_entry_point,
            color_target: wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            },
            depth_stencil: create_equal_keep_depth_state(),
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
        create_equal_keep_depth_state()
    };
    create_shape_pipeline(
        device,
        ShapePipelineDescriptor {
            uses_coverage: false,
            evaluates_path_coverage: false,
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
