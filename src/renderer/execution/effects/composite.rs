use super::bindings::create_effect_input_bind_group_layout;
use super::shaders::FULLSCREEN_TRIANGLE_VS;
use wgpu::{
    BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType, BlendState,
    BufferBindingType, ColorTargetState, ColorWrites, CompareFunction, DepthBiasState,
    DepthStencilState, Device, FragmentState, MultisampleState, PipelineLayoutDescriptor,
    PrimitiveState, PrimitiveTopology, RenderPipeline, RenderPipelineDescriptor,
    ShaderModuleDescriptor, ShaderSource, ShaderStages, StencilFaceState, StencilOperation,
    StencilState, TextureFormat, TextureSampleType, TextureViewDimension, VertexState,
};
/// Samples effect results for compositing into the parent target.
pub(in crate::renderer) const COMPOSITE_FS: &str =
    include_str!("../../../shaders/composite_fs.wgsl");

const BACKDROP_LAYER_COMPOSITE_FS: &str =
    include_str!("../../../shaders/backdrop_layer_composite_fs.wgsl");

pub(in crate::renderer) struct CompositePipelineResources {
    pub pipeline: RenderPipeline,
    pub bind_group_layout: BindGroupLayout,
}

/// Combines the fullscreen vertex shader and passthrough fragment shader.
pub(in crate::renderer) fn build_composite_wgsl() -> String {
    format!("{FULLSCREEN_TRIANGLE_VS}\n{COMPOSITE_FS}")
}

/// Compiles the composite pipeline, which samples the effect result and respects the parent clip.
pub(in crate::renderer) fn compile_composite_pipeline(
    device: &Device,
    format: TextureFormat,
) -> CompositePipelineResources {
    let wgsl = build_composite_wgsl();

    let shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("composite_shader"),
        source: ShaderSource::Wgsl(wgsl.into()),
    });

    let input_bind_group_layout = create_effect_input_bind_group_layout(device);

    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("composite_pipeline_layout"),
        bind_group_layouts: &[&input_bind_group_layout],
        push_constant_ranges: &[],
    });

    // Respect the parent's clip without changing stencil values.
    let stencil_face = StencilFaceState {
        compare: CompareFunction::Equal,
        fail_op: StencilOperation::Keep,
        depth_fail_op: StencilOperation::Keep,
        pass_op: StencilOperation::Keep,
    };

    let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
        label: Some("composite_pipeline"),
        layout: Some(&pipeline_layout),
        vertex: VertexState {
            module: &shader,
            entry_point: Some("vs_triangle"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(FragmentState {
            module: &shader,
            entry_point: Some("fs_composite"),
            compilation_options: Default::default(),
            targets: &[Some(ColorTargetState {
                format,
                blend: Some(BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                write_mask: ColorWrites::ALL,
            })],
        }),
        primitive: PrimitiveState {
            topology: PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: Some(DepthStencilState {
            format: TextureFormat::Depth24PlusStencil8,
            depth_write_enabled: false,
            depth_compare: CompareFunction::Always,
            stencil: StencilState {
                front: stencil_face,
                back: stencil_face,
                read_mask: 0xff,
                write_mask: 0x00,
            },
            bias: DepthBiasState::default(),
        }),
        multisample: MultisampleState::default(),
        multiview: None,
        cache: None,
    });

    CompositePipelineResources {
        pipeline,
        bind_group_layout: input_bind_group_layout,
    }
}

/// Compile a fullscreen texture-sampling pipeline without stencil/depth usage.
/// Used for capture downsampling before running the user effect shader.
pub(in crate::renderer) fn compile_texture_blit_pipeline(
    device: &Device,
    format: TextureFormat,
    input_bind_group_layout: &BindGroupLayout,
) -> RenderPipeline {
    let wgsl = build_composite_wgsl();

    let shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("texture_blit_shader"),
        source: ShaderSource::Wgsl(wgsl.into()),
    });

    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("texture_blit_pipeline_layout"),
        bind_group_layouts: &[input_bind_group_layout],
        push_constant_ranges: &[],
    });

    device.create_render_pipeline(&RenderPipelineDescriptor {
        label: Some("texture_blit_pipeline"),
        layout: Some(&pipeline_layout),
        vertex: VertexState {
            module: &shader,
            entry_point: Some("vs_triangle"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(FragmentState {
            module: &shader,
            entry_point: Some("fs_composite"),
            compilation_options: Default::default(),
            targets: &[Some(ColorTargetState {
                format,
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
        }),
        primitive: PrimitiveState {
            topology: PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: MultisampleState::default(),
        multiview: None,
        cache: None,
    })
}

/// Compile a fullscreen pipeline that overlays an already-rendered transparent group prefix
/// onto a backdrop capture using premultiplied-alpha blending.
pub(in crate::renderer) fn compile_backdrop_layer_composite_pipeline(
    device: &Device,
    format: TextureFormat,
) -> CompositePipelineResources {
    let shader_source = format!("{FULLSCREEN_TRIANGLE_VS}\n{BACKDROP_LAYER_COMPOSITE_FS}");
    let shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("backdrop_layer_composite_shader"),
        source: ShaderSource::Wgsl(shader_source.into()),
    });
    let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("backdrop_layer_composite_bind_group_layout"),
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("backdrop_layer_composite_pipeline_layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
        label: Some("backdrop_layer_composite_pipeline"),
        layout: Some(&pipeline_layout),
        vertex: VertexState {
            module: &shader,
            entry_point: Some("vs_triangle"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(FragmentState {
            module: &shader,
            entry_point: Some("fs_backdrop_layer"),
            compilation_options: Default::default(),
            targets: &[Some(ColorTargetState {
                format,
                blend: Some(BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                write_mask: ColorWrites::ALL,
            })],
        }),
        primitive: PrimitiveState::default(),
        depth_stencil: None,
        multisample: MultisampleState::default(),
        multiview: None,
        cache: None,
    });

    CompositePipelineResources {
        pipeline,
        bind_group_layout,
    }
}
