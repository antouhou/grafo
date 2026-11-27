use super::filtering::ClipFilterPipeline;
use std::sync::OnceLock;
use wgpu::{
    BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType, BlendState,
    ColorTargetState, ColorWrites, CompareFunction, DepthStencilState, Device, FragmentState,
    MultisampleState, PipelineLayoutDescriptor, PrimitiveState, RenderPipeline,
    RenderPipelineDescriptor, ShaderModuleDescriptor, ShaderSource, ShaderStages, StencilFaceState,
    StencilOperation, StencilState, TextureFormat, TextureSampleType, TextureViewDimension,
    VertexState,
};
use wgpu::{PipelineLayout, ShaderModule};

pub(super) fn image_entry(
    binding: u32,
    multisampled: bool,
    filterable: bool,
) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::FRAGMENT,
        ty: BindingType::Texture {
            sample_type: TextureSampleType::Float { filterable },
            view_dimension: TextureViewDimension::D2,
            multisampled,
        },
        count: None,
    }
}

pub(super) struct ClipPipelineCompiler<'a> {
    device: &'a Device,
    layout: &'a PipelineLayout,
    shader: &'a ShaderModule,
    format: TextureFormat,
    sample_count: u32,
}

impl<'a> ClipPipelineCompiler<'a> {
    pub fn new(
        device: &'a Device,
        layout: &'a PipelineLayout,
        shader: &'a ShaderModule,
        format: TextureFormat,
        sample_count: u32,
    ) -> Self {
        Self {
            device,
            layout,
            shader,
            format,
            sample_count,
        }
    }

    pub fn compile(
        &self,
        vertex: &str,
        fragment: &str,
        stencil: Option<StencilFaceState>,
    ) -> RenderPipeline {
        let mut depth_stencil = None;
        if let Some(face) = stencil {
            depth_stencil = Some(DepthStencilState {
                format: TextureFormat::Depth24PlusStencil8,
                depth_write_enabled: Some(false),
                depth_compare: Some(CompareFunction::Always),
                stencil: StencilState {
                    front: face,
                    back: face,
                    read_mask: 0xff,
                    write_mask: if face.pass_op == StencilOperation::Replace {
                        0xff
                    } else {
                        0
                    },
                },
                bias: Default::default(),
            });
        }
        self.device
            .create_render_pipeline(&RenderPipelineDescriptor {
                label: Some("clip_group"),
                layout: Some(self.layout),
                vertex: VertexState {
                    module: self.shader,
                    entry_point: Some(vertex),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(FragmentState {
                    module: self.shader,
                    entry_point: Some(fragment),
                    compilation_options: Default::default(),
                    targets: &[Some(ColorTargetState {
                        format: self.format,
                        blend: (fragment == "isolated_fragment")
                            .then_some(BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                        write_mask: if fragment == "import_stencil" {
                            ColorWrites::empty()
                        } else {
                            ColorWrites::ALL
                        },
                    })],
                }),
                primitive: PrimitiveState::default(),
                depth_stencil,
                multisample: MultisampleState {
                    count: self.sample_count,
                    ..Default::default()
                },
                multiview_mask: None,
                cache: None,
            })
    }
}

pub(in crate::wgpu_backend) struct ClipPipelines {
    pub(super) filtering: OnceLock<Box<ClipFilterPipeline>>,
    pub export_stencil: RenderPipeline,
    pub import_stencil: RenderPipeline,
    pub copy: RenderPipeline,
    pub initialize: RenderPipeline,
    pub close: RenderPipeline,
    pub close_isolated: RenderPipeline,
    pub capture: RenderPipeline,
    pub images: BindGroupLayout,
}

impl ClipPipelines {
    pub fn new(
        device: &Device,
        format: TextureFormat,
        sample_count: u32,
        coverage_layout: &BindGroupLayout,
    ) -> Self {
        let images = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("clip_images"),
            entries: &[
                image_entry(0, sample_count > 1, false),
                image_entry(1, sample_count > 1, false),
                image_entry(2, sample_count > 1, false),
            ],
        });
        let source = include_str!("../../../shaders/clip_group.wgsl")
            .replace(
                "CLIP_TEXTURE",
                if sample_count > 1 {
                    "texture_multisampled_2d<f32>"
                } else {
                    "texture_2d<f32>"
                },
            )
            .replace(
                "CLIP_SAMPLE",
                if sample_count > 1 { "i32(sample)" } else { "0" },
            );
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("clip_group"),
            source: ShaderSource::Wgsl(source.into()),
        });
        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("clip_group"),
            bind_group_layouts: &[Some(coverage_layout), Some(&images)],
            immediate_size: 0,
        });
        let compiler = ClipPipelineCompiler::new(device, &layout, &shader, format, sample_count);
        let ignore = Some(StencilFaceState::IGNORE);
        let test = Some(StencilFaceState {
            compare: CompareFunction::Equal,
            ..StencilFaceState::IGNORE
        });
        let write = Some(StencilFaceState {
            pass_op: StencilOperation::Replace,
            ..StencilFaceState::IGNORE
        });
        Self {
            filtering: OnceLock::new(),
            export_stencil: compiler.compile("initialize_vertex", "export_stencil", test),
            import_stencil: compiler.compile("initialize_vertex", "import_stencil", write),
            copy: compiler.compile("initialize_vertex", "initialize_fragment", None),
            initialize: compiler.compile("initialize_vertex", "initialize_fragment", ignore),
            close: compiler.compile("clip_vertex", "clip_fragment", test),
            close_isolated: compiler.compile("clip_vertex", "isolated_fragment", test),
            capture: compiler.compile("clip_vertex", "capture_fragment", ignore),
            images,
        }
    }
}
