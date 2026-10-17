use crate::core::vertex::CustomVertex;
use crate::wgpu_backend::execution::shapes::coverage::CoverageVertex;
use crate::wgpu_backend::pipeline;
use wgpu::{
    BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType,
    BufferBindingType, ColorTargetState, ColorWrites, Device, FragmentState, MultisampleState,
    PipelineLayoutDescriptor, PrimitiveState, RenderPipeline, RenderPipelineDescriptor,
    ShaderModuleDescriptor, ShaderSource, ShaderStages, TextureFormat, VertexState,
};
const SHAPE_MASK_SHADER: &str = include_str!("../../../shaders/shape_mask.wgsl");

fn create_mask_bind_group_layout(device: &Device) -> BindGroupLayout {
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("shape_mask_bind_group_layout"),
        entries: &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::VERTEX,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    })
}

fn create_mask_pipeline(
    device: &Device,
    format: TextureFormat,
    bind_group_layout: &BindGroupLayout,
    uses_coverage: bool,
) -> RenderPipeline {
    let shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("shape_mask_shader"),
        source: ShaderSource::Wgsl(
            format!(
                "{}\n{}",
                include_str!("../../../shaders/path_coverage.wgsl"),
                SHAPE_MASK_SHADER
            )
            .into(),
        ),
    });
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("shape_mask_pipeline_layout"),
        bind_group_layouts: &[bind_group_layout],
        push_constant_ranges: &[],
    });

    device.create_render_pipeline(&RenderPipelineDescriptor {
        label: Some("shape_mask_pipeline"),
        layout: Some(&pipeline_layout),
        vertex: VertexState {
            module: &shader,
            entry_point: Some(if uses_coverage {
                "mask_coverage_vertex"
            } else {
                "mask_vertex"
            }),
            compilation_options: Default::default(),
            buffers: &[if uses_coverage {
                CoverageVertex::desc()
            } else {
                CustomVertex::desc()
            }],
        },
        fragment: Some(FragmentState {
            module: &shader,
            entry_point: Some(if uses_coverage {
                "mask_coverage_fragment"
            } else {
                "mask_fragment"
            }),
            compilation_options: Default::default(),
            targets: &[Some(ColorTargetState {
                format,
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
        }),
        primitive: PrimitiveState::default(),
        depth_stencil: None,
        multisample: MultisampleState::default(),
        multiview: None,
        cache: None,
    })
}

pub(in crate::wgpu_backend) struct ShapeMaskEffectRendererResources {
    pub coverage_layout: BindGroupLayout,
    pub coverage_pipeline: RenderPipeline,
    pub mask_bind_group_layout: BindGroupLayout,
    pub mask_pipeline: RenderPipeline,
}

impl ShapeMaskEffectRendererResources {
    pub(in crate::wgpu_backend) fn new(device: &Device, format: TextureFormat) -> Self {
        let mask_bind_group_layout = create_mask_bind_group_layout(device);
        let mask_pipeline = create_mask_pipeline(device, format, &mask_bind_group_layout, false);
        let coverage_layout = pipeline::create_uniform_bind_group_layout(device);
        let coverage_pipeline = create_mask_pipeline(device, format, &coverage_layout, true);
        Self {
            coverage_layout,
            coverage_pipeline,
            mask_bind_group_layout,
            mask_pipeline,
        }
    }

    pub(in crate::wgpu_backend) fn recreate_pipeline(
        &mut self,
        device: &Device,
        format: TextureFormat,
    ) {
        self.mask_pipeline =
            create_mask_pipeline(device, format, &self.mask_bind_group_layout, false);
        self.coverage_pipeline = create_mask_pipeline(device, format, &self.coverage_layout, true);
    }
}
