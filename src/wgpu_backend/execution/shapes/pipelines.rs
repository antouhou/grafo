use crate::core::vertex::{CustomVertex, InstanceTransform};
use crate::wgpu_backend::pipeline;
use crate::wgpu_backend::resources::ShapePipelines;
use crate::wgpu_backend::vertex::{InstanceColor, InstanceMetadata};
use smallvec::{smallvec, SmallVec};
use std::sync::{Arc, OnceLock};
use wgpu::{
    BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType, BlendState,
    BufferBindingType, ColorTargetState, ColorWrites, DepthStencilState, Device, FragmentState,
    MultisampleState, PipelineCompilationOptions, PipelineLayout, PipelineLayoutDescriptor,
    RenderPipeline, RenderPipelineDescriptor, ShaderModule, ShaderModuleDescriptor, ShaderSource,
    ShaderStages, TextureFormat, TextureSampleType, TextureViewDimension, VertexBufferLayout,
    VertexState,
};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(in crate::wgpu_backend) enum ShapeGeometryType {
    Tessellated,
    CoverageInterior,
    CoverageBoundary,
}

impl ShapeGeometryType {
    pub fn coverage(is_boundary: bool) -> Self {
        if is_boundary {
            Self::CoverageBoundary
        } else {
            Self::CoverageInterior
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(in crate::wgpu_backend) enum StencilAction {
    Keep,
    Increment,
    IncrementOnly,
    Decrement,
}

impl StencilAction {
    fn depth_state(self) -> DepthStencilState {
        match self {
            Self::Keep => pipeline::create_equal_keep_depth_state(),
            Self::Increment | Self::IncrementOnly => pipeline::create_equal_increment_depth_state(),
            Self::Decrement => pipeline::create_equal_decrement_depth_state(),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(in crate::wgpu_backend) enum ShapeShaderType {
    Solid,
    Textured,
    Gradient,
    UnderFillTexture,
    GradientUnderFillTexture,
}

impl ShapeShaderType {
    pub fn uses_texture_bindings(self) -> bool {
        self != Self::Solid
    }

    fn uses_gradient(self) -> bool {
        matches!(self, Self::Gradient | Self::GradientUnderFillTexture)
    }

    fn vertex_entry_point(self, geometry_type: ShapeGeometryType, is_solid: bool) -> &'static str {
        let uses_coverage = geometry_type != ShapeGeometryType::Tessellated;
        match (uses_coverage, is_solid, self.uses_gradient()) {
            (false, true, _) => "vs_solid",
            (true, true, _) => "vs_coverage_solid",
            (false, false, false) => "vs_main",
            (false, false, true) => "vs_main_gradient",
            (true, false, false) => "vs_coverage",
            (true, false, true) => "vs_coverage_gradient",
        }
    }

    fn fragment_entry_point(self, writes_color: bool) -> &'static str {
        if !writes_color {
            "fs_stencil_only"
        } else {
            match self {
                Self::Solid => "fs_solid",
                Self::Textured => "fs_main",
                Self::Gradient => "fs_main_gradient",
                Self::UnderFillTexture => "fs_texture_material",
                Self::GradientUnderFillTexture => "fs_texture_material_gradient",
            }
        }
    }
}

pub(in crate::wgpu_backend) struct ShapePipelineLayouts {
    pub uniform: BindGroupLayout,
    pub background_texture: Arc<BindGroupLayout>,
    pub foreground_texture: Arc<BindGroupLayout>,
    pub gradient: BindGroupLayout,
    pub under_fill: BindGroupLayout,
    pub gradient_under_fill: BindGroupLayout,
}

impl ShapePipelineLayouts {
    pub fn new(device: &Device, gradient: Option<BindGroupLayout>) -> Self {
        Self {
            uniform: pipeline::create_uniform_bind_group_layout(device),
            background_texture: Arc::new(pipeline::create_shape_texture_bind_group_layout(
                device,
                "shape_background_texture",
            )),
            foreground_texture: Arc::new(pipeline::create_shape_texture_bind_group_layout(
                device,
                "shape_foreground_texture",
            )),
            gradient: gradient
                .unwrap_or_else(|| pipeline::create_gradient_bind_group_layout(device)),
            under_fill: pipeline::create_texture_material_bind_group_layout(device),
            gradient_under_fill: pipeline::create_gradient_texture_material_bind_group_layout(
                device,
            ),
        }
    }

    fn create_pipeline_layout(
        &self,
        device: &Device,
        shader: ShapeShaderType,
        is_solid: bool,
        coverage_layout: &BindGroupLayout,
    ) -> PipelineLayout {
        let material_layout = match shader {
            ShapeShaderType::Gradient => Some(&self.gradient),
            ShapeShaderType::UnderFillTexture => Some(&self.under_fill),
            ShapeShaderType::GradientUnderFillTexture => Some(&self.gradient_under_fill),
            _ => None,
        };
        let bind_group_layouts = [
            coverage_layout,
            &self.background_texture,
            &self.foreground_texture,
            material_layout.unwrap_or(&self.uniform),
        ];
        let layout_count = if is_solid {
            1
        } else if material_layout.is_some() {
            4
        } else {
            3
        };
        device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("shape_pipeline_layout"),
            bind_group_layouts: &bind_group_layouts[..layout_count],
            push_constant_ranges: &[],
        })
    }
}

fn vertex_buffer_layouts(
    geometry_type: ShapeGeometryType,
    is_solid: bool,
) -> SmallVec<[VertexBufferLayout<'static>; 4]> {
    if geometry_type != ShapeGeometryType::Tessellated {
        return smallvec![
            InstanceColor::desc(),
            if is_solid {
                InstanceMetadata::coverage_desc()
            } else {
                InstanceMetadata::desc()
            },
        ];
    }
    let mut layouts = smallvec![
        if is_solid {
            CustomVertex::solid_desc()
        } else {
            CustomVertex::desc()
        },
        InstanceTransform::desc(),
        InstanceColor::desc(),
    ];
    if !is_solid {
        layouts.push(InstanceMetadata::desc());
    }
    layouts
}

struct ShapePipelineCompiler {
    device: Device,
    shader: ShaderModule,
    format: TextureFormat,
    sample_count: u32,
    coverage_layout: Option<BindGroupLayout>,
}

impl ShapePipelineCompiler {
    fn create(
        &self,
        layouts: &ShapePipelineLayouts,
        geometry_type: ShapeGeometryType,
        shader: ShapeShaderType,
        stencil: StencilAction,
        fragment_override: Option<&'static str>,
    ) -> RenderPipeline {
        let is_solid = shader == ShapeShaderType::Solid;
        let layout = layouts.create_pipeline_layout(
            &self.device,
            shader,
            is_solid,
            self.coverage_layout.as_ref().unwrap_or(&layouts.uniform),
        );
        let vertex_entry_point = shader.vertex_entry_point(geometry_type, is_solid);
        let writes_color = matches!(stencil, StencilAction::Keep | StencilAction::Increment);
        let fragment_entry_point =
            fragment_override.unwrap_or_else(|| shader.fragment_entry_point(writes_color));
        let vertex_buffer_layouts = vertex_buffer_layouts(geometry_type, is_solid);
        self.device
            .create_render_pipeline(&RenderPipelineDescriptor {
                label: Some(fragment_entry_point),
                layout: Some(&layout),
                vertex: VertexState {
                    module: &self.shader,
                    entry_point: Some(vertex_entry_point),
                    compilation_options: Default::default(),
                    buffers: &vertex_buffer_layouts,
                },
                fragment: Some(FragmentState {
                    module: &self.shader,
                    entry_point: Some(fragment_entry_point),
                    compilation_options: PipelineCompilationOptions {
                        constants: &[(
                            "evaluates_path_coverage",
                            f64::from(geometry_type == ShapeGeometryType::CoverageBoundary),
                        )],
                        ..Default::default()
                    },
                    targets: &[Some(ColorTargetState {
                        format: self.format,
                        blend: Some(BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                        write_mask: if writes_color {
                            ColorWrites::ALL
                        } else {
                            ColorWrites::empty()
                        },
                    })],
                }),
                primitive: Default::default(),
                depth_stencil: Some(stencil.depth_state()),
                multisample: MultisampleState {
                    count: self.sample_count,
                    ..Default::default()
                },
                multiview: None,
                cache: None,
            })
    }
}

#[derive(Default)]
struct MaterialPipelines {
    solid: OnceLock<RenderPipeline>,
    textured: OnceLock<RenderPipeline>,
    gradient: OnceLock<RenderPipeline>,
    under_fill_texture: OnceLock<RenderPipeline>,
    gradient_under_fill_texture: OnceLock<RenderPipeline>,
}

impl MaterialPipelines {
    fn slot(&self, shader: ShapeShaderType) -> &OnceLock<RenderPipeline> {
        match shader {
            ShapeShaderType::Solid => &self.solid,
            ShapeShaderType::Textured => &self.textured,
            ShapeShaderType::Gradient => &self.gradient,
            ShapeShaderType::UnderFillTexture => &self.under_fill_texture,
            ShapeShaderType::GradientUnderFillTexture => &self.gradient_under_fill_texture,
        }
    }
}

#[derive(Default)]
struct GeometryPipelines {
    keep: MaterialPipelines,
    increment: MaterialPipelines,
    increment_only: OnceLock<RenderPipeline>,
    decrement: OnceLock<RenderPipeline>,
}

pub(in crate::wgpu_backend) struct ShapePipelineVariants {
    compiler: ShapePipelineCompiler,
    tessellated: GeometryPipelines,
    coverage_interior: GeometryPipelines,
    coverage_boundary: GeometryPipelines,
    stroke_compiler: OnceLock<ShapePipelineCompiler>,
    stroke_solid: OnceLock<RenderPipeline>,
    stroke_gradient: OnceLock<RenderPipeline>,
    stroke_color: OnceLock<RenderPipeline>,
}

impl ShapePipelineVariants {
    pub fn new(device: &Device, format: TextureFormat, sample_count: u32) -> Self {
        let compiler = ShapePipelineCompiler {
            device: device.clone(),
            shader: device.create_shader_module(ShaderModuleDescriptor {
                label: Some("shape_shader"),
                source: ShaderSource::Wgsl(pipeline::SHAPE_SHADER.into()),
            }),
            format,
            sample_count,
            coverage_layout: None,
        };
        Self {
            compiler,
            tessellated: GeometryPipelines::default(),
            coverage_interior: GeometryPipelines::default(),
            coverage_boundary: GeometryPipelines::default(),
            stroke_compiler: OnceLock::new(),
            stroke_solid: OnceLock::new(),
            stroke_gradient: OnceLock::new(),
            stroke_color: OnceLock::new(),
        }
    }
}

impl ShapePipelines {
    fn stroke_compiler(&self) -> &ShapePipelineCompiler {
        self.variants.stroke_compiler.get_or_init(|| {
            let source = &self.variants.compiler;
            let layout = source
                .device
                .create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: Some("stroke_coverage"),
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
                                sample_type: TextureSampleType::Float { filterable: false },
                                view_dimension: TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        pipeline::storage_binding(2, ShaderStages::VERTEX),
                        pipeline::storage_binding(3, ShaderStages::VERTEX),
                    ],
                });
            ShapePipelineCompiler {
                device: source.device.clone(),
                shader: source.device.create_shader_module(ShaderModuleDescriptor {
                    label: Some("stroke_coverage"),
                    source: ShaderSource::Wgsl(pipeline::STROKE_SHADER.into()),
                }),
                format: source.format,
                sample_count: source.sample_count,
                coverage_layout: Some(layout),
            }
        })
    }

    pub(in crate::wgpu_backend) fn stroke_layout(&self) -> &BindGroupLayout {
        self.stroke_compiler()
            .coverage_layout
            .as_ref()
            .expect("stroke atlas layout")
    }

    pub(in crate::wgpu_backend) fn stroke_pipeline(
        &self,
        shader: ShapeShaderType,
        rasterizes_color: bool,
    ) -> &RenderPipeline {
        let (slot, shader, fragment) = if rasterizes_color {
            (
                &self.variants.stroke_color,
                ShapeShaderType::Solid,
                Some("fs_stroke_color"),
            )
        } else {
            let slot = match shader {
                ShapeShaderType::Solid => &self.variants.stroke_solid,
                ShapeShaderType::Gradient => &self.variants.stroke_gradient,
                _ => unreachable!("strokes use solid or gradient paint"),
            };
            (slot, shader, None)
        };
        slot.get_or_init(|| {
            self.stroke_compiler().create(
                &self.layouts,
                ShapeGeometryType::CoverageBoundary,
                shader,
                StencilAction::Keep,
                fragment,
            )
        })
    }

    /// Creates a variant on first use and reuses it until format or MSAA changes.
    pub(in crate::wgpu_backend) fn get(
        &self,
        geometry_type: ShapeGeometryType,
        shader: ShapeShaderType,
        stencil: StencilAction,
    ) -> &RenderPipeline {
        let pipelines = match geometry_type {
            ShapeGeometryType::Tessellated => &self.variants.tessellated,
            ShapeGeometryType::CoverageInterior => &self.variants.coverage_interior,
            ShapeGeometryType::CoverageBoundary => &self.variants.coverage_boundary,
        };
        let (slot, shader) = match stencil {
            StencilAction::Keep => (pipelines.keep.slot(shader), shader),
            StencilAction::Increment => (pipelines.increment.slot(shader), shader),
            StencilAction::IncrementOnly => (&pipelines.increment_only, ShapeShaderType::Solid),
            StencilAction::Decrement => (&pipelines.decrement, ShapeShaderType::Solid),
        };
        slot.get_or_init(|| {
            self.variants
                .compiler
                .create(&self.layouts, geometry_type, shader, stencil, None)
        })
    }
}
