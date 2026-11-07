use crate::core::vertex::{CustomVertex, InstanceTransform};
use crate::wgpu_backend::pipeline;
use crate::wgpu_backend::resources::ShapePipelines;
use crate::wgpu_backend::vertex::{InstanceColor, InstanceMetadata};
use smallvec::{smallvec, SmallVec};
use std::sync::{Arc, OnceLock};
use wgpu::{
    BindGroupLayout, BlendState, ColorTargetState, ColorWrites, CompareFunction, DepthStencilState,
    Device, FragmentState, MultisampleState, PipelineCompilationOptions, PipelineLayout,
    PipelineLayoutDescriptor, RenderPipeline, RenderPipelineDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, StencilState, TextureFormat, VertexBufferLayout,
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
    Decrement,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum DepthAction {
    /// Leaves the target's depth unchanged. Shapes composite in command order.
    Ignore,
    /// For direct stroke draws that keep stencil, writes depth 0 with Less against
    /// clear depth 1, shading each sample once.
    RejectOverlap,
    /// Immediately redraws at depth 1 with Greater to clear those marks, without color
    /// or stencil writes. Must cover the same samples before another shape is drawn.
    Reset,
}

impl DepthAction {
    fn state(self, stencil: StencilAction) -> DepthStencilState {
        let mut state = match stencil {
            StencilAction::Keep => pipeline::create_equal_keep_depth_state(),
            StencilAction::Increment => pipeline::create_equal_increment_depth_state(),
            StencilAction::Decrement => pipeline::create_equal_decrement_depth_state(),
        };
        match self {
            Self::Ignore => {}
            Self::RejectOverlap => {
                state.depth_write_enabled = true;
                state.depth_compare = CompareFunction::Less;
            }
            Self::Reset => {
                state.depth_write_enabled = true;
                state.depth_compare = CompareFunction::Greater;
                // Marks pass 1 > 0. Clear samples fail 1 > 1, so stencil can be ignored.
                state.stencil = StencilState::default();
            }
        }
        state
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
            "fs_no_color"
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
    ) -> PipelineLayout {
        let material_layout = match shader {
            ShapeShaderType::Gradient => Some(&self.gradient),
            ShapeShaderType::UnderFillTexture => Some(&self.under_fill),
            ShapeShaderType::GradientUnderFillTexture => Some(&self.gradient_under_fill),
            _ => None,
        };
        let bind_group_layouts = [
            &self.uniform,
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
}

impl ShapePipelineCompiler {
    fn create(
        &self,
        layouts: &ShapePipelineLayouts,
        geometry_type: ShapeGeometryType,
        shader: ShapeShaderType,
        stencil: StencilAction,
        depth: DepthAction,
    ) -> RenderPipeline {
        let is_solid = shader == ShapeShaderType::Solid;
        let layout = layouts.create_pipeline_layout(&self.device, shader, is_solid);
        let vertex_entry_point = shader.vertex_entry_point(geometry_type, is_solid);
        let writes_color = stencil != StencilAction::Decrement && depth != DepthAction::Reset;
        let fragment_entry_point = shader.fragment_entry_point(writes_color);
        let vertex_buffer_layouts = vertex_buffer_layouts(geometry_type, is_solid);
        self.device
            .create_render_pipeline(&RenderPipelineDescriptor {
                label: Some(fragment_entry_point),
                layout: Some(&layout),
                vertex: VertexState {
                    module: &self.shader,
                    entry_point: Some(vertex_entry_point),
                    compilation_options: PipelineCompilationOptions {
                        // Pair Greater with the clear depth. The material draw uses default 0.
                        constants: if depth == DepthAction::Reset {
                            &[("tessellated_depth", 1.0)]
                        } else {
                            &[]
                        },
                        ..Default::default()
                    },
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
                depth_stencil: Some(depth.state(stencil)),
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
    decrement: OnceLock<RenderPipeline>,
}

pub(in crate::wgpu_backend) struct ShapePipelineVariants {
    compiler: ShapePipelineCompiler,
    tessellated: GeometryPipelines,
    coverage_interior: GeometryPipelines,
    coverage_boundary: GeometryPipelines,
    overlapping: MaterialPipelines,
    depth_reset: OnceLock<RenderPipeline>,
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
        };
        Self {
            compiler,
            tessellated: GeometryPipelines::default(),
            coverage_interior: GeometryPipelines::default(),
            coverage_boundary: GeometryPipelines::default(),
            overlapping: Default::default(),
            depth_reset: OnceLock::new(),
        }
    }
}

impl ShapePipelines {
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
            StencilAction::Decrement => (&pipelines.decrement, ShapeShaderType::Solid),
        };
        slot.get_or_init(|| {
            self.variants.compiler.create(
                &self.layouts,
                geometry_type,
                shader,
                stencil,
                DepthAction::Ignore,
            )
        })
    }

    pub(in crate::wgpu_backend) fn get_overlapping(
        &self,
        shader: ShapeShaderType,
    ) -> &RenderPipeline {
        self.variants.overlapping.slot(shader).get_or_init(|| {
            self.variants.compiler.create(
                &self.layouts,
                ShapeGeometryType::Tessellated,
                shader,
                StencilAction::Keep,
                DepthAction::RejectOverlap,
            )
        })
    }

    pub(in crate::wgpu_backend) fn get_depth_reset(&self) -> &RenderPipeline {
        self.variants.depth_reset.get_or_init(|| {
            self.variants.compiler.create(
                &self.layouts,
                ShapeGeometryType::Tessellated,
                ShapeShaderType::Solid,
                StencilAction::Keep,
                DepthAction::Reset,
            )
        })
    }
}
