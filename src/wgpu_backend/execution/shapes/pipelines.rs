use super::coverage::CoverageVertex;
use crate::core::vertex::{CustomVertex, InstanceTransform};
use crate::wgpu_backend::pipeline;
use crate::wgpu_backend::resources::ShapePipelines;
use crate::wgpu_backend::types::Pipeline;
use crate::wgpu_backend::vertex::{InstanceColor, InstanceMetadata};
use std::sync::Arc;
use wgpu::{
    BindGroupLayout, BlendState, ColorTargetState, ColorWrites, Device, FragmentState,
    MultisampleState, PipelineCompilationOptions, PipelineLayout, PipelineLayoutDescriptor,
    RenderPipeline, RenderPipelineDescriptor, ShaderModule, ShaderModuleDescriptor, ShaderSource,
    TextureFormat, VertexBufferLayout, VertexState,
};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(in crate::wgpu_backend) enum ShapeGeometry {
    Tessellated,
    CoverageInterior,
    CoverageBoundary,
}

impl ShapeGeometry {
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
pub(in crate::wgpu_backend) enum ShapeShader {
    Solid,
    Textured,
    Gradient,
    UnderFillTexture,
    GradientUnderFillTexture,
}

impl ShapeShader {
    pub fn uses_texture_bindings(self) -> bool {
        self != Self::Solid
    }

    fn uses_gradient(self) -> bool {
        matches!(self, Self::Gradient | Self::GradientUnderFillTexture)
    }

    fn vertex_entry_point(self, geometry: ShapeGeometry, is_solid: bool) -> &'static str {
        let uses_coverage = geometry != ShapeGeometry::Tessellated;
        match (uses_coverage, is_solid, self.uses_gradient()) {
            (false, true, _) => "vs_solid",
            (true, true, _) => "vs_coverage_solid",
            (false, false, false) => "vs_main",
            (false, false, true) => "vs_main_gradient",
            (true, false, false) => "vs_coverage",
            (true, false, true) => "vs_coverage_gradient",
        }
    }

    fn fragment_entry_point(self, stencil: StencilAction) -> &'static str {
        if stencil == StencilAction::Decrement {
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

    pub fn tracked_pipeline(self, stencil: StencilAction) -> Pipeline {
        match (self, stencil) {
            (_, StencilAction::Decrement) => Pipeline::StencilDecrement,
            (Self::Solid, StencilAction::Keep) => Pipeline::LeafDrawSolid,
            (Self::Solid, StencilAction::Increment) => Pipeline::StencilIncrementSolid,
            (Self::Textured, StencilAction::Keep) => Pipeline::LeafDraw,
            (Self::Textured, StencilAction::Increment) => Pipeline::StencilIncrement,
            (Self::Gradient, StencilAction::Keep) => Pipeline::LeafDrawGradient,
            (Self::Gradient, StencilAction::Increment) => Pipeline::StencilIncrementGradient,
            (Self::UnderFillTexture, StencilAction::Keep) => Pipeline::LeafDrawTexture,
            (Self::UnderFillTexture, StencilAction::Increment) => Pipeline::StencilIncrementTexture,
            (Self::GradientUnderFillTexture, StencilAction::Keep) => {
                Pipeline::LeafDrawGradientTexture
            }
            (Self::GradientUnderFillTexture, StencilAction::Increment) => {
                Pipeline::StencilIncrementGradientTexture
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
        shader: ShapeShader,
        is_solid: bool,
    ) -> PipelineLayout {
        let material_layout = match shader {
            ShapeShader::Gradient => Some(&self.gradient),
            ShapeShader::UnderFillTexture => Some(&self.under_fill),
            ShapeShader::GradientUnderFillTexture => Some(&self.gradient_under_fill),
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

fn vertex_buffers(geometry: ShapeGeometry, is_solid: bool) -> [VertexBufferLayout<'static>; 4] {
    let uses_coverage = geometry != ShapeGeometry::Tessellated;
    if uses_coverage && is_solid {
        return [
            CoverageVertex::solid_desc(),
            InstanceColor::desc(),
            InstanceMetadata::coverage_desc(),
            InstanceMetadata::desc(),
        ];
    }
    [
        match (uses_coverage, is_solid) {
            (false, false) => CustomVertex::desc(),
            (false, true) => CustomVertex::solid_desc(),
            (true, false) => CoverageVertex::desc(),
            (true, true) => CoverageVertex::solid_desc(),
        },
        InstanceTransform::desc(),
        InstanceColor::desc(),
        if is_solid {
            InstanceMetadata::coverage_desc()
        } else {
            InstanceMetadata::desc()
        },
    ]
}

struct ShapePipelineCompiler<'a> {
    device: &'a Device,
    layouts: &'a ShapePipelineLayouts,
    shader: ShaderModule,
    format: TextureFormat,
    sample_count: u32,
}

impl ShapePipelineCompiler<'_> {
    fn create(
        &self,
        geometry: ShapeGeometry,
        shader: ShapeShader,
        stencil: StencilAction,
    ) -> RenderPipeline {
        let is_solid = shader == ShapeShader::Solid && stencil != StencilAction::Decrement;
        let layout = self
            .layouts
            .create_pipeline_layout(self.device, shader, is_solid);
        let vertex_entry_point = shader.vertex_entry_point(geometry, is_solid);
        let fragment_entry_point = shader.fragment_entry_point(stencil);
        let buffers = vertex_buffers(geometry, is_solid);
        self.device
            .create_render_pipeline(&RenderPipelineDescriptor {
                label: Some(fragment_entry_point),
                layout: Some(&layout),
                vertex: VertexState {
                    module: &self.shader,
                    entry_point: Some(vertex_entry_point),
                    compilation_options: Default::default(),
                    buffers: &buffers[..if is_solid { 3 } else { 4 }],
                },
                fragment: Some(FragmentState {
                    module: &self.shader,
                    entry_point: Some(fragment_entry_point),
                    compilation_options: PipelineCompilationOptions {
                        constants: &[(
                            "evaluates_path_coverage",
                            f64::from(geometry == ShapeGeometry::CoverageBoundary),
                        )],
                        ..Default::default()
                    },
                    targets: &[Some(ColorTargetState {
                        format: self.format,
                        blend: Some(BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                        write_mask: if stencil == StencilAction::Decrement {
                            ColorWrites::empty()
                        } else {
                            ColorWrites::ALL
                        },
                    })],
                }),
                primitive: Default::default(),
                depth_stencil: Some(match stencil {
                    StencilAction::Keep => pipeline::create_equal_keep_depth_state(),
                    StencilAction::Increment => pipeline::create_equal_increment_depth_state(),
                    StencilAction::Decrement => pipeline::create_equal_decrement_depth_state(),
                }),
                multisample: MultisampleState {
                    count: self.sample_count,
                    ..Default::default()
                },
                multiview: None,
                cache: None,
            })
    }
}

struct GeometryPipelines {
    shaders: [[RenderPipeline; 2]; 5],
    decrement: RenderPipeline,
}

pub(in crate::wgpu_backend) struct ShapePipelineVariants {
    geometry: [GeometryPipelines; 3],
}

impl ShapePipelineVariants {
    fn new(
        device: &Device,
        layouts: &ShapePipelineLayouts,
        format: TextureFormat,
        sample_count: u32,
    ) -> Self {
        let compiler = ShapePipelineCompiler {
            device,
            layouts,
            shader: device.create_shader_module(ShaderModuleDescriptor {
                label: Some("shape_shader"),
                source: ShaderSource::Wgsl(pipeline::SHAPE_SHADER.into()),
            }),
            format,
            sample_count,
        };
        let geometry = [
            ShapeGeometry::Tessellated,
            ShapeGeometry::CoverageInterior,
            ShapeGeometry::CoverageBoundary,
        ]
        .map(|geometry| GeometryPipelines {
            shaders: [
                ShapeShader::Solid,
                ShapeShader::Textured,
                ShapeShader::Gradient,
                ShapeShader::UnderFillTexture,
                ShapeShader::GradientUnderFillTexture,
            ]
            .map(|shader| {
                [StencilAction::Keep, StencilAction::Increment]
                    .map(|stencil| compiler.create(geometry, shader, stencil))
            }),
            decrement: compiler.create(geometry, ShapeShader::Solid, StencilAction::Decrement),
        });
        Self { geometry }
    }
}

impl ShapePipelines {
    /// Creates the shape variants before any draw commands are encoded.
    pub(in crate::wgpu_backend) fn prepare(
        &mut self,
        device: &Device,
        format: TextureFormat,
        sample_count: u32,
    ) {
        if self.variants.is_none() {
            self.variants = Some(ShapePipelineVariants::new(
                device,
                &self.layouts,
                format,
                sample_count,
            ));
        }
    }

    pub(in crate::wgpu_backend) fn get(
        &self,
        geometry: ShapeGeometry,
        shader: ShapeShader,
        stencil: StencilAction,
    ) -> &RenderPipeline {
        let variants = self
            .variants
            .as_ref()
            .expect("shape pipelines are prepared before drawing");
        let pipelines = &variants.geometry[geometry as usize];
        match stencil {
            StencilAction::Decrement => &pipelines.decrement,
            _ => &pipelines.shaders[shader as usize][stencil as usize],
        }
    }
}
