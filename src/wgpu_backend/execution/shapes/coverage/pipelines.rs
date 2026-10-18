use super::CoverageVertex;
use crate::commands::ShapeDrawMaterial;
use crate::wgpu_backend::pipeline::{self, ShapePipelineDescriptor};
use std::mem;
use wgpu::{
    BindGroupLayout, BlendState, ColorTargetState, ColorWrites, Device, RenderPipeline,
    TextureFormat, VertexAttribute, VertexBufferLayout, VertexFormat, VertexStepMode,
};

impl CoverageVertex {
    pub(in crate::wgpu_backend) fn desc<'a>() -> VertexBufferLayout<'a> {
        const ATTRIBUTES: [VertexAttribute; 5] = [
            VertexAttribute {
                format: VertexFormat::Float32x2,
                offset: 0,
                shader_location: 0,
            },
            VertexAttribute {
                format: VertexFormat::Float32x3,
                offset: 8,
                shader_location: 2,
            },
            VertexAttribute {
                format: VertexFormat::Float32x3,
                offset: 20,
                shader_location: 7,
            },
            VertexAttribute {
                format: VertexFormat::Uint32,
                offset: mem::offset_of!(CoverageVertex, coverage_start) as u64,
                shader_location: 8,
            },
            VertexAttribute {
                format: VertexFormat::Uint32x2,
                offset: mem::offset_of!(CoverageVertex, tile_origin) as u64,
                shader_location: 12,
            },
        ];
        VertexBufferLayout {
            array_stride: mem::size_of::<Self>() as u64,
            step_mode: VertexStepMode::Vertex,
            attributes: &ATTRIBUTES,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum StencilAction {
    Keep,
    Increment,
    Decrement,
}

pub(in crate::wgpu_backend) struct CoverageMaterialPipelines {
    solid: [RenderPipeline; 2],
    gradient: [RenderPipeline; 2],
    texture: [RenderPipeline; 2],
    gradient_texture: [RenderPipeline; 2],
    pub decrement: RenderPipeline,
}

impl CoverageMaterialPipelines {
    pub(in crate::wgpu_backend) fn new(
        device: &Device,
        format: TextureFormat,
        sample_count: u32,
        evaluates_path_coverage: bool,
        shared: [&BindGroupLayout; 3],
        materials: [&BindGroupLayout; 3],
    ) -> Self {
        let [gradient, texture, gradient_texture] = materials;
        let create = |material_layout: Option<&BindGroupLayout>,
                      uses_gradient: bool,
                      has_under_fill_texture: bool,
                      stencil: StencilAction| {
            let mut layouts = [shared[0], shared[1], shared[2], shared[0]];
            let layout_count = if let Some(material) = material_layout {
                layouts[3] = material;
                4
            } else {
                3
            };
            pipeline::create_shape_pipeline(
                device,
                ShapePipelineDescriptor {
                    label: Some("analytic_path_coverage"),
                    bind_group_layouts: &layouts[..layout_count],
                    vertex_entry_point: if uses_gradient {
                        "vs_coverage_gradient"
                    } else {
                        "vs_coverage"
                    },
                    fragment_entry_point: if stencil == StencilAction::Decrement {
                        "fs_stencil_only"
                    } else if has_under_fill_texture {
                        if uses_gradient {
                            "fs_texture_material_gradient"
                        } else {
                            "fs_texture_material"
                        }
                    } else if uses_gradient {
                        "fs_main_gradient"
                    } else {
                        "fs_main"
                    },
                    color_target: ColorTargetState {
                        format,
                        blend: Some(BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                        write_mask: if stencil == StencilAction::Decrement {
                            ColorWrites::empty()
                        } else {
                            ColorWrites::ALL
                        },
                    },
                    depth_stencil: match stencil {
                        StencilAction::Increment => pipeline::create_equal_increment_depth_state(),
                        StencilAction::Decrement => pipeline::create_equal_decrement_depth_state(),
                        StencilAction::Keep => pipeline::create_equal_keep_depth_state(),
                    },
                    sample_count,
                    uses_coverage: true,
                    evaluates_path_coverage,
                },
            )
        };
        Self {
            solid: [StencilAction::Keep, StencilAction::Increment]
                .map(|stencil| create(None, false, false, stencil)),
            gradient: [StencilAction::Keep, StencilAction::Increment]
                .map(|stencil| create(Some(gradient), true, false, stencil)),
            texture: [StencilAction::Keep, StencilAction::Increment]
                .map(|stencil| create(Some(texture), false, true, stencil)),
            gradient_texture: [StencilAction::Keep, StencilAction::Increment]
                .map(|stencil| create(Some(gradient_texture), true, true, stencil)),
            decrement: create(None, false, false, StencilAction::Decrement),
        }
    }

    pub(in crate::wgpu_backend) fn material(
        &self,
        material: ShapeDrawMaterial,
        increments_stencil: bool,
    ) -> &RenderPipeline {
        let pipelines = match (
            material.has_gradient_fill,
            material.under_fill_texture.is_some(),
        ) {
            (false, false) => &self.solid,
            (true, false) => &self.gradient,
            (false, true) => &self.texture,
            (true, true) => &self.gradient_texture,
        };
        &pipelines[increments_stencil as usize]
    }
}

pub(in crate::wgpu_backend) struct CoveragePipelines {
    pub solid: CoverageMaterialPipelines,
    pub boundary: CoverageMaterialPipelines,
}

impl CoveragePipelines {
    pub(in crate::wgpu_backend) fn new(
        device: &Device,
        format: TextureFormat,
        sample_count: u32,
        shared: [&BindGroupLayout; 3],
        gradient: &BindGroupLayout,
        texture: &BindGroupLayout,
        gradient_texture: &BindGroupLayout,
    ) -> Self {
        let create = |evaluates_path_coverage| {
            CoverageMaterialPipelines::new(
                device,
                format,
                sample_count,
                evaluates_path_coverage,
                shared,
                [gradient, texture, gradient_texture],
            )
        };
        Self {
            solid: create(false),
            boundary: create(true),
        }
    }
}
