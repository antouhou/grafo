use crate::commands::TexturePlacement;
use crate::core::vertex::{CustomVertex, InstanceTransform};
use crate::core::BlendMode;
use crate::wgpu_backend::pipeline;
use crate::wgpu_backend::vertex::{InstanceColor, InstanceMetadata};
use std::sync::OnceLock;
use wgpu::{
    BindGroupLayout, BlendComponent, BlendFactor, BlendOperation, BlendState, ColorTargetState,
    ColorWrites, Device, FragmentState, MultisampleState, PipelineCompilationOptions,
    PipelineLayout, PipelineLayoutDescriptor, RenderPipeline, RenderPipelineDescriptor,
    ShaderModule, ShaderModuleDescriptor, ShaderSource, TextureFormat, VertexState,
};

const BLEND_SHADER: &str = include_str!("../../../shaders/texture_blend.wgsl");

fn blend_component(source: BlendFactor, destination: BlendFactor) -> BlendComponent {
    BlendComponent {
        src_factor: source,
        dst_factor: destination,
        operation: BlendOperation::Add,
    }
}

fn fixed_blend_state(mode: BlendMode) -> Option<BlendState> {
    let (source, destination) = match mode {
        BlendMode::Clear => (BlendFactor::Zero, BlendFactor::Zero),
        BlendMode::Source => (BlendFactor::One, BlendFactor::Zero),
        BlendMode::Destination => (BlendFactor::Zero, BlendFactor::One),
        BlendMode::SourceOver => (BlendFactor::One, BlendFactor::OneMinusSrcAlpha),
        BlendMode::DestinationOver => (BlendFactor::OneMinusDstAlpha, BlendFactor::One),
        BlendMode::SourceIn => (BlendFactor::DstAlpha, BlendFactor::Zero),
        BlendMode::DestinationIn => (BlendFactor::Zero, BlendFactor::SrcAlpha),
        BlendMode::SourceOut => (BlendFactor::OneMinusDstAlpha, BlendFactor::Zero),
        BlendMode::DestinationOut => (BlendFactor::Zero, BlendFactor::OneMinusSrcAlpha),
        BlendMode::SourceAtop => (BlendFactor::DstAlpha, BlendFactor::OneMinusSrcAlpha),
        BlendMode::DestinationAtop => (BlendFactor::OneMinusDstAlpha, BlendFactor::SrcAlpha),
        BlendMode::Xor => (BlendFactor::OneMinusDstAlpha, BlendFactor::OneMinusSrcAlpha),
        BlendMode::Plus => (BlendFactor::One, BlendFactor::One),
        BlendMode::Screen => {
            return Some(BlendState {
                color: blend_component(BlendFactor::One, BlendFactor::OneMinusSrc),
                alpha: BlendState::PREMULTIPLIED_ALPHA_BLENDING.alpha,
            });
        }
        BlendMode::Exclusion => {
            return Some(BlendState {
                color: blend_component(BlendFactor::OneMinusDst, BlendFactor::OneMinusSrc),
                alpha: BlendState::PREMULTIPLIED_ALPHA_BLENDING.alpha,
            });
        }
        BlendMode::Overlay
        | BlendMode::Darken
        | BlendMode::Lighten
        | BlendMode::ColorDodge
        | BlendMode::ColorBurn
        | BlendMode::HardLight
        | BlendMode::SoftLight
        | BlendMode::Difference
        | BlendMode::Multiply
        | BlendMode::Hue
        | BlendMode::Saturation
        | BlendMode::Color
        | BlendMode::Luminosity => return None,
    };
    let component = blend_component(source, destination);
    Some(BlendState {
        color: component,
        alpha: component,
    })
}

fn shader_mode_id(mode: BlendMode) -> u32 {
    match mode {
        BlendMode::Clear => 0,
        BlendMode::Source => 1,
        BlendMode::Destination => 2,
        BlendMode::SourceOver => 3,
        BlendMode::DestinationOver => 4,
        BlendMode::SourceIn => 5,
        BlendMode::DestinationIn => 6,
        BlendMode::SourceOut => 7,
        BlendMode::DestinationOut => 8,
        BlendMode::SourceAtop => 9,
        BlendMode::DestinationAtop => 10,
        BlendMode::Xor => 11,
        BlendMode::Plus => 12,
        BlendMode::Screen => 13,
        BlendMode::Overlay => 14,
        BlendMode::Darken => 15,
        BlendMode::Lighten => 16,
        BlendMode::ColorDodge => 17,
        BlendMode::ColorBurn => 18,
        BlendMode::HardLight => 19,
        BlendMode::SoftLight => 20,
        BlendMode::Difference => 21,
        BlendMode::Exclusion => 22,
        BlendMode::Multiply => 23,
        BlendMode::Hue => 24,
        BlendMode::Saturation => 25,
        BlendMode::Color => 26,
        BlendMode::Luminosity => 27,
    }
}

pub(in crate::wgpu_backend) struct BlendPipelineResources {
    device: Device,
    shader: ShaderModule,
    source_layout: PipelineLayout,
    sampled_layout: PipelineLayout,
    format: TextureFormat,
    sample_count: u32,
    pipelines: [[OnceLock<RenderPipeline>; 2]; 28],
}

impl BlendPipelineResources {
    pub(in crate::wgpu_backend) fn new(
        device: &Device,
        format: TextureFormat,
        sample_count: u32,
        uniform_layout: &BindGroupLayout,
        texture_layout: &BindGroupLayout,
    ) -> Self {
        let source_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("source_blend_pipeline_layout"),
            bind_group_layouts: &[uniform_layout, texture_layout],
            push_constant_ranges: &[],
        });
        let sampled_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("sampled_blend_pipeline_layout"),
            bind_group_layouts: &[uniform_layout, texture_layout, texture_layout],
            push_constant_ranges: &[],
        });
        Self {
            device: device.clone(),
            shader: device.create_shader_module(ShaderModuleDescriptor {
                label: Some("texture_blend_shader"),
                source: ShaderSource::Wgsl(BLEND_SHADER.into()),
            }),
            source_layout,
            sampled_layout,
            format,
            sample_count,
            pipelines: Default::default(),
        }
    }

    fn create(&self, mode: BlendMode, placement: TexturePlacement) -> RenderPipeline {
        let fixed_blend = fixed_blend_state(mode);
        let samples_destination = fixed_blend.is_none();
        let vertex_layouts = [
            CustomVertex::desc(),
            InstanceTransform::desc(),
            InstanceColor::desc(),
            InstanceMetadata::desc(),
        ];
        let (vertex_entry_point, vertex_buffers) = match placement {
            TexturePlacement::Local { .. } => ("vs_local", vertex_layouts.as_slice()),
            TexturePlacement::Target => ("vs_target", &[][..]),
        };
        let constants = [("blend_mode", f64::from(shader_mode_id(mode)))];
        self.device
            .create_render_pipeline(&RenderPipelineDescriptor {
                label: Some("texture_blend_pipeline"),
                layout: Some(if samples_destination {
                    &self.sampled_layout
                } else {
                    &self.source_layout
                }),
                vertex: VertexState {
                    module: &self.shader,
                    entry_point: Some(vertex_entry_point),
                    compilation_options: Default::default(),
                    buffers: vertex_buffers,
                },
                fragment: Some(FragmentState {
                    module: &self.shader,
                    entry_point: Some(if samples_destination {
                        "fs_blend"
                    } else {
                        "fs_source"
                    }),
                    compilation_options: PipelineCompilationOptions {
                        constants: if samples_destination { &constants } else { &[] },
                        ..Default::default()
                    },
                    targets: &[Some(ColorTargetState {
                        format: self.format,
                        blend: Some(
                            fixed_blend.unwrap_or(BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                        ),
                        write_mask: ColorWrites::ALL,
                    })],
                }),
                primitive: Default::default(),
                depth_stencil: Some(pipeline::create_equal_keep_depth_state()),
                multisample: MultisampleState {
                    count: self.sample_count,
                    ..Default::default()
                },
                multiview: None,
                cache: None,
            })
    }

    pub(in crate::wgpu_backend) fn get(
        &self,
        mode: BlendMode,
        placement: TexturePlacement,
    ) -> &RenderPipeline {
        let placement_index = match placement {
            TexturePlacement::Local { .. } => 0,
            TexturePlacement::Target => 1,
        };
        self.pipelines[shader_mode_id(mode) as usize][placement_index]
            .get_or_init(|| self.create(mode, placement))
    }
}

#[cfg(test)]
mod tests;
