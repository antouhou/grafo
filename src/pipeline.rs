//! Pipeline creation and management for the Grafo library.
//!
//! This module provides functions to create and manage rendering pipelines.

use crate::vertex::{CustomVertex, TexturedVertex};
use wgpu::util::DeviceExt;
use wgpu::{
    BindGroup, BindGroupLayout, Device, RenderPass, RenderPipeline, StencilFaceState, StoreOp,
    Texture, TextureView,
};

/// A structure for coordinate normalization on the GPU. We pass pixel coordinates to the GPU,
///  but GPU needs coordinates to be normalized between 0 and 1.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Uniforms {
    pub canvas_size: [f32; 2],
}

impl Uniforms {
    pub fn new(width: f32, height: f32) -> Self {
        Self {
            canvas_size: [width, height],
        }
    }
}

fn create_equal_increment_stencil_state() -> wgpu::StencilState {
    // In this stencil state we will only draw where the stencil value is equal to the reference value,
    //  and all outside areas are zeroed.
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
    // In this stencil state we will only draw where the stencil value is equal to the reference value,
    //  and all outside areas are zeroed.
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

// /// This stencil state always replaces the stencil value with the reference value.
// fn create_always_stencil_state() -> wgpu::StencilState {
//     // In this stencil state we will only draw where the stencil value is equal to the reference value,
//     //  and all outside areas are zeroed.
//     let face_state = wgpu::StencilFaceState {
//         compare: wgpu::CompareFunction::Always,
//         fail_op: wgpu::StencilOperation::Replace,
//         depth_fail_op: wgpu::StencilOperation::Replace,
//         pass_op: wgpu::StencilOperation::Replace,
//     };
//
//     wgpu::StencilState {
//         front: face_state,
//         back: face_state,
//         read_mask: 0xff,
//         write_mask: 0xff,
//     }
// }

// /// This stencil state always replaces the stencil value with the reference value.
// fn create_always_decrement_stencil_state() -> wgpu::StencilState {
//     // In this stencil state we will only draw where the stencil value is equal to the reference value,
//     //  and all outside areas are zeroed.
//     let face_state = wgpu::StencilFaceState {
//         compare: wgpu::CompareFunction::Always,
//         fail_op: wgpu::StencilOperation::Keep,
//         depth_fail_op: wgpu::StencilOperation::Keep,
//         pass_op: wgpu::StencilOperation::DecrementClamp,
//     };
//
//     wgpu::StencilState {
//         front: face_state,
//         back: face_state,
//         read_mask: 0xff,
//         write_mask: 0xff,
//     }
// }

/// Creates a bind group so uniforms can be processed. Look at Uniforms struct for more info.
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

/// This depth stencil state ignores stencil and draws only where the depth value is equal to the
/// reference value
pub fn create_depth_stencil_state_for_text() -> wgpu::DepthStencilState {
    wgpu::DepthStencilState {
        format: wgpu::TextureFormat::Depth24PlusStencil8,
        depth_write_enabled: true,
        // Draw only where depth value is equal to the reference value set by the shape when it
        //  was drawn. When the shape is drawn, it always replaces the depth value with the reference
        //  value for itself
        depth_compare: wgpu::CompareFunction::Equal,
        stencil: wgpu::StencilState {
            front: StencilFaceState::IGNORE,
            back: StencilFaceState::IGNORE,
            read_mask: 0,
            write_mask: 0,
        },
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

// pub fn create_always_replace_depth_stencil_state() -> wgpu::DepthStencilState {
//     wgpu::DepthStencilState {
//         format: wgpu::TextureFormat::Depth24PlusStencil8,
//         depth_write_enabled: true,
//         depth_compare: wgpu::CompareFunction::Always,
//         stencil: create_always_stencil_state(),
//         bias: wgpu::DepthBiasState::default(),
//     }
// }

pub fn write_on_equal_depth_stencil_state() -> wgpu::DepthStencilState {
    let face_state = wgpu::StencilFaceState {
        compare: wgpu::CompareFunction::Equal,
        fail_op: wgpu::StencilOperation::Keep,
        depth_fail_op: wgpu::StencilOperation::Keep,
        pass_op: wgpu::StencilOperation::Replace,
    };

    let stencil = wgpu::StencilState {
        front: face_state,
        back: face_state,
        read_mask: 0xff,
        write_mask: 0xff,
    };

    wgpu::DepthStencilState {
        format: wgpu::TextureFormat::Depth24PlusStencil8,
        depth_write_enabled: false,
        depth_compare: wgpu::CompareFunction::Always,
        stencil,
        bias: wgpu::DepthBiasState::default(),
    }
}

pub fn always_pass_and_keep_stencil_state() -> wgpu::DepthStencilState {
    let face_state = wgpu::StencilFaceState {
        compare: wgpu::CompareFunction::Always,
        fail_op: wgpu::StencilOperation::Keep,
        depth_fail_op: wgpu::StencilOperation::Keep,
        pass_op: wgpu::StencilOperation::Keep,
    };

    let stencil = wgpu::StencilState {
        front: face_state,
        back: face_state,
        read_mask: 0xff,
        write_mask: 0xff,
    };

    wgpu::DepthStencilState {
        format: wgpu::TextureFormat::Depth24PlusStencil8,
        depth_write_enabled: false,
        depth_compare: wgpu::CompareFunction::Always,
        stencil,
        bias: wgpu::DepthBiasState::default(),
    }
}

// pub fn create_always_decrement_depth_stencil_state() -> wgpu::DepthStencilState {
//     wgpu::DepthStencilState {
//         format: wgpu::TextureFormat::Depth24PlusStencil8,
//         depth_write_enabled: true,
//         depth_compare: wgpu::CompareFunction::Always,
//         stencil: create_always_decrement_stencil_state(),
//         bias: wgpu::DepthBiasState::default(),
//     }
// }

pub enum PipelineType {
    /// Keeps values where the stencil is equal to the reference value, zeros outside areas.
    /// I.e. keeps intersection between stencil buffer and what's being rendered.
    EqualIncrementStencil,
    // /// Always replaces the stencil value with the reference value.
    // AlwaysReplaceStencil,
    // /// Always decrements the stencil value.
    // AlwaysDecrementStencil,
    /// Decrements the stencil value where the stencil is equal to the reference value.
    EqualDecrementStencil,
}

pub fn create_pipeline(
    canvas_logical_size: (f32, f32),
    device: &Device,
    config: &wgpu::SurfaceConfiguration,
    pipeline_type: PipelineType,
) -> (Uniforms, BindGroup, RenderPipeline) {
    let (depth_stencil_state, targets) = match pipeline_type {
        PipelineType::EqualIncrementStencil => (
            create_equal_increment_depth_state(),
            [Some(wgpu::ColorTargetState {
                format: config.format,
                blend: Some(wgpu::BlendState {
                    color: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::SrcAlpha,
                        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        operation: wgpu::BlendOperation::Add,
                    },
                    alpha: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        operation: wgpu::BlendOperation::Add,
                    },
                }),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        ),
        // PipelineType::AlwaysReplaceStencil => (
        //     create_always_replace_depth_stencil_state(),
        //     [Some(wgpu::ColorTargetState {
        //         format: config.format,
        //         blend: Some(wgpu::BlendState {
        //             color: wgpu::BlendComponent {
        //                 src_factor: wgpu::BlendFactor::SrcAlpha,
        //                 dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
        //                 operation: wgpu::BlendOperation::Add,
        //             },
        //             alpha: wgpu::BlendComponent {
        //                 src_factor: wgpu::BlendFactor::One,
        //                 dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
        //                 operation: wgpu::BlendOperation::Add,
        //             },
        //         }),
        //         write_mask: wgpu::ColorWrites::ALL,
        //     })],
        // ),
        // PipelineType::AlwaysDecrementStencil => {
        //     (create_always_decrement_depth_stencil_state(), [None])
        // }
        PipelineType::EqualDecrementStencil => (
            create_equal_decrement_depth_state(),
            [Some(wgpu::ColorTargetState {
                format: config.format,
                blend: Some(wgpu::BlendState {
                    color: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::SrcAlpha,
                        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        operation: wgpu::BlendOperation::Add,
                    },
                    alpha: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        operation: wgpu::BlendOperation::Add,
                    },
                }),
                write_mask: wgpu::ColorWrites::empty(),
            })],
        ),
    };
    let uniforms = Uniforms::new(canvas_logical_size.0, canvas_logical_size.1);

    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&[uniforms]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // Bind group for uniforms
    let bind_group_layout = create_uniform_bind_group_layout(device);

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform_buffer.as_entire_binding(),
        }],
        label: None,
    });

    // Create the render pipeline
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/shader.wgsl").into()),
    });

    let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: Some(&render_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[CustomVertex::desc()],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_main",
            targets: &targets,
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: Some(depth_stencil_state),
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    });

    (uniforms, bind_group, render_pipeline)
}

pub fn create_render_pass<'a, 'b: 'a>(
    encoder: &'a mut wgpu::CommandEncoder,
    output_texture_view: &'b TextureView,
    depth_texture_view: &'b TextureView,
) -> RenderPass<'a> {
    let render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: None,
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: output_texture_view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: StoreOp::Store,
            },
        })],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: depth_texture_view,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Clear(1.0), // Clear to maximum depth
                store: StoreOp::Store,
            }),
            stencil_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Clear(0), // Clear to 0
                store: StoreOp::Store,
            }),
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
    });

    render_pass
}

pub fn create_and_depth_texture(device: &Device, size: (u32, u32)) -> Texture {
    let size = wgpu::Extent3d {
        width: size.0,
        height: size.1,
        depth_or_array_layers: 1,
    };

    device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth24PlusStencil8,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

// Renders buffer to texture and increments stencil value where the buffer is drawn.
pub fn render_buffer_to_texture2<'a>(
    vertex_buffer: &'a wgpu::Buffer,
    index_buffer: &'a wgpu::Buffer,
    num_indices: u32,
    incrementing_pass: &mut RenderPass<'a>,
    parent_stencil_reference: u32,
) {
    incrementing_pass.set_stencil_reference(parent_stencil_reference);

    incrementing_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
    incrementing_pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint16);
    incrementing_pass.draw_indexed(0..num_indices, 0, 0..1);
}

pub fn create_texture_pipeline(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
) -> (BindGroupLayout, RenderPipeline, RenderPipeline) {
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                // This should match the filterable field of the
                // corresponding Texture entry above.
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
        label: Some("texture_bind_group_layout"),
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/texture_shader.wgsl").into()),
    });

    let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Render Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    // TODO
    let targets = [Some(wgpu::ColorTargetState {
        format: config.format,
        blend: Some(wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::SrcAlpha,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
        }),
        write_mask: wgpu::ColorWrites::ALL,
    })];

    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: Some(&render_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[TexturedVertex::desc()],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_main",
            targets: &targets,
        }),
        primitive: wgpu::PrimitiveState::default(),
        // TODO: add stencil test
        depth_stencil: Some(write_on_equal_depth_stencil_state()),
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    });

    let always_render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: Some(&render_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[TexturedVertex::desc()],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_main",
            targets: &targets,
        }),
        primitive: wgpu::PrimitiveState::default(),
        // TODO: add stencil test
        depth_stencil: Some(always_pass_and_keep_stencil_state()),
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    });

    (bind_group_layout, render_pipeline, always_render_pipeline)
}
