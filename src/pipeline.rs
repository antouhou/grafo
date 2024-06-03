use crate::vertex::CustomVertex;
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

/// This stencil state always replaces the stencil value with the reference value.
fn create_always_stencil_state() -> wgpu::StencilState {
    // In this stencil state we will only draw where the stencil value is equal to the reference value,
    //  and all outside areas are zeroed.
    let face_state = wgpu::StencilFaceState {
        compare: wgpu::CompareFunction::Always,
        fail_op: wgpu::StencilOperation::Replace,
        depth_fail_op: wgpu::StencilOperation::Replace,
        pass_op: wgpu::StencilOperation::Replace,
    };

    wgpu::StencilState {
        front: face_state,
        back: face_state,
        read_mask: 0xff,
        write_mask: 0xff,
    }
}

/// This stencil state always replaces the stencil value with the reference value.
fn create_always_decrement_stencil_state() -> wgpu::StencilState {
    // In this stencil state we will only draw where the stencil value is equal to the reference value,
    //  and all outside areas are zeroed.
    let face_state = wgpu::StencilFaceState {
        compare: wgpu::CompareFunction::Always,
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

/// Creates a bind group so uniforms can be processed. Look at Uniforms struct for more info.
pub fn create_uniform_bind_group_layout(device: &Device) -> BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("uniform_bind_group_layout"),
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

pub fn create_always_replace_depth_stencil_state() -> wgpu::DepthStencilState {
    wgpu::DepthStencilState {
        format: wgpu::TextureFormat::Depth24PlusStencil8,
        depth_write_enabled: true,
        depth_compare: wgpu::CompareFunction::Always,
        stencil: create_always_stencil_state(),
        bias: wgpu::DepthBiasState::default(),
    }
}

pub fn create_always_decrement_depth_stencil_state() -> wgpu::DepthStencilState {
    wgpu::DepthStencilState {
        format: wgpu::TextureFormat::Depth24PlusStencil8,
        depth_write_enabled: true,
        depth_compare: wgpu::CompareFunction::Always,
        stencil: create_always_decrement_stencil_state(),
        bias: wgpu::DepthBiasState::default(),
    }
}

pub enum PipelineType {
    /// Keeps values where the stencil is equal to the reference value, zeros outside areas.
    /// I.e. keeps intersection between stencil buffer and what's being rendered.
    EqualIncrementStencil,
    /// Always replaces the stencil value with the reference value.
    AlwaysReplaceStencil,
    AlwaysDecrementStencil,
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
        PipelineType::AlwaysReplaceStencil => (
            create_always_replace_depth_stencil_state(),
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
        PipelineType::AlwaysDecrementStencil => {
            (create_always_decrement_depth_stencil_state(), [None])
        }
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
        label: Some("Uniform Buffer"),
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
        label: Some("uniform_bind_group"),
    });

    // Create the render pipeline
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("./shader.wgsl").into()),
    });

    let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Render Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Render Pipeline"),
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

pub fn create_and_pass<'a, 'b: 'a>(
    encoder: &'a mut wgpu::CommandEncoder,
    output_texture_view: &'b TextureView,
    depth_texture_view: &'b TextureView,
) -> RenderPass<'a> {
    let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("Render Pass"),
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

pub fn create_stencil_decrementing_pass<'a, 'b: 'a>(
    encoder: &'a mut wgpu::CommandEncoder,
    depth_texture_view: &'b TextureView,
) -> RenderPass<'a> {
    let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("Stencil Decrementing Render Pass"),
        color_attachments: &[None],
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
        label: Some("Depth Texture"),
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

pub fn create_and_depth_texture_view(depth_texture: &Texture) -> wgpu::TextureView {
    depth_texture.create_view(&wgpu::TextureViewDescriptor::default())
}

pub fn create_and_output_texture(device: &Device, size: (u32, u32)) -> Texture {
    let size = wgpu::Extent3d {
        width: size.0,
        height: size.1,
        depth_or_array_layers: 1,
    };

    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Output Texture"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

pub fn create_and_output_texture_view(output_texture: &Texture) -> wgpu::TextureView {
    output_texture.create_view(&wgpu::TextureViewDescriptor::default())
}

pub struct AndResult {
    // pub output_texture: Texture,
    // pub output_texture_view: TextureView,
    pub depth_texture: Texture,
    pub depth_texture_view: TextureView,
}

pub fn render_buffer_to_texture(
    pipeline: &RenderPipeline,
    bind_group: &BindGroup,
    device: &Device,
    canvas_size: (u32, u32),
    vertex_buffer: &wgpu::Buffer,
    index_buffer: &wgpu::Buffer,
    num_indices: u32,
    queue: &wgpu::Queue,
    parent_depth_texture: Option<&Texture>,
    output_texture_view: &TextureView,
    always_decrement_pipeline: &RenderPipeline,
    always_decrement_bind_group: &BindGroup,
    canvas_sized_quad: &wgpu::Buffer,
    mut encoder: &mut wgpu::CommandEncoder,
) -> AndResult {
    // let output_texture = create_and_output_texture(device, canvas_size);
    // let output_texture_view = create_and_output_texture_view(&output_texture);
    let depth_texture = create_and_depth_texture(device, canvas_size);

    if let Some(parent_depth_texture) = parent_depth_texture {
        // Copy parent texture to current texture
        encoder.copy_texture_to_texture(
            wgpu::ImageCopyTexture {
                texture: parent_depth_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyTexture {
                texture: &depth_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: canvas_size.0,
                height: canvas_size.1,
                depth_or_array_layers: 1,
            },
        );
    }

    let depth_texture_view = create_and_depth_texture_view(&depth_texture);

    {
        let mut render_pass =
            create_and_pass(&mut encoder, &output_texture_view, &depth_texture_view);
        // Since parent texture is filled with 1s only where the parent has been drawn and that
        //  the stencil test is logical AND, the resulting depth texture would be filled with 1s
        //  only on the intersection between the parent and the current shape.
        render_pass.set_stencil_reference(1);

        render_pass.set_pipeline(pipeline);
        render_pass.set_bind_group(0, bind_group, &[]);
        render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
        render_pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        render_pass.draw_indexed(0..num_indices, 0, 0..1);
    }

    // If there's a parent texture, that means that intersection will be all 2's. We need
    //  to decrement the stencil value across the whole texture to make it 1's and 0's again
    //  for the next shape to be drawn.
    if parent_depth_texture.is_some() {
        {
            let indices: [u16; 6] = [0, 1, 2, 2, 1, 3];
            let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Index Buffer"),
                contents: bytemuck::cast_slice(&indices),
                usage: wgpu::BufferUsages::INDEX,
            });
            let mut render_pass =
                create_stencil_decrementing_pass(&mut encoder, &depth_texture_view);

            render_pass.set_pipeline(always_decrement_pipeline);
            render_pass.set_bind_group(0, always_decrement_bind_group, &[]);

            render_pass.set_vertex_buffer(0, canvas_sized_quad.slice(..));
            render_pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            render_pass.draw_indexed(0..6, 0, 0..1);
        }
    }

    AndResult {
        // output_texture,
        // output_texture_view,
        depth_texture,
        depth_texture_view,
    }
}

// Renders buffer to texture and increments stencil value where the buffer is drawn.
pub fn render_buffer_to_texture2<'a>(
    vertex_buffer: &'a wgpu::Buffer,
    index_buffer: &'a wgpu::Buffer,
    num_indices: u32,
    mut incrementing_pass: &mut RenderPass<'a>,
    parent_stencil_reference: u32,
) {
    incrementing_pass.set_stencil_reference(parent_stencil_reference);

    // TODO: This should be done beforehand
    // incrementing_pass.set_pipeline(pipeline);
    // incrementing_pass.set_bind_group(0, bind_group, &[]);

    incrementing_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
    incrementing_pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint16);
    incrementing_pass.draw_indexed(0..num_indices, 0, 0..1);
}

pub unsafe fn erase_stencil<'a>(
    vertex_buffer: &'a wgpu::Buffer,
    index_buffer: &'a wgpu::Buffer,
    num_indices: u32,
    mut decrementing_pass: &mut RenderPass<'a>,
    stencil_reference: u32,
) {
    decrementing_pass.set_stencil_reference(stencil_reference);

    decrementing_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
    decrementing_pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint16);
    decrementing_pass.draw_indexed(0..num_indices, 0, 0..1);
}
