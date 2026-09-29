use super::execution::targets;
use crate::core::UnsignedPhysicalRect;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindingResource, Color, ColorTargetState,
    ColorWrites, CommandEncoder, CompareFunction, DepthStencilState, Device, Extent3d,
    FragmentState, LoadOp, MultisampleState, Operations, RenderPassColorAttachment,
    RenderPassDepthStencilAttachment, RenderPassDescriptor, RenderPipeline,
    RenderPipelineDescriptor, ShaderModuleDescriptor, ShaderSource, StoreOp, Texture,
    TextureDescriptor, TextureDimension, TextureFormat, TextureUsages, TextureView,
    TextureViewDescriptor, VertexState,
};

/// One persistent image shared by surface presentation and pixmap readback.
pub(super) struct RetainedOutput {
    pub texture: Texture,
    pub view: TextureView,
    clear_pipeline: RenderPipeline,
    present_pipeline: RenderPipeline,
    present_binding: BindGroup,
}

impl RetainedOutput {
    pub fn new(device: &Device, size: (u32, u32), format: TextureFormat, samples: u32) -> Self {
        let texture = device.create_texture(&TextureDescriptor {
            label: Some("retained_output"),
            size: Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format,
            usage: TextureUsages::RENDER_ATTACHMENT
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&TextureViewDescriptor::default());
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("retained_output"),
            source: ShaderSource::Wgsl(include_str!("../shaders/retained_output.wgsl").into()),
        });
        let pipeline = |entry_point, samples, depth_stencil| {
            device.create_render_pipeline(&RenderPipelineDescriptor {
                label: Some(entry_point),
                layout: None,
                vertex: VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(FragmentState {
                    module: &shader,
                    entry_point: Some(entry_point),
                    compilation_options: Default::default(),
                    targets: &[Some(ColorTargetState {
                        format,
                        blend: None,
                        write_mask: ColorWrites::ALL,
                    })],
                }),
                primitive: Default::default(),
                depth_stencil,
                multisample: MultisampleState {
                    count: samples,
                    ..Default::default()
                },
                multiview: None,
                cache: None,
            })
        };
        let clear_pipeline = pipeline(
            "fs_clear",
            samples,
            Some(DepthStencilState {
                format: TextureFormat::Depth24PlusStencil8,
                depth_write_enabled: false,
                depth_compare: CompareFunction::Always,
                stencil: Default::default(),
                bias: Default::default(),
            }),
        );
        let present_pipeline = pipeline("fs_present", 1, None);
        let present_binding = device.create_bind_group(&BindGroupDescriptor {
            label: Some("retained_output"),
            layout: &present_pipeline.get_bind_group_layout(0),
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&view),
            }],
        });
        Self {
            texture,
            view,
            clear_pipeline,
            present_pipeline,
            present_binding,
        }
    }

    /// Clears changed color pixels and resets stencil before replaying the scene.
    pub fn clear(
        &self,
        encoder: &mut CommandEncoder,
        multisample_view: Option<&TextureView>,
        depth_stencil_view: &TextureView,
        scissor: UnsignedPhysicalRect,
        is_new: bool,
    ) {
        let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("clear_dirty_output"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: multisample_view.unwrap_or(&self.view),
                resolve_target: multisample_view.map(|_| &self.view),
                ops: Operations {
                    load: if is_new {
                        LoadOp::Clear(Color::TRANSPARENT)
                    } else {
                        LoadOp::Load
                    },
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                view: depth_stencil_view,
                depth_ops: Some(Operations {
                    load: LoadOp::Clear(1.0),
                    store: StoreOp::Store,
                }),
                stencil_ops: Some(Operations {
                    load: LoadOp::Clear(0),
                    store: StoreOp::Store,
                }),
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        if !is_new {
            targets::set_scissor(&mut pass, scissor);
            pass.set_pipeline(&self.clear_pipeline);
            pass.draw(0..3, 0..1);
        }
    }

    pub fn present(&self, encoder: &mut CommandEncoder, output: &TextureView) {
        let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("present_retained_output"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: output,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(Color::TRANSPARENT),
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(&self.present_pipeline);
        pass.set_bind_group(0, &self.present_binding, &[]);
        pass.draw(0..3, 0..1);
    }
}
