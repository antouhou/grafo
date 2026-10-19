use super::draws::DrawPass;
use super::shapes::{ShapeGeometryType, ShapeShaderType, StencilAction};
use crate::commands::{DrawClip, ShapeTextureBinding, TextureComposite, TexturePlacement};
use crate::core::slot_pool::SlotPool;
use crate::core::vertex::{CustomVertex, InstanceTransform, TextureUvTransform};
use crate::wgpu_backend::vertex::{InstanceColor, InstanceMetadata};
use bytemuck::{Pod, Zeroable};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{Buffer, BufferUsages, Device, IndexFormat, Queue};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CompositeInstance {
    transform: InstanceTransform,
    color: InstanceColor,
    metadata: InstanceMetadata,
}

struct QuadBuffers {
    vertices: Buffer,
    indices: Buffer,
}

/// Shared quad geometry and reusable instance slots, independent of queue nodes.
#[derive(Default)]
pub(in crate::wgpu_backend) struct CompositeExecutionResources {
    quad: Option<QuadBuffers>,
    instances: SlotPool<Option<Buffer>>,
}

impl CompositeExecutionResources {
    pub(super) fn begin_render(&mut self) {
        self.instances.reset();
    }

    fn prepare(
        &mut self,
        device: &Device,
        queue: &Queue,
        transform: InstanceTransform,
        sampling: TextureUvTransform,
    ) -> (&QuadBuffers, &Buffer) {
        let quad = self.quad.get_or_insert_with(|| QuadBuffers {
            vertices: device.create_buffer_init(&BufferInitDescriptor {
                label: Some("composite_quad_vertices"),
                contents: bytemuck::cast_slice(&CustomVertex::unit_quad()),
                usage: BufferUsages::VERTEX,
            }),
            indices: device.create_buffer_init(&BufferInitDescriptor {
                label: Some("composite_quad_indices"),
                contents: bytemuck::cast_slice(&[0u16, 1, 2, 0, 2, 3]),
                usage: BufferUsages::INDEX,
            }),
        });
        let instance = CompositeInstance {
            transform,
            color: InstanceColor::zeroed(),
            metadata: InstanceMetadata {
                coverage_translation: [0.0; 2],
                texture_flags: 1.0,
                texture_uv_transform_layer0: sampling,
                texture_uv_transform_layer1: TextureUvTransform::IDENTITY,
            },
        };
        let contents = bytemuck::bytes_of(&instance);
        let slot = self.instances.acquire(|| None);
        let buffer = match slot {
            Some(buffer) => {
                queue.write_buffer(buffer, 0, contents);
                buffer
            }
            None => slot.insert(device.create_buffer_init(&BufferInitDescriptor {
                label: Some("composite_instance"),
                contents,
                usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            })),
        };
        (quad, buffer)
    }
}

impl DrawPass<'_, '_> {
    pub(super) fn composite_local_texture(
        &mut self,
        command: TextureComposite,
        clip: DrawClip,
        resources: &mut CompositeExecutionResources,
        device: &Device,
        queue: &Queue,
    ) {
        let TexturePlacement::Local {
            transform,
            sampling,
        } = command.placement
        else {
            unreachable!("local composites have a transform and sampling");
        };
        let (quad, buffer) = resources.prepare(device, queue, transform, sampling);
        self.set_scissor(clip.scissor);
        let pipelines = &self.pipelines.shapes;
        self.render_pass.set_pipeline(pipelines.get(
            ShapeGeometryType::Tessellated,
            ShapeShaderType::Textured,
            StencilAction::Keep,
        ));
        self.render_pass
            .set_bind_group(0, &pipelines.and_bind_group, &[]);
        self.render_pass.set_texture_bind_group(
            1,
            ShapeTextureBinding::None,
            &pipelines.default_shape_texture_bind_groups[1],
        );
        self.render_pass.set_texture_bind_group(
            0,
            ShapeTextureBinding::Intermediate(command.texture),
            self.textures.bind_group(command.texture),
        );
        self.render_pass
            .set_vertex_buffer(0, quad.vertices.slice(..));
        self.render_pass
            .set_index_buffer(quad.indices.slice(..), IndexFormat::Uint16);
        let color_offset = InstanceTransform::STRIDE;
        let metadata_offset = color_offset + InstanceColor::STRIDE;
        self.render_pass
            .set_vertex_buffer(1, buffer.slice(..color_offset));
        self.render_pass
            .set_vertex_buffer(2, buffer.slice(color_offset..metadata_offset));
        self.render_pass
            .set_vertex_buffer(3, buffer.slice(metadata_offset..));
        self.render_pass
            .set_stencil_reference(clip.stencil_reference);
        self.render_pass.draw_indexed(0..6, 0, 0..1);
    }
}
