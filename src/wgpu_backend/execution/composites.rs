use super::draws::{self, DrawPass};
use crate::commands::{DrawClip, ShapeTextureBinding, TextureComposite, TexturePlacement};
use crate::core::vertex::{CustomVertex, InstanceTransform, TextureUvTransform};
use crate::wgpu_backend::types::Pipeline;
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
    instances: Vec<Buffer>,
    used: usize,
}

impl CompositeExecutionResources {
    pub(super) fn begin_render(&mut self) {
        self.used = 0;
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
                texture_flags: 1.0,
                texture_uv_transform_layer0: sampling,
                texture_uv_transform_layer1: TextureUvTransform::IDENTITY,
            },
        };
        let contents = bytemuck::bytes_of(&instance);
        // Draws encoded before submission must not overwrite each other's instance data.
        if self.used == self.instances.len() {
            self.instances
                .push(device.create_buffer_init(&BufferInitDescriptor {
                    label: Some("composite_instance"),
                    contents,
                    usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                }));
        } else {
            queue.write_buffer(&self.instances[self.used], 0, contents);
        }
        let buffer = &self.instances[self.used];
        self.used += 1;
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
        if self.pipeline_tracker.current != Pipeline::LeafDraw {
            self.render_pass.set_pipeline(&pipelines.leaf_draw_pipeline);
            self.render_pass
                .set_bind_group(0, &pipelines.and_bind_group, &[]);
            self.render_pass.set_bind_group(
                2,
                &*pipelines.default_shape_texture_bind_groups[1],
                &[],
            );
            self.bound_textures.mark_bound(1, ShapeTextureBinding::None);
            self.pipeline_tracker.switch_to(Pipeline::LeafDraw);
        }
        self.render_pass
            .set_bind_group(1, self.textures.bind_group(command.texture), &[]);
        self.bound_textures
            .mark_bound(0, ShapeTextureBinding::Intermediate(command.texture));
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
        draws::bind_aggregated_geometry_buffers(self.render_pass, self.buffers);
    }
}
