use super::draws::{self, DrawPass};
use super::targets;
use crate::backend::types::Pipeline;
use crate::backend::vertex::{InstanceColor, InstanceMetadata};
use crate::commands::{
    RenderCommand, RenderOperation, RenderPlan, ShapeTextureBinding, TextureComposite,
    TexturePlacement,
};
use crate::core::vertex::{CustomVertex, InstanceTransform, TextureUvTransform};
use std::num::NonZeroU64;
use std::ops::Range;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{Buffer, BufferDescriptor, BufferUsages, Device, IndexFormat, Queue};

struct QuadBuffers {
    vertices: Buffer,
    indices: Buffer,
}

#[derive(Clone, Copy)]
pub(super) struct CompositeInstanceBuffer {
    count: usize,
}

/// Shared GPU quad geometry and one reusable instance buffer, independent of queue nodes.
#[derive(Default)]
pub(in crate::backend) struct CompositeExecutionResources {
    quad: Option<QuadBuffers>,
    instances: Option<Buffer>,
}

impl CompositeExecutionResources {
    pub(super) fn prepare(
        &mut self,
        device: &Device,
        queue: &Queue,
        commands: &RenderPlan,
    ) -> Option<CompositeInstanceBuffer> {
        if commands.composite_draws.is_empty() {
            return None;
        }
        self.quad.get_or_insert_with(|| QuadBuffers {
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
        let count = commands.composite_draws.len();
        let color_offset = count * InstanceTransform::STRIDE as usize;
        let metadata_offset = color_offset + count * InstanceColor::STRIDE as usize;
        let size = metadata_offset + count * InstanceMetadata::STRIDE as usize;
        let descriptor = BufferDescriptor {
            label: Some("composite_instances"),
            size: size as u64,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        };
        if self
            .instances
            .as_ref()
            .is_none_or(|buffer| buffer.size() < size as u64)
        {
            self.instances = Some(device.create_buffer(&descriptor));
        }
        let mut upload = queue
            .write_buffer_with(
                self.instances
                    .as_ref()
                    .expect("composite buffer was allocated"),
                0,
                NonZeroU64::new(size as u64).expect("composite instances are nonempty"),
            )
            .expect("composite upload fits its buffer");
        upload[color_offset..metadata_offset].fill(0);
        for (instance, &index) in commands.composite_draws.iter().enumerate() {
            let RenderOperation::CompositeTexture(composite) =
                &commands.instructions[index].operation
            else {
                unreachable!("composite instance must reference texture parameters");
            };
            let TexturePlacement::Local {
                transform,
                sampling,
            } = composite.placement
            else {
                unreachable!("only local composites need instances");
            };
            let metadata = InstanceMetadata {
                draw_order: 0.0,
                texture_flags: 1.0,
                texture_uv_transform_layer0: sampling,
                texture_uv_transform_layer1: TextureUvTransform::IDENTITY,
            };
            let transform_start = instance * InstanceTransform::STRIDE as usize;
            upload[transform_start..transform_start + InstanceTransform::STRIDE as usize]
                .copy_from_slice(bytemuck::bytes_of(&transform));
            let metadata_start = metadata_offset + instance * InstanceMetadata::STRIDE as usize;
            upload[metadata_start..metadata_start + InstanceMetadata::STRIDE as usize]
                .copy_from_slice(bytemuck::bytes_of(&metadata));
        }
        Some(CompositeInstanceBuffer { count })
    }
}

impl DrawPass<'_, '_> {
    /// Consecutive local composites can share one instanced quad draw.
    pub(super) fn execute_texture_composites(
        &mut self,
        instructions: &[RenderCommand],
        resources: &CompositeExecutionResources,
        instances: CompositeInstanceBuffer,
        first_instance: u32,
    ) -> usize {
        let first = &instructions[0];
        let RenderOperation::CompositeTexture(command) = &first.operation else {
            unreachable!("composite batch starts with a texture");
        };
        let command = *command;
        let texture = self.textures.resolve_id(command.texture);
        let mut count = 1;
        for next in &instructions[1..] {
            let RenderOperation::CompositeTexture(next_composite) = &next.operation else {
                break;
            };

            if !matches!(next_composite.placement, TexturePlacement::Local { .. }) {
                break;
            }
            if next.clip != first.clip
                || self.textures.resolve_id(next_composite.texture) != texture
            {
                break;
            }
            count += 1;
        }
        targets::set_scissor(self.render_pass, first.clip.scissor);
        self.composite_local_texture(
            first.clip.stencil_reference,
            command,
            resources,
            instances,
            first_instance..first_instance + count as u32,
        );
        count
    }

    fn composite_local_texture(
        &mut self,
        stencil_reference: u32,
        command: TextureComposite,
        resources: &CompositeExecutionResources,
        instances: CompositeInstanceBuffer,
        range: Range<u32>,
    ) {
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
        let quad = resources
            .quad
            .as_ref()
            .expect("composite geometry was prepared");
        self.render_pass
            .set_vertex_buffer(0, quad.vertices.slice(..));
        self.render_pass
            .set_index_buffer(quad.indices.slice(..), IndexFormat::Uint16);
        let buffer = resources
            .instances
            .as_ref()
            .expect("composite instances were uploaded");
        let color_offset = instances.count as u64 * InstanceTransform::STRIDE;
        let metadata_offset = color_offset + instances.count as u64 * InstanceColor::STRIDE;
        self.render_pass
            .set_vertex_buffer(1, buffer.slice(..color_offset));
        self.render_pass
            .set_vertex_buffer(2, buffer.slice(color_offset..metadata_offset));
        self.render_pass
            .set_vertex_buffer(3, buffer.slice(metadata_offset..));
        self.render_pass.set_stencil_reference(stencil_reference);
        self.render_pass.draw_indexed(0..6, 0, range);
        draws::bind_aggregated_geometry_buffers(self.render_pass, self.buffers);
    }
}
