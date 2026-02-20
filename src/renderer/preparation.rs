use super::types::decide_buffer_sizing;
use super::*;

fn upsert_gpu_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &mut Option<wgpu::Buffer>,
    label: &'static str,
    bytes: &[u8],
    usage: wgpu::BufferUsages,
) {
    let decision =
        decide_buffer_sizing(buffer.as_ref().map(|existing| existing.size()), bytes.len());

    if decision.should_reallocate {
        *buffer = Some(crate::pipeline::create_buffer_init(
            device,
            Some(label),
            bytes,
            usage,
        ));
    } else if let Some(existing_buffer) = buffer.as_ref() {
        queue.write_buffer(existing_buffer, 0, bytes);
    }
}

fn append_aggregated_geometry(
    temp_vertices: &mut Vec<crate::vertex::CustomVertex>,
    temp_indices: &mut Vec<u16>,
    vertices: &[crate::vertex::CustomVertex],
    indices: &[u16],
) -> Option<(usize, usize)> {
    if vertices.is_empty() || indices.is_empty() {
        return None;
    }

    let vertex_start = temp_vertices.len();
    if vertex_start > u16::MAX as usize {
        warn!(
            "Aggregated vertex count ({}) exceeds u16 limit. Rendering artifacts may occur.",
            vertex_start
        );
    }

    let index_start = temp_indices.len();
    let vertex_offset = vertex_start as u16;
    temp_vertices.extend_from_slice(vertices);

    for &index in indices {
        temp_indices.push(index + vertex_offset);
    }

    Some((index_start, indices.len()))
}

fn append_instance_data(
    temp_instance_transforms: &mut Vec<InstanceTransform>,
    temp_instance_colors: &mut Vec<InstanceColor>,
    temp_instance_metadata: &mut Vec<InstanceMetadata>,
    transform: Option<InstanceTransform>,
    color_override: Option<[f32; 4]>,
) -> usize {
    let instance_index = temp_instance_transforms.len();
    temp_instance_transforms.push(transform.unwrap_or_else(InstanceTransform::identity));
    temp_instance_colors.push(InstanceColor {
        color: color_override.unwrap_or([1.0, 1.0, 1.0, 1.0]),
    });
    temp_instance_metadata.push(InstanceMetadata {
        draw_order: instance_index as f32,
    });
    instance_index
}

impl<'a> Renderer<'a> {
    fn ensure_identity_instance_buffers(&mut self) {
        if self.identity_instance_transform_buffer.is_none() {
            let identity = InstanceTransform::identity();
            self.identity_instance_transform_buffer = Some(crate::pipeline::create_buffer_init(
                &self.device,
                Some("Identity Instance Transform Buffer"),
                bytemuck::cast_slice(&[identity]),
                BufferUsages::VERTEX | BufferUsages::COPY_DST,
            ));
        }

        if self.identity_instance_color_buffer.is_none() {
            let white = InstanceColor::white();
            self.identity_instance_color_buffer = Some(crate::pipeline::create_buffer_init(
                &self.device,
                Some("Identity Instance Color Buffer"),
                bytemuck::cast_slice(&[white]),
                BufferUsages::VERTEX | BufferUsages::COPY_DST,
            ));
        }

        if self.identity_instance_metadata_buffer.is_none() {
            let metadata = InstanceMetadata::default();
            self.identity_instance_metadata_buffer = Some(crate::pipeline::create_buffer_init(
                &self.device,
                Some("Identity Instance Metadata Buffer"),
                bytemuck::cast_slice(&[metadata]),
                BufferUsages::VERTEX | BufferUsages::COPY_DST,
            ));
        }
    }

    pub(super) fn prepare_render(&mut self) {
        self.temp_vertices.clear();
        self.temp_indices.clear();
        self.temp_instance_transforms.clear();
        self.temp_instance_colors.clear();
        self.temp_instance_metadata.clear();

        for (_node_id, draw_command) in self.draw_tree.iter_mut() {
            match draw_command {
                DrawCommand::Shape(shape) => {
                    let vertex_buffers =
                        shape.tessellate(&mut self.tessellator, &mut self.buffers_pool_manager);

                    if let Some((index_start, index_count)) = append_aggregated_geometry(
                        &mut self.temp_vertices,
                        &mut self.temp_indices,
                        &vertex_buffers.vertices,
                        &vertex_buffers.indices,
                    ) {
                        shape.index_buffer_range = Some((index_start, index_count));
                        let instance_index = append_instance_data(
                            &mut self.temp_instance_transforms,
                            &mut self.temp_instance_colors,
                            &mut self.temp_instance_metadata,
                            shape.transform(),
                            shape.instance_color_override(),
                        );
                        *shape.instance_index_mut() = Some(instance_index);
                    } else {
                        shape.is_empty = true;
                    }
                }
                DrawCommand::CachedShape(cached_shape_data) => {
                    if let Some(cached_shape) = self.shape_cache.get_mut(&cached_shape_data.id) {
                        let vertex_buffers = &cached_shape.vertex_buffers;

                        if let Some((index_start, index_count)) = append_aggregated_geometry(
                            &mut self.temp_vertices,
                            &mut self.temp_indices,
                            &vertex_buffers.vertices,
                            &vertex_buffers.indices,
                        ) {
                            cached_shape_data.index_buffer_range = Some((index_start, index_count));
                            let instance_index = append_instance_data(
                                &mut self.temp_instance_transforms,
                                &mut self.temp_instance_colors,
                                &mut self.temp_instance_metadata,
                                cached_shape_data.transform(),
                                cached_shape_data.instance_color_override(),
                            );
                            *cached_shape_data.instance_index_mut() = Some(instance_index);
                        } else {
                            cached_shape_data.is_empty = true;
                        }
                    } else {
                        warn!("Cached shape not found in cache");
                    }
                }
            }
        }

        if !self.temp_vertices.is_empty() {
            upsert_gpu_buffer(
                &self.device,
                &self.queue,
                &mut self.aggregated_vertex_buffer,
                "Aggregated Vertex Buffer",
                bytemuck::cast_slice(&self.temp_vertices),
                BufferUsages::VERTEX | BufferUsages::COPY_DST,
            );
        }

        if !self.temp_indices.is_empty() {
            upsert_gpu_buffer(
                &self.device,
                &self.queue,
                &mut self.aggregated_index_buffer,
                "Aggregated Index Buffer",
                bytemuck::cast_slice(&self.temp_indices),
                BufferUsages::INDEX | BufferUsages::COPY_DST,
            );
        }

        self.ensure_identity_instance_buffers();

        if !self.temp_instance_transforms.is_empty() {
            upsert_gpu_buffer(
                &self.device,
                &self.queue,
                &mut self.aggregated_instance_transform_buffer,
                "Aggregated Instance Transform Buffer",
                bytemuck::cast_slice(&self.temp_instance_transforms),
                BufferUsages::VERTEX | BufferUsages::COPY_DST,
            );
        }

        if !self.temp_instance_colors.is_empty() {
            upsert_gpu_buffer(
                &self.device,
                &self.queue,
                &mut self.aggregated_instance_color_buffer,
                "Aggregated Instance Color Buffer",
                bytemuck::cast_slice(&self.temp_instance_colors),
                BufferUsages::VERTEX | BufferUsages::COPY_DST,
            );
        }

        if !self.temp_instance_metadata.is_empty() {
            upsert_gpu_buffer(
                &self.device,
                &self.queue,
                &mut self.aggregated_instance_metadata_buffer,
                "Aggregated Instance Metadata Buffer",
                bytemuck::cast_slice(&self.temp_instance_metadata),
                BufferUsages::VERTEX | BufferUsages::COPY_DST,
            );
        }
    }
}
