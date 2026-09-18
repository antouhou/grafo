use super::*;
use crate::renderer::types::GeometryBufferError;
use crate::vertex::CustomVertex;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{BufferDescriptor, COPY_BUFFER_ALIGNMENT};

#[derive(Copy, Clone)]
pub(crate) struct InstanceTextureData {
    pub(crate) texture_presence: [bool; 2],
    pub(crate) texture_uv_transforms: [TextureUvTransform; 2],
}

fn upsert_gpu_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &mut Option<wgpu::Buffer>,
    label: &'static str,
    bytes: &[u8],
    usage: wgpu::BufferUsages,
) {
    match buffer.as_ref() {
        Some(existing_buffer) if existing_buffer.size() >= bytes.len() as u64 => {
            queue.write_buffer(existing_buffer, 0, bytes);
        }
        None if bytes.is_empty() => {
            // Keep empty scenes bindable until geometry fills these buffers.
            *buffer = Some(device.create_buffer(&BufferDescriptor {
                label: Some(label),
                size: COPY_BUFFER_ALIGNMENT,
                usage,
                mapped_at_creation: false,
            }));
        }
        _ => {
            *buffer = Some(device.create_buffer_init(&BufferInitDescriptor {
                label: Some(label),
                contents: bytes,
                usage,
            }))
        }
    }
}

fn append_aggregated_geometry(
    temp_vertices: &mut Vec<CustomVertex>,
    temp_indices: &mut Vec<u16>,
    vertices: &[CustomVertex],
    indices: &[u16],
) -> Result<Option<GeometryBufferRange>, GeometryBufferError> {
    if vertices.is_empty() || indices.is_empty() {
        return Ok(None);
    }

    let vertex_start = i32::try_from(temp_vertices.len())
        .map_err(|_| GeometryBufferError::VertexOffsetOverflow)?;
    let index_end = temp_indices
        .len()
        .checked_add(indices.len())
        .and_then(|end| u32::try_from(end).ok())
        .ok_or(GeometryBufferError::IndexRangeOverflow)?;
    let index_start = temp_indices.len() as u32;
    temp_vertices.extend_from_slice(vertices);
    temp_indices.extend_from_slice(indices);

    Ok(Some(GeometryBufferRange {
        index_start,
        index_count: index_end - index_start,
        vertex_start,
    }))
}

pub(crate) fn append_aggregated_geometry_for_shape(
    cached_shape_data: &CachedShapeDrawData,
    temp_vertices: &mut Vec<CustomVertex>,
    temp_indices: &mut Vec<u16>,
    geometry_dedup_map: &mut HashMap<u64, GeometryBufferRange>,
) -> Result<Option<GeometryBufferRange>, GeometryBufferError> {
    let geometry_id = cached_shape_data.cached_shape.geometry_id;
    // Geometry deduplication: if we already appended this cache
    // key's vertices/indices, reuse the same range.
    if let Some(&existing_range) = geometry_id.and_then(|id| geometry_dedup_map.get(&id)) {
        Ok(Some(existing_range))
    } else {
        let cached_shape = &cached_shape_data.cached_shape;
        let vertex_buffers = cached_shape.vertex_buffers();
        let range = append_aggregated_geometry(
            temp_vertices,
            temp_indices,
            &vertex_buffers.vertices,
            &vertex_buffers.indices,
        )?;
        if let (Some(id), Some(range)) = (geometry_id, range) {
            geometry_dedup_map.insert(id, range);
        }
        Ok(range)
    }
}

pub(crate) fn append_instance_data(
    temp_instance_transforms: &mut Vec<InstanceTransform>,
    temp_instance_colors: &mut Vec<InstanceColor>,
    temp_instance_metadata: &mut Vec<InstanceMetadata>,
    transform: Option<InstanceTransform>,
    color_override: Option<[f32; 4]>,
    texture_data: InstanceTextureData,
) -> usize {
    let instance_index = temp_instance_transforms.len();
    temp_instance_transforms.push(transform.unwrap_or_else(InstanceTransform::identity));
    temp_instance_colors.push(InstanceColor {
        color: color_override.unwrap_or([0.0, 0.0, 0.0, 0.0]),
    });
    let texture_flags = (texture_data.texture_presence[0] as u32)
        | ((texture_data.texture_presence[1] as u32) << 1);
    temp_instance_metadata.push(InstanceMetadata {
        draw_order: instance_index as f32,
        texture_flags: texture_flags as f32,
        texture_uv_transform_layer0: texture_data.texture_uv_transforms[0],
        texture_uv_transform_layer1: texture_data.texture_uv_transforms[1],
    });
    instance_index
}

impl<'a> Renderer<'a> {
    fn ensure_identity_instance_buffers(&mut self) {
        let buffers = &mut self.state.buffers;
        if buffers.identity_instance_transform_buffer.is_none() {
            let identity = InstanceTransform::identity();
            buffers.identity_instance_transform_buffer =
                Some(self.device.create_buffer_init(&BufferInitDescriptor {
                    label: Some("Identity Instance Transform Buffer"),
                    contents: bytemuck::cast_slice(&[identity]),
                    usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                }));
        }

        if buffers.identity_instance_color_buffer.is_none() {
            let transparent = InstanceColor::transparent();
            buffers.identity_instance_color_buffer =
                Some(self.device.create_buffer_init(&BufferInitDescriptor {
                    label: Some("Identity Instance Color Buffer"),
                    contents: bytemuck::cast_slice(&[transparent]),
                    usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                }));
        }

        if buffers.identity_instance_metadata_buffer.is_none() {
            let metadata = InstanceMetadata::default();
            buffers.identity_instance_metadata_buffer =
                Some(self.device.create_buffer_init(&BufferInitDescriptor {
                    label: Some("Identity Instance Metadata Buffer"),
                    contents: bytemuck::cast_slice(&[metadata]),
                    usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                }));
        }
    }

    pub(super) fn clear_buffers(&mut self) {
        self.temp_vertices.clear();
        self.temp_indices.clear();
        self.temp_instance_transforms.clear();
        self.temp_instance_colors.clear();
        self.temp_instance_metadata.clear();
        self.geometry_dedup_map.clear();
    }

    pub(super) fn upload_buffers_for_frame(&mut self) {
        let buffers = &mut self.state.buffers;
        if !self.temp_vertices.is_empty() || buffers.aggregated_vertex_buffer.is_none() {
            upsert_gpu_buffer(
                &self.device,
                &self.queue,
                &mut buffers.aggregated_vertex_buffer,
                "Aggregated Vertex Buffer",
                bytemuck::cast_slice(&self.temp_vertices),
                BufferUsages::VERTEX | BufferUsages::COPY_DST,
            );
        }

        if !self.temp_indices.is_empty() || buffers.aggregated_index_buffer.is_none() {
            upsert_gpu_buffer(
                &self.device,
                &self.queue,
                &mut buffers.aggregated_index_buffer,
                "Aggregated Index Buffer",
                bytemuck::cast_slice(&self.temp_indices),
                BufferUsages::INDEX | BufferUsages::COPY_DST,
            );
        }

        self.ensure_identity_instance_buffers();
        let buffers = &mut self.state.buffers;

        if !self.temp_instance_transforms.is_empty() {
            upsert_gpu_buffer(
                &self.device,
                &self.queue,
                &mut buffers.aggregated_instance_transform_buffer,
                "Aggregated Instance Transform Buffer",
                bytemuck::cast_slice(&self.temp_instance_transforms),
                BufferUsages::VERTEX | BufferUsages::COPY_DST,
            );
        }

        if !self.temp_instance_colors.is_empty() {
            upsert_gpu_buffer(
                &self.device,
                &self.queue,
                &mut buffers.aggregated_instance_color_buffer,
                "Aggregated Instance Color Buffer",
                bytemuck::cast_slice(&self.temp_instance_colors),
                BufferUsages::VERTEX | BufferUsages::COPY_DST,
            );
        }

        if !self.temp_instance_metadata.is_empty() {
            upsert_gpu_buffer(
                &self.device,
                &self.queue,
                &mut buffers.aggregated_instance_metadata_buffer,
                "Aggregated Instance Metadata Buffer",
                bytemuck::cast_slice(&self.temp_instance_metadata),
                BufferUsages::VERTEX | BufferUsages::COPY_DST,
            );
        }
    }

    pub(super) fn prepare_render(&mut self) -> Result<(), GeometryBufferError> {
        self.begin_frame_scratch();
        // Include prepared effect leaves in this upload without making them part
        // of the durable user draw queue.
        let base_vertex_count = self.temp_vertices.len();
        let base_index_count = self.temp_indices.len();
        let base_instance_count = self.temp_instance_transforms.len();
        self.prepare_shape_effect_leaves()?;
        self.upload_buffers_for_frame();
        self.temp_vertices.truncate(base_vertex_count);
        self.temp_indices.truncate(base_index_count);
        self.temp_instance_transforms.truncate(base_instance_count);
        self.temp_instance_colors.truncate(base_instance_count);
        self.temp_instance_metadata.truncate(base_instance_count);
        Ok(())
    }
}
