use super::execution::shapes::ShapeDrawResources;
use super::*;
use crate::renderer::types::GeometryBufferError;
use crate::shape::ShapeTextureBinding;
use crate::vertex::CustomVertex;
use crate::ShapeTextureFitMode;
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
    // Reuse the range if this geometry is already in the frame's buffers.
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
    /// Appends shape data to shared buffers and resolves material bindings.
    pub(super) fn append_shape_resources(
        &mut self,
        cached_shape_data: &mut CachedShapeDrawData,
    ) -> Result<ShapeDrawResources, GeometryBufferError> {
        let mut resources = ShapeDrawResources::default();
        self.refresh_geometry_cache(cached_shape_data);
        resources.refresh_gradient_bind_group(
            &mut cached_shape_data.fill,
            &mut self.state.shape_execution.gradient_cache,
            &self.device,
            &self.queue,
            &self.pipeline_resources.shapes.gradient_bind_group_layout,
            &self.pipeline_resources.shapes.gradient_ramp_sampler,
        );
        let geometry_range = append_aggregated_geometry_for_shape(
            cached_shape_data,
            &mut self.state.shape_execution.vertices,
            &mut self.state.shape_execution.indices,
            &mut self.state.shape_execution.geometry_ranges,
        )?;
        if let Some(geometry_range) = geometry_range {
            resources.geometry_buffer_range = Some(geometry_range);
            let texture_uv_transforms = self.compute_texture_uv_transforms(
                cached_shape_data.cached_shape.texture_mapping_size(),
                cached_shape_data,
            );
            let instance_index = append_instance_data(
                &mut self.state.shape_execution.instance_transforms,
                &mut self.state.shape_execution.instance_colors,
                &mut self.state.shape_execution.instance_metadata,
                cached_shape_data.transform,
                cached_shape_data.color_override,
                InstanceTextureData {
                    texture_presence: cached_shape_data
                        .texture_bindings
                        .each_ref()
                        .map(ShapeTextureBinding::is_present),
                    texture_uv_transforms,
                },
            );
            resources.instance_index = Some(instance_index);
        }
        Ok(resources)
    }

    fn compute_texture_uv_transforms(
        &self,
        texture_mapping_size: [f32; 2],
        shape: &CachedShapeDrawData,
    ) -> [TextureUvTransform; 2] {
        [
            self.compute_texture_uv_transform_for_layer(
                shape.texture_bindings[0].managed_texture_id(),
                shape.texture_fit_modes[0],
                texture_mapping_size,
            ),
            self.compute_texture_uv_transform_for_layer(
                shape.texture_bindings[1].managed_texture_id(),
                shape.texture_fit_modes[1],
                texture_mapping_size,
            ),
        ]
    }

    fn compute_texture_uv_transform_for_layer(
        &self,
        texture_id: Option<u64>,
        texture_fit_mode: ShapeTextureFitMode,
        texture_mapping_size: [f32; 2],
    ) -> TextureUvTransform {
        if texture_fit_mode == ShapeTextureFitMode::Stretch {
            return TextureUvTransform::IDENTITY;
        }

        let Some(texture_id) = texture_id else {
            return TextureUvTransform::IDENTITY;
        };

        let Some((texture_width, texture_height)) = self
            .pipeline_resources
            .shapes
            .texture_manager
            .texture_dimensions(texture_id)
        else {
            return TextureUvTransform::IDENTITY;
        };

        let original_size_uv_scale = self.compute_texture_uv_scale_from_dimensions(
            texture_mapping_size,
            (texture_width, texture_height),
        );

        let normalization_factor = match texture_fit_mode {
            ShapeTextureFitMode::Stretch => return TextureUvTransform::IDENTITY,
            ShapeTextureFitMode::Cover => {
                f32::max(original_size_uv_scale[0], original_size_uv_scale[1])
            }
            ShapeTextureFitMode::Contain => {
                f32::min(original_size_uv_scale[0], original_size_uv_scale[1])
            }
            ShapeTextureFitMode::OriginalSize => {
                return TextureUvTransform {
                    scale: original_size_uv_scale,
                    offset: [0.0, 0.0],
                };
            }
        };

        if !normalization_factor.is_finite() || normalization_factor <= f32::EPSILON {
            return TextureUvTransform::IDENTITY;
        }

        let scale = [
            original_size_uv_scale[0] / normalization_factor,
            original_size_uv_scale[1] / normalization_factor,
        ];

        TextureUvTransform {
            scale,
            offset: [(1.0 - scale[0]) / 2.0, (1.0 - scale[1]) / 2.0],
        }
    }

    fn compute_texture_uv_scale_from_dimensions(
        &self,
        texture_mapping_size: [f32; 2],
        texture_dimensions: (u32, u32),
    ) -> [f32; 2] {
        [
            texture_mapping_size[0] * self.state.scale_factor as f32
                / texture_dimensions.0.max(1) as f32,
            texture_mapping_size[1] * self.state.scale_factor as f32
                / texture_dimensions.1.max(1) as f32,
        ]
    }

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

    pub(super) fn upload_buffers_for_frame(&mut self) {
        let buffers = &mut self.state.buffers;
        if !self.state.shape_execution.vertices.is_empty()
            || buffers.aggregated_vertex_buffer.is_none()
        {
            upsert_gpu_buffer(
                &self.device,
                &self.queue,
                &mut buffers.aggregated_vertex_buffer,
                "Aggregated Vertex Buffer",
                bytemuck::cast_slice(&self.state.shape_execution.vertices),
                BufferUsages::VERTEX | BufferUsages::COPY_DST,
            );
        }

        if !self.state.shape_execution.indices.is_empty()
            || buffers.aggregated_index_buffer.is_none()
        {
            upsert_gpu_buffer(
                &self.device,
                &self.queue,
                &mut buffers.aggregated_index_buffer,
                "Aggregated Index Buffer",
                bytemuck::cast_slice(&self.state.shape_execution.indices),
                BufferUsages::INDEX | BufferUsages::COPY_DST,
            );
        }

        self.ensure_identity_instance_buffers();
        let buffers = &mut self.state.buffers;

        if !self.state.shape_execution.instance_transforms.is_empty() {
            upsert_gpu_buffer(
                &self.device,
                &self.queue,
                &mut buffers.aggregated_instance_transform_buffer,
                "Aggregated Instance Transform Buffer",
                bytemuck::cast_slice(&self.state.shape_execution.instance_transforms),
                BufferUsages::VERTEX | BufferUsages::COPY_DST,
            );
        }

        if !self.state.shape_execution.instance_colors.is_empty() {
            upsert_gpu_buffer(
                &self.device,
                &self.queue,
                &mut buffers.aggregated_instance_color_buffer,
                "Aggregated Instance Color Buffer",
                bytemuck::cast_slice(&self.state.shape_execution.instance_colors),
                BufferUsages::VERTEX | BufferUsages::COPY_DST,
            );
        }

        if !self.state.shape_execution.instance_metadata.is_empty() {
            upsert_gpu_buffer(
                &self.device,
                &self.queue,
                &mut buffers.aggregated_instance_metadata_buffer,
                "Aggregated Instance Metadata Buffer",
                bytemuck::cast_slice(&self.state.shape_execution.instance_metadata),
                BufferUsages::VERTEX | BufferUsages::COPY_DST,
            );
        }
    }

    pub(super) fn prepare_render(&mut self) -> Result<(), GeometryBufferError> {
        self.begin_frame_scratch();
        // Include prepared effect leaves in this upload without making them part
        // of the durable user draw queue.
        let base_vertex_count = self.state.shape_execution.vertices.len();
        let base_index_count = self.state.shape_execution.indices.len();
        let base_instance_count = self.state.shape_execution.instance_transforms.len();
        self.prepare_shape_effect_leaves()?;
        self.upload_buffers_for_frame();
        let execution = &mut self.state.shape_execution;
        execution.vertices.truncate(base_vertex_count);
        execution.indices.truncate(base_index_count);
        execution.instance_transforms.truncate(base_instance_count);
        execution.instance_colors.truncate(base_instance_count);
        execution.instance_metadata.truncate(base_instance_count);
        Ok(())
    }
}
