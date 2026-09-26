use super::{ShapeDrawLocation, ShapeDrawResources, ShapeExecutionResources};
use crate::core::shape::ShapeInstance;
use crate::core::vertex::{CustomVertex, InstanceTransform, TextureUvTransform};
use crate::wgpu_backend::resources::ShapePipelines;
use crate::wgpu_backend::texture_manager::WgpuTextureManager;
use crate::wgpu_backend::types::GeometryBufferError;
use crate::wgpu_backend::vertex::{GeometryBufferRange, InstanceColor, InstanceMetadata};
use crate::ShapeTextureFitMode;
use ahash::HashMap;
use wgpu::{Device, Queue};

#[derive(Copy, Clone)]
pub(crate) struct InstanceTextureData {
    pub(crate) texture_presence: [bool; 2],
    pub(crate) texture_uv_transforms: [TextureUvTransform; 2],
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
    cached_shape_data: &ShapeInstance,
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

fn compute_texture_uv_transform_for_layer(
    texture_id: Option<u64>,
    texture_fit_mode: ShapeTextureFitMode,
    texture_mapping_size: [f32; 2],
    texture_manager: &WgpuTextureManager,
    scale_factor: f64,
) -> TextureUvTransform {
    if texture_fit_mode == ShapeTextureFitMode::Stretch {
        return TextureUvTransform::IDENTITY;
    }

    let Some(texture_id) = texture_id else {
        return TextureUvTransform::IDENTITY;
    };

    let Some((texture_width, texture_height)) = texture_manager.texture_dimensions(texture_id)
    else {
        return TextureUvTransform::IDENTITY;
    };

    texture_fit_mode.compute_uv_transform(
        texture_mapping_size,
        (texture_width, texture_height),
        scale_factor,
    )
}

fn compute_texture_uv_transforms(
    texture_mapping_size: [f32; 2],
    shape: &ShapeInstance,
    texture_manager: &WgpuTextureManager,
    scale_factor: f64,
) -> [TextureUvTransform; 2] {
    [
        compute_texture_uv_transform_for_layer(
            shape.textures[0].texture_id,
            shape.textures[0].fit_mode,
            texture_mapping_size,
            texture_manager,
            scale_factor,
        ),
        compute_texture_uv_transform_for_layer(
            shape.textures[1].texture_id,
            shape.textures[1].fit_mode,
            texture_mapping_size,
            texture_manager,
            scale_factor,
        ),
    ]
}

impl ShapeExecutionResources {
    /// Appends shape data to shared buffers and resolves material bindings.
    pub(in crate::wgpu_backend) fn prepare_draw(
        &mut self,
        cached_shape_data: &ShapeInstance,
        device: &Device,
        queue: &Queue,
        pipelines: &ShapePipelines,
        scale_factor: f64,
    ) -> Result<ShapeDrawResources, GeometryBufferError> {
        let mut resources = ShapeDrawResources::default();
        resources.refresh_gradient_material(
            &cached_shape_data.fill,
            &mut self.gradient_cache,
            device,
            queue,
            &pipelines.gradient_bind_group_layout,
            &pipelines.linear_clamp_sampler,
        );
        let geometry_range = append_aggregated_geometry_for_shape(
            cached_shape_data,
            &mut self.vertices,
            &mut self.indices,
            &mut self.geometry_ranges,
        )?;
        if let Some(geometry_range) = geometry_range {
            let texture_uv_transforms = compute_texture_uv_transforms(
                cached_shape_data.cached_shape.texture_mapping_size(),
                cached_shape_data,
                &pipelines.texture_manager,
                scale_factor,
            );
            let instance_index = append_instance_data(
                &mut self.instance_transforms,
                &mut self.instance_colors,
                &mut self.instance_metadata,
                cached_shape_data.transform,
                cached_shape_data.color_override,
                InstanceTextureData {
                    texture_presence: cached_shape_data
                        .textures
                        .each_ref()
                        .map(|texture| texture.texture_id.is_some()),
                    texture_uv_transforms,
                },
            );
            resources.location = Some(ShapeDrawLocation {
                geometry_range,
                instance_index,
            });
        }
        Ok(resources)
    }
}
