use super::types::decide_buffer_sizing;
use super::*;
use crate::pipeline::create_buffer_init;
use crate::vertex::CustomVertex;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PreparationOutcome {
    Ready,
    Suspended,
}

#[derive(Debug, thiserror::Error)]
pub enum RenderError {
    #[error("no prepared rendering is available to commit")]
    NotPrepared,
    #[error("the presentation deadline passed before surface acquisition")]
    DeadlineMissed,
    #[error("a headless renderer cannot present to a surface")]
    Headless,
    #[error(transparent)]
    Surface(#[from] wgpu::SurfaceError),
}

pub(super) struct PreparedBuffers {
    vertex_count: usize,
    index_count: usize,
    instance_count: usize,
}

#[derive(Copy, Clone)]
pub(crate) struct InstanceTextureData {
    pub(crate) texture_presence: [bool; 2],
    pub(crate) texture_uv_scales: [[f32; 2]; 2],
}

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
        *buffer = Some(create_buffer_init(device, Some(label), bytes, usage));
    } else if let Some(existing_buffer) = buffer.as_ref() {
        queue.write_buffer(existing_buffer, 0, bytes);
    }
}

fn append_aggregated_geometry(
    temp_vertices: &mut Vec<CustomVertex>,
    temp_indices: &mut Vec<u16>,
    vertices: &[CustomVertex],
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

pub(crate) fn append_aggregated_geometry_for_shape(
    cached_shape_data: &CachedShapeDrawData,
    temp_vertices: &mut Vec<CustomVertex>,
    temp_indices: &mut Vec<u16>,
    geometry_dedup_map: &mut HashMap<u64, (usize, usize)>,
) -> Option<(usize, usize)> {
    let geometry_id = cached_shape_data.cached_shape.geometry_id;
    // Geometry deduplication: if we already appended this cache
    // key's vertices/indices, reuse the same range.
    if let Some(&existing_range) = geometry_id.and_then(|id| geometry_dedup_map.get(&id)) {
        Some(existing_range)
    } else {
        let cached_shape = &cached_shape_data.cached_shape;
        let vertex_buffers = cached_shape.vertex_buffers();
        let range = append_aggregated_geometry(
            temp_vertices,
            temp_indices,
            &vertex_buffers.vertices,
            &vertex_buffers.indices,
        );
        if let (Some(id), Some(range)) = (geometry_id, range) {
            geometry_dedup_map.insert(id, range);
        }
        range
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
        texture_uv_scale_layer0: texture_data.texture_uv_scales[0],
        texture_uv_scale_layer1: texture_data.texture_uv_scales[1],
    });
    instance_index
}

impl<'a> Renderer<'a> {
    fn ensure_identity_instance_buffers(&mut self) {
        if self.identity_instance_transform_buffer.is_none() {
            let identity = InstanceTransform::identity();
            self.identity_instance_transform_buffer = Some(create_buffer_init(
                &self.device,
                Some("Identity Instance Transform Buffer"),
                bytemuck::cast_slice(&[identity]),
                BufferUsages::VERTEX | BufferUsages::COPY_DST,
            ));
        }

        if self.identity_instance_color_buffer.is_none() {
            let transparent = InstanceColor::transparent();
            self.identity_instance_color_buffer = Some(create_buffer_init(
                &self.device,
                Some("Identity Instance Color Buffer"),
                bytemuck::cast_slice(&[transparent]),
                BufferUsages::VERTEX | BufferUsages::COPY_DST,
            ));
        }

        if self.identity_instance_metadata_buffer.is_none() {
            let metadata = InstanceMetadata::default();
            self.identity_instance_metadata_buffer = Some(create_buffer_init(
                &self.device,
                Some("Identity Instance Metadata Buffer"),
                bytemuck::cast_slice(&[metadata]),
                BufferUsages::VERTEX | BufferUsages::COPY_DST,
            ));
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

    /// Prepares the current draw queue without acquiring a drawable or uploading its buffers.
    /// Calling this again discards the previous preparation and reuses the same storage.
    pub fn prepare(&mut self) -> PreparationOutcome {
        #[cfg(feature = "render_metrics")]
        let started_at = std::time::Instant::now();
        self.discard_preparation();
        if self.physical_size.0 == 0 || self.physical_size.1 == 0 {
            return PreparationOutcome::Suspended;
        }
        self.begin_frame_scratch();
        self.prepared_buffers = Some(PreparedBuffers {
            vertex_count: self.temp_vertices.len(),
            index_count: self.temp_indices.len(),
            instance_count: self.temp_instance_transforms.len(),
        });
        self.prepare_shape_effect_leaves();
        #[cfg(feature = "render_metrics")]
        {
            self.preparation_cpu_time = started_at.elapsed();
        }
        PreparationOutcome::Ready
    }

    /// Cancels preparation without acquiring, submitting, or presenting a surface image.
    pub fn discard_preparation(&mut self) {
        if let Some(prepared) = self.prepared_buffers.take() {
            self.restore_draw_buffers(prepared);
            self.scratch_mut().shape_effect_leaves.clear();
        }
    }

    pub(super) fn upload_prepared_buffers(&mut self) -> Result<(), RenderError> {
        let prepared = self
            .prepared_buffers
            .take()
            .ok_or(RenderError::NotPrepared)?;
        self.upload_effect_params();
        self.upload_buffers_for_frame();
        self.restore_draw_buffers(prepared);
        Ok(())
    }

    fn restore_draw_buffers(&mut self, prepared: PreparedBuffers) {
        self.temp_vertices.truncate(prepared.vertex_count);
        self.temp_indices.truncate(prepared.index_count);
        self.temp_instance_transforms
            .truncate(prepared.instance_count);
        self.temp_instance_colors.truncate(prepared.instance_count);
        self.temp_instance_metadata
            .truncate(prepared.instance_count);
    }
}
