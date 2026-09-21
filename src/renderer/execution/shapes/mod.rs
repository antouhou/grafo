use crate::gradient::gpu::{GpuMaterialParams, GradientCache};
use crate::gradient::types::Fill;
use crate::pipeline::BackdropSamplingUniform;
use crate::vertex::{
    CustomVertex, GeometryBufferRange, InstanceColor, InstanceMetadata, InstanceTransform,
};
use ahash::{HashMap, HashMapExt};
use std::sync::Arc;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{BindGroup, BindGroupLayout, Buffer, BufferUsages, Device, Queue, Sampler, TextureView};

/// GPU locations and material bindings for one CPU shape description.
#[derive(Debug, Default)]
pub(crate) struct ShapeDrawResources {
    pub(crate) geometry_buffer_range: Option<GeometryBufferRange>,
    pub(crate) instance_index: Option<usize>,
    pub(crate) gradient_bind_group: Option<Arc<BindGroup>>,
    backdrop_material_params_buffer: Option<Buffer>,
    backdrop_gradient_bind_group: Option<BindGroup>,
    backdrop_gradient_texture_id: Option<u64>,
}

impl ShapeDrawResources {
    pub(crate) fn new(geometry_buffer_range: GeometryBufferRange, instance_index: usize) -> Self {
        Self {
            geometry_buffer_range: Some(geometry_buffer_range),
            instance_index: Some(instance_index),
            ..Self::default()
        }
    }

    /// Discards material bindings tied to a replaced pipeline layout.
    pub(crate) fn invalidate_material_bindings(&mut self) {
        self.gradient_bind_group = None;
        self.backdrop_gradient_bind_group = None;
        self.backdrop_gradient_texture_id = None;
    }

    pub(crate) fn refresh_gradient_bind_group(
        &mut self,
        fill: &mut Option<Fill>,
        gradient_cache: &mut GradientCache,
        device: &Device,
        queue: &Queue,
        layout: &BindGroupLayout,
        sampler: &Sampler,
    ) {
        self.gradient_bind_group = match fill.as_mut() {
            Some(Fill::Gradient(gradient)) => Some(gradient_cache.get_or_create_bind_group(
                &mut gradient.data,
                device,
                queue,
                layout,
                sampler,
            )),
            _ => None,
        };
    }

    pub(crate) fn prepare_gradient_backdrop_material_params_buffer(
        &mut self,
        fill: &Option<Fill>,
        device: &Device,
        queue: &Queue,
        backdrop_sampling_uniform: BackdropSamplingUniform,
    ) -> Option<Buffer> {
        let params = {
            let gradient = match fill.as_ref() {
                Some(Fill::Gradient(gradient)) => gradient,
                _ => return None,
            };

            GpuMaterialParams::from_gradient_data(&gradient.data)
                .with_backdrop_sampling(backdrop_sampling_uniform)
        };

        if let Some(existing_buffer) = self.backdrop_material_params_buffer.as_ref() {
            queue.write_buffer(existing_buffer, 0, bytemuck::bytes_of(&params));
        } else {
            self.backdrop_material_params_buffer =
                Some(device.create_buffer_init(&BufferInitDescriptor {
                    label: Some("gradient_backdrop_material_params_buffer"),
                    contents: bytemuck::bytes_of(&params),
                    usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                }));
        }

        self.backdrop_material_params_buffer.clone()
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prepare_backdrop_gradient_bind_group(
        &mut self,
        fill: &mut Option<Fill>,
        gradient_cache: &mut GradientCache,
        device: &Device,
        queue: &Queue,
        layout: &BindGroupLayout,
        material_params_buffer: &Buffer,
        gradient_sampler: &Sampler,
        backdrop_texture_id: u64,
        backdrop_view: &TextureView,
        backdrop_sampler: &Sampler,
    ) -> Option<&BindGroup> {
        if self.backdrop_gradient_texture_id != Some(backdrop_texture_id) {
            let gradient = match fill.as_mut() {
                Some(Fill::Gradient(gradient)) => gradient,
                _ => return None,
            };

            self.backdrop_gradient_bind_group =
                Some(gradient_cache.create_backdrop_gradient_bind_group(
                    &mut gradient.data,
                    device,
                    queue,
                    layout,
                    material_params_buffer,
                    gradient_sampler,
                    backdrop_view,
                    backdrop_sampler,
                ));
            self.backdrop_gradient_texture_id = Some(backdrop_texture_id);
        }

        self.backdrop_gradient_bind_group.as_ref()
    }
}

/// Reusable execution storage, separate from draw descriptions and their tree.
pub(crate) struct ShapeExecutionResources {
    pub(crate) draws: HashMap<usize, ShapeDrawResources>,
    pub(crate) effect_leaves: HashMap<usize, ShapeDrawResources>,
    pub(crate) gradient_cache: GradientCache,
    pub(crate) vertices: Vec<CustomVertex>,
    pub(crate) indices: Vec<u16>,
    pub(crate) geometry_ranges: HashMap<u64, GeometryBufferRange>,
    pub(crate) instance_transforms: Vec<InstanceTransform>,
    pub(crate) instance_colors: Vec<InstanceColor>,
    pub(crate) instance_metadata: Vec<InstanceMetadata>,
}

impl ShapeExecutionResources {
    pub(crate) fn new() -> Self {
        Self {
            draws: HashMap::new(),
            effect_leaves: HashMap::new(),
            gradient_cache: GradientCache::new(),
            vertices: Vec::new(),
            indices: Vec::new(),
            geometry_ranges: HashMap::new(),
            instance_transforms: Vec::new(),
            instance_colors: Vec::new(),
            instance_metadata: Vec::new(),
        }
    }

    /// Releases queued resources while retaining allocation capacity and shared gradients.
    pub(crate) fn clear_draw_queue(&mut self) {
        self.draws.clear();
        self.effect_leaves.clear();
        self.vertices.clear();
        self.indices.clear();
        self.geometry_ranges.clear();
        self.instance_transforms.clear();
        self.instance_colors.clear();
        self.instance_metadata.clear();
    }
}
