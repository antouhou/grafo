use super::composites::CompositeExecutionResources;
use crate::commands::ShapeDrawId;
use crate::core::cache::CachedTessellation;
use crate::core::gradient::types::Fill;
use crate::core::vertex::{CustomVertex, InstanceTransform};
use crate::wgpu_backend::gradient::{GradientCache, GradientMaterial};
use crate::wgpu_backend::vertex::{GeometryBufferRange, InstanceColor, InstanceMetadata};
use ahash::{HashMap, HashMapExt};
use compaction::DrawBufferCompactionStorage;
use materials::TextureMaterialPool;
pub(in crate::wgpu_backend) use pipelines::TextureMaterialPipelines;
pub(crate) use sampling::TextureSamplingUniform;
use std::sync::Arc;
use wgpu::{BindGroup, BindGroupLayout, Device, Queue, Sampler};

mod buffers;
mod compaction;
mod materials;
mod pipelines;
pub(in crate::wgpu_backend) mod preparation;
mod sampling;

#[derive(Debug, Clone, Copy)]
pub(crate) struct ShapeBufferLocation {
    pub(crate) geometry_range: GeometryBufferRange,
    pub(crate) instance_index: usize,
}

/// GPU locations and material bindings for one CPU shape description.
#[derive(Debug, Default)]
pub(in crate::wgpu_backend) struct ShapeDrawResources {
    /// Retained only for draws with a shape effect, to identify cached coverage masks.
    pub(crate) mask_tessellation: Option<Arc<CachedTessellation>>,
    pub(crate) geometry_buffer_location: Option<ShapeBufferLocation>,
    gradient_material: Option<Arc<GradientMaterial>>,
    texture_material_bind_group: Option<BindGroup>,
}

impl ShapeDrawResources {
    pub(crate) fn clear_under_fill_binding(&mut self) {
        self.texture_material_bind_group = None;
    }

    pub(crate) fn refresh_gradient_material(
        &mut self,
        fill: &Option<Fill>,
        gradient_cache: &mut GradientCache,
        device: &Device,
        queue: &Queue,
        layout: &BindGroupLayout,
        sampler: &Sampler,
    ) {
        self.gradient_material = match fill.as_ref() {
            Some(Fill::Gradient(gradient)) => Some(gradient_cache.get_or_create_material(
                gradient.data(),
                device,
                queue,
                layout,
                sampler,
            )),
            _ => None,
        };
    }
}

/// Reusable execution storage, separate from draw descriptions and their tree.
pub(crate) struct ShapeExecutionResources {
    pub(in crate::wgpu_backend) composites: CompositeExecutionResources,
    pub(in crate::wgpu_backend) texture_materials: TextureMaterialPool,
    pub(crate) draws: HashMap<usize, ShapeDrawResources>,
    pub(crate) gradient_cache: GradientCache,
    pub(crate) vertices: Vec<CustomVertex>,
    pub(crate) indices: Vec<u16>,
    pub(crate) geometry_ranges: HashMap<u64, GeometryBufferRange>,
    pub(crate) instance_transforms: Vec<InstanceTransform>,
    pub(crate) instance_colors: Vec<InstanceColor>,
    pub(crate) instance_metadata: Vec<InstanceMetadata>,
    compaction: DrawBufferCompactionStorage,
    has_unused_draw_buffers: bool,
}

impl ShapeExecutionResources {
    pub(in crate::wgpu_backend) fn register_draw(
        &mut self,
        id: ShapeDrawId,
        resources: ShapeDrawResources,
    ) {
        if self
            .draws
            .insert(id.0, resources)
            .and_then(|previous| previous.geometry_buffer_location)
            .is_some()
        {
            self.has_unused_draw_buffers = true;
        }
    }

    pub(in crate::wgpu_backend) fn remove_draws(&mut self, ids: &[ShapeDrawId]) {
        for id in ids {
            let removed_draw = self.draws.remove(&id.0);
            if removed_draw
                .and_then(|draw| draw.geometry_buffer_location)
                .is_some()
            {
                self.has_unused_draw_buffers = true;
            }
        }
    }

    pub(in crate::wgpu_backend) fn draw_resources(&self, id: ShapeDrawId) -> &ShapeDrawResources {
        &self.draws[&id.0]
    }

    pub(crate) fn new() -> Self {
        Self {
            texture_materials: TextureMaterialPool::default(),
            composites: CompositeExecutionResources::default(),
            draws: HashMap::new(),
            gradient_cache: GradientCache::new(),
            vertices: Vec::new(),
            indices: Vec::new(),
            geometry_ranges: HashMap::new(),
            instance_transforms: Vec::new(),
            instance_colors: Vec::new(),
            instance_metadata: Vec::new(),
            compaction: DrawBufferCompactionStorage::default(),
            has_unused_draw_buffers: false,
        }
    }

    /// Drops draw references while retaining GPU material slots, gradients, and CPU capacity.
    pub(crate) fn clear_draw_queue(&mut self) {
        self.draws.clear();
        self.vertices.clear();
        self.indices.clear();
        self.geometry_ranges.clear();
        self.instance_transforms.clear();
        self.instance_colors.clear();
        self.instance_metadata.clear();
        self.has_unused_draw_buffers = false;
    }
}

#[cfg(test)]
mod tests;
