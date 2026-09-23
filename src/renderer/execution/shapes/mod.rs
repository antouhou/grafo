use super::composites::CompositeExecutionResources;
use crate::cache::CachedTessellation;
use crate::gradient::gpu::{GradientCache, GradientMaterial};
use crate::gradient::types::Fill;
use crate::renderer::commands::ShapeDrawId;
use crate::vertex::{
    CustomVertex, GeometryBufferRange, InstanceColor, InstanceMetadata, InstanceTransform,
};
use ahash::{HashMap, HashMapExt};
use materials::TextureMaterialPool;
pub(in crate::renderer) use pipelines::TextureMaterialPipelines;
pub(crate) use sampling::TextureSamplingUniform;
use std::sync::Arc;
use wgpu::{BindGroup, BindGroupLayout, Device, Queue, Sampler};

mod buffers;
mod materials;
mod pipelines;
pub(in crate::renderer) mod preparation;
mod sampling;

#[derive(Debug, Clone, Copy)]
pub(crate) struct ShapeDrawLocation {
    pub(crate) geometry_range: GeometryBufferRange,
    pub(crate) instance_index: usize,
}

/// GPU locations and material bindings for one CPU shape description.
#[derive(Debug, Default)]
pub(crate) struct ShapeDrawResources {
    /// Retained only for draws with a shape effect, to identify cached coverage masks.
    pub(crate) mask_tessellation: Option<Arc<CachedTessellation>>,
    pub(crate) location: Option<ShapeDrawLocation>,
    gradient_material: Option<Arc<GradientMaterial>>,
    texture_material_bind_group: Option<BindGroup>,
}

impl ShapeDrawResources {
    /// Discards material bindings tied to a replaced pipeline layout.
    pub(crate) fn invalidate_material_bindings(&mut self) {
        self.gradient_material = None;
        self.texture_material_bind_group = None;
    }

    pub(crate) fn clear_under_fill_binding(&mut self) {
        self.texture_material_bind_group = None;
    }

    pub(crate) fn refresh_gradient_material(
        &mut self,
        fill: &mut Option<Fill>,
        gradient_cache: &mut GradientCache,
        device: &Device,
        queue: &Queue,
        layout: &BindGroupLayout,
        sampler: &Sampler,
    ) {
        self.gradient_material = match fill.as_mut() {
            Some(Fill::Gradient(gradient)) => Some(gradient_cache.get_or_create_material(
                &mut gradient.data,
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
    pub(in crate::renderer) composites: CompositeExecutionResources,
    pub(in crate::renderer) texture_materials: TextureMaterialPool,
    pub(crate) draws: HashMap<usize, ShapeDrawResources>,
    pub(crate) gradient_cache: GradientCache,
    pub(crate) vertices: Vec<CustomVertex>,
    pub(crate) indices: Vec<u16>,
    pub(crate) geometry_ranges: HashMap<u64, GeometryBufferRange>,
    pub(crate) instance_transforms: Vec<InstanceTransform>,
    pub(crate) instance_colors: Vec<InstanceColor>,
    pub(crate) instance_metadata: Vec<InstanceMetadata>,
}

impl ShapeExecutionResources {
    pub(in crate::renderer) fn draw_resources(&self, id: ShapeDrawId) -> &ShapeDrawResources {
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
    }
}
