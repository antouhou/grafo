use super::execution::effects::{apply_effect_passes, EffectPassRunConfig};
use super::execution::shapes::ShapeDrawResources;
use super::rect_utils::compute_downsampled_dimensions;
use super::state::Buffers;
use super::types::{DrawTreeNode, GeometryBufferError};
use super::Renderer;
use crate::cache::{CachedTessellation, FrameCache};
use crate::effect::{self, OffscreenTexturePool, PooledTexture, ShapeEffectConfig};
use crate::renderer::preparation::{self, InstanceTextureData};
use crate::shape::{CachedShapeDrawData, CachedShapeHandle, ShapeTextureBinding};
use crate::vertex::{CustomVertex, GeometryBufferRange, InstanceTransform, TextureUvTransform};
use crate::{ShapeDrawCommandOptions, Size};
use bytemuck::{Pod, Zeroable};
use lyon::tessellation::VertexBuffers;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BufferUsages, Color, CommandEncoder, Device, IndexFormat, LoadOp, Operations,
    RenderPassColorAttachment, RenderPassDescriptor, StoreOp, TextureView,
};

const SHAPE_EFFECT_MASK_SHADER: &str = include_str!("../shaders/shape_effect_mask.wgsl");

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(super) struct ShapeEffectMaskUniform {
    local_origin: [f32; 2],
    logical_size: [f32; 2],
    scale_factor: f32,
    fringe_width: f32,
    padding: [f32; 2],
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub(super) struct ShapeEffectRasterRect {
    /// Shape-local origin scaled to physical pixels. Node transforms apply later,
    /// so moving the shape on screen does not change this value.
    pub local_physical_origin: [i32; 2],
    /// Mask/effect texture size in texels. Smaller than the full-resolution
    /// physical extent when the effect config downsamples the rasterization.
    pub texture_size: [u32; 2],
    pub local_bounds: [(f32, f32); 2],
}

impl ShapeEffectRasterRect {
    pub(super) fn mask_uniform(
        self,
        scale_factor: f64,
        fringe_width: f32,
    ) -> ShapeEffectMaskUniform {
        ShapeEffectMaskUniform {
            local_origin: [self.local_bounds[0].0, self.local_bounds[0].1],
            logical_size: [
                self.local_bounds[1].0 - self.local_bounds[0].0,
                self.local_bounds[1].1 - self.local_bounds[0].1,
            ],
            scale_factor: scale_factor as f32,
            fringe_width,
            padding: [0.0; 2],
        }
    }
}

pub(super) fn compute_shape_effect_raster_rect(
    local_bounds: [(f32, f32); 2],
    config: ShapeEffectConfig,
    scale_factor: f64,
    fringe_width: f32,
) -> Option<ShapeEffectRasterRect> {
    let bounds_and_outsets = [
        local_bounds[0].0,
        local_bounds[0].1,
        local_bounds[1].0,
        local_bounds[1].1,
        config.left_outset,
        config.top_outset,
        config.right_outset,
        config.bottom_outset,
    ];
    if !scale_factor.is_finite()
        || scale_factor <= 0.0
        || !fringe_width.is_finite()
        || fringe_width < 0.0
        || !config.downsample.is_finite()
        || config.downsample <= 0.0
        || config.downsample > 1.0
        || !bounds_and_outsets.iter().all(|value| value.is_finite())
    {
        return None;
    }

    let minimum_x = local_bounds[0].0.min(local_bounds[1].0) - config.left_outset;
    let minimum_y = local_bounds[0].1.min(local_bounds[1].1) - config.top_outset;
    let maximum_x = local_bounds[0].0.max(local_bounds[1].0) + config.right_outset;
    let maximum_y = local_bounds[0].1.max(local_bounds[1].1) + config.bottom_outset;
    if ![minimum_x, minimum_y, maximum_x, maximum_y]
        .iter()
        .all(|value| value.is_finite())
    {
        return None;
    }

    let guard = f64::from(fringe_width).ceil();
    let physical_minimum_x = (f64::from(minimum_x) * scale_factor).floor() - guard;
    let physical_minimum_y = (f64::from(minimum_y) * scale_factor).floor() - guard;
    let physical_maximum_x = (f64::from(maximum_x) * scale_factor).ceil() + guard;
    let physical_maximum_y = (f64::from(maximum_y) * scale_factor).ceil() + guard;

    let coordinates = [
        physical_minimum_x,
        physical_minimum_y,
        physical_maximum_x,
        physical_maximum_y,
    ];
    if !coordinates.iter().all(|value| {
        value.is_finite() && *value >= f64::from(i32::MIN) && *value <= f64::from(i32::MAX)
    }) {
        return None;
    }

    let local_physical_origin = [physical_minimum_x as i32, physical_minimum_y as i32];
    let physical_width = physical_maximum_x - physical_minimum_x;
    let physical_height = physical_maximum_y - physical_minimum_y;
    if physical_width <= 0.0
        || physical_height <= 0.0
        || physical_width > f64::from(u32::MAX)
        || physical_height > f64::from(u32::MAX)
    {
        return None;
    }

    let full_resolution_size = Size::new(physical_width as u32, physical_height as u32);
    let texture_size = compute_downsampled_dimensions(full_resolution_size, config.downsample);
    Some(ShapeEffectRasterRect {
        local_physical_origin,
        texture_size: texture_size.to_array(),
        local_bounds: [
            (
                physical_minimum_x as f32 / scale_factor as f32,
                physical_minimum_y as f32 / scale_factor as f32,
            ),
            (
                physical_maximum_x as f32 / scale_factor as f32,
                physical_maximum_y as f32 / scale_factor as f32,
            ),
        ],
    })
}

pub(super) struct PreparedShapeEffectLeaf {
    pub(super) draw_data: CachedShapeDrawData,
    pub(super) raster_rect: ShapeEffectRasterRect,
}

/// Identifies a mask by geometry and rasterization settings, allowing reuse across
/// effects and parameter changes.
#[derive(Clone)]
pub(super) struct ShapeEffectMaskCacheKey {
    pub tessellation: Arc<CachedTessellation>,
    /// Local raster origin from [`ShapeEffectRasterRect::local_physical_origin`].
    /// Node transforms do not invalidate this entry.
    pub local_raster_origin: [i32; 2],
    pub raster_size: [u32; 2],
    pub scale_factor_bits: u64,
    pub fringe_width_bits: u32,
    pub downsample_bits: u32,
    pub texture_format: wgpu::TextureFormat,
}

impl PartialEq for ShapeEffectMaskCacheKey {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.tessellation, &other.tessellation)
            && self.local_raster_origin == other.local_raster_origin
            && self.raster_size == other.raster_size
            && self.scale_factor_bits == other.scale_factor_bits
            && self.fringe_width_bits == other.fringe_width_bits
            && self.downsample_bits == other.downsample_bits
            && self.texture_format == other.texture_format
    }
}

impl Eq for ShapeEffectMaskCacheKey {}

impl Hash for ShapeEffectMaskCacheKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (Arc::as_ptr(&self.tessellation) as usize).hash(state);
        self.local_raster_origin.hash(state);
        self.raster_size.hash(state);
        self.scale_factor_bits.hash(state);
        self.fringe_width_bits.hash(state);
        self.downsample_bits.hash(state);
        self.texture_format.hash(state);
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub(super) struct ShapeEffectCacheKey {
    pub mask_key: ShapeEffectMaskCacheKey,
    pub effect_id: u64,
    pub params: Arc<[u8]>,
}

pub(super) struct CachedShapeEffectMask {
    pub texture: PooledTexture,
}

pub(super) type ShapeEffectMaskCache =
    FrameCache<ShapeEffectMaskCacheKey, Arc<CachedShapeEffectMask>>;

pub(super) struct CachedShapeEffect {
    pub texture: PooledTexture,
    pub texture_bind_group: Arc<wgpu::BindGroup>,
}

pub(super) type ShapeEffectResultCache = FrameCache<ShapeEffectCacheKey, Arc<CachedShapeEffect>>;

struct ShapeEffectMaskDraw {
    cache_key: ShapeEffectMaskCacheKey,
    geometry_range: GeometryBufferRange,
    uniform: ShapeEffectMaskUniform,
}

pub(super) struct ShapeEffectRendererResources {
    pub mask_bind_group_layout: wgpu::BindGroupLayout,
    pub mask_pipeline: wgpu::RenderPipeline,
    pub quad_tessellation: Arc<CachedTessellation>,
}

impl ShapeEffectRendererResources {
    pub(super) fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let mask_bind_group_layout = create_mask_bind_group_layout(device);
        let mask_pipeline = create_mask_pipeline(device, format, &mask_bind_group_layout);
        let quad_tessellation = create_shape_effect_quad_tessellation();
        Self {
            mask_bind_group_layout,
            mask_pipeline,
            quad_tessellation,
        }
    }

    pub(super) fn recreate_pipeline(&mut self, device: &wgpu::Device, format: wgpu::TextureFormat) {
        self.mask_pipeline = create_mask_pipeline(device, format, &self.mask_bind_group_layout);
    }
}

fn create_mask_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("shape_effect_mask_bind_group_layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    })
}

fn create_mask_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    bind_group_layout: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("shape_effect_mask_shader"),
        source: wgpu::ShaderSource::Wgsl(SHAPE_EFFECT_MASK_SHADER.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("shape_effect_mask_pipeline_layout"),
        bind_group_layouts: &[bind_group_layout],
        push_constant_ranges: &[],
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("shape_effect_mask_pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("mask_vertex"),
            compilation_options: Default::default(),
            buffers: &[CustomVertex::desc()],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("mask_fragment"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    })
}

pub(super) fn create_quad_vertices(local_bounds: [(f32, f32); 2]) -> [CustomVertex; 4] {
    let [(minimum_x, minimum_y), (maximum_x, maximum_y)] = local_bounds;
    [
        CustomVertex {
            position: [minimum_x, minimum_y],
            tex_coords: [0.0, 0.0],
            normal: [0.0; 2],
            coverage: 1.0,
        },
        CustomVertex {
            position: [maximum_x, minimum_y],
            tex_coords: [1.0, 0.0],
            normal: [0.0; 2],
            coverage: 1.0,
        },
        CustomVertex {
            position: [maximum_x, maximum_y],
            tex_coords: [1.0, 1.0],
            normal: [0.0; 2],
            coverage: 1.0,
        },
        CustomVertex {
            position: [minimum_x, maximum_y],
            tex_coords: [0.0, 1.0],
            normal: [0.0; 2],
            coverage: 1.0,
        },
    ]
}

fn create_shape_effect_quad_tessellation() -> Arc<CachedTessellation> {
    let local_bounds = [(0.0, 0.0), (1.0, 1.0)];
    let quad_vertices = create_quad_vertices(local_bounds);
    Arc::new(CachedTessellation {
        vertex_buffers: Arc::new(VertexBuffers {
            vertices: quad_vertices.to_vec(),
            indices: vec![0, 1, 2, 0, 2, 3],
        }),
        local_bounds,
        texture_mapping_size: [1.0, 1.0],
    })
}

fn shape_effect_quad_transform(
    local_bounds: [(f32, f32); 2],
    source_transform: Option<InstanceTransform>,
) -> InstanceTransform {
    let [(minimum_x, minimum_y), (maximum_x, maximum_y)] = local_bounds;
    let bounds_transform = InstanceTransform::affine_2d(
        maximum_x - minimum_x,
        0.0,
        0.0,
        maximum_y - minimum_y,
        minimum_x,
        minimum_y,
    );
    source_transform.map_or(bounds_transform, |transform| {
        bounds_transform.then(&transform)
    })
}

pub(super) fn create_mask_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    uniform_buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("shape_effect_mask_bind_group"),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform_buffer.as_entire_binding(),
        }],
    })
}

fn render_shape_effect_mask(
    device: &Device,
    encoder: &mut CommandEncoder,
    resources: &ShapeEffectRendererResources,
    buffers: &Buffers,
    target: &TextureView,
    draw: &ShapeEffectMaskDraw,
) {
    let uniform_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("shape_effect_mask_uniform"),
        contents: bytemuck::bytes_of(&draw.uniform),
        usage: BufferUsages::UNIFORM,
    });
    let bind_group =
        create_mask_bind_group(device, &resources.mask_bind_group_layout, &uniform_buffer);
    let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
        label: Some("shape_effect_mask_pass"),
        color_attachments: &[Some(RenderPassColorAttachment {
            view: target,
            resolve_target: None,
            ops: Operations {
                load: LoadOp::Clear(Color::TRANSPARENT),
                store: StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
    });
    render_pass.set_pipeline(&resources.mask_pipeline);
    render_pass.set_bind_group(0, &bind_group, &[]);
    render_pass.set_vertex_buffer(0, buffers.vertex_buffer().slice(..));
    render_pass.set_index_buffer(buffers.index_buffer().slice(..), IndexFormat::Uint16);
    buffers.draw_indexed(&mut render_pass, draw.geometry_range, 0..1);
}

fn resolve_shape_effect_mask(
    device: &Device,
    encoder: &mut CommandEncoder,
    resources: &ShapeEffectRendererResources,
    buffers: &Buffers,
    texture_pool: &mut OffscreenTexturePool,
    cache: &mut ShapeEffectMaskCache,
    draw: ShapeEffectMaskDraw,
) -> (Arc<CachedShapeEffectMask>, bool) {
    if let Some(mask) = cache.get(&draw.cache_key) {
        return (mask, true);
    }

    let [width, height] = draw.cache_key.raster_size;
    let texture =
        texture_pool.acquire_color_only(device, width, height, draw.cache_key.texture_format, 1);
    render_shape_effect_mask(
        device,
        encoder,
        resources,
        buffers,
        &texture.color_view,
        &draw,
    );
    let mask = Arc::new(CachedShapeEffectMask { texture });
    cache.insert(draw.cache_key, Arc::clone(&mask));
    (mask, false)
}

fn render_shape_effect(
    device: &Device,
    encoder: &mut CommandEncoder,
    texture_pool: &mut OffscreenTexturePool,
    params: &[u8],
    config: EffectPassRunConfig<'_>,
    textures_to_recycle: &mut Vec<PooledTexture>,
) -> Arc<CachedShapeEffect> {
    let parameter_buffer = (!params.is_empty()).then(|| {
        device.create_buffer_init(&BufferInitDescriptor {
            label: Some("shape_effect_params_buffer"),
            contents: params,
            usage: BufferUsages::UNIFORM,
        })
    });
    let parameter_bind_group = parameter_buffer.as_ref().and_then(|buffer| {
        config
            .loaded_effect
            .params_bind_group_layout
            .as_ref()
            .map(|layout| effect::create_params_bind_group(device, layout, buffer))
    });
    let effect_output = apply_effect_passes(
        device,
        encoder,
        texture_pool,
        EffectPassRunConfig {
            params_bind_group: parameter_bind_group.as_ref(),
            ..config
        },
    );
    let (final_texture, texture_bind_group) =
        effect_output.into_final_and_recyclable(textures_to_recycle);
    Arc::new(CachedShapeEffect {
        texture: final_texture,
        texture_bind_group: Arc::new(
            texture_bind_group.expect("shape effect generation must create a texture bind group"),
        ),
    })
}

impl<'a> Renderer<'a> {
    pub(super) fn prepare_shape_effect_leaves(&mut self) -> Result<(), GeometryBufferError> {
        let maximum_texture_dimension = self.device.limits().max_texture_dimension_2d;
        let maximum_texel_count = u64::from(self.state.physical_size.0)
            .saturating_mul(u64::from(self.state.physical_size.1))
            .saturating_mul(4);
        let mut quad_geometry_range = None;
        for (&node_id, shape_effect) in &self.state.shape_effects {
            let Some(draw_tree_node) = self.state.draw_tree.get(node_id) else {
                continue;
            };
            let DrawTreeNode::CachedShape(source_shape) = draw_tree_node else {
                continue;
            };
            let local_bounds = source_shape.cached_shape.tessellation.local_bounds;
            let source_transform = source_shape.transform;
            let Some(raster_rect) = compute_shape_effect_raster_rect(
                local_bounds,
                shape_effect.config,
                self.state.scale_factor,
                self.fringe_width,
            ) else {
                tracing::warn!(
                    node_id,
                    effect_id = shape_effect.effect_id,
                    "skipping shape effect with invalid raster bounds"
                );
                continue;
            };
            let [width, height] = raster_rect.texture_size;
            let texel_count = u64::from(width).saturating_mul(u64::from(height));
            if width > maximum_texture_dimension
                || height > maximum_texture_dimension
                || texel_count > maximum_texel_count
            {
                tracing::warn!(
                    node_id,
                    effect_id = shape_effect.effect_id,
                    width,
                    height,
                    maximum_texture_dimension,
                    maximum_texel_count,
                    scale_factor = self.state.scale_factor,
                    "skipping oversized shape effect texture"
                );
                continue;
            }

            let quad_handle = CachedShapeHandle {
                tessellation: Arc::clone(&self.pipeline_resources.shape_effects.quad_tessellation),
                is_rect: true,
                rect_bounds: Some([(0.0, 0.0), (1.0, 1.0)]),
                geometry_id: None,
            };
            let mut leaf = CachedShapeDrawData::new(quad_handle, &ShapeDrawCommandOptions::new());
            let transform = shape_effect_quad_transform(raster_rect.local_bounds, source_transform);
            leaf.transform = Some(transform);
            let geometry_range = match quad_geometry_range {
                Some(geometry_range) => geometry_range,
                None => {
                    let Some(geometry_range) = preparation::append_aggregated_geometry_for_shape(
                        &leaf,
                        &mut self.state.shape_execution.vertices,
                        &mut self.state.shape_execution.indices,
                        &mut self.state.shape_execution.geometry_ranges,
                    )?
                    else {
                        continue;
                    };
                    quad_geometry_range = Some(geometry_range);
                    geometry_range
                }
            };
            let instance_index = preparation::append_instance_data(
                &mut self.state.shape_execution.instance_transforms,
                &mut self.state.shape_execution.instance_colors,
                &mut self.state.shape_execution.instance_metadata,
                Some(transform),
                None,
                InstanceTextureData {
                    texture_presence: [true, false],
                    texture_uv_transforms: [TextureUvTransform::IDENTITY; 2],
                },
            );
            self.state.shape_execution.effect_leaves.insert(
                node_id,
                ShapeDrawResources::new(geometry_range, instance_index),
            );
            self.state.scratch.shape_effect_leaves.insert(
                node_id,
                PreparedShapeEffectLeaf {
                    draw_data: leaf,
                    raster_rect,
                },
            );
        }
        Ok(())
    }

    pub(super) fn resolve_shape_effects(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        textures_to_recycle: &mut Vec<PooledTexture>,
    ) {
        if self.state.buffers.aggregated_vertex_buffer.is_none()
            || self.state.buffers.aggregated_index_buffer.is_none()
        {
            return;
        }

        let effect_sampler = self
            .pipeline_resources
            .effect_sampler
            .as_ref()
            .expect("shape effects require the shared effect sampler");
        let shape_effect_leaves = &mut self.state.scratch.shape_effect_leaves;
        #[cfg(feature = "render_metrics")]
        let metrics = &mut self.state.shape_effect_cache_metrics;

        for (&node_id, shape_effect_instance) in &self.state.shape_effects {
            let Some(leaf) = shape_effect_leaves.get_mut(&node_id) else {
                continue;
            };
            let Some(DrawTreeNode::CachedShape(cached_shape)) = self.state.draw_tree.get(node_id)
            else {
                continue;
            };
            let Some(geometry_range) =
                self.state.shape_execution.draws[&node_id].geometry_buffer_range
            else {
                continue;
            };

            let raster_rect = leaf.raster_rect;
            let [width, height] = raster_rect.texture_size;
            let mask_cache_key = ShapeEffectMaskCacheKey {
                tessellation: Arc::clone(&cached_shape.cached_shape.tessellation),
                local_raster_origin: raster_rect.local_physical_origin,
                raster_size: raster_rect.texture_size,
                scale_factor_bits: self.state.scale_factor.to_bits(),
                fringe_width_bits: self.fringe_width.to_bits(),
                downsample_bits: shape_effect_instance.config.downsample.to_bits(),
                texture_format: self.config.format,
            };
            let cache_key = ShapeEffectCacheKey {
                mask_key: mask_cache_key.clone(),
                effect_id: shape_effect_instance.effect_id,
                params: Arc::clone(&shape_effect_instance.params),
            };
            let cached_result = if let Some(cached_result) =
                self.state.shape_effect_cache.get(&cache_key)
            {
                // Keep the mask alive too, so parameter changes can reuse it next frame.
                self.state.shape_effect_mask_cache.get(&mask_cache_key);
                #[cfg(feature = "render_metrics")]
                {
                    metrics.hits += 1;
                }
                cached_result
            } else {
                let Some(loaded_effect) = self.loaded_effects.get(&shape_effect_instance.effect_id)
                else {
                    continue;
                };
                #[cfg(feature = "render_metrics")]
                {
                    metrics.misses += 1;
                }

                let (cached_mask, _mask_was_cached) = resolve_shape_effect_mask(
                    &self.device,
                    encoder,
                    &self.pipeline_resources.shape_effects,
                    &self.state.buffers,
                    &mut self.state.texture_pool,
                    &mut self.state.shape_effect_mask_cache,
                    ShapeEffectMaskDraw {
                        cache_key: mask_cache_key,
                        geometry_range,
                        uniform: raster_rect
                            .mask_uniform(self.state.scale_factor, self.fringe_width),
                    },
                );
                #[cfg(feature = "render_metrics")]
                if _mask_was_cached {
                    metrics.mask_hits += 1;
                } else {
                    metrics.generated_masks += 1;
                }

                let cached_result = render_shape_effect(
                    &self.device,
                    encoder,
                    &mut self.state.texture_pool,
                    &shape_effect_instance.params,
                    EffectPassRunConfig {
                        loaded_effect,
                        params_bind_group: None,
                        source_view: &cached_mask.texture.color_view,
                        effect_sampler,
                        composite_bind_group_layout: &self
                            .pipeline_resources
                            .shapes
                            .shape_texture_bind_group_layout_background,
                        create_composite_bind_group: true,
                        width,
                        height,
                        texture_format: self.config.format,
                        label: "shape_effect",
                    },
                    textures_to_recycle,
                );
                self.state
                    .shape_effect_cache
                    .insert(cache_key, Arc::clone(&cached_result));
                #[cfg(feature = "render_metrics")]
                {
                    metrics.executed_passes += loaded_effect.passes.len() as u64;
                }
                cached_result
            };

            leaf.draw_data.texture_bindings[0] = ShapeTextureBinding::Direct {
                texture_id: cached_result.texture.texture_id,
                bind_group: Arc::clone(&cached_result.texture_bind_group),
            };
        }

        shape_effect_leaves.retain(|_, leaf| leaf.draw_data.texture_bindings[0].is_present());
    }
}

#[cfg(test)]
mod tests {
    use super::{
        compute_shape_effect_raster_rect, shape_effect_quad_transform, ShapeEffectCacheKey,
        ShapeEffectMaskCacheKey,
    };
    use crate::cache::CachedTessellation;
    use crate::effect::ShapeEffectConfig;
    use crate::vertex::CustomVertex;
    use lyon::tessellation::VertexBuffers;
    use std::sync::Arc;

    fn tessellation() -> Arc<CachedTessellation> {
        Arc::new(CachedTessellation {
            vertex_buffers: Arc::new(VertexBuffers::<CustomVertex, u16>::new()),
            local_bounds: [(0.0, 0.0), (10.0, 10.0)],
            texture_mapping_size: [10.0, 10.0],
        })
    }

    fn cache_key(tessellation: Arc<CachedTessellation>, params: Arc<[u8]>) -> ShapeEffectCacheKey {
        ShapeEffectCacheKey {
            mask_key: ShapeEffectMaskCacheKey {
                tessellation,
                local_raster_origin: [-1, -1],
                raster_size: [12, 12],
                scale_factor_bits: 1.0f64.to_bits(),
                fringe_width_bits: 0.75f32.to_bits(),
                downsample_bits: 1.0f32.to_bits(),
                texture_format: wgpu::TextureFormat::Bgra8UnormSrgb,
            },
            effect_id: 7,
            params,
        }
    }

    #[test]
    fn raster_rect_rounds_outward_and_adds_fringe_guard() {
        let raster_rect = compute_shape_effect_raster_rect(
            [(1.25, 2.75), (10.1, 20.2)],
            ShapeEffectConfig::new().outsets(1.0, 2.0, 3.0, 4.0),
            2.0,
            0.75,
        )
        .unwrap();

        assert_eq!(raster_rect.local_physical_origin, [-1, 0]);
        assert_eq!(raster_rect.texture_size, [29, 50]);
        assert_eq!(raster_rect.local_bounds, [(-0.5, 0.0), (14.0, 25.0)]);
    }

    #[test]
    fn raster_rect_downsample_shrinks_texture_but_not_coverage() {
        let full_resolution = compute_shape_effect_raster_rect(
            [(1.25, 2.75), (10.1, 20.2)],
            ShapeEffectConfig::new().outsets(1.0, 2.0, 3.0, 4.0),
            2.0,
            0.75,
        )
        .unwrap();
        let downsampled = compute_shape_effect_raster_rect(
            [(1.25, 2.75), (10.1, 20.2)],
            ShapeEffectConfig::new()
                .outsets(1.0, 2.0, 3.0, 4.0)
                .downsample(0.5),
            2.0,
            0.75,
        )
        .unwrap();

        assert_eq!(downsampled.texture_size, [15, 25]);
        assert_eq!(
            downsampled.local_physical_origin,
            full_resolution.local_physical_origin
        );
        assert_eq!(downsampled.local_bounds, full_resolution.local_bounds);
    }

    #[test]
    fn raster_rect_downsample_keeps_at_least_one_texel() {
        let raster_rect = compute_shape_effect_raster_rect(
            [(0.0, 0.0), (1.0, 1.0)],
            ShapeEffectConfig::new().downsample(0.1),
            1.0,
            0.75,
        )
        .unwrap();

        assert!(raster_rect.texture_size[0] >= 1);
        assert!(raster_rect.texture_size[1] >= 1);
    }

    #[test]
    fn raster_rect_rejects_out_of_range_downsample() {
        for downsample in [0.0, -0.5, f32::NAN, 1.5] {
            assert!(compute_shape_effect_raster_rect(
                [(0.0, 0.0), (10.0, 10.0)],
                ShapeEffectConfig::new().downsample(downsample),
                1.0,
                0.75,
            )
            .is_none());
        }
    }

    #[test]
    fn raster_rect_rejects_non_finite_inputs() {
        assert!(compute_shape_effect_raster_rect(
            [(0.0, 0.0), (f32::NAN, 10.0)],
            ShapeEffectConfig::default(),
            1.0,
            0.75,
        )
        .is_none());
    }

    #[test]
    fn cache_key_uses_tessellation_identity_and_exact_parameter_bytes() {
        let shared_tessellation = tessellation();
        let first_key = cache_key(Arc::clone(&shared_tessellation), Arc::from([1u8, 2, 3, 4]));
        let equal_key = cache_key(Arc::clone(&shared_tessellation), Arc::from([1u8, 2, 3, 4]));
        let different_params =
            cache_key(Arc::clone(&shared_tessellation), Arc::from([1u8, 2, 3, 5]));
        let different_tessellation = cache_key(tessellation(), Arc::from([1u8, 2, 3, 4]));

        assert!(first_key == equal_key);
        assert!(first_key != different_params);
        assert!(first_key != different_tessellation);
    }

    #[test]
    fn cache_key_differs_when_only_downsample_changes() {
        let shared_tessellation = tessellation();
        let full_resolution_key =
            cache_key(Arc::clone(&shared_tessellation), Arc::from([1u8, 2, 3, 4]));
        let mut downsampled_key =
            cache_key(Arc::clone(&shared_tessellation), Arc::from([1u8, 2, 3, 4]));
        downsampled_key.mask_key.downsample_bits = 0.5f32.to_bits();

        assert!(full_resolution_key != downsampled_key);
    }

    #[test]
    fn shape_effect_quad_transform_maps_unit_quad_before_source_transform() {
        let transform = shape_effect_quad_transform(
            [(-3.0, -4.0), (11.0, 15.0)],
            Some(crate::vertex::InstanceTransform::translation(5.0, 7.0)),
        );

        assert_eq!(transform.col0, [14.0, 0.0, 0.0, 0.0]);
        assert_eq!(transform.col1, [0.0, 19.0, 0.0, 0.0]);
        assert_eq!(transform.col3, [2.0, 3.0, 0.0, 1.0]);
    }
}
