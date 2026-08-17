// The bytemuck derive emits private compile-time helpers that trigger false unused warnings.
#![allow(unused)]

use std::hash::{Hash, Hasher};
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};

use crate::cache::{CachedTessellation, FrameCache};
use crate::effect::{
    self, create_effect_input_bind_group_layout, PooledTexture, ShapeEffectConfig,
};
use crate::vertex::{CustomVertex, InstanceTransform};

use super::passes::{apply_effect_passes, EffectPassRunConfig};
use super::types::{Buffers, DrawCommand, Pipelines};
use super::Renderer;

const SHAPE_EFFECT_MASK_SHADER: &str = r#"
struct MaskUniforms {
    local_origin: vec2<f32>,
    logical_size: vec2<f32>,
    scale_factor: f32,
    fringe_width: f32,
    _padding: vec2<f32>,
};

struct MaskVertexInput {
    @location(0) position: vec2<f32>,
    @location(8) normal: vec2<f32>,
    @location(9) coverage: f32,
};

struct MaskVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) coverage: f32,
};

@group(0) @binding(0) var<uniform> uniforms: MaskUniforms;

@vertex
fn mask_vertex(input: MaskVertexInput) -> MaskVertexOutput {
    var local_position = input.position;
    if input.coverage < 1.0 {
        local_position += input.normal * (uniforms.fringe_width / uniforms.scale_factor);
    }

    let target_position = local_position - uniforms.local_origin;
    var output: MaskVertexOutput;
    output.position = vec4<f32>(
        2.0 * target_position.x / uniforms.logical_size.x - 1.0,
        1.0 - 2.0 * target_position.y / uniforms.logical_size.y,
        0.0,
        1.0,
    );
    output.coverage = input.coverage;
    return output;
}

@fragment
fn mask_fragment(input: MaskVertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(input.coverage);
}
"#;

const SHAPE_EFFECT_COMPOSITE_SHADER: &str = r#"
struct CanvasUniforms {
    canvas_size: vec2<f32>,
    scale_factor: f32,
    fringe_width: f32,
};

struct CompositeVertexInput {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(3) transform_column_0: vec4<f32>,
    @location(4) transform_column_1: vec4<f32>,
    @location(5) transform_column_2: vec4<f32>,
    @location(6) transform_column_3: vec4<f32>,
};

struct CompositeVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@group(0) @binding(0) var<uniform> canvas: CanvasUniforms;
@group(1) @binding(0) var effect_texture: texture_2d<f32>;
@group(1) @binding(1) var effect_sampler: sampler;

@vertex
fn composite_vertex(input: CompositeVertexInput) -> CompositeVertexOutput {
    let transform = mat4x4<f32>(
        input.transform_column_0,
        input.transform_column_1,
        input.transform_column_2,
        input.transform_column_3,
    );
    let transformed = transform * vec4<f32>(input.position, 0.0, 1.0);
    let inverse_w = 1.0 / max(abs(transformed.w), 1e-6);
    let logical_position = transformed.xy * inverse_w;

    var output: CompositeVertexOutput;
    output.position = vec4<f32>(
        2.0 * logical_position.x / canvas.canvas_size.x - 1.0,
        1.0 - 2.0 * logical_position.y / canvas.canvas_size.y,
        1.0,
        1.0,
    );
    output.uv = input.uv;
    return output;
}

@fragment
fn composite_fragment(input: CompositeVertexOutput) -> @location(0) vec4<f32> {
    return textureSample(effect_texture, effect_sampler, input.uv);
}
"#;

#[repr(C)]
#[allow(unused)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(super) struct ShapeEffectVertex {
    pub position: [f32; 2],
    pub uv: [f32; 2],
}

impl ShapeEffectVertex {
    fn descriptor<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x2,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x2,
                    offset: std::mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
                    shader_location: 1,
                },
            ],
        }
    }
}

#[repr(C)]
#[allow(unused)]
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
    pub physical_origin: [i32; 2],
    pub physical_size: [u32; 2],
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

    let physical_origin = [physical_minimum_x as i32, physical_minimum_y as i32];
    let physical_width = physical_maximum_x - physical_minimum_x;
    let physical_height = physical_maximum_y - physical_minimum_y;
    if physical_width <= 0.0
        || physical_height <= 0.0
        || physical_width > f64::from(u32::MAX)
        || physical_height > f64::from(u32::MAX)
    {
        return None;
    }

    let physical_size = [physical_width as u32, physical_height as u32];
    Some(ShapeEffectRasterRect {
        physical_origin,
        physical_size,
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

#[derive(Clone)]
pub(super) struct ShapeEffectCacheKey {
    pub effect_id: u64,
    pub tessellation: Arc<CachedTessellation>,
    pub params: Arc<[u8]>,
    pub raster_origin: [i32; 2],
    pub raster_size: [u32; 2],
    pub scale_factor_bits: u64,
    pub fringe_width_bits: u32,
    pub texture_format: wgpu::TextureFormat,
}

impl PartialEq for ShapeEffectCacheKey {
    fn eq(&self, other: &Self) -> bool {
        self.effect_id == other.effect_id
            && Arc::ptr_eq(&self.tessellation, &other.tessellation)
            && self.params.as_ref() == other.params.as_ref()
            && self.raster_origin == other.raster_origin
            && self.raster_size == other.raster_size
            && self.scale_factor_bits == other.scale_factor_bits
            && self.fringe_width_bits == other.fringe_width_bits
            && self.texture_format == other.texture_format
    }
}

impl Eq for ShapeEffectCacheKey {}

impl Hash for ShapeEffectCacheKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.effect_id.hash(state);
        (Arc::as_ptr(&self.tessellation) as usize).hash(state);
        self.params.as_ref().hash(state);
        self.raster_origin.hash(state);
        self.raster_size.hash(state);
        self.scale_factor_bits.hash(state);
        self.fringe_width_bits.hash(state);
        self.texture_format.hash(state);
    }
}

pub(super) struct CachedShapeEffect {
    pub texture: PooledTexture,
    pub texture_bind_group: wgpu::BindGroup,
    pub quad_vertex_buffer: wgpu::Buffer,
    pub local_bounds: [(f32, f32); 2],
}

pub(super) type ShapeEffectResultCache = FrameCache<ShapeEffectCacheKey, Arc<CachedShapeEffect>>;

pub(super) struct ShapeEffectRendererResources {
    pub mask_bind_group_layout: wgpu::BindGroupLayout,
    pub texture_bind_group_layout: wgpu::BindGroupLayout,
    pub mask_pipeline: wgpu::RenderPipeline,
    pub composite_pipeline: wgpu::RenderPipeline,
    pub quad_index_buffer: wgpu::Buffer,
}

impl ShapeEffectRendererResources {
    pub(super) fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        sample_count: u32,
        canvas_uniform_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let mask_bind_group_layout = create_mask_bind_group_layout(device);
        let texture_bind_group_layout = create_effect_input_bind_group_layout(device);
        let mask_pipeline = create_mask_pipeline(device, format, &mask_bind_group_layout);
        let composite_pipeline = create_composite_pipeline(
            device,
            format,
            sample_count,
            canvas_uniform_bind_group_layout,
            &texture_bind_group_layout,
        );
        let quad_index_buffer = crate::pipeline::create_buffer_init(
            device,
            Some("shape_effect_quad_indices"),
            bytemuck::cast_slice(&[0u16, 1, 2, 0, 2, 3]),
            wgpu::BufferUsages::INDEX,
        );

        Self {
            mask_bind_group_layout,
            texture_bind_group_layout,
            mask_pipeline,
            composite_pipeline,
            quad_index_buffer,
        }
    }

    pub(super) fn recreate_pipelines(
        &mut self,
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        sample_count: u32,
        canvas_uniform_bind_group_layout: &wgpu::BindGroupLayout,
    ) {
        self.mask_pipeline = create_mask_pipeline(device, format, &self.mask_bind_group_layout);
        self.composite_pipeline = create_composite_pipeline(
            device,
            format,
            sample_count,
            canvas_uniform_bind_group_layout,
            &self.texture_bind_group_layout,
        );
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

fn create_composite_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    sample_count: u32,
    canvas_uniform_bind_group_layout: &wgpu::BindGroupLayout,
    texture_bind_group_layout: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("shape_effect_composite_shader"),
        source: wgpu::ShaderSource::Wgsl(SHAPE_EFFECT_COMPOSITE_SHADER.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("shape_effect_composite_pipeline_layout"),
        bind_group_layouts: &[canvas_uniform_bind_group_layout, texture_bind_group_layout],
        push_constant_ranges: &[],
    });
    let stencil_face = wgpu::StencilFaceState {
        compare: wgpu::CompareFunction::Equal,
        fail_op: wgpu::StencilOperation::Keep,
        depth_fail_op: wgpu::StencilOperation::Keep,
        pass_op: wgpu::StencilOperation::Keep,
    };

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("shape_effect_composite_pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("composite_vertex"),
            compilation_options: Default::default(),
            buffers: &[ShapeEffectVertex::descriptor(), InstanceTransform::desc()],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("composite_fragment"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth24PlusStencil8,
            depth_write_enabled: false,
            depth_compare: wgpu::CompareFunction::Always,
            stencil: wgpu::StencilState {
                front: stencil_face,
                back: stencil_face,
                read_mask: 0xff,
                write_mask: 0,
            },
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState {
            count: sample_count,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview: None,
        cache: None,
    })
}

pub(super) fn create_quad_vertices(local_bounds: [(f32, f32); 2]) -> [ShapeEffectVertex; 4] {
    let [(minimum_x, minimum_y), (maximum_x, maximum_y)] = local_bounds;
    [
        ShapeEffectVertex {
            position: [minimum_x, minimum_y],
            uv: [0.0, 0.0],
        },
        ShapeEffectVertex {
            position: [maximum_x, minimum_y],
            uv: [1.0, 0.0],
        },
        ShapeEffectVertex {
            position: [maximum_x, maximum_y],
            uv: [1.0, 1.0],
        },
        ShapeEffectVertex {
            position: [minimum_x, maximum_y],
            uv: [0.0, 1.0],
        },
    ]
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

impl<'a> Renderer<'a> {
    pub(super) fn resolve_shape_effects(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        resolved_results: &mut ahash::HashMap<usize, Arc<CachedShapeEffect>>,
        textures_to_recycle: &mut Vec<PooledTexture>,
        #[cfg(feature = "render_metrics")] metrics: &mut super::metrics::ShapeEffectCacheMetrics,
    ) {
        let Some(aggregated_vertex_buffer) = self.aggregated_vertex_buffer.as_ref() else {
            return;
        };
        let Some(aggregated_index_buffer) = self.aggregated_index_buffer.as_ref() else {
            return;
        };
        let effect_sampler = self
            .effect_sampler
            .as_ref()
            .expect("shape effects require the shared effect sampler");
        let maximum_texture_dimension = self.device.limits().max_texture_dimension_2d;
        let maximum_texel_count = u64::from(self.physical_size.0)
            .saturating_mul(u64::from(self.physical_size.1))
            .saturating_mul(4);

        for (&node_id, shape_effect_instance) in &self.shape_effects {
            let Some(DrawCommand::CachedShape(cached_shape)) = self.draw_tree.get(node_id) else {
                continue;
            };
            let Some(index_buffer_range) = cached_shape.index_buffer_range else {
                continue;
            };
            if cached_shape.is_empty {
                continue;
            }

            let Some(raster_rect) = compute_shape_effect_raster_rect(
                cached_shape.cached_shape.tessellation.local_bounds,
                shape_effect_instance.config,
                self.scale_factor,
                self.fringe_width,
            ) else {
                tracing::warn!(
                    node_id,
                    effect_id = shape_effect_instance.effect_id,
                    "skipping shape effect with invalid raster bounds"
                );
                continue;
            };
            let [width, height] = raster_rect.physical_size;
            let texel_count = u64::from(width).saturating_mul(u64::from(height));
            if width > maximum_texture_dimension
                || height > maximum_texture_dimension
                || texel_count > maximum_texel_count
            {
                tracing::warn!(
                    node_id,
                    effect_id = shape_effect_instance.effect_id,
                    width,
                    height,
                    maximum_texture_dimension,
                    maximum_texel_count,
                    scale_factor = self.scale_factor,
                    "skipping oversized shape effect texture"
                );
                continue;
            }

            let cache_key = ShapeEffectCacheKey {
                effect_id: shape_effect_instance.effect_id,
                tessellation: Arc::clone(&cached_shape.cached_shape.tessellation),
                params: Arc::clone(&shape_effect_instance.params),
                raster_origin: raster_rect.physical_origin,
                raster_size: raster_rect.physical_size,
                scale_factor_bits: self.scale_factor.to_bits(),
                fringe_width_bits: self.fringe_width.to_bits(),
                texture_format: self.config.format,
            };
            if let Some(cached_result) = self.shape_effect_cache.get(&cache_key) {
                #[cfg(feature = "render_metrics")]
                {
                    metrics.hits += 1;
                }
                resolved_results.insert(node_id, cached_result);
                continue;
            }

            #[cfg(feature = "render_metrics")]
            {
                metrics.misses += 1;
                metrics.generated_masks += 1;
            }

            let Some(loaded_effect) = self.loaded_effects.get(&shape_effect_instance.effect_id)
            else {
                continue;
            };
            let mask_texture = self.offscreen_texture_pool.acquire_color_only(
                &self.device,
                width,
                height,
                self.config.format,
                1,
            );
            let mask_uniform = raster_rect.mask_uniform(self.scale_factor, self.fringe_width);
            let mask_uniform_buffer = crate::pipeline::create_buffer_init(
                &self.device,
                Some("shape_effect_mask_uniform"),
                bytemuck::bytes_of(&mask_uniform),
                wgpu::BufferUsages::UNIFORM,
            );
            let mask_bind_group = create_mask_bind_group(
                &self.device,
                &self.shape_effect_resources.mask_bind_group_layout,
                &mask_uniform_buffer,
            );

            {
                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("shape_effect_mask_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &mask_texture.color_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                render_pass.set_pipeline(&self.shape_effect_resources.mask_pipeline);
                render_pass.set_bind_group(0, &mask_bind_group, &[]);
                render_pass.set_vertex_buffer(0, aggregated_vertex_buffer.slice(..));
                render_pass
                    .set_index_buffer(aggregated_index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                let index_start = index_buffer_range.0 as u32;
                let index_end = (index_buffer_range.0 + index_buffer_range.1) as u32;
                render_pass.draw_indexed(index_start..index_end, 0, 0..1);
            }

            let parameter_buffer = (!shape_effect_instance.params.is_empty()).then(|| {
                crate::pipeline::create_buffer_init(
                    &self.device,
                    Some("shape_effect_params_buffer"),
                    shape_effect_instance.params.as_ref(),
                    wgpu::BufferUsages::UNIFORM,
                )
            });
            let parameter_bind_group = parameter_buffer.as_ref().and_then(|buffer| {
                loaded_effect
                    .params_bind_group_layout
                    .as_ref()
                    .map(|layout| effect::create_params_bind_group(&self.device, layout, buffer))
            });
            let effect_output = apply_effect_passes(
                &self.device,
                encoder,
                &mut self.offscreen_texture_pool,
                EffectPassRunConfig {
                    loaded_effect,
                    params_bind_group: parameter_bind_group.as_ref(),
                    source_view: &mask_texture.color_view,
                    effect_sampler,
                    composite_bind_group_layout: &self
                        .shape_effect_resources
                        .texture_bind_group_layout,
                    create_composite_bind_group: true,
                    width,
                    height,
                    texture_format: self.config.format,
                    label_prefix: "shape_effect",
                },
            );
            #[cfg(feature = "render_metrics")]
            {
                metrics.executed_passes += loaded_effect.passes.len() as u64;
            }
            let (final_texture, texture_bind_group, mut recyclable_work_textures) =
                effect_output.into_final_and_recyclable();
            let quad_vertices = create_quad_vertices(raster_rect.local_bounds);
            let quad_vertex_buffer = crate::pipeline::create_buffer_init(
                &self.device,
                Some("shape_effect_quad_vertices"),
                bytemuck::cast_slice(&quad_vertices),
                wgpu::BufferUsages::VERTEX,
            );
            let cached_result = Arc::new(CachedShapeEffect {
                texture: final_texture,
                texture_bind_group: texture_bind_group
                    .expect("shape effect generation must create a texture bind group"),
                quad_vertex_buffer,
                local_bounds: raster_rect.local_bounds,
            });

            textures_to_recycle.push(mask_texture);
            textures_to_recycle.append(&mut recyclable_work_textures);
            self.shape_effect_cache
                .insert(cache_key, Arc::clone(&cached_result));
            resolved_results.insert(node_id, cached_result);
        }
    }
}

pub(super) fn composite_cached_shape_effect(
    render_pass: &mut wgpu::RenderPass<'_>,
    cached_effect: &CachedShapeEffect,
    instance_index: Option<usize>,
    parent_stencil: u32,
    pipelines: &Pipelines<'_>,
    buffers: &Buffers<'_>,
) {
    render_pass.set_pipeline(pipelines.shape_effect_composite_pipeline);
    render_pass.set_bind_group(0, pipelines.and_bind_group, &[]);
    render_pass.set_bind_group(1, &cached_effect.texture_bind_group, &[]);
    render_pass.set_vertex_buffer(0, cached_effect.quad_vertex_buffer.slice(..));
    if let (Some(instance_index), Some(instance_transform_buffer)) =
        (instance_index, buffers.aggregated_instance_transform_buffer)
    {
        let stride = std::mem::size_of::<InstanceTransform>() as u64;
        let offset = instance_index as u64 * stride;
        render_pass.set_vertex_buffer(1, instance_transform_buffer.slice(offset..offset + stride));
    } else {
        render_pass.set_vertex_buffer(1, buffers.identity_instance_transform_buffer.slice(..));
    }
    render_pass.set_index_buffer(
        pipelines.shape_effect_quad_index_buffer.slice(..),
        wgpu::IndexFormat::Uint16,
    );
    render_pass.set_stencil_reference(parent_stencil);
    render_pass.draw_indexed(0..6, 0, 0..1);
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use lyon::tessellation::VertexBuffers;

    use super::{compute_shape_effect_raster_rect, ShapeEffectCacheKey};
    use crate::cache::CachedTessellation;
    use crate::effect::ShapeEffectConfig;
    use crate::vertex::CustomVertex;

    fn tessellation() -> Arc<CachedTessellation> {
        Arc::new(CachedTessellation {
            vertex_buffers: Arc::new(VertexBuffers::<CustomVertex, u16>::new()),
            local_bounds: [(0.0, 0.0), (10.0, 10.0)],
            texture_mapping_size: [10.0, 10.0],
        })
    }

    fn cache_key(tessellation: Arc<CachedTessellation>, params: Arc<[u8]>) -> ShapeEffectCacheKey {
        ShapeEffectCacheKey {
            effect_id: 7,
            tessellation,
            params,
            raster_origin: [-1, -1],
            raster_size: [12, 12],
            scale_factor_bits: 1.0f64.to_bits(),
            fringe_width_bits: 0.75f32.to_bits(),
            texture_format: wgpu::TextureFormat::Bgra8UnormSrgb,
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

        assert_eq!(raster_rect.physical_origin, [-1, 0]);
        assert_eq!(raster_rect.physical_size, [29, 50]);
        assert_eq!(raster_rect.local_bounds, [(-0.5, 0.0), (14.0, 25.0)]);
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
}
