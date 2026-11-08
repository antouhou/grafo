use super::{AtlasBatch, AtlasPage};
use crate::core::vertex::{CustomVertex, InstanceTransform};
use crate::wgpu_backend::execution::shapes::buffers::upsert_gpu_buffer;
use crate::wgpu_backend::execution::shapes::coverage::prepare_indices;
use crate::wgpu_backend::resources::ShapePipelines;
use bytemuck::{Pod, Zeroable};
use lru::LruCache;
use std::num::NonZeroUsize;
use wgpu::{
    Buffer, BufferUsages, Color, ColorTargetState, ColorWrites, CommandEncoder, Device, Extent3d,
    FragmentState, IndexFormat, LoadOp, MultisampleState, Operations, PipelineLayoutDescriptor,
    Queue, RenderPassColorAttachment, RenderPassDescriptor, RenderPipeline,
    RenderPipelineDescriptor, ShaderModuleDescriptor, ShaderSource, StoreOp, TextureDescriptor,
    TextureDimension, TextureFormat, TextureUsages, TextureView, VertexBufferLayout, VertexState,
    VertexStepMode,
};

const MASK_FORMAT: TextureFormat = TextureFormat::R8Unorm;
const COLOR_FORMAT: TextureFormat = TextureFormat::Rgba16Float;
const STROKE_SAMPLES: u32 = 4;

const GRADIENT_RASTER_SHADER: &str = concat!(
    include_str!("../../../../../shaders/path_coverage.wgsl"),
    "\n",
    include_str!("../../../../../shaders/shader.wgsl"),
    "\n",
    include_str!("../../../../../shaders/stroke_raster.wgsl"),
    "\n",
    include_str!("../../../../../shaders/stroke_raster_gradient.wgsl"),
);

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct RasterInstance {
    pub(super) transform_col0: [f32; 4],
    pub(super) transform_col1: [f32; 4],
    pub(super) transform_col3: [f32; 4],
    pub(super) screen_to_clip: [f32; 4],
    pub(super) scale: f32,
    vertices_are_projected: u32,
}

impl RasterInstance {
    pub(super) fn new(
        transform: InstanceTransform,
        scale: f64,
        vertices_are_projected: bool,
    ) -> Self {
        Self {
            transform_col0: transform.col0,
            transform_col1: transform.col1,
            transform_col3: transform.col3,
            screen_to_clip: [0.0; 4],
            scale: scale as f32,
            vertices_are_projected: u32::from(vertices_are_projected),
        }
    }
}

fn create_pipeline(
    device: &Device,
    pipelines: &ShapePipelines,
    rasterizes_color: bool,
) -> RenderPipeline {
    let (source, vertex, fragment, format) = if rasterizes_color {
        (
            GRADIENT_RASTER_SHADER,
            "vs_stroke_raster_gradient",
            "fs_stroke_raster_gradient",
            COLOR_FORMAT,
        )
    } else {
        (
            include_str!("../../../../../shaders/stroke_raster.wgsl"),
            "vs_stroke_raster",
            "fs_stroke_raster",
            MASK_FORMAT,
        )
    };
    let shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("stroke_raster"),
        source: ShaderSource::Wgsl(source.into()),
    });
    let layout = rasterizes_color.then(|| {
        device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("stroke_gradient_raster"),
            bind_group_layouts: &[
                &pipelines.layouts.uniform,
                &pipelines.layouts.background_texture,
                &pipelines.layouts.foreground_texture,
                &pipelines.layouts.gradient,
            ],
            push_constant_ranges: &[],
        })
    });
    device.create_render_pipeline(&RenderPipelineDescriptor {
        label: Some("stroke_mask_raster"),
        layout: layout.as_ref(),
        vertex: VertexState {
            module: &shader,
            entry_point: Some(vertex),
            compilation_options: Default::default(),
            buffers: &[
                if rasterizes_color {
                    CustomVertex::desc()
                } else {
                    CustomVertex::solid_desc()
                },
                VertexBufferLayout {
                    array_stride: size_of::<RasterInstance>() as u64,
                    step_mode: VertexStepMode::Instance,
                    attributes: &wgpu::vertex_attr_array![
                        3 => Float32x4, 4 => Float32x4, 5 => Float32x4, 6 => Float32x4,
                        7 => Float32, 8 => Uint32
                    ],
                },
            ],
        },
        fragment: Some(FragmentState {
            module: &shader,
            entry_point: Some(fragment),
            compilation_options: Default::default(),
            targets: &[Some(ColorTargetState {
                format,
                // Each covered sample stores one color, including where triangles overlap.
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
        }),
        primitive: Default::default(),
        depth_stencil: None,
        multisample: MultisampleState {
            count: STROKE_SAMPLES,
            ..Default::default()
        },
        multiview: None,
        cache: None,
    })
}

fn atlas_texture(
    device: &Device,
    size: [u32; 2],
    format: TextureFormat,
    multisampled: bool,
) -> TextureView {
    device
        .create_texture(&TextureDescriptor {
            label: Some(if multisampled {
                "stroke_atlas_samples"
            } else {
                "stroke_atlas_resolved"
            }),
            size: Extent3d {
                width: size[0],
                height: size[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: if multisampled { STROKE_SAMPLES } else { 1 },
            dimension: TextureDimension::D2,
            format,
            usage: TextureUsages::RENDER_ATTACHMENT
                | if multisampled {
                    TextureUsages::empty()
                } else {
                    TextureUsages::TEXTURE_BINDING
                },
            view_formats: &[],
        })
        .create_view(&Default::default())
}

struct AtlasRasterPipeline {
    pipeline: RenderPipeline,
    multisampled_pages: LruCache<[u32; 2], TextureView>,
}

impl AtlasRasterPipeline {
    fn new(device: &Device, pipelines: &ShapePipelines, rasterizes_color: bool) -> Self {
        Self {
            pipeline: create_pipeline(device, pipelines, rasterizes_color),
            multisampled_pages: LruCache::new(NonZeroUsize::new(2).expect("scratch page capacity")),
        }
    }
}

#[derive(Default)]
pub(super) struct RasterResources {
    mask_pipeline: Option<AtlasRasterPipeline>,
    gradient_pipeline: Option<AtlasRasterPipeline>,
    vertices: Option<Buffer>,
    indices: Option<Buffer>,
    instances: Option<Buffer>,
    pub(super) quad_indices: Option<Buffer>,
    quad_index_storage: Vec<u32>,
}

impl RasterResources {
    pub(super) fn upload(&mut self, batch: &AtlasBatch, device: &Device, queue: &Queue) {
        for (buffer, label, bytes, usage) in [
            (
                &mut self.vertices,
                "stroke_vertices",
                bytemuck::cast_slice(batch.vertices.as_slice()),
                BufferUsages::VERTEX,
            ),
            (
                &mut self.indices,
                "stroke_indices",
                bytemuck::cast_slice(batch.indices.as_slice()),
                BufferUsages::INDEX,
            ),
            (
                &mut self.instances,
                "stroke_instances",
                bytemuck::cast_slice(batch.instances.as_slice()),
                BufferUsages::VERTEX,
            ),
        ] {
            upsert_gpu_buffer(
                device,
                queue,
                buffer,
                label,
                bytes,
                usage | BufferUsages::COPY_DST,
            );
        }
        let largest_page = batch
            .pages
            .iter()
            .map(|page| page.draws.len())
            .max()
            .unwrap_or(0);
        prepare_indices(
            device,
            queue,
            &mut self.quad_indices,
            &mut self.quad_index_storage,
            largest_page,
        );
    }

    pub(super) fn rasterize_page(
        &mut self,
        page: &AtlasPage,
        batch: &AtlasBatch,
        device: &Device,
        pipelines: &ShapePipelines,
        encoder: &mut CommandEncoder,
    ) -> TextureView {
        let (pipeline, format) = if page.rasterizes_color {
            (&mut self.gradient_pipeline, COLOR_FORMAT)
        } else {
            (&mut self.mask_pipeline, MASK_FORMAT)
        };
        let raster = pipeline.get_or_insert_with(|| {
            AtlasRasterPipeline::new(device, pipelines, page.rasterizes_color)
        });
        let multisampled_page = raster
            .multisampled_pages
            .get_or_insert(page.size, || atlas_texture(device, page.size, format, true));
        let resolved = atlas_texture(device, page.size, format, false);
        let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("stroke_coverage_atlas"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: multisampled_page,
                resolve_target: Some(&resolved),
                ops: Operations {
                    load: LoadOp::Clear(Color::TRANSPARENT),
                    store: StoreOp::Discard,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(&raster.pipeline);
        if page.rasterizes_color {
            pass.set_bind_group(0, &pipelines.and_bind_group, &[]);
            for (layer, binding) in pipelines
                .default_shape_texture_bind_groups
                .iter()
                .enumerate()
            {
                pass.set_bind_group(1 + layer as u32, binding.as_ref(), &[]);
            }
        }
        let vertices = self.vertices.as_ref().expect("stroke mesh was uploaded");
        let indices = self.indices.as_ref().expect("stroke indices were uploaded");
        pass.set_index_buffer(indices.slice(..), IndexFormat::Uint32);
        pass.set_vertex_buffer(
            1,
            self.instances
                .as_ref()
                .expect("stroke instances were uploaded")
                .slice(..),
        );
        for index in page.draws.clone() {
            let draw = &batch.draws[index];
            if let Some(gradient) = &batch.masks[draw.mask_index].mask.rasterized_gradient {
                pass.set_bind_group(3, &gradient.bind_group, &[]);
            }
            let [x, y] = draw.atlas_origin;
            let [width, height] = draw.size;
            pass.set_viewport(x as f32, y as f32, width as f32, height as f32, 0.0, 1.0);
            pass.set_scissor_rect(x, y, width, height);
            pass.set_vertex_buffer(0, vertices.slice(draw.mesh.vertices.clone()));
            pass.draw_indexed(draw.mesh.indices.clone(), 0, index as u32..index as u32 + 1);
        }
        drop(pass);
        resolved
    }
}
