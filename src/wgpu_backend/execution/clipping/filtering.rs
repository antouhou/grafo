use super::pipelines::{image_entry, ClipPipelineCompiler};
use crate::commands::{EffectApplication, IntermediateTextureId};
use crate::core::PhysicalRect;
use crate::wgpu_backend::execution::effects::PooledTexture;
use crate::wgpu_backend::execution::instructions::{ExecutionContext, ExecutionResources};
use crate::wgpu_backend::execution::render_pass::RenderPass;
use crate::wgpu_backend::execution::shapes::coverage::CoverageAllocation;
use crate::wgpu_backend::execution::textures::PlannedTexture;
use std::sync::Arc;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, Color, CommandEncoder, Device, LoadOp,
    Operations, PipelineLayoutDescriptor, RenderPassColorAttachment, RenderPassDescriptor,
    RenderPipeline, Sampler, SamplerBindingType, ShaderModuleDescriptor, ShaderSource,
    ShaderStages, StoreOp, TextureFormat, TextureView,
};

pub(in crate::wgpu_backend::execution) struct ClipBackdropComposition {
    pub bounds: PhysicalRect,
    pub base_bounds: PhysicalRect,
    pub coverage: Arc<CoverageAllocation>,
    pub images: [TextureView; 3],
}

pub(super) struct ClipFilterPipeline {
    pipeline: RenderPipeline,
    inherited: RenderPipeline,
    images: BindGroupLayout,
    sampler: BindGroup,
}

impl ClipFilterPipeline {
    fn new(
        device: &Device,
        format: TextureFormat,
        coverage: &BindGroupLayout,
        sampler: &Sampler,
    ) -> Self {
        let images = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("clip_filter_images"),
            entries: &[
                image_entry(0, false, true),
                image_entry(1, false, true),
                image_entry(2, false, true),
            ],
        });
        let sampler_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("clip_filter_sampler"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Sampler(SamplerBindingType::Filtering),
                count: None,
            }],
        });
        let sampler = device.create_bind_group(&BindGroupDescriptor {
            label: Some("clip_filter_sampler"),
            layout: &sampler_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::Sampler(sampler),
            }],
        });
        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("clip_filter"),
            bind_group_layouts: &[Some(coverage), Some(&images), Some(&sampler_layout)],
            immediate_size: 0,
        });
        let mut source = include_str!("../../../shaders/clip_group.wgsl")
            .replace("CLIP_TEXTURE", "texture_2d<f32>")
            .replace("CLIP_SAMPLE", "0");
        source.push_str(include_str!("../../../shaders/clip_filter.wgsl"));
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("clip_filter"),
            source: ShaderSource::Wgsl(source.into()),
        });
        let compiler = ClipPipelineCompiler::new(device, &layout, &shader, format, 1);
        Self {
            pipeline: compiler.compile("clip_vertex", "filter_fragment", None),
            inherited: compiler.compile("clip_vertex", "inherited_fragment", None),
            images,
            sampler,
        }
    }
}

impl ExecutionContext<'_> {
    fn clip_filter_pipeline(&self) -> &ClipFilterPipeline {
        self.clip_pipelines().filtering.get_or_init(|| {
            Box::new(ClipFilterPipeline::new(
                self.device,
                self.format,
                &self.pipelines.shapes.layouts.uniform,
                self.effects.sampler,
            ))
        })
    }
}

impl ExecutionResources<'_> {
    pub(in crate::wgpu_backend::execution) fn compose_inherited_backdrop(
        &mut self,
        encoder: &mut CommandEncoder,
        output: &PooledTexture,
        composition: ClipBackdropComposition,
    ) {
        let pipeline = self.context.clip_filter_pipeline();
        let bounds = composition.bounds;
        let origin = composition.base_bounds.min;
        let binding = self.textures.clip_bindings.coverage(
            self.context.device,
            self.context.queue,
            &self.context.pipelines.shapes,
            &composition.coverage,
            [
                [
                    bounds.min.x as f32,
                    bounds.min.y as f32,
                    bounds.width() as f32,
                    bounds.height() as f32,
                ],
                [origin.x as f32, origin.y as f32, 0.0, 0.0],
            ],
        );
        let images = self.textures.clip_bindings.images_with_layout(
            self.context.device,
            &pipeline.images,
            [
                &composition.images[0],
                &composition.images[1],
                &composition.images[2],
            ],
        );
        let mut pass = RenderPass::new(encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("clip_inherited_backdrop"),
            color_attachments: &[Some(RenderPassColorAttachment {
                depth_slice: None,
                view: &output.color_view,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Load,
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        }));
        pass.set_bind_group(0, &binding, &[]);
        pass.set_bind_group(1, &images, &[]);
        pass.set_bind_group(2, &pipeline.sampler, &[]);
        composition.coverage.buffers().bind(&mut pass);
        composition
            .coverage
            .draw_with_pipeline(&mut pass, &pipeline.inherited, 0..1);
    }

    /// Transfers filtered colors back into the clip's working pixels.
    pub(in crate::wgpu_backend::execution) fn finish_clip_backdrop_filter(
        &mut self,
        encoder: &mut CommandEncoder,
        effect: &EffectApplication<'_>,
    ) {
        let Some(filter) = self.textures.clip_filter.take() else {
            return;
        };
        if filter.input != effect.input {
            return;
        }
        let working = filter.working;
        let origin = filter.bounds.min;
        let coverage = filter.coverage;
        let IntermediateTextureId::Planned(input) = effect.input else {
            unreachable!("backdrop input is planned")
        };
        let IntermediateTextureId::Planned(output) = effect.output else {
            unreachable!("backdrop output is planned")
        };
        let PlannedTexture::Work(input) = self.textures.planned[input] else {
            unreachable!("backdrop input is complete")
        };
        let PlannedTexture::Work(filtered_index) = self.textures.planned[output] else {
            unreachable!("backdrop output is complete")
        };
        let original = self.textures.work_textures[input].color_view.clone();
        let filtered = self.textures.work_textures[filtered_index]
            .color_view
            .clone();
        let size = filter.bounds.size().to_u32();
        let mut corrected = self.textures.pool.acquire_color_only(
            self.context.device,
            size.width,
            size.height,
            self.context.format,
            1,
        );
        let pipeline = self.context.clip_filter_pipeline();
        let binding = self.textures.clip_bindings.coverage(
            self.context.device,
            self.context.queue,
            &self.context.pipelines.shapes,
            &coverage,
            [
                [
                    filter.bounds.min.x as f32,
                    filter.bounds.min.y as f32,
                    size.width as f32,
                    size.height as f32,
                ],
                [origin.x as f32, origin.y as f32, 0.0, 0.0],
            ],
        );
        let images = self.textures.clip_bindings.images_with_layout(
            self.context.device,
            &pipeline.images,
            [&working, &filtered, &original],
        );
        let mut pass = RenderPass::new(encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("clip_filter_coverage"),
            color_attachments: &[Some(RenderPassColorAttachment {
                depth_slice: None,
                view: &corrected.color_view,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(Color::TRANSPARENT),
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        }));
        pass.set_bind_group(0, &binding, &[]);
        pass.set_bind_group(1, &images, &[]);
        pass.set_bind_group(2, &pipeline.sampler, &[]);
        coverage.buffers().bind(&mut pass);
        coverage.draw_with_pipeline(&mut pass, &pipeline.pipeline, 0..1);
        drop(pass);
        corrected.composite_bind_group(
            self.context.device,
            self.context.effects.composite_layout,
            self.context.effects.sampler,
        );
        self.textures.planned[output] = PlannedTexture::Work(self.textures.work_textures.len());
        self.textures.work_textures.push(corrected);
    }
}
