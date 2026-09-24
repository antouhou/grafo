use super::effects::{
    apply_effect_passes, EffectExecutionResources, EffectPassRunConfig, EffectRegistry,
};
use super::shapes::ShapeExecutionResources;
use super::textures::{
    CachedShapeEffectMask, IntermediateTexture, IntermediateTextureResources, ShapeEffectCacheKey,
    ShapeEffectMaskCacheKey,
};
use super::{draws, targets};
use crate::commands::{
    EffectApplication, EffectParameters, IntermediateTextureId, MaskTarget, RenderPlan,
    ShapeMaskDraw,
};
use crate::renderer::backend::resources::Buffers;
#[cfg(feature = "render_metrics")]
use crate::renderer::metrics::ShapeEffectCacheMetrics;
use bytemuck::{Pod, Zeroable};
pub(in crate::renderer) use pipelines::ShapeEffectRendererResources;
use std::sync::Arc;
use wgpu::{
    BindGroupLayout, Color, CommandEncoder, Device, LoadOp, Operations, Queue,
    RenderPassColorAttachment, RenderPassDescriptor, Sampler, StoreOp, TextureFormat,
};

mod pipelines;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct MaskUniform {
    local_origin: [f32; 2],
    logical_size: [f32; 2],
    scale_factor: f32,
    fringe_width: f32,
    padding: [f32; 2],
}

#[derive(Clone)]
pub(super) struct CompletedMask {
    pub(super) texture: IntermediateTextureId,
    key: ShapeEffectMaskCacheKey,
    #[cfg(feature = "render_metrics")]
    was_cached: bool,
}

pub(in crate::renderer) struct ShapeEffectExecutionResources<'a> {
    pub device: &'a Device,
    pub queue: &'a Queue,
    pub registry: &'a EffectRegistry,
    pub sampler: &'a Sampler,
    pub composite_layout: &'a BindGroupLayout,
    pub format: TextureFormat,
    pub pipelines: &'a ShapeEffectRendererResources,
    pub buffers: &'a Buffers,
    pub shapes: &'a ShapeExecutionResources,
    pub effects: &'a mut EffectExecutionResources,
    pub textures: &'a mut IntermediateTextureResources,
    #[cfg(feature = "render_metrics")]
    pub metrics: &'a mut ShapeEffectCacheMetrics,
}

impl ShapeEffectExecutionResources<'_> {
    pub(super) fn draw_mask(
        &mut self,
        encoder: &mut CommandEncoder,
        target: MaskTarget,
        command: ShapeMaskDraw,
    ) -> CompletedMask {
        let shape = self.shapes.draw_resources(command.shape);
        let key = ShapeEffectMaskCacheKey {
            tessellation: Arc::clone(
                shape
                    .mask_tessellation
                    .as_ref()
                    .expect("shape effect geometry was prepared when attached"),
            ),
            local_raster_origin: command.local_physical_origin,
            raster_size: target.size,
            scale_factor_bits: command.scale_factor.to_bits(),
            fringe_width_bits: command.fringe_width.to_bits(),
            downsample_bits: command.downsample.to_bits(),
            texture_format: self.format,
        };
        let (_, _was_cached) =
            self.textures
                .shape_effect_masks
                .get_or_insert_with(key.clone(), || {
                    let [width, height] = target.size;
                    let texture = self.textures.pool.acquire_color_only(
                        self.device,
                        width,
                        height,
                        self.format,
                        1,
                    );
                    let uniform = MaskUniform {
                        local_origin: [command.local_bounds[0].0, command.local_bounds[0].1],
                        logical_size: [
                            command.local_bounds[1].0 - command.local_bounds[0].0,
                            command.local_bounds[1].1 - command.local_bounds[0].1,
                        ],
                        scale_factor: command.scale_factor as f32,
                        fringe_width: command.fringe_width,
                        padding: [0.0; 2],
                    };
                    let binding = self.effects.parameters.prepare(
                        self.device,
                        self.queue,
                        &self.pipelines.mask_bind_group_layout,
                        bytemuck::bytes_of(&uniform),
                    );
                    let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                        label: Some("shape_effect_mask"),
                        color_attachments: &[Some(RenderPassColorAttachment {
                            view: &texture.color_view,
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
                    targets::set_scissor(&mut pass, command.clip.scissor);
                    pass.set_stencil_reference(command.clip.stencil_reference);
                    draws::draw_shape_mask(
                        &mut pass,
                        shape
                            .location
                            .expect("planned mask has uploaded geometry")
                            .geometry_range,
                        &self.pipelines.mask_pipeline,
                        binding,
                        self.buffers,
                    );
                    drop(pass);
                    CachedShapeEffectMask { texture }
                });
        CompletedMask {
            texture: target.texture,
            key,
            #[cfg(feature = "render_metrics")]
            was_cached: _was_cached,
        }
    }

    pub(super) fn apply_effect(
        &mut self,
        encoder: &mut CommandEncoder,
        command: &EffectApplication,
        commands: &RenderPlan,
        mask: CompletedMask,
    ) {
        assert_eq!(mask.texture, command.input);
        let EffectParameters::Shared(index) = command.parameters else {
            unreachable!("shape effects share immutable parameter bytes");
        };
        let parameters = &commands.shared_effect_parameters[index];
        let key = ShapeEffectCacheKey {
            mask_key: mask.key,
            effect_id: command.effect_id,
            params: Arc::clone(parameters),
        };
        let texture = if let Some(texture) = self.textures.shape_effect_results.get(&key) {
            #[cfg(feature = "render_metrics")]
            {
                self.metrics.hits += 1;
            }
            texture
        } else {
            #[cfg(feature = "render_metrics")]
            {
                self.metrics.misses += 1;
                if mask.was_cached {
                    self.metrics.mask_hits += 1;
                } else {
                    self.metrics.generated_masks += 1;
                }
                self.metrics.executed_passes += self.registry.pass_count(command.effect_id) as u64;
            }
            let mask_texture = &mut self
                .textures
                .shape_effect_masks
                .get_mut(&key.mask_key)
                .expect("mask was completed before its effect")
                .texture;
            let source_bind_group = mask_texture.input_bind_group(
                self.device,
                self.registry.input_bind_group_layout(),
                self.sampler,
            );
            let output = apply_effect_passes(
                self.registry,
                self.device,
                self.queue,
                &mut self.effects.parameters,
                encoder,
                &mut self.textures.pool,
                EffectPassRunConfig {
                    effect_id: command.effect_id,
                    params: parameters,
                    source_bind_group,
                    effect_sampler: self.sampler,
                    composite_bind_group_layout: self.composite_layout,
                    create_composite_bind_group: true,
                    width: key.mask_key.raster_size[0],
                    height: key.mask_key.raster_size[1],
                    texture_format: self.format,
                    label: "shape_effect",
                },
            );
            let (texture, bind_group) = output.into_final_output(&mut self.textures.work_textures);
            self.textures.insert_cached(
                key,
                IntermediateTexture {
                    texture,
                    bind_group,
                },
            )
        };
        self.textures.insert_cached_output(command.output, texture);
    }
}
