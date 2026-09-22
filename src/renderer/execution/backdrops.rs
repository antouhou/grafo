use super::effects::{
    self, EffectExecutionResources, EffectPassRunConfig, OffscreenTexturePool, PooledTexture,
};
use super::textures::{IntermediateTexture, IntermediateTextureResources};
use crate::effect::BackdropEffectInstance;
use crate::renderer::plan::backdrops::BackdropCaptureRegion;
use crate::renderer::rect_utils;
use crate::renderer::types::{BackdropContext, BackdropSource};
use crate::shape::{ShapeTextureBinding, ShapeTextureLayer, TextureSampling};
use crate::Size;
use wgpu::{
    BindGroup, Color, CommandEncoder, Extent3d, LoadOp, Operations, Origin3d,
    RenderPassColorAttachment, RenderPassDescriptor, StoreOp, TexelCopyTextureInfo, TextureAspect,
    TextureView,
};

fn clear_capture(encoder: &mut CommandEncoder, output_view: &TextureView) {
    encoder.begin_render_pass(&RenderPassDescriptor {
        label: Some("backdrop_capture_clear"),
        color_attachments: &[Some(RenderPassColorAttachment {
            view: output_view,
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
}

fn composite_foreground_layer(
    encoder: &mut CommandEncoder,
    context: &BackdropContext<'_>,
    output_view: &TextureView,
    bind_group: &BindGroup,
) {
    let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
        label: Some("backdrop_layer_composite_pass"),
        color_attachments: &[Some(RenderPassColorAttachment {
            view: output_view,
            resolve_target: None,
            ops: Operations {
                load: LoadOp::Load,
                store: StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
    });
    render_pass.set_pipeline(context.backdrop_layer_composite_pipeline);
    render_pass.set_bind_group(0, bind_group, &[]);
    render_pass.draw(0..3, 0..1);
}

/// Copies resolved source pixels and leaves offscreen padding transparent.
fn capture_backdrop(
    encoder: &mut CommandEncoder,
    context: &BackdropContext<'_>,
    source: BackdropSource<'_>,
    region: BackdropCaptureRegion,
    resources: &mut EffectExecutionResources,
    texture_pool: &mut OffscreenTexturePool,
) -> PooledTexture {
    let capture_size = region.bounds.size().to_u32();
    let capture_texture = texture_pool.acquire_color_only(
        context.device,
        capture_size.width,
        capture_size.height,
        context.config_format,
        1,
    );
    if region.source_rect.map(|rect| rect.size()) != Some(capture_size) {
        clear_capture(encoder, &capture_texture.color_view);
    }
    let base_texture = source.base_texture();
    if let Some(source_rect) = region.source_rect {
        encoder.copy_texture_to_texture(
            TexelCopyTextureInfo {
                texture: base_texture,
                mip_level: 0,
                origin: Origin3d {
                    x: source_rect.min.x,
                    y: source_rect.min.y,
                    z: 0,
                },
                aspect: TextureAspect::All,
            },
            TexelCopyTextureInfo {
                texture: &capture_texture.color_texture,
                mip_level: 0,
                origin: Origin3d {
                    x: region.copy_destination_origin.x,
                    y: region.copy_destination_origin.y,
                    z: 0,
                },
                aspect: TextureAspect::All,
            },
            Extent3d {
                width: source_rect.width(),
                height: source_rect.height(),
                depth_or_array_layers: 1,
            },
        );
    }

    if let Some(foreground_view) = source.foreground_view() {
        let layer_params = effects::backdrop_layer_params(
            region.bounds.min.to_tuple(),
            (base_texture.width(), base_texture.height()),
        );
        let bind_group = resources.backdrop_composites.prepare(
            context.device,
            context.queue,
            context.backdrop_layer_composite_bind_group_layout,
            foreground_view,
            layer_params,
        );
        composite_foreground_layer(encoder, context, &capture_texture.color_view, bind_group);
    }
    capture_texture
}

fn downsample_capture(
    encoder: &mut CommandEncoder,
    context: &BackdropContext<'_>,
    input: &mut PooledTexture,
    output_size: Size,
    texture_pool: &mut OffscreenTexturePool,
) -> PooledTexture {
    let output_texture = texture_pool.acquire_color_only(
        context.device,
        output_size.width,
        output_size.height,
        context.config_format,
        1,
    );
    let binding = input.composite_bind_group(
        context.device,
        context.composite_bind_group_layout,
        context.effect_sampler,
    );
    let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
        label: Some("backdrop_capture_downsample"),
        color_attachments: &[Some(RenderPassColorAttachment {
            view: &output_texture.color_view,
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
    render_pass.set_pipeline(context.texture_blit_pipeline);
    render_pass.set_bind_group(0, binding, &[]);
    render_pass.draw(0..3, 0..1);
    drop(render_pass);
    output_texture
}

/// Executes an accepted capture without changing the traversal's target or clip state.
pub(in crate::renderer) fn apply_backdrop_effect(
    encoder: &mut CommandEncoder,
    context: &BackdropContext<'_>,
    source: BackdropSource<'_>,
    region: BackdropCaptureRegion,
    effect: &BackdropEffectInstance,
    resources: &mut EffectExecutionResources,
    textures: &mut IntermediateTextureResources,
) -> ShapeTextureLayer {
    let mut capture_texture = capture_backdrop(
        encoder,
        context,
        source,
        region,
        resources,
        &mut textures.pool,
    );
    let capture_size = region.bounds.size().to_u32();
    let effect_input_size =
        rect_utils::compute_downsampled_dimensions(capture_size, effect.config.downsample);
    let mut downsampled_texture = if effect_input_size != capture_size {
        Some(downsample_capture(
            encoder,
            context,
            &mut capture_texture,
            effect_input_size,
            &mut textures.pool,
        ))
    } else {
        None
    };
    let source_bind_group = downsampled_texture
        .as_mut()
        .unwrap_or(&mut capture_texture)
        .input_bind_group(
            context.device,
            context.effect_registry.input_bind_group_layout(),
            context.effect_sampler,
        );
    let effect_output = effects::apply_effect_passes(
        context.effect_registry,
        context.device,
        context.queue,
        &mut resources.parameters,
        encoder,
        &mut textures.pool,
        EffectPassRunConfig {
            effect_id: effect.effect.effect_id,
            params: &effect.effect.params,
            source_bind_group,
            effect_sampler: context.effect_sampler,
            composite_bind_group_layout: context.composite_bind_group_layout,
            create_composite_bind_group: false,
            width: effect_input_size.width,
            height: effect_input_size.height,
            texture_format: context.config_format,
            label: "backdrop_effect",
        },
    );

    textures.work_textures.push(capture_texture);
    if let Some(downsampled_texture) = downsampled_texture {
        textures.work_textures.push(downsampled_texture);
    }
    let (texture, bind_group) = effect_output.into_final_output(&mut textures.work_textures);
    let texture_id = textures.insert_transient(IntermediateTexture {
        texture,
        bind_group,
    });
    ShapeTextureLayer {
        texture: ShapeTextureBinding::Intermediate(texture_id),
        sampling: TextureSampling::TargetPixels(region.bounds),
    }
}
