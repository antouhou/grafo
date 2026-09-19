use super::effects::{self, EffectPassRunConfig};
use crate::effect::{self, EffectInstance, OffscreenTexturePool, PooledTexture};
use crate::gradient::gpu::GradientCache;
use crate::renderer::types::{BackdropContext, BackdropSource, DrawCommand};
use wgpu::{
    BindGroup, BindGroupLayout, Buffer, Color, CommandEncoder, Device, Extent3d, LoadOp,
    Operations, Origin3d, RenderPassColorAttachment, RenderPassDescriptor, RenderPipeline, Sampler,
    StoreOp, TexelCopyTextureInfo, TextureAspect, TextureView,
};

mod capture;

pub(super) struct BackdropPreparation<'a> {
    pub(super) context: &'a BackdropContext<'a>,
    pub(super) source: BackdropSource<'a>,
    pub(super) composite_bind_group_layout: &'a BindGroupLayout,
    pub(super) scale_factor: f64,
    pub(super) physical_size: (u32, u32),
}

#[derive(Default)]
pub(super) struct BackdropMaterialBindings {
    pub(super) solid: Option<BindGroup>,
    pub(super) gradient: Option<BindGroup>,
}

fn clear_texture_to_transparent(
    encoder: &mut CommandEncoder,
    output_view: &TextureView,
    label: &str,
) {
    encoder.begin_render_pass(&RenderPassDescriptor {
        label: Some(label),
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

#[allow(clippy::too_many_arguments)]
fn blit_texture_to_texture(
    device: &Device,
    encoder: &mut CommandEncoder,
    pipeline: &RenderPipeline,
    bind_group_layout: &BindGroupLayout,
    input_view: &TextureView,
    output_view: &TextureView,
    sampler: &Sampler,
    label: &str,
) {
    let bind_group = effect::create_texture_sample_bind_group(
        device,
        bind_group_layout,
        input_view,
        sampler,
        Some(label),
    );

    let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
        label: Some(label),
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
    render_pass.set_pipeline(pipeline);
    render_pass.set_bind_group(0, &bind_group, &[]);
    render_pass.draw(0..3, 0..1);
}

fn composite_backdrop_foreground_layer(
    device: &Device,
    encoder: &mut CommandEncoder,
    pipeline: &RenderPipeline,
    bind_group_layout: &BindGroupLayout,
    foreground_view: &TextureView,
    output_view: &TextureView,
    params_buffer: &Buffer,
) {
    let bind_group = effect::create_backdrop_layer_composite_bind_group(
        device,
        bind_group_layout,
        foreground_view,
        params_buffer,
    );
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
    render_pass.set_pipeline(pipeline);
    render_pass.set_bind_group(0, &bind_group, &[]);
    render_pass.draw(0..3, 0..1);
}

/// Capture the backdrop and retain every work texture until render submission.
/// Empty bindings preserve the caller's ordinary shape-material fallback.
pub(super) fn prepare_backdrop(
    encoder: &mut CommandEncoder,
    draw_command: &mut DrawCommand,
    effect_instance: &mut EffectInstance,
    texture_pool: &mut OffscreenTexturePool,
    gradient_cache: &mut GradientCache,
    work_textures: &mut Vec<PooledTexture>,
    config: BackdropPreparation<'_>,
) -> BackdropMaterialBindings {
    let context = config.context;
    let backdrop_config = effect_instance.backdrop_config.unwrap_or_default();
    let Some(capture_region) = capture::compute_backdrop_capture_region(
        draw_command,
        backdrop_config,
        config.scale_factor,
        config.physical_size,
        context.max_texture_dimension_2d,
    ) else {
        return BackdropMaterialBindings::default();
    };
    let mut material_bindings = BackdropMaterialBindings::default();

    let backdrop_sampling_uniform = capture_region.sample_uniform();
    let (capture_width, capture_height) = capture_region.capture_size;
    let backdrop_capture_texture = texture_pool.acquire_color_only(
        context.device,
        capture_width,
        capture_height,
        context.config_format,
        1,
    );
    if capture_region.copy_size != capture_region.capture_size {
        clear_texture_to_transparent(
            encoder,
            &backdrop_capture_texture.color_view,
            "backdrop_capture_clear",
        );
    }
    if let Some((copy_source_x, copy_source_y)) = capture_region.copy_source_origin {
        encoder.copy_texture_to_texture(
            TexelCopyTextureInfo {
                texture: config.source.base_texture(),
                mip_level: 0,
                origin: Origin3d {
                    x: copy_source_x,
                    y: copy_source_y,
                    z: 0,
                },
                aspect: TextureAspect::All,
            },
            TexelCopyTextureInfo {
                texture: &backdrop_capture_texture.color_texture,
                mip_level: 0,
                origin: Origin3d {
                    x: capture_region.copy_destination_origin.0,
                    y: capture_region.copy_destination_origin.1,
                    z: 0,
                },
                aspect: TextureAspect::All,
            },
            Extent3d {
                width: capture_region.copy_size.0,
                height: capture_region.copy_size.1,
                depth_or_array_layers: 1,
            },
        );
    }

    if let Some(foreground_view) = config.source.foreground_view() {
        let layer_params =
            effect::backdrop_layer_params(capture_region.capture_origin, config.physical_size);
        let layer_params_buffer = effect::prepare_backdrop_layer_params_buffer(
            context.device,
            context.queue,
            &mut effect_instance.backdrop_layer_params_buffer,
            layer_params,
        );
        composite_backdrop_foreground_layer(
            context.device,
            encoder,
            context.backdrop_layer_composite_pipeline,
            context.backdrop_layer_composite_bind_group_layout,
            foreground_view,
            &backdrop_capture_texture.color_view,
            &layer_params_buffer,
        );
    }

    let effect_input_size = effects::compute_downsampled_dimensions(
        (capture_width, capture_height),
        backdrop_config.downsample,
    );
    let mut downsampled_capture_texture: Option<PooledTexture> = None;

    if effect_input_size != (capture_width, capture_height) {
        let downsampled_capture_target = texture_pool.acquire_color_only(
            context.device,
            effect_input_size.0,
            effect_input_size.1,
            context.config_format,
            1,
        );
        blit_texture_to_texture(
            context.device,
            encoder,
            context.texture_blit_pipeline,
            config.composite_bind_group_layout,
            &backdrop_capture_texture.color_view,
            &downsampled_capture_target.color_view,
            context.effect_sampler,
            "backdrop_capture_downsample",
        );
        downsampled_capture_texture = Some(downsampled_capture_target);
    }

    let loaded_effect = context
        .loaded_effects
        .get(&effect_instance.effect_id)
        .expect("loaded backdrop effect must exist");
    let effect_output = effects::apply_effect_passes(
        context.device,
        encoder,
        texture_pool,
        EffectPassRunConfig {
            loaded_effect,
            params_bind_group: effect_instance
                .parameter_resources
                .as_ref()
                .map(|resources| &resources.bind_group),
            source_view: downsampled_capture_texture
                .as_ref()
                .map(|texture| &texture.color_view)
                .unwrap_or(&backdrop_capture_texture.color_view),
            effect_sampler: context.effect_sampler,
            composite_bind_group_layout: config.composite_bind_group_layout,
            create_composite_bind_group: false,
            width: effect_input_size.0,
            height: effect_input_size.1,
            texture_format: context.config_format,
            label: "backdrop_effect",
        },
    );

    let uses_gradient_backdrop = draw_command.has_gradient_fill();
    if let DrawCommand::CachedShape(cached_shape) = draw_command {
        if uses_gradient_backdrop {
            let gradient_backdrop_material_params_buffer = cached_shape
                .prepare_gradient_backdrop_material_params_buffer(
                    context.device,
                    context.queue,
                    backdrop_sampling_uniform,
                )
                .expect("gradient backdrop shapes must prepare a backdrop material params buffer");
            let backdrop_view = effect_output.final_output_view();
            material_bindings.gradient = cached_shape
                .prepare_backdrop_gradient_bind_group(
                    gradient_cache,
                    context.device,
                    context.queue,
                    context.backdrop_gradient_bind_group_layout,
                    &gradient_backdrop_material_params_buffer,
                    context.gradient_ramp_sampler,
                    effect_output.final_output_texture_id(),
                    backdrop_view,
                    context.effect_sampler,
                )
                .cloned();
        } else {
            let solid_backdrop_material_params_buffer =
                effect::prepare_solid_backdrop_material_params_buffer(
                    context.device,
                    context.queue,
                    &mut effect_instance.backdrop_material_params_buffer,
                    backdrop_sampling_uniform,
                );

            if effect_instance.backdrop_texture_id != Some(effect_output.final_output_texture_id())
            {
                effect_instance.backdrop_texture_bind_group =
                    Some(effect::create_backdrop_texture_sample_bind_group(
                        context.device,
                        context.backdrop_texture_bind_group_layout,
                        &solid_backdrop_material_params_buffer,
                        effect_output.final_output_view(),
                        context.effect_sampler,
                        Some("backdrop_shape_background_bind_group"),
                    ));
                effect_instance.backdrop_texture_id = Some(effect_output.final_output_texture_id());
            }

            material_bindings.solid = effect_instance.backdrop_texture_bind_group.clone();
        }
    }

    work_textures.push(backdrop_capture_texture);
    if let Some(downsampled_capture_texture) = downsampled_capture_texture {
        work_textures.push(downsampled_capture_texture);
    }
    effect_output.push_work_textures_into(work_textures);
    material_bindings
}
