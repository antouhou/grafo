use super::effects::{self, EffectExecutionResources, OffscreenTexturePool, PooledTexture};
use super::instructions::ExecutionResources;
use super::targets::SurfaceTarget;
use super::textures::IntermediateTextureResources;
use crate::commands::{CaptureConfig, CaptureSource, Target};
use crate::core::effect::CaptureRegion;
use crate::core::{PhysicalRect, Size};
use crate::wgpu_backend::types::{BackdropContext, BackdropSource};
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
    region: CaptureRegion,
    resources: &mut EffectExecutionResources,
    texture_pool: &mut OffscreenTexturePool,
) -> PooledTexture {
    let capture_size = region.bounds.size().to_u32();
    let capture_texture = texture_pool.acquire_color_only(
        context.effects.device,
        capture_size.width,
        capture_size.height,
        context.effects.format,
        1,
    );
    let (base_texture, base_bounds) = source.base();
    let copy = region.copy_from(base_bounds);
    if copy.map(|(rect, _)| rect.size()) != Some(capture_size) {
        clear_capture(encoder, &capture_texture.color_view);
    }
    if let Some((source_rect, destination)) = copy {
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
                    x: destination.x,
                    y: destination.y,
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

    if let Some((foreground_view, foreground_bounds)) = source.foreground() {
        let layer_params = effects::backdrop_layer_params(
            (region.bounds.min - foreground_bounds.min).to_tuple(),
            foreground_bounds.size().to_u32().to_tuple(),
        );
        let bind_group = resources.backdrop_composites.prepare(
            context.effects.device,
            context.effects.queue,
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
        context.effects.device,
        output_size.width,
        output_size.height,
        context.effects.format,
        1,
    );
    let binding = input.composite_bind_group(
        context.effects.device,
        context.effects.composite_layout,
        context.effects.sampler,
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

/// Captures and resamples source pixels into the command's logical output.
fn execute_capture(
    encoder: &mut CommandEncoder,
    context: &BackdropContext<'_>,
    source: BackdropSource<'_>,
    command: CaptureConfig,
    resources: &mut EffectExecutionResources,
    textures: &mut IntermediateTextureResources,
) {
    let mut texture = capture_backdrop(
        encoder,
        context,
        source,
        command.region,
        resources,
        &mut textures.pool,
    );
    if command.sampling_size != command.region.bounds.size().to_u32() {
        let downsampled = downsample_capture(
            encoder,
            context,
            &mut texture,
            command.sampling_size,
            &mut textures.pool,
        );
        textures.work_textures.push(texture);
        texture = downsampled;
    }
    texture.composite_bind_group(
        context.effects.device,
        context.effects.composite_layout,
        context.effects.sampler,
    );
    textures.insert_planned(command.output, texture);
}

impl ExecutionResources<'_> {
    pub(super) fn capture(
        &mut self,
        encoder: &mut CommandEncoder,
        capture: CaptureConfig,
        surface: &SurfaceTarget<'_>,
    ) {
        let target = self
            .textures
            .active_targets
            .last_mut()
            .expect("capture requires an open target");
        target.attachments(surface).clear_if_needed(encoder);
        target.needs_clear = false;
        if matches!(capture.source, CaptureSource::Target)
            && matches!(target.target, Target::Texture { region, .. } if region == capture.region)
        {
            let mut output = downsample_capture(
                encoder,
                &self.context.backdrops,
                target
                    .texture
                    .as_mut()
                    .expect("texture target owns its input"),
                capture.sampling_size,
                &mut self.textures.pool,
            );
            let context = &self.context.effects;
            output.composite_bind_group(context.device, context.composite_layout, context.sampler);
            self.textures.insert_planned(capture.output, output);
            return;
        }
        let foreground = target.attachments(surface).resolved_view().clone();
        let target_bounds = match target.target {
            Target::Texture { region, .. } => region.bounds,
            _ => {
                let texture = surface
                    .capture_texture
                    .expect("surface has a capture texture");
                PhysicalRect::from_size((texture.width() as i32, texture.height() as i32).into())
            }
        };
        let base = match capture.source {
            CaptureSource::Target => match &target.texture {
                Some(texture) => texture
                    .resolve_texture
                    .as_ref()
                    .unwrap_or(&texture.color_texture)
                    .clone(),
                None => surface
                    .capture_texture
                    .expect("surface has a capture texture")
                    .clone(),
            },
            CaptureSource::Layered { base, .. } => self.textures.texture(base).clone(),
        };
        let source = match capture.source {
            CaptureSource::Target => BackdropSource::Flattened {
                texture: &base,
                origin: target_bounds.min.to_array(),
            },
            CaptureSource::Layered { origin, .. } => BackdropSource::Layered {
                base_texture: &base,
                base_origin: origin,
                foreground_view: &foreground,
                foreground_bounds: target_bounds,
            },
        };
        execute_capture(
            encoder,
            &self.context.backdrops,
            source,
            capture,
            self.effects,
            self.textures,
        );
    }
}
