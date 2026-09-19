use crate::effect::{self, LoadedEffect, OffscreenTexturePool, PooledTexture};
use wgpu::{
    BindGroup, BindGroupLayout, Color, CommandEncoder, Device, LoadOp, Operations,
    RenderPassColorAttachment, RenderPassDescriptor, Sampler, StoreOp, TextureFormat, TextureView,
};

pub(in crate::renderer) struct AppliedEffectOutput {
    pub(in crate::renderer) composite_bind_group: Option<BindGroup>,
    pub(in crate::renderer) final_output_texture: PooledTexture,
    pub(in crate::renderer) recyclable_texture: Option<PooledTexture>,
}

impl AppliedEffectOutput {
    pub(in crate::renderer) fn into_final_and_recyclable(
        self,
        recyclable: &mut Vec<PooledTexture>,
    ) -> (PooledTexture, Option<BindGroup>) {
        if let Some(recyclable_texture) = self.recyclable_texture {
            recyclable.push(recyclable_texture);
        }
        (self.final_output_texture, self.composite_bind_group)
    }

    pub(in crate::renderer) fn push_work_textures_into(
        self,
        output_textures: &mut Vec<PooledTexture>,
    ) -> Option<BindGroup> {
        output_textures.push(self.final_output_texture);
        if let Some(recyclable_texture) = self.recyclable_texture {
            output_textures.push(recyclable_texture);
        }
        self.composite_bind_group
    }

    pub(in crate::renderer) fn final_output_view(&self) -> &TextureView {
        &self.final_output_texture.color_view
    }

    pub(in crate::renderer) fn final_output_texture_id(&self) -> u64 {
        self.final_output_texture.texture_id
    }
}

pub(in crate::renderer) struct EffectPassRunConfig<'a> {
    pub(in crate::renderer) loaded_effect: &'a LoadedEffect,
    pub(in crate::renderer) params_bind_group: Option<&'a BindGroup>,
    pub(in crate::renderer) source_view: &'a TextureView,
    pub(in crate::renderer) effect_sampler: &'a Sampler,
    pub(in crate::renderer) composite_bind_group_layout: &'a BindGroupLayout,
    pub(in crate::renderer) create_composite_bind_group: bool,
    pub(in crate::renderer) width: u32,
    pub(in crate::renderer) height: u32,
    pub(in crate::renderer) texture_format: TextureFormat,
    pub(in crate::renderer) label: &'static str,
}

pub(in crate::renderer) fn apply_effect_passes(
    device: &Device,
    encoder: &mut CommandEncoder,
    texture_pool: &mut OffscreenTexturePool,
    config: EffectPassRunConfig<'_>,
) -> AppliedEffectOutput {
    let number_of_passes = config.loaded_effect.passes.len();

    let effect_texture_a = texture_pool.acquire_color_only(
        device,
        config.width,
        config.height,
        config.texture_format,
        1,
    );

    let effect_texture_b = if number_of_passes > 1 {
        Some(texture_pool.acquire_color_only(
            device,
            config.width,
            config.height,
            config.texture_format,
            1,
        ))
    } else {
        None
    };

    let mut previous_input_view: &TextureView = config.source_view;

    for (pass_index, effect_pass) in config.loaded_effect.passes.iter().enumerate() {
        let output_view = if pass_index % 2 == 0 {
            &effect_texture_a.color_view
        } else {
            &effect_texture_b.as_ref().unwrap().color_view
        };

        let input_bind_group = effect::create_texture_sample_bind_group(
            device,
            &config.loaded_effect.input_bind_group_layout,
            previous_input_view,
            config.effect_sampler,
            Some(config.label),
        );

        let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some(config.label),
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

        pass.set_pipeline(&effect_pass.pipeline);
        pass.set_bind_group(0, &input_bind_group, &[]);

        if effect_pass.has_params {
            if let Some(params_bind_group) = config.params_bind_group {
                pass.set_bind_group(1, params_bind_group, &[]);
            }
        }

        pass.draw(0..3, 0..1);
        previous_input_view = output_view;
    }

    let composite_bind_group = config.create_composite_bind_group.then(|| {
        effect::create_texture_sample_bind_group(
            device,
            config.composite_bind_group_layout,
            previous_input_view,
            config.effect_sampler,
            Some(config.label),
        )
    });

    let (final_output_texture, recyclable_texture) = if number_of_passes % 2 == 1 {
        (effect_texture_a, effect_texture_b)
    } else {
        (
            effect_texture_b.expect("an even number of effect passes needs a second texture"),
            Some(effect_texture_a),
        )
    };

    AppliedEffectOutput {
        composite_bind_group,
        final_output_texture,
        recyclable_texture,
    }
}

pub(in crate::renderer) fn compute_downsampled_dimensions(
    capture_size: (u32, u32),
    downsample: f32,
) -> (u32, u32) {
    (
        ((capture_size.0 as f32) * downsample).ceil().max(1.0) as u32,
        ((capture_size.1 as f32) * downsample).ceil().max(1.0) as u32,
    )
}
