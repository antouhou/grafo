use super::types::TraversalEvent;
use super::*;

pub(super) struct AppliedEffectOutput {
    pub(super) composite_bind_group: wgpu::BindGroup,
    pub(super) primary_work_texture: wgpu::Texture,
    pub(super) secondary_work_texture: Option<wgpu::Texture>,
}

impl AppliedEffectOutput {
    pub(super) fn push_work_textures_into(
        self,
        output_textures: &mut Vec<wgpu::Texture>,
    ) -> wgpu::BindGroup {
        output_textures.push(self.primary_work_texture);
        if let Some(secondary_work_texture) = self.secondary_work_texture {
            output_textures.push(secondary_work_texture);
        }
        self.composite_bind_group
    }
}

pub(super) struct EffectPassRunConfig<'a> {
    pub(super) loaded_effect: &'a LoadedEffect,
    pub(super) params_bind_group: Option<&'a wgpu::BindGroup>,
    pub(super) source_view: &'a wgpu::TextureView,
    pub(super) effect_sampler: &'a wgpu::Sampler,
    pub(super) composite_bind_group_layout: &'a wgpu::BindGroupLayout,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) texture_format: wgpu::TextureFormat,
    pub(super) label_prefix: &'a str,
}

pub(super) fn apply_effect_passes(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    config: EffectPassRunConfig<'_>,
) -> AppliedEffectOutput {
    let number_of_passes = config.loaded_effect.passes.len();

    let effect_texture_a = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(&format!("{}_work_a", config.label_prefix)),
        size: wgpu::Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: config.texture_format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let effect_view_a = effect_texture_a.create_view(&wgpu::TextureViewDescriptor::default());

    let effect_texture_b = if number_of_passes > 1 {
        Some(device.create_texture(&wgpu::TextureDescriptor {
            label: Some(&format!("{}_work_b", config.label_prefix)),
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: config.texture_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        }))
    } else {
        None
    };
    let effect_view_b = effect_texture_b
        .as_ref()
        .map(|texture| texture.create_view(&wgpu::TextureViewDescriptor::default()));

    let mut previous_input_view: &wgpu::TextureView = config.source_view;

    for (pass_index, effect_pass) in config.loaded_effect.passes.iter().enumerate() {
        let output_view = if pass_index % 2 == 0 {
            &effect_view_a
        } else {
            effect_view_b.as_ref().unwrap()
        };

        let input_bind_group = effect::create_texture_sample_bind_group(
            device,
            &config.loaded_effect.input_bind_group_layout,
            previous_input_view,
            config.effect_sampler,
            Some(&format!("{}_pass_input_bg", config.label_prefix)),
        );

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(&format!(
                "{}_apply_pass_{}",
                config.label_prefix, pass_index
            )),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: output_view,
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

    let composite_bind_group = effect::create_texture_sample_bind_group(
        device,
        config.composite_bind_group_layout,
        previous_input_view,
        config.effect_sampler,
        Some(&format!("{}_composite_bg", config.label_prefix)),
    );

    AppliedEffectOutput {
        composite_bind_group,
        primary_work_texture: effect_texture_a,
        secondary_work_texture: effect_texture_b,
    }
}

fn bind_shape_texture_layers(
    render_pass: &mut wgpu::RenderPass<'_>,
    texture_ids: [Option<u64>; 2],
    texture_manager: &TextureManager,
    shape_texture_bind_group_layout_background: &wgpu::BindGroupLayout,
    shape_texture_bind_group_layout_foreground: &wgpu::BindGroupLayout,
    default_shape_texture_bind_groups: &[Arc<wgpu::BindGroup>; 2],
    shape_texture_layout_epoch: u64,
) {
    for (layer, &texture_id) in texture_ids.iter().enumerate() {
        if let Some(texture_id) = texture_id {
            if texture_manager.is_texture_loaded(texture_id) {
                if let Ok(bind_group) = texture_manager.get_or_create_shape_bind_group(
                    if layer == 0 {
                        shape_texture_bind_group_layout_background
                    } else {
                        shape_texture_bind_group_layout_foreground
                    },
                    shape_texture_layout_epoch,
                    texture_id,
                ) {
                    render_pass.set_bind_group(1 + layer as u32, &*bind_group, &[]);
                }
            }
        } else {
            render_pass.set_bind_group(
                1 + layer as u32,
                &*default_shape_texture_bind_groups[layer],
                &[],
            );
        }
    }
}

pub(super) fn bind_instance_buffers(
    render_pass: &mut wgpu::RenderPass<'_>,
    shape: &(impl DrawShapeCommand + ?Sized),
    buffers: &crate::renderer::types::Buffers,
) {
    if let Some(instance_idx) = shape.instance_index() {
        if let Some(instance_transform_buffer) = buffers.aggregated_instance_transform_buffer {
            let stride = std::mem::size_of::<InstanceTransform>() as u64;
            let offset = instance_idx as u64 * stride;
            render_pass
                .set_vertex_buffer(1, instance_transform_buffer.slice(offset..offset + stride));
        } else {
            render_pass.set_vertex_buffer(1, buffers.identity_instance_transform_buffer.slice(..));
        }

        if let Some(instance_color_buffer) = buffers.aggregated_instance_color_buffer {
            let stride = std::mem::size_of::<InstanceColor>() as u64;
            let offset = instance_idx as u64 * stride;
            render_pass.set_vertex_buffer(2, instance_color_buffer.slice(offset..offset + stride));
        } else {
            render_pass.set_vertex_buffer(2, buffers.identity_instance_color_buffer.slice(..));
        }

        if let Some(instance_metadata_buffer) = buffers.aggregated_instance_metadata_buffer {
            let stride = std::mem::size_of::<InstanceMetadata>() as u64;
            let offset = instance_idx as u64 * stride;
            render_pass
                .set_vertex_buffer(3, instance_metadata_buffer.slice(offset..offset + stride));
        } else {
            render_pass.set_vertex_buffer(3, buffers.identity_instance_metadata_buffer.slice(..));
        }
    } else {
        render_pass.set_vertex_buffer(1, buffers.identity_instance_transform_buffer.slice(..));
        render_pass.set_vertex_buffer(2, buffers.identity_instance_color_buffer.slice(..));
        render_pass.set_vertex_buffer(3, buffers.identity_instance_metadata_buffer.slice(..));
    }
}

pub(super) fn handle_increment_pass<'rp>(
    render_pass: &mut wgpu::RenderPass<'rp>,
    currently_set_pipeline: &mut crate::renderer::types::Pipeline,
    stencil_stack: &mut Vec<u32>,
    shape: &mut (impl DrawShapeCommand + ?Sized),
    pipelines: &crate::renderer::types::Pipelines,
    buffers: &crate::renderer::types::Buffers,
) {
    if let Some(index_range) = shape.index_buffer_range() {
        if shape.is_empty() {
            return;
        }

        if !matches!(
            currently_set_pipeline,
            crate::renderer::types::Pipeline::StencilIncrement
        ) {
            render_pass.set_pipeline(pipelines.and_pipeline);
            render_pass.set_bind_group(0, pipelines.and_bind_group, &[]);
            render_pass.set_bind_group(1, &*pipelines.default_shape_texture_bind_groups[0], &[]);
            render_pass.set_bind_group(2, &*pipelines.default_shape_texture_bind_groups[1], &[]);

            if !matches!(
                currently_set_pipeline,
                crate::renderer::types::Pipeline::StencilDecrement
            ) {
                render_pass.set_vertex_buffer(0, buffers.aggregated_vertex_buffer.slice(..));
                render_pass.set_index_buffer(
                    buffers.aggregated_index_buffer.slice(..),
                    wgpu::IndexFormat::Uint16,
                );
            }

            *currently_set_pipeline = crate::renderer::types::Pipeline::StencilIncrement;
        }

        bind_shape_texture_layers(
            render_pass,
            [shape.texture_id(0), shape.texture_id(1)],
            pipelines.texture_manager,
            pipelines.shape_texture_bind_group_layout_background,
            pipelines.shape_texture_bind_group_layout_foreground,
            pipelines.default_shape_texture_bind_groups,
            pipelines.shape_texture_layout_epoch,
        );

        bind_instance_buffers(render_pass, shape, buffers);

        let parent_stencil = stencil_stack.last().copied().unwrap_or(0);
        render_buffer_range_to_texture(index_range, render_pass, parent_stencil);

        let this_stencil = parent_stencil + 1;
        *shape.stencil_ref_mut() = Some(this_stencil);
        stencil_stack.push(this_stencil);
    }
}

pub(super) fn handle_decrement_pass<'rp>(
    render_pass: &mut wgpu::RenderPass<'rp>,
    currently_set_pipeline: &mut crate::renderer::types::Pipeline,
    stencil_stack: &mut Vec<u32>,
    shape: &mut (impl DrawShapeCommand + ?Sized),
    pipelines: &crate::renderer::types::Pipelines,
    buffers: &crate::renderer::types::Buffers,
) {
    if let Some(index_range) = shape.index_buffer_range() {
        if shape.is_empty() {
            return;
        }

        if !matches!(
            currently_set_pipeline,
            crate::renderer::types::Pipeline::StencilDecrement
        ) {
            render_pass.set_pipeline(pipelines.decrementing_pipeline);
            render_pass.set_bind_group(0, pipelines.decrementing_bind_group, &[]);
            render_pass.set_bind_group(1, &*pipelines.default_shape_texture_bind_groups[0], &[]);
            render_pass.set_bind_group(2, &*pipelines.default_shape_texture_bind_groups[1], &[]);

            if !matches!(
                currently_set_pipeline,
                crate::renderer::types::Pipeline::StencilIncrement
            ) {
                render_pass.set_vertex_buffer(0, buffers.aggregated_vertex_buffer.slice(..));
                render_pass.set_index_buffer(
                    buffers.aggregated_index_buffer.slice(..),
                    wgpu::IndexFormat::Uint16,
                );
            }

            *currently_set_pipeline = crate::renderer::types::Pipeline::StencilDecrement;
        }

        bind_instance_buffers(render_pass, shape, buffers);

        let this_shape_stencil = shape.stencil_ref_mut().unwrap_or(0);
        render_buffer_range_to_texture(index_range, render_pass, this_shape_stencil);

        if shape.stencil_ref_mut().is_some() {
            stencil_stack.pop();
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn render_segments(
    draw_tree: &mut easy_tree::Tree<DrawCommand>,
    encoder: &mut wgpu::CommandEncoder,
    events: &[TraversalEvent],
    stencil_refs: &HashMap<usize, u32>,
    parent_stencils: &HashMap<usize, u32>,
    effect_results: &HashMap<usize, wgpu::BindGroup>,
    color_view: &wgpu::TextureView,
    color_resolve_target: Option<&wgpu::TextureView>,
    depth_stencil_view: &wgpu::TextureView,
    copy_source_texture: &wgpu::Texture,
    clear_first: bool,
    pipelines: &crate::renderer::types::Pipelines,
    buffers: &crate::renderer::types::Buffers,
    backdrop_ctx: &crate::renderer::types::BackdropContext,
    backdrop_work_textures: &mut Vec<wgpu::Texture>,
    stencil_stack_scratch: &mut Vec<u32>,
) {
    let mut event_idx = 0;
    let mut is_first_segment = clear_first;
    let mut currently_set_pipeline = crate::renderer::types::Pipeline::None;
    let (width, height) = backdrop_ctx.physical_size;
    stencil_stack_scratch.clear();
    backdrop_work_textures.clear();

    while event_idx < events.len() {
        let mut segment_end = events.len();
        let mut backdrop_node_id: Option<usize> = None;
        for (idx, event) in events.iter().enumerate().skip(event_idx) {
            if let TraversalEvent::Pre(node_id) = event {
                if backdrop_ctx.backdrop_effects.contains_key(node_id) {
                    segment_end = idx;
                    backdrop_node_id = Some(*node_id);
                    break;
                }
            }
        }

        if event_idx < segment_end {
            let mut render_pass = crate::pipeline::begin_render_pass_with_load_ops(
                encoder,
                Some(if is_first_segment {
                    "segment_clear_pass"
                } else {
                    "segment_load_pass"
                }),
                color_view,
                color_resolve_target,
                depth_stencil_view,
                crate::pipeline::RenderPassLoadOperations {
                    color_load_op: if is_first_segment {
                        wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT)
                    } else {
                        wgpu::LoadOp::Load
                    },
                    depth_load_op: if is_first_segment {
                        wgpu::LoadOp::Clear(1.0)
                    } else {
                        wgpu::LoadOp::Load
                    },
                    stencil_load_op: if is_first_segment {
                        wgpu::LoadOp::Clear(0)
                    } else {
                        wgpu::LoadOp::Load
                    },
                },
            );

            for event in events.iter().take(segment_end).skip(event_idx) {
                match event {
                    TraversalEvent::Pre(node_id) => {
                        let node_id = *node_id;

                        if let Some(result_bind_group) = effect_results.get(&node_id) {
                            let parent_stencil =
                                parent_stencils.get(&node_id).copied().unwrap_or(0);
                            render_pass.set_pipeline(backdrop_ctx.composite_pipeline);
                            render_pass.set_bind_group(0, result_bind_group, &[]);
                            render_pass.set_stencil_reference(parent_stencil);
                            render_pass.draw(0..3, 0..1);
                            currently_set_pipeline = crate::renderer::types::Pipeline::None;
                            continue;
                        }

                        if let Some(draw_command) = draw_tree.get_mut(node_id) {
                            let parent_stencil =
                                parent_stencils.get(&node_id).copied().unwrap_or(0);
                            let this_stencil = stencil_refs.get(&node_id).copied().unwrap_or(1);

                            stencil_stack_scratch.clear();
                            stencil_stack_scratch.push(parent_stencil);

                            match draw_command {
                                DrawCommand::Shape(shape) => {
                                    handle_increment_pass(
                                        &mut render_pass,
                                        &mut currently_set_pipeline,
                                        stencil_stack_scratch,
                                        shape,
                                        pipelines,
                                        buffers,
                                    );
                                    *shape.stencil_ref_mut() = Some(this_stencil);
                                }
                                DrawCommand::CachedShape(shape) => {
                                    handle_increment_pass(
                                        &mut render_pass,
                                        &mut currently_set_pipeline,
                                        stencil_stack_scratch,
                                        shape,
                                        pipelines,
                                        buffers,
                                    );
                                    *shape.stencil_ref_mut() = Some(this_stencil);
                                }
                            }
                        }
                    }
                    TraversalEvent::Post(node_id) => {
                        let node_id = *node_id;

                        if effect_results.contains_key(&node_id) {
                            continue;
                        }

                        if let Some(draw_command) = draw_tree.get_mut(node_id) {
                            let this_stencil = stencil_refs.get(&node_id).copied().unwrap_or(1);

                            stencil_stack_scratch.clear();
                            stencil_stack_scratch.push(this_stencil);

                            match draw_command {
                                DrawCommand::Shape(shape) => {
                                    *shape.stencil_ref_mut() = Some(this_stencil);
                                    handle_decrement_pass(
                                        &mut render_pass,
                                        &mut currently_set_pipeline,
                                        stencil_stack_scratch,
                                        shape,
                                        pipelines,
                                        buffers,
                                    );
                                }
                                DrawCommand::CachedShape(shape) => {
                                    *shape.stencil_ref_mut() = Some(this_stencil);
                                    handle_decrement_pass(
                                        &mut render_pass,
                                        &mut currently_set_pipeline,
                                        stencil_stack_scratch,
                                        shape,
                                        pipelines,
                                        buffers,
                                    );
                                }
                            }
                        }
                    }
                }
            }

            is_first_segment = false;
        }

        event_idx = segment_end;

        if let Some(backdrop_node_id) = backdrop_node_id {
            if is_first_segment {
                crate::pipeline::begin_render_pass_with_load_ops(
                    encoder,
                    Some("segment_initial_clear"),
                    color_view,
                    color_resolve_target,
                    depth_stencil_view,
                    crate::pipeline::RenderPassLoadOperations {
                        color_load_op: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        depth_load_op: wgpu::LoadOp::Clear(1.0),
                        stencil_load_op: wgpu::LoadOp::Clear(0),
                    },
                );
            }

            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: copy_source_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: backdrop_ctx.backdrop_snapshot_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );

            let effect_instance = backdrop_ctx
                .backdrop_effects
                .get(&backdrop_node_id)
                .unwrap();
            let loaded_effect = backdrop_ctx
                .loaded_effects
                .get(&effect_instance.effect_id)
                .unwrap();

            let effect_output = apply_effect_passes(
                backdrop_ctx.device,
                encoder,
                EffectPassRunConfig {
                    loaded_effect,
                    params_bind_group: effect_instance.params_bind_group.as_ref(),
                    source_view: backdrop_ctx.backdrop_snapshot_view,
                    effect_sampler: backdrop_ctx.effect_sampler,
                    composite_bind_group_layout: backdrop_ctx.composite_bgl,
                    width,
                    height,
                    texture_format: backdrop_ctx.config_format,
                    label_prefix: "backdrop_effect",
                },
            );
            let backdrop_composite_bind_group =
                effect_output.push_work_textures_into(backdrop_work_textures);

            let mut render_pass = crate::pipeline::begin_render_pass_with_load_ops(
                encoder,
                Some("backdrop_shape_pass"),
                color_view,
                color_resolve_target,
                depth_stencil_view,
                crate::pipeline::RenderPassLoadOperations {
                    color_load_op: wgpu::LoadOp::Load,
                    depth_load_op: wgpu::LoadOp::Load,
                    stencil_load_op: wgpu::LoadOp::Load,
                },
            );

            let parent_stencil = parent_stencils.get(&backdrop_node_id).copied().unwrap_or(0);
            let this_stencil = stencil_refs.get(&backdrop_node_id).copied().unwrap_or(1);

            if let Some(draw_command) = draw_tree.get_mut(backdrop_node_id) {
                render_pass.set_pipeline(backdrop_ctx.stencil_only_pipeline);
                render_pass.set_bind_group(0, backdrop_ctx.and_bind_group, &[]);
                render_pass.set_bind_group(
                    1,
                    &*backdrop_ctx.default_shape_texture_bind_groups[0],
                    &[],
                );
                render_pass.set_bind_group(
                    2,
                    &*backdrop_ctx.default_shape_texture_bind_groups[1],
                    &[],
                );
                render_pass.set_vertex_buffer(0, buffers.aggregated_vertex_buffer.slice(..));
                render_pass.set_index_buffer(
                    buffers.aggregated_index_buffer.slice(..),
                    wgpu::IndexFormat::Uint16,
                );

                let shape_index_range = match draw_command {
                    DrawCommand::Shape(shape) => {
                        bind_instance_buffers(&mut render_pass, shape, buffers);
                        shape.index_buffer_range()
                    }
                    DrawCommand::CachedShape(shape) => {
                        bind_instance_buffers(&mut render_pass, shape, buffers);
                        shape.index_buffer_range()
                    }
                };

                if let Some(shape_index_range) = shape_index_range {
                    render_pass.set_stencil_reference(parent_stencil);
                    let start = shape_index_range.0 as u32;
                    let end = (shape_index_range.0 + shape_index_range.1) as u32;
                    render_pass.draw_indexed(start..end, 0, 0..1);
                }

                match draw_command {
                    DrawCommand::Shape(shape) => *shape.stencil_ref_mut() = Some(this_stencil),
                    DrawCommand::CachedShape(shape) => {
                        *shape.stencil_ref_mut() = Some(this_stencil)
                    }
                }
            }

            render_pass.set_pipeline(backdrop_ctx.composite_pipeline);
            render_pass.set_bind_group(0, &backdrop_composite_bind_group, &[]);
            render_pass.set_stencil_reference(this_stencil);
            render_pass.draw(0..3, 0..1);

            if let Some(draw_command) = draw_tree.get_mut(backdrop_node_id) {
                render_pass.set_pipeline(backdrop_ctx.backdrop_color_pipeline);
                render_pass.set_bind_group(0, backdrop_ctx.and_bind_group, &[]);
                render_pass.set_bind_group(
                    1,
                    &*backdrop_ctx.default_shape_texture_bind_groups[0],
                    &[],
                );
                render_pass.set_bind_group(
                    2,
                    &*backdrop_ctx.default_shape_texture_bind_groups[1],
                    &[],
                );
                render_pass.set_vertex_buffer(0, buffers.aggregated_vertex_buffer.slice(..));
                render_pass.set_index_buffer(
                    buffers.aggregated_index_buffer.slice(..),
                    wgpu::IndexFormat::Uint16,
                );

                let (texture_ids, shape_index_range) = match draw_command {
                    DrawCommand::Shape(shape) => {
                        bind_instance_buffers(&mut render_pass, shape, buffers);
                        (
                            [shape.texture_id(0), shape.texture_id(1)],
                            shape.index_buffer_range(),
                        )
                    }
                    DrawCommand::CachedShape(shape) => {
                        bind_instance_buffers(&mut render_pass, shape, buffers);
                        (
                            [shape.texture_id(0), shape.texture_id(1)],
                            shape.index_buffer_range(),
                        )
                    }
                };

                bind_shape_texture_layers(
                    &mut render_pass,
                    texture_ids,
                    backdrop_ctx.texture_manager,
                    backdrop_ctx.shape_texture_bind_group_layout_background,
                    backdrop_ctx.shape_texture_bind_group_layout_foreground,
                    backdrop_ctx.default_shape_texture_bind_groups,
                    backdrop_ctx.shape_texture_layout_epoch,
                );

                if let Some(shape_index_range) = shape_index_range {
                    render_pass.set_stencil_reference(this_stencil);
                    let start = shape_index_range.0 as u32;
                    let end = (shape_index_range.0 + shape_index_range.1) as u32;
                    render_pass.draw_indexed(start..end, 0, 0..1);
                }
            }

            currently_set_pipeline = crate::renderer::types::Pipeline::None;
            is_first_segment = false;
            event_idx += 1;
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn render_scene_behind_group(
    draw_tree: &mut easy_tree::Tree<DrawCommand>,
    encoder: &mut wgpu::CommandEncoder,
    effect_results: &HashMap<usize, wgpu::BindGroup>,
    exclude_subtree_id: usize,
    color_view: &wgpu::TextureView,
    color_resolve_target: Option<&wgpu::TextureView>,
    depth_stencil_view: &wgpu::TextureView,
    composite_pipeline: Option<&wgpu::RenderPipeline>,
    pipelines: &crate::renderer::types::Pipelines,
    buffers: &crate::renderer::types::Buffers,
    stencil_stack_scratch: &mut Vec<u32>,
    skipped_stack_scratch: &mut Vec<usize>,
) {
    let render_pass = crate::pipeline::begin_render_pass_with_load_ops(
        encoder,
        Some("behind_group_pass"),
        color_view,
        color_resolve_target,
        depth_stencil_view,
        crate::pipeline::RenderPassLoadOperations {
            color_load_op: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
            depth_load_op: wgpu::LoadOp::Clear(1.0),
            stencil_load_op: wgpu::LoadOp::Clear(0),
        },
    );

    stencil_stack_scratch.clear();
    let current_pipeline = crate::renderer::types::Pipeline::None;
    skipped_stack_scratch.clear();

    let mut data = (
        render_pass,
        stencil_stack_scratch,
        current_pipeline,
        skipped_stack_scratch,
    );

    let effect_results_ref = effect_results;
    let exclude_id = exclude_subtree_id;

    draw_tree.traverse_mut(
        |shape_id, draw_command, data| {
            let (render_pass, stencil_stack, currently_set_pipeline, skipped_stack) = data;

            if shape_id == exclude_id {
                skipped_stack.push(shape_id);
                return;
            }

            if let Some(result_bind_group) = effect_results_ref.get(&shape_id) {
                let parent_stencil = stencil_stack.last().copied().unwrap_or(0);
                if let Some(composite_pipeline) = composite_pipeline {
                    render_pass.set_pipeline(composite_pipeline);
                    render_pass.set_bind_group(0, result_bind_group, &[]);
                    render_pass.set_stencil_reference(parent_stencil);
                    render_pass.draw(0..3, 0..1);
                    *currently_set_pipeline = crate::renderer::types::Pipeline::None;
                }
                skipped_stack.push(shape_id);
                return;
            }

            if !skipped_stack.is_empty() {
                return;
            }

            match draw_command {
                DrawCommand::Shape(shape) => {
                    handle_increment_pass(
                        render_pass,
                        currently_set_pipeline,
                        stencil_stack,
                        shape,
                        pipelines,
                        buffers,
                    );
                }
                DrawCommand::CachedShape(shape) => {
                    handle_increment_pass(
                        render_pass,
                        currently_set_pipeline,
                        stencil_stack,
                        shape,
                        pipelines,
                        buffers,
                    );
                }
            }
        },
        |shape_id, draw_command, data| {
            let (render_pass, stencil_stack, currently_set_pipeline, skipped_stack) = data;

            if skipped_stack.last().copied() == Some(shape_id) {
                skipped_stack.pop();
                return;
            }

            if !skipped_stack.is_empty() {
                return;
            }

            match draw_command {
                DrawCommand::Shape(shape) => {
                    handle_decrement_pass(
                        render_pass,
                        currently_set_pipeline,
                        stencil_stack,
                        shape,
                        pipelines,
                        buffers,
                    );
                }
                DrawCommand::CachedShape(shape) => {
                    handle_decrement_pass(
                        render_pass,
                        currently_set_pipeline,
                        stencil_stack,
                        shape,
                        pipelines,
                        buffers,
                    );
                }
            }
        },
        &mut data,
    );
}
