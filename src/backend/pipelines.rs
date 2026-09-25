use super::execution::effects;
use super::resources::{BackdropPipelineResources, ShapePipelines};
use super::{WgpuBackend, WgpuContext};
use crate::backend::pipeline::{
    create_gradient_bind_group_layout, create_gradient_increment_pipeline,
    create_gradient_stencil_keep_color_pipeline, create_pipeline,
    create_stencil_keep_color_pipeline, create_stencil_only_pipeline, PipelineType,
};
use crate::core::util::to_logical;
use std::sync::Arc;
use wgpu::{BindGroupLayout, Device, SurfaceConfiguration, TextureFormat};

fn create_transparent_texture_view_and_sampler(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &'static str,
) -> (wgpu::TextureView, wgpu::Sampler) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let transparent: [u8; 4] = [0, 0, 0, 0];
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &transparent,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4),
            rows_per_image: Some(1),
        },
        wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
    );

    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });
    (view, sampler)
}

impl ShapePipelines {
    pub(in crate::backend) fn new(
        context: &WgpuContext,
        config: &SurfaceConfiguration,
        physical_size: (u32, u32),
        scale_factor: f64,
        fringe_width: f32,
        msaa_sample_count: u32,
        gradient_layout: Option<BindGroupLayout>,
    ) -> Self {
        let device = &context.device;
        let queue = &context.queue;
        let canvas_logical_size = to_logical(physical_size, scale_factor);

        let (
            and_uniforms,
            and_uniform_buffer,
            and_bind_group,
            background_texture_layout,
            foreground_texture_layout,
            and_pipeline,
        ) = create_pipeline(
            canvas_logical_size,
            scale_factor,
            fringe_width,
            device,
            config,
            PipelineType::EqualIncrementStencil,
            msaa_sample_count,
        );

        let (
            decrementing_uniforms,
            decrementing_uniform_buffer,
            decrementing_bind_group,
            _,
            _,
            decrementing_pipeline,
        ) = create_pipeline(
            canvas_logical_size,
            scale_factor,
            fringe_width,
            device,
            config,
            PipelineType::EqualDecrementStencil,
            msaa_sample_count,
        );

        let gradient_bind_group_layout =
            gradient_layout.unwrap_or_else(|| create_gradient_bind_group_layout(device));
        let and_gradient_pipeline = create_gradient_increment_pipeline(
            device,
            config.format,
            msaa_sample_count,
            &and_pipeline.get_bind_group_layout(0),
            &background_texture_layout,
            &foreground_texture_layout,
            &gradient_bind_group_layout,
        );

        let leaf_draw_pipeline = create_stencil_keep_color_pipeline(
            device,
            config.format,
            msaa_sample_count,
            &and_pipeline.get_bind_group_layout(0),
            &background_texture_layout,
            &foreground_texture_layout,
        );
        let leaf_draw_gradient_pipeline = create_gradient_stencil_keep_color_pipeline(
            device,
            config.format,
            msaa_sample_count,
            &and_pipeline.get_bind_group_layout(0),
            &background_texture_layout,
            &foreground_texture_layout,
            &gradient_bind_group_layout,
        );

        let linear_clamp_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("linear_clamp_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let default_background_texture_bind_group =
            WgpuBackend::create_default_shape_texture_bind_group(
                device,
                queue,
                &background_texture_layout,
            );
        let default_foreground_texture_bind_group =
            WgpuBackend::create_default_shape_texture_bind_group(
                device,
                queue,
                &foreground_texture_layout,
            );
        let stencil_only_pipeline = create_stencil_only_pipeline(
            device,
            config.format,
            msaa_sample_count,
            &and_pipeline.get_bind_group_layout(0),
            &background_texture_layout,
            &foreground_texture_layout,
        );

        Self {
            and_pipeline: Arc::new(and_pipeline),
            and_gradient_pipeline: Arc::new(and_gradient_pipeline),
            and_bind_group,
            decrementing_pipeline: Arc::new(decrementing_pipeline),
            decrementing_bind_group,
            leaf_draw_pipeline: Arc::new(leaf_draw_pipeline),
            leaf_draw_gradient_pipeline: Arc::new(leaf_draw_gradient_pipeline),
            shape_texture_bind_group_layout_background: Arc::new(background_texture_layout),
            shape_texture_bind_group_layout_foreground: Arc::new(foreground_texture_layout),
            default_shape_texture_bind_groups: [
                Arc::new(default_background_texture_bind_group),
                Arc::new(default_foreground_texture_bind_group),
            ],
            texture_manager: context.texture_manager.clone(),
            and_uniforms,
            and_uniform_buffer,
            decrementing_uniforms,
            decrementing_uniform_buffer,
            under_fill_pipelines: None,
            stencil_only_pipeline,
            gradient_bind_group_layout,
            linear_clamp_sampler,
        }
    }
}

impl BackdropPipelineResources {
    pub(in crate::backend) fn new(
        device: &Device,
        format: TextureFormat,
        composite_layout: &BindGroupLayout,
    ) -> Self {
        Self {
            texture_blit_pipeline: effects::compile_texture_blit_pipeline(
                device,
                format,
                composite_layout,
            ),
            layer_composite_resources: effects::compile_backdrop_layer_composite_pipeline(
                device, format,
            ),
        }
    }
}

impl WgpuBackend {
    fn create_default_shape_texture_bind_group(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        shape_texture_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> wgpu::BindGroup {
        let (view, sampler) = create_transparent_texture_view_and_sampler(
            device,
            queue,
            "default_transparent_texture",
        );

        device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: shape_texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
            label: Some("default_shape_texture_bind_group_transparent"),
        })
    }
}
