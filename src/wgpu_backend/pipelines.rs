use super::execution::effects;
use super::execution::shape_mask_effects::ShapeMaskEffectRendererResources;
use super::execution::shapes::coverage;
use super::execution::shapes::{ShapePipelineLayouts, ShapePipelineVariants};
use super::resources::{BackdropPipelineResources, RendererPipelineResources, ShapePipelines};
use super::{WgpuBackend, WgpuContext};
use crate::core::util::to_logical;
use crate::wgpu_backend::pipeline::Uniforms;
use std::sync::Arc;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    AddressMode, BindGroupLayout, BufferUsages, Device, FilterMode, SamplerDescriptor,
    TextureFormat,
};

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
    pub(in crate::wgpu_backend) fn new(
        context: &WgpuContext,
        format: TextureFormat,
        physical_size: (u32, u32),
        scale_factor: f64,
        fringe_width: f32,
        msaa_sample_count: u32,
        gradient_layout: Option<BindGroupLayout>,
    ) -> Self {
        let device = &context.device;
        let queue = &context.queue;
        let canvas_logical_size = to_logical(physical_size, scale_factor);

        let layouts = ShapePipelineLayouts::new(device, gradient_layout);
        let and_uniforms = Uniforms::new(
            canvas_logical_size.0,
            canvas_logical_size.1,
            scale_factor as f32,
            fringe_width,
        );
        let decrementing_uniforms = and_uniforms;
        let create_uniform_buffer = |uniforms: &Uniforms| {
            device.create_buffer_init(&BufferInitDescriptor {
                label: Some("shape_uniforms"),
                contents: bytemuck::bytes_of(uniforms),
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            })
        };
        let and_uniform_buffer = create_uniform_buffer(&and_uniforms);
        let decrementing_uniform_buffer = create_uniform_buffer(&decrementing_uniforms);
        let empty_coverage = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("empty_coverage"),
            contents: &[0; 64],
            usage: BufferUsages::STORAGE,
        });
        let and_bind_group = coverage::create_bind_group(
            device,
            &layouts.uniform,
            &and_uniform_buffer,
            [&empty_coverage; 3],
        );
        let decrementing_bind_group = coverage::create_bind_group(
            device,
            &layouts.uniform,
            &decrementing_uniform_buffer,
            [&empty_coverage; 3],
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
                &layouts.background_texture,
            );
        let default_foreground_texture_bind_group =
            WgpuBackend::create_default_shape_texture_bind_group(
                device,
                queue,
                &layouts.foreground_texture,
            );
        Self {
            variants: ShapePipelineVariants::new(device, format, msaa_sample_count),
            layouts,
            and_bind_group,
            decrementing_bind_group,
            default_shape_texture_bind_groups: [
                Arc::new(default_background_texture_bind_group),
                Arc::new(default_foreground_texture_bind_group),
            ],
            texture_manager: context.texture_manager.clone(),
            and_uniforms,
            and_uniform_buffer,
            decrementing_uniforms,
            decrementing_uniform_buffer,
            linear_clamp_sampler,
        }
    }
}

impl BackdropPipelineResources {
    pub(in crate::wgpu_backend) fn new(
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

impl RendererPipelineResources {
    pub(in crate::wgpu_backend) fn new(
        context: &WgpuContext,
        format: TextureFormat,
        physical_size: (u32, u32),
        scale_factor: f64,
        fringe_width: f32,
        msaa_sample_count: u32,
    ) -> Self {
        let device = &context.device;
        let composite_resources =
            effects::compile_composite_pipeline(device, format, msaa_sample_count);
        let backdrops =
            BackdropPipelineResources::new(device, format, &composite_resources.bind_group_layout);
        Self {
            shapes: ShapePipelines::new(
                context,
                format,
                physical_size,
                scale_factor,
                fringe_width,
                msaa_sample_count,
                None,
            ),
            shape_mask_effects: ShapeMaskEffectRendererResources::new(device, format),
            effect_sampler: device.create_sampler(&SamplerDescriptor {
                address_mode_u: AddressMode::ClampToEdge,
                address_mode_v: AddressMode::ClampToEdge,
                address_mode_w: AddressMode::ClampToEdge,
                mag_filter: FilterMode::Linear,
                min_filter: FilterMode::Linear,
                mipmap_filter: FilterMode::Linear,
                ..Default::default()
            }),
            composite_resources,
            backdrops,
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
