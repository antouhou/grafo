use std::cmp::max;
use wgpu::{BindGroup, BindGroupLayout, Device};
use wgpu::util::DeviceExt;
use crate::renderer::MathRect;
use crate::util::normalize_rect;
use crate::vertex::{CustomVertex, TexturedVertex};

pub(crate) struct ImageDrawData {
    pub(crate) image_data: Vec<u8>,
    pub(crate) image_size: (u32, u32),
    pub(crate) logical_rect: MathRect,
    pub(crate) clip_to_shape: Option<usize>,
    pub(crate) texture: Option<wgpu::Texture>,
    pub(crate) bind_group: Option<BindGroup>,
    pub(crate) vertex_buffer: Option<wgpu::Buffer>,
    pub(crate) index_buffer: Option<wgpu::Buffer>,
    pub(crate) num_indices: Option<u32>,
}

impl ImageDrawData {
    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bind_group_layout: &BindGroupLayout,
        canvas_physical_size: (u32, u32),
        scale_factor: f32,
    ) {
        let texture_dimensions = self.image_size;

        if (texture_dimensions.0 * texture_dimensions.1) != (self.image_data.len() / 4) as u32 {
            panic!("Image size and data size mismatch");
        }

        println!("texture_dimensions: {:?}", texture_dimensions);

        let texture_extent = wgpu::Extent3d {
            width: texture_dimensions.0,
            height: texture_dimensions.1,
            depth_or_array_layers: 1,
        };

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: texture_extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            // sRGBA, as we're going to work with RGBA images
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            // TEXTURE_BINDING to use texture in the shader, COPY_DST to copy data to the texture
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        queue.write_texture(
            // Tells wgpu where to copy the pixel data
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            // The actual pixel data
            &self.image_data,
            // The layout of the texture
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * texture_dimensions.0),
                rows_per_image: Some(texture_dimensions.1),
            },
            texture_extent,
        );

        self.texture = Some(texture);

        let view = self.create_texture_view().expect("Texture to be prepared for rendering");
        let sampler = ImageDrawData::create_sampler(&device);
        let bind_group = ImageDrawData::create_bind_group(
            &device,
            &bind_group_layout,
            &view,
            &sampler,
        );

        self.bind_group = Some(bind_group);
        self.prepare_vertices(device, canvas_physical_size, scale_factor);
    }

    fn prepare_vertices(&mut self, device: &Device, canvas_physical_size: (u32, u32), scale_factor: f32) {
        let normalized_rect = normalize_rect(&self.logical_rect, canvas_physical_size, scale_factor);
        let min_x = normalized_rect.min.x;
        let min_y = normalized_rect.min.y;
        let max_x = normalized_rect.max.x;
        let max_y = normalized_rect.max.y;

        println!("min_x: {}, min_y: {}, max_x: {}, max_y: {}", min_x, min_y, max_x, max_y);

        // let top_left = [-1.0, 1.0];
        // let top_right = [1.0, 1.0];
        // let bottom_right = [1.0, -1.0];
        // let bottom_left = [-1.0, -1.0];

        let top_left = [max_x, max_y];
        let top_right = [min_x, max_y];
        let bottom_right = [min_x, min_y];
        let bottom_left = [max_x, min_y];

        let top_left_tex = [0.0, 0.0];
        let top_right_tex = [1.0, 0.0];
        let bottom_right_tex = [1.0, 1.0];
        let bottom_left_tex = [0.0, 1.0];

        let quad = [
            TexturedVertex {
                position: top_left,
                tex_coords: top_left_tex,
            },
            TexturedVertex {
                position: top_right,
                tex_coords: top_right_tex,
            },
            TexturedVertex {
                position: bottom_right,
                tex_coords: bottom_right_tex,
            },
            TexturedVertex {
                position: bottom_left,
                tex_coords: bottom_left_tex,
            },
        ];

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&quad),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let indices: &[u16] = &[
            0, 1, 2, // first triangle
            2, 3, 0, // second triangle
        ];

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        self.vertex_buffer = Some(vertex_buffer);
        self.index_buffer = Some(index_buffer);
        self.num_indices = Some(indices.len() as u32);
    }

    fn create_sampler(device: &wgpu::Device) -> wgpu::Sampler {
        device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        })
    }

    fn create_texture_view(&self) -> Option<wgpu::TextureView> {
        Some(
            self.texture
                .as_ref()?
                .create_view(&wgpu::TextureViewDescriptor::default()),
        )
    }

    fn create_bind_group(
        device: &wgpu::Device,
        texture_bind_group_layout: &wgpu::BindGroupLayout,
        texture_view: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
            label: Some("diffuse_bind_group"),
        })
    }
}