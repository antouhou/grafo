use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use wgpu::util::DeviceExt;
use crate::MathRect;
use crate::util::{normalize_rect};
use crate::vertex::TexturedVertex;

#[derive(Debug)]
pub enum TextureManagerError {
    TextureNotFound(u64),
}

#[derive(Clone)]
pub struct TextureManager {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    sampler: Arc<wgpu::Sampler>,
    bind_group_layout: Arc<RwLock<wgpu::BindGroupLayout>>,
    /// Textures is raw image data, without any screen position information
    texture_storage: Arc<RwLock<HashMap<u64, wgpu::Texture>>>,
}

impl TextureManager {
    pub(crate) fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>, bind_group_layout: wgpu::BindGroupLayout) -> Self {
        let sampler = Self::create_sampler(&device);
        Self {
            device,
            queue,
            sampler: Arc::new(sampler),
            bind_group_layout: Arc::new(RwLock::new(bind_group_layout)),
            texture_storage: Arc::new(RwLock::new(HashMap::new())),
        }
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

    pub(crate) fn set_bind_group_layout(&self, bind_group_layout: wgpu::BindGroupLayout) {
        *self.bind_group_layout.write().unwrap() = bind_group_layout;
    }

    pub fn allocate_texture(
        &self,
        texture_id: u64,
        texture_dimensions: (u32, u32),
    ) {
        let texture_extent = wgpu::Extent3d {
            width: texture_dimensions.0,
            height: texture_dimensions.1,
            depth_or_array_layers: 1,
        };

        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
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

        self.texture_storage.write().unwrap().insert(texture_id, texture);
    }

    /// Allocates texture and loads data into it
    pub fn allocate_texture_with_data(
        &self,
        texture_id: u64,
        texture_dimensions: (u32, u32),
        texture_data: &[u8],
    ) {
        self.allocate_texture(texture_id, texture_dimensions);
        self.load_data_into_texture(texture_id, texture_data, texture_dimensions).unwrap();
    }

    /// Loads data to already allocated texture
    pub fn load_data_into_texture(&self, texture_id: u64, texture_data: &[u8], texture_dimensions: (u32, u32)) -> Result<(), TextureManagerError> {
        let texture_storage = self.texture_storage.read().unwrap();
        let texture = texture_storage.get(&texture_id).ok_or(TextureManagerError::TextureNotFound(texture_id))?;

        let texture_extent = wgpu::Extent3d {
            width: texture_dimensions.0,
            height: texture_dimensions.1,
            depth_or_array_layers: 1,
        };

        self.write_image_bytes_to_texture(texture, texture_dimensions, texture_extent, texture_data);

        Ok(())
    }

    fn write_image_bytes_to_texture(
        &self,
        texture: &wgpu::Texture,
        texture_dimensions: (u32, u32),
        texture_extent: wgpu::Extent3d,
        texture_data_bytes: &[u8],
    ) {
        self.queue.write_texture(
            // Tells wgpu where to copy the pixel data
            wgpu::ImageCopyTexture {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            // The actual pixel data
            texture_data_bytes,
            // The layout of the texture
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * texture_dimensions.0),
                rows_per_image: Some(texture_dimensions.1),
            },
            texture_extent,
        );
    }

    fn create_view(texture: &wgpu::Texture) -> wgpu::TextureView {
        texture.create_view(&wgpu::TextureViewDescriptor::default())
    }

    fn create_bind_group(
        &self,
        texture_bind_group_layout: &wgpu::BindGroupLayout,
        texture_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
            label: Some("diffuse_bind_group"),
        })
    }

    fn create_index_buffer(&self) -> wgpu::Buffer {
        // To draw an image we need to use only two triangles, so indices are always
        //  going to be the same
        let indices: &[u16] = &[
            0, 1, 2, // first triangle
            2, 3, 0, // second triangle
        ];

        let index_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        index_buffer
    }

    /// This creates vertex buffer for a quad that will be used to draw the texture. Not that
    ///  there can be any number of those buffers, each representing an area of the screen
    fn create_vertex_buffer(
        &self,
        canvas_physical_size: (u32, u32),
        logical_area_to_render_texture: &MathRect,
        scale_factor: f32,
    ) -> wgpu::Buffer {
        let vertex_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            // To draw an image, we always need 4 quad vertices, so the size is always the same
            size: 64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let normalized_rect =
            normalize_rect(logical_area_to_render_texture, canvas_physical_size, scale_factor);
        let min_x = normalized_rect.min.x;
        let min_y = normalized_rect.min.y;
        let max_x = normalized_rect.max.x;
        let max_y = normalized_rect.max.y;

        let top_left = [max_x, max_y];
        let top_right = [min_x, max_y];
        let bottom_right = [min_x, min_y];
        let bottom_left = [max_x, min_y];

        let top_left_tex_coords = [1.0, 1.0];
        let top_right_tex_coords = [0.0, 1.0];
        let bottom_right_text_coords = [0.0, 0.0];
        let bottom_left_tex_coords = [1.0, 0.0];

        let quad = [
            TexturedVertex {
                position: top_left,
                tex_coords: top_left_tex_coords,
            },
            TexturedVertex {
                position: top_right,
                tex_coords: top_right_tex_coords,
            },
            TexturedVertex {
                position: bottom_right,
                tex_coords: bottom_right_text_coords,
            },
            TexturedVertex {
                position: bottom_left,
                tex_coords: bottom_left_tex_coords,
            },
        ];

        self.queue.write_buffer(&vertex_buffer, 0, bytemuck::cast_slice(&quad));

        vertex_buffer
    }

    pub(crate) fn create_everything_to_render_texture(
        &self,
        texture_id: u64,
        canvas_physical_size: (u32, u32),
        logical_screen_area: &MathRect,
        scale_factor: f32,
    ) -> Result<(
        wgpu::Buffer,
        wgpu::Buffer,
        wgpu::BindGroup,
    ), TextureManagerError> {
        let vertex_buffer = self.create_vertex_buffer(canvas_physical_size, logical_screen_area, scale_factor);
        let index_buffer = self.create_index_buffer();

        let texture_storage = &self.texture_storage.read().unwrap();
        let texture = texture_storage.get(&texture_id).ok_or(TextureManagerError::TextureNotFound(texture_id))?;
        let texture_view = Self::create_view(texture);
        let bind_group_layout = self.bind_group_layout.read().unwrap();
        let bind_group = self.create_bind_group(&bind_group_layout, &texture_view);

        Ok((vertex_buffer, index_buffer, bind_group))
    }

    pub fn is_texture_loaded(&self, texture_id: u64) -> bool {
        self.texture_storage.read().unwrap().contains_key(&texture_id)
    }
}