use std::sync::{Arc, RwLock};

pub struct DepthStencilTextureViewer {
    read_buffer: Option<wgpu::Buffer>,
    buffer_size: wgpu::BufferAddress,
    texture_size: (u32, u32),
    texture_data: Arc<RwLock<Vec<[u8; 4]>>>,
}

impl DepthStencilTextureViewer {
    pub fn new(device: &wgpu::Device, texture_size: (u32, u32)) -> Self {
        let buffer_size = (texture_size.0 * texture_size.1 * 4) as wgpu::BufferAddress;
        let read_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Read Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let pixel = [0u8; 4];
        let texture_data = Arc::new(RwLock::new(vec![pixel; buffer_size as usize]));
        Self {
            read_buffer: Some(read_buffer),
            buffer_size,
            texture_size,
            texture_data,
        }
    }

    /// Called before submitting the frame
    pub fn copy_depth_stencil_texture_to_buffer(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        depth_stencil_texture: &wgpu::Texture,
    ) {
        let texture_extent = wgpu::Extent3d {
            width: self.texture_size.0,
            height: self.texture_size.1,
            depth_or_array_layers: 1,
        };

        let buffer = self.read_buffer.as_ref().unwrap();

        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: depth_stencil_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::DepthOnly,
            },
            wgpu::ImageCopyBuffer {
                buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(self.texture_size.0 * 4),
                    rows_per_image: Some(self.texture_size.1),
                },
            },
            texture_extent,
        );
    }

    /// Before you can view the depth stencil texture, you need to call this method
    ///  to copy texture data to memory
    pub fn save_depth_stencil_texture(&mut self, device: &wgpu::Device) {
        let read_buffer = Arc::new(self.read_buffer.take().unwrap());
        self.read_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Read Buffer"),
            size: self.buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }));

        let read_buffer_2 = read_buffer.clone();
        let texture_data = self.texture_data.clone();

        read_buffer_2
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                if let Ok(()) = result {
                    let data = read_buffer.slice(..).get_mapped_range();
                    let mut data_guard = texture_data.write().unwrap();
                    data.chunks(4).enumerate().for_each(|(i, chunk)| {
                        data_guard[i] = [chunk[0], chunk[1], chunk[2], chunk[3]];
                    });
                }
            });
    }

    pub fn print_texture_data_at_pixel(&self, x: u32, y: u32) {
        let data = self.texture_data.read().unwrap();
        let index = (y * self.texture_size.0 + x) as usize;

        let pixel_data = data[index];
        let depth_bits =
            u32::from_le_bytes([pixel_data[0], pixel_data[1], pixel_data[2], pixel_data[3]]);
        let depth_value = f32::from_bits(depth_bits);

        println!("{:?}", depth_value);
    }
}
