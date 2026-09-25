use super::WgpuBackendError;
use crate::render_backend::TextureManager;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use wgpu::Texture;

#[derive(Debug, thiserror::Error)]
pub enum TextureManagerError {
    #[error("Texture {0} not found")]
    TextureNotFound(u64),
}

/// GPU textures that can be allocated and updated from multiple threads.
///
/// Cloned managers share texture storage. All dimensions are `(width, height)` in pixels.
///
/// # Examples
///
/// ```rust,no_run
/// use grafo::{premultiply_rgba8_srgb_inplace, TextureManager};
/// # fn example(renderer: &grafo::Renderer<'_>) {
/// let texture_manager = renderer.texture_manager();
/// let texture_id = 42;
/// let texture_dimensions = (256, 256);
/// let mut data = vec![255u8; 256 * 256 * 4];
///
/// texture_manager.allocate_texture_with_data(texture_id, texture_dimensions, &data);
///
/// data.fill(128);
/// premultiply_rgba8_srgb_inplace(&mut data);
/// texture_manager.load_data_into_texture(texture_id, texture_dimensions, &data).unwrap();
/// assert!(texture_manager.is_texture_loaded(texture_id));
/// # }
/// ```
#[derive(Clone)]
pub struct WgpuTextureManager {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    sampler: Arc<wgpu::Sampler>,
    // Operations needing both locks acquire storage before the bind-group cache.
    texture_storage: Arc<RwLock<HashMap<u64, wgpu::Texture>>>,
    /// Both shape texture layers use the same layout across renderers and MSAA settings.
    shape_bind_group_cache: Arc<RwLock<BindGroupCache>>,
}

type BindGroupCache = HashMap<u64, Arc<wgpu::BindGroup>>;

impl WgpuTextureManager {
    pub(crate) fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        let sampler = Self::create_sampler(&device);
        Self {
            device,
            queue,
            sampler: Arc::new(sampler),
            texture_storage: Arc::new(RwLock::new(HashMap::new())),
            shape_bind_group_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Returns the number of stored textures and cached bind groups, in that order.
    pub fn size(&self) -> (usize, usize) {
        let texture_storage = self.texture_storage.read().unwrap();
        let bind_group_cache = self.shape_bind_group_cache.read().unwrap();
        (texture_storage.len(), bind_group_cache.len())
    }

    fn create_sampler(device: &wgpu::Device) -> wgpu::Sampler {
        device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        })
    }

    fn create_texture(&self, texture_dimensions: (u32, u32)) -> wgpu::Texture {
        let texture_extent = wgpu::Extent3d {
            width: texture_dimensions.0,
            height: texture_dimensions.1,
            depth_or_array_layers: 1,
        };

        self.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: texture_extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        })
    }

    fn insert_texture(&self, texture_id: u64, texture: wgpu::Texture) {
        let mut texture_storage = self.texture_storage.write().unwrap();
        let mut bind_group_cache = self.shape_bind_group_cache.write().unwrap();
        // Invalidate old bindings while both locks exclude concurrent cache insertion.
        bind_group_cache.remove(&texture_id);
        texture_storage.insert(texture_id, texture);
    }

    fn write_pixels_to_texture(
        &self,
        texture: &wgpu::Texture,
        texture_dimensions: (u32, u32),
        texture_extent: wgpu::Extent3d,
        texture_data_bytes: &[u8],
    ) {
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            texture_data_bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * texture_dimensions.0),
                rows_per_image: Some(texture_dimensions.1),
            },
            texture_extent,
        );
    }

    /// Returns a cached bind group for the texture, creating it with the shape layout if needed.
    pub(crate) fn get_or_create_shape_bind_group(
        &self,
        layout: &wgpu::BindGroupLayout,
        texture_id: u64,
    ) -> Result<Arc<wgpu::BindGroup>, TextureManagerError> {
        if let Some(bg) = self
            .shape_bind_group_cache
            .read()
            .unwrap()
            .get(&texture_id)
            .cloned()
        {
            return Ok(bg);
        }

        // Hold storage through cache insertion so replacement cannot leave a stale binding.
        let storage = self.texture_storage.read().unwrap();
        let texture = storage
            .get(&texture_id)
            .ok_or(TextureManagerError::TextureNotFound(texture_id))?;
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = Arc::new(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
            label: Some("shape_texture_bind_group_cached"),
        }));

        self.shape_bind_group_cache
            .write()
            .unwrap()
            .insert(texture_id, bind_group.clone());

        Ok(bind_group)
    }

    pub(crate) fn texture(&self, texture_id: u64) -> Option<Texture> {
        self.texture_storage
            .read()
            .expect("texture storage lock poisoned")
            .get(&texture_id)
            .cloned()
    }

    pub(crate) fn texture_dimensions(&self, texture_id: u64) -> Option<(u32, u32)> {
        self.texture_storage
            .read()
            .unwrap()
            .get(&texture_id)
            .map(|texture| {
                let size = texture.size();
                (size.width, size.height)
            })
    }
}

impl TextureManager for WgpuTextureManager {
    type Error = WgpuBackendError;

    fn clear(&self) {
        let mut texture_storage = self.texture_storage.write().unwrap();
        let mut bind_group_cache = self.shape_bind_group_cache.write().unwrap();
        texture_storage.clear();
        bind_group_cache.clear();
    }

    fn allocate_texture(&self, texture_id: u64, texture_dimensions: (u32, u32)) {
        self.insert_texture(texture_id, self.create_texture(texture_dimensions));
    }

    fn allocate_texture_with_data(
        &self,
        texture_id: u64,
        texture_dimensions: (u32, u32),
        texture_data: &[u8],
    ) {
        let texture = self.create_texture(texture_dimensions);
        self.write_pixels_to_texture(&texture, texture_dimensions, texture.size(), texture_data);
        self.insert_texture(texture_id, texture);
    }

    fn load_data_into_texture(
        &self,
        texture_id: u64,
        texture_dimensions: (u32, u32),
        texture_data: &[u8],
    ) -> Result<(), Self::Error> {
        let texture_storage = self.texture_storage.read().unwrap();
        let texture = texture_storage
            .get(&texture_id)
            .ok_or(TextureManagerError::TextureNotFound(texture_id))?;

        let texture_extent = wgpu::Extent3d {
            width: texture_dimensions.0,
            height: texture_dimensions.1,
            depth_or_array_layers: 1,
        };

        self.write_pixels_to_texture(texture, texture_dimensions, texture_extent, texture_data);

        Ok(())
    }

    fn remove_texture(&self, texture_id: u64) {
        let mut texture_storage = self.texture_storage.write().unwrap();
        let mut bind_group_cache = self.shape_bind_group_cache.write().unwrap();
        bind_group_cache.remove(&texture_id);
        texture_storage.remove(&texture_id);
    }

    fn is_texture_loaded(&self, texture_id: u64) -> bool {
        self.texture_storage
            .read()
            .unwrap()
            .contains_key(&texture_id)
    }
}
