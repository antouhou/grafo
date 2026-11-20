use super::WgpuBackendError;
use crate::render_backend::texture_manager::{
    TextureManager, TexturePixels, TextureStorageFormat, TextureUpdatePublisher,
};
use crate::render_backend::PixelBufferError;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use upload::TextureUpload;
use wgpu::{
    Extent3d, MipmapFilterMode, Texture, TextureDescriptor, TextureDimension, TextureFormat,
    TextureUsages, TextureViewDescriptor,
};

mod upload;

#[derive(Debug, thiserror::Error)]
pub enum TextureManagerError {
    #[error("Texture dimensions {size:?} must each be between 1 and {max_dimension}")]
    InvalidDimensions {
        size: (u32, u32),
        max_dimension: u32,
    },
    #[error("Texture {0} not found")]
    TextureNotFound(u64),
    #[error(transparent)]
    InvalidPixels(#[from] PixelBufferError),
    #[error("Upload of {size:?} pixels at {origin:?} exceeds texture dimensions {texture_size:?}")]
    OutOfBounds {
        size: (u32, u32),
        origin: (u32, u32),
        texture_size: (u32, u32),
    },
}

struct ManagedTexture {
    texture: Texture,
    storage: TextureStorageFormat,
}

/// GPU textures that can be allocated and updated from multiple threads.
///
/// Cloned managers share texture storage. All dimensions are `(width, height)` in pixels.
/// Uploads to [`TextureStorageFormat::Rgba16Float`] clamp converted colors to the finite binary16 range.
///
/// # Examples
///
/// ```rust,no_run
/// use grafo::{AlphaMode, Scene, TextureManager, TexturePixels, WgpuBackendError};
/// # fn example(scene: &Scene) -> Result<(), WgpuBackendError> {
/// let texture_manager = scene.texture_manager();
/// let texture_id = 42;
/// let texture_dimensions = (256, 256);
/// let mut data = vec![255u8; 256 * 256 * 4];
///
/// texture_manager.allocate_texture_with_data(texture_id, texture_dimensions, &data)?;
///
/// data.fill(128);
/// let pixels = TexturePixels::rgba8_srgb(&data, texture_dimensions, AlphaMode::Straight);
/// texture_manager.load_data_into_texture(texture_id, (0, 0), pixels)?;
/// assert!(texture_manager.is_texture_loaded(texture_id));
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct WgpuTextureManager {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    sampler: Arc<wgpu::Sampler>,
    // Operations needing both locks acquire storage before the bind-group cache.
    texture_storage: Arc<RwLock<HashMap<u64, ManagedTexture>>>,
    /// Both shape texture layers use the same layout across scenes and MSAA settings.
    shape_bind_group_cache: Arc<RwLock<BindGroupCache>>,
    updates: TextureUpdatePublisher,
}

type BindGroupCache = HashMap<u64, Arc<wgpu::BindGroup>>;

impl WgpuTextureManager {
    pub(crate) fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        updates: TextureUpdatePublisher,
    ) -> Self {
        let sampler = Self::create_sampler(&device);
        Self {
            device,
            queue,
            sampler: Arc::new(sampler),
            texture_storage: Arc::new(RwLock::new(HashMap::new())),
            shape_bind_group_cache: Arc::new(RwLock::new(HashMap::new())),
            updates,
        }
    }

    /// Returns the number of stored textures and cached bind groups, in that order.
    pub fn size(&self) -> (usize, usize) {
        let texture_storage = self.texture_storage.read().unwrap();
        let bind_group_cache = self.shape_bind_group_cache.read().unwrap();
        (texture_storage.len(), bind_group_cache.len())
    }

    /// Releases cached bindings and spare storage while preserving all loaded textures.
    pub fn clear_caches(&self) {
        let mut texture_storage = self
            .texture_storage
            .write()
            .expect("texture storage lock poisoned");
        let mut bind_group_cache = self
            .shape_bind_group_cache
            .write()
            .expect("shape bind-group cache lock poisoned");
        bind_group_cache.clear();
        bind_group_cache.shrink_to_fit();
        texture_storage.shrink_to_fit();
        self.updates.clear_caches();
    }

    fn create_sampler(device: &wgpu::Device) -> wgpu::Sampler {
        device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: MipmapFilterMode::Linear,
            ..Default::default()
        })
    }

    fn allocate(
        &self,
        texture_id: u64,
        size: (u32, u32),
        storage: TextureStorageFormat,
        pixels: Option<TexturePixels<'_>>,
    ) -> Result<(), TextureManagerError> {
        let max_dimension = self.device.limits().max_texture_dimension_2d;
        let valid_dimensions = 1..=max_dimension;
        if !valid_dimensions.contains(&size.0) || !valid_dimensions.contains(&size.1) {
            return Err(TextureManagerError::InvalidDimensions {
                size,
                max_dimension,
            });
        }
        let upload = pixels
            .map(|pixels| TextureUpload::new(storage, pixels))
            .transpose()?;
        let texture_extent = Extent3d {
            width: size.0,
            height: size.1,
            depth_or_array_layers: 1,
        };

        let texture = self.device.create_texture(&TextureDescriptor {
            label: None,
            size: texture_extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: match storage {
                TextureStorageFormat::Rgba8Srgb => TextureFormat::Rgba8UnormSrgb,
                TextureStorageFormat::Rgba16Float => TextureFormat::Rgba16Float,
            },
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        if let Some(upload) = upload {
            upload.write(&self.queue, &texture, (0, 0));
        }
        self.insert_texture(texture_id, ManagedTexture { texture, storage });
        Ok(())
    }

    fn insert_texture(&self, texture_id: u64, texture: ManagedTexture) {
        let mut texture_storage = self.texture_storage.write().unwrap();
        let mut bind_group_cache = self.shape_bind_group_cache.write().unwrap();
        // Invalidate old bindings while both locks exclude concurrent cache insertion.
        bind_group_cache.remove(&texture_id);
        texture_storage.insert(texture_id, texture);
        self.updates.publish(texture_id);
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
        let entry = storage
            .get(&texture_id)
            .ok_or(TextureManagerError::TextureNotFound(texture_id))?;
        let view = entry.texture.create_view(&TextureViewDescriptor::default());
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
            .map(|entry| entry.texture.clone())
    }

    pub(crate) fn texture_dimensions(&self, texture_id: u64) -> Option<(u32, u32)> {
        self.texture_storage
            .read()
            .unwrap()
            .get(&texture_id)
            .map(|entry| (entry.texture.width(), entry.texture.height()))
    }
}

impl TextureManager for WgpuTextureManager {
    type Error = WgpuBackendError;

    fn clear(&self) {
        let mut texture_storage = self.texture_storage.write().unwrap();
        let mut bind_group_cache = self.shape_bind_group_cache.write().unwrap();
        for &texture_id in texture_storage.keys() {
            self.updates.publish(texture_id);
        }
        texture_storage.clear();
        bind_group_cache.clear();
    }

    fn allocate_texture_with_format(
        &self,
        texture_id: u64,
        texture_dimensions: (u32, u32),
        storage: TextureStorageFormat,
    ) -> Result<(), Self::Error> {
        self.allocate(texture_id, texture_dimensions, storage, None)
            .map_err(Into::into)
    }

    fn allocate_texture_with_pixels(
        &self,
        texture_id: u64,
        storage: TextureStorageFormat,
        pixels: TexturePixels<'_>,
    ) -> Result<(), Self::Error> {
        self.allocate(texture_id, pixels.size, storage, Some(pixels))
            .map_err(Into::into)
    }

    fn load_data_into_texture(
        &self,
        texture_id: u64,
        origin: (u32, u32),
        pixels: TexturePixels<'_>,
    ) -> Result<(), Self::Error> {
        let texture_storage = self.texture_storage.read().unwrap();
        let entry = texture_storage
            .get(&texture_id)
            .ok_or(TextureManagerError::TextureNotFound(texture_id))?;

        let texture_size = (entry.texture.width(), entry.texture.height());
        if u64::from(origin.0) + u64::from(pixels.size.0) > u64::from(texture_size.0)
            || u64::from(origin.1) + u64::from(pixels.size.1) > u64::from(texture_size.1)
        {
            return Err(TextureManagerError::OutOfBounds {
                origin,
                size: pixels.size,
                texture_size,
            }
            .into());
        }
        let upload =
            TextureUpload::new(entry.storage, pixels).map_err(TextureManagerError::from)?;
        upload.write(&self.queue, &entry.texture, origin);
        self.updates.publish(texture_id);

        Ok(())
    }

    fn remove_texture(&self, texture_id: u64) {
        let mut texture_storage = self.texture_storage.write().unwrap();
        let mut bind_group_cache = self.shape_bind_group_cache.write().unwrap();
        bind_group_cache.remove(&texture_id);
        if texture_storage.remove(&texture_id).is_some() {
            self.updates.publish(texture_id);
        }
    }

    fn is_texture_loaded(&self, texture_id: u64) -> bool {
        self.texture_storage
            .read()
            .unwrap()
            .contains_key(&texture_id)
    }
}
