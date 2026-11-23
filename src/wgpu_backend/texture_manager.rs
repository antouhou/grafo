use super::WgpuBackendError;
use crate::render_backend::texture_manager::{
    TextureManager, TexturePixelLayout, TexturePixels, TextureStorageFormat, TextureStreamError,
    TextureUpdatePublisher,
};
use crate::render_backend::PixelBufferError;
use futures_core::Stream;
use operations::TextureUploadOperations;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};
use thiserror::Error;
pub use upload::transfer::TextureTransferError;
use upload::transfer::TextureUploadQueue;
use wgpu::{
    Extent3d, MipmapFilterMode, Texture, TextureDescriptor, TextureDimension, TextureFormat,
    TextureUsages, TextureViewDescriptor,
};

mod operations;
mod streaming;
mod upload;

#[derive(Debug, Error)]
pub enum TextureManagerError {
    #[error("Texture {0} already has an active upload")]
    TextureBusy(u64),
    #[error("Texture upload was cancelled")]
    UploadCancelled,
    #[error(transparent)]
    Transfer(#[from] TextureTransferError),
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

/// Allocates and updates GPU textures.
///
/// Cloned managers share textures. All dimensions are `(width, height)` in pixels.
/// Native managers can be shared between threads. Browser managers must stay on their creating thread.
/// Uploads to [`TextureStorageFormat::Rgba16Float`] clamp converted colors to the finite binary16 range.
/// Synchronous uploads may block on native targets.
/// Prefer streaming methods for large browser uploads to reduce temporary memory use.
/// For concurrent uploads on a single-threaded executor, await the streaming methods
/// so earlier uploads can resume.
/// Successful uploads are available to subsequent rendering. GPU execution may still be in progress.
/// Only one upload can target an ID at a time. Removing or clearing textures cancels pending uploads.
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
    operations: TextureUploadOperations,
    uploads: Arc<TextureUploadQueue>,
    is_device_lost: Arc<AtomicBool>,
}

type BindGroupCache = HashMap<u64, Arc<wgpu::BindGroup>>;

impl WgpuTextureManager {
    pub(crate) fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        updates: TextureUpdatePublisher,
        is_device_lost: Arc<AtomicBool>,
    ) -> Self {
        let sampler = Self::create_sampler(&device);
        let operations = TextureUploadOperations::default();
        let uploads = Arc::new(TextureUploadQueue::new(
            Arc::clone(&device),
            Arc::clone(&queue),
            Arc::clone(&is_device_lost),
            operations.failure_signal(),
        ));
        Self {
            device,
            queue,
            sampler: Arc::new(sampler),
            texture_storage: Arc::new(RwLock::new(HashMap::new())),
            shape_bind_group_cache: Arc::new(RwLock::new(HashMap::new())),
            updates,
            operations,
            uploads,
            is_device_lost,
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

    fn ensure_available(&self) -> Result<(), WgpuBackendError> {
        if self.is_device_lost.load(Ordering::Relaxed) {
            Err(WgpuBackendError::DeviceLost)
        } else {
            Ok(())
        }
    }

    pub(super) fn device_loss_notifier(&self) -> impl Fn() + Send + Sync + 'static {
        self.uploads.device_loss_notifier()
    }

    fn validate_dimensions(&self, size: (u32, u32)) -> Result<(), TextureManagerError> {
        let max_dimension = self.device.limits().max_texture_dimension_2d;
        let valid_dimensions = 1..=max_dimension;
        if !valid_dimensions.contains(&size.0) || !valid_dimensions.contains(&size.1) {
            return Err(TextureManagerError::InvalidDimensions {
                size,
                max_dimension,
            });
        }
        Ok(())
    }

    fn create_texture(
        &self,
        size: (u32, u32),
        storage: TextureStorageFormat,
        usage: TextureUsages,
    ) -> Result<Texture, WgpuBackendError> {
        self.ensure_available()?;
        self.validate_dimensions(size)?;
        Ok(self.device.create_texture(&TextureDescriptor {
            label: Some("source_texture"),
            size: Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: match storage {
                TextureStorageFormat::Rgba8Srgb => TextureFormat::Rgba8UnormSrgb,
                TextureStorageFormat::Rgba16Float => TextureFormat::Rgba16Float,
            },
            usage: usage | TextureUsages::COPY_DST,
            view_formats: &[],
        }))
    }

    fn allocate(
        &self,
        texture_id: u64,
        size: (u32, u32),
        storage: TextureStorageFormat,
        pixels: Option<TexturePixels<'_>>,
    ) -> Result<(), WgpuBackendError> {
        self.ensure_available()?;
        self.validate_dimensions(size)?;
        if let Some(pixels) = pixels {
            pixels
                .validated_bytes()
                .map_err(TextureManagerError::from)?;
        }
        let operation = self.operations.begin(texture_id)?;
        let texture = self.create_texture(size, storage, TextureUsages::TEXTURE_BINDING)?;
        if let Some(pixels) = pixels {
            self.upload_slice(&operation, &texture, storage, pixels)?;
        }
        operation.commit_allocation(self, ManagedTexture { texture, storage })
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
        self.operations.clear(
            &self.texture_storage,
            &self.shape_bind_group_cache,
            &self.updates,
        );
    }

    fn allocate_texture_with_format(
        &self,
        texture_id: u64,
        texture_dimensions: (u32, u32),
        storage: TextureStorageFormat,
    ) -> Result<(), Self::Error> {
        self.allocate(texture_id, texture_dimensions, storage, None)
    }

    fn allocate_texture_with_pixels(
        &self,
        texture_id: u64,
        storage: TextureStorageFormat,
        pixels: TexturePixels<'_>,
    ) -> Result<(), Self::Error> {
        self.allocate(texture_id, pixels.layout.size, storage, Some(pixels))
    }

    fn load_data_into_texture(
        &self,
        texture_id: u64,
        origin: (u32, u32),
        pixels: TexturePixels<'_>,
    ) -> Result<(), Self::Error> {
        self.update_slice(texture_id, origin, pixels)
    }

    async fn allocate_texture_with_stream<Source, Chunk, SourceError>(
        &self,
        texture_id: u64,
        storage: TextureStorageFormat,
        layout: TexturePixelLayout,
        source: Source,
    ) -> Result<(), TextureStreamError<Self::Error, SourceError>>
    where
        Source: Stream<Item = Result<Chunk, SourceError>>,
        Chunk: AsRef<[u8]>,
    {
        self.ensure_available()?;
        self.validate_dimensions(layout.size)?;
        layout.required_byte_len()?;
        let operation = self.operations.begin(texture_id)?;
        let texture = self.create_texture(layout.size, storage, TextureUsages::TEXTURE_BINDING)?;
        self.upload_stream(&operation, &texture, storage, layout, source)
            .await?;
        operation
            .commit_allocation(self, ManagedTexture { texture, storage })
            .map_err(TextureStreamError::Backend)
    }

    async fn stream_data_into_texture<Source, Chunk, SourceError>(
        &self,
        texture_id: u64,
        origin: (u32, u32),
        layout: TexturePixelLayout,
        source: Source,
    ) -> Result<(), TextureStreamError<Self::Error, SourceError>>
    where
        Source: Stream<Item = Result<Chunk, SourceError>>,
        Chunk: AsRef<[u8]>,
    {
        self.ensure_available()?;
        let operation = self.operations.begin(texture_id)?;
        let (destination, storage) = self.update_destination(texture_id, origin, layout.size)?;
        layout.required_byte_len()?;
        let patch = self.create_texture(layout.size, storage, TextureUsages::COPY_SRC)?;
        self.upload_stream(&operation, &patch, storage, layout, source)
            .await?;
        operation
            .commit_update(self, &destination, &patch, origin)
            .map_err(TextureStreamError::Backend)
    }

    fn remove_texture(&self, texture_id: u64) {
        self.operations.remove(
            texture_id,
            &self.texture_storage,
            &self.shape_bind_group_cache,
            &self.updates,
        );
    }

    fn is_texture_loaded(&self, texture_id: u64) -> bool {
        self.texture_storage
            .read()
            .unwrap()
            .contains_key(&texture_id)
    }
}
