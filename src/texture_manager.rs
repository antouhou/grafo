use std::collections::HashMap;
use std::sync::{Arc, RwLock};

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
/// use grafo::premultiply_rgba8_srgb_inplace;
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
pub struct TextureManager {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    sampler: Arc<wgpu::Sampler>,
    // Operations needing both locks acquire storage before the bind-group cache.
    texture_storage: Arc<RwLock<HashMap<u64, wgpu::Texture>>>,
    /// Cache for shape texture bind groups keyed by (texture_id, layout_epoch)
    shape_bind_group_cache: Arc<RwLock<BindGroupCache>>,
}

type BindGroupCache = HashMap<(u64, u64), Arc<wgpu::BindGroup>>;

impl TextureManager {
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

    /// Removes all textures and cached bind groups from the shared storage.
    pub fn clear(&self) {
        let mut texture_storage = self.texture_storage.write().unwrap();
        let mut bind_group_cache = self.shape_bind_group_cache.write().unwrap();
        texture_storage.clear();
        bind_group_cache.clear();
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

    /// Allocates an RGBA8 sRGB texture, replacing any texture with the same ID.
    ///
    /// Upload pixels with [`Self::load_data_into_texture`], or allocate and upload together
    /// with [`Self::allocate_texture_with_data`].
    pub fn allocate_texture(&self, texture_id: u64, texture_dimensions: (u32, u32)) {
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
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let mut texture_storage = self.texture_storage.write().unwrap();
        let mut bind_group_cache = self.shape_bind_group_cache.write().unwrap();
        // Invalidate old bindings while both locks exclude concurrent cache insertion.
        bind_group_cache.retain(|(cached_texture_id, _), _| *cached_texture_id != texture_id);
        texture_storage.insert(texture_id, texture);
    }

    /// Allocates and uploads a texture, replacing any texture with the same ID.
    ///
    /// See [`Self::load_data_into_texture`] for the required pixel format.
    pub fn allocate_texture_with_data(
        &self,
        texture_id: u64,
        texture_dimensions: (u32, u32),
        texture_data: &[u8],
    ) {
        self.allocate_texture(texture_id, texture_dimensions);
        self.load_data_into_texture(texture_id, texture_dimensions, texture_data)
            .unwrap();
    }

    /// Uploads RGBA8 sRGB pixels to the top-left corner of an allocated texture.
    ///
    /// `texture_dimensions` is the upload size and must fit inside the texture. Supply
    /// four bytes per pixel with no padding between rows. RGB must be premultiplied
    /// by alpha in linear space, then encoded as sRGB. Use [`premultiply_rgba8_srgb_inplace`]
    /// to convert straight-alpha input before uploading.
    ///
    /// Returns an error if `texture_id` has not been allocated.
    pub fn load_data_into_texture(
        &self,
        texture_id: u64,
        texture_dimensions: (u32, u32),
        texture_data: &[u8],
    ) -> Result<(), TextureManagerError> {
        let texture_storage = self.texture_storage.read().unwrap();
        let texture = texture_storage
            .get(&texture_id)
            .ok_or(TextureManagerError::TextureNotFound(texture_id))?;

        let texture_extent = wgpu::Extent3d {
            width: texture_dimensions.0,
            height: texture_dimensions.1,
            depth_or_array_layers: 1,
        };

        self.write_image_bytes_to_texture(
            texture,
            texture_dimensions,
            texture_extent,
            texture_data,
        );

        Ok(())
    }

    /// Removes the texture identified by `texture_id` from the manager.
    pub fn remove_texture(&self, texture_id: u64) {
        let mut texture_storage = self.texture_storage.write().unwrap();
        let mut bind_group_cache = self.shape_bind_group_cache.write().unwrap();
        bind_group_cache.retain(|(cached_texture_id, _), _| *cached_texture_id != texture_id);
        texture_storage.remove(&texture_id);
    }

    fn write_image_bytes_to_texture(
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

    /// Returns a cached bind group for the texture and layout, creating it if needed.
    pub(crate) fn get_or_create_shape_bind_group(
        &self,
        layout: &wgpu::BindGroupLayout,
        layout_epoch: u64,
        texture_id: u64,
    ) -> Result<Arc<wgpu::BindGroup>, TextureManagerError> {
        if let Some(bg) = self
            .shape_bind_group_cache
            .read()
            .unwrap()
            .get(&(texture_id, layout_epoch))
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
            .insert((texture_id, layout_epoch), bind_group.clone());

        Ok(bind_group)
    }

    /// Returns whether the ID has an allocated texture, even if no pixels were uploaded.
    pub fn is_texture_loaded(&self, texture_id: u64) -> bool {
        self.texture_storage
            .read()
            .unwrap()
            .contains_key(&texture_id)
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

fn srgb_to_linear_u8(c: u8) -> f32 {
    let x = c as f32 / 255.0;
    if x <= 0.04045 {
        x / 12.92
    } else {
        ((x + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb_u8(x: f32) -> u8 {
    let x = x.clamp(0.0, 1.0);
    let y = if x <= 0.0031308 {
        x * 12.92
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    };
    (y.clamp(0.0, 1.0) * 255.0 + 0.5).floor() as u8
}

/// Converts straight-alpha RGBA8 sRGB pixels to premultiplied alpha in place.
///
/// Multiplies RGB by alpha in linear space, then encodes it as sRGB. Alpha bytes are unchanged.
///
/// # Panics
///
/// Panics if the slice length is not a multiple of four.
pub fn premultiply_rgba8_srgb_inplace(pixels: &mut [u8]) {
    assert!(
        pixels.len().is_multiple_of(4),
        "RGBA8 data length must be multiple of 4"
    );
    for px in pixels.chunks_mut(4) {
        let r_lin = srgb_to_linear_u8(px[0]);
        let g_lin = srgb_to_linear_u8(px[1]);
        let b_lin = srgb_to_linear_u8(px[2]);
        let a = px[3] as f32 / 255.0;

        let r_pma = r_lin * a;
        let g_pma = g_lin * a;
        let b_pma = b_lin * a;

        px[0] = linear_to_srgb_u8(r_pma);
        px[1] = linear_to_srgb_u8(g_pma);
        px[2] = linear_to_srgb_u8(b_pma);
    }
}

#[cfg(test)]
mod tests {
    use super::TextureManager;
    use crate::{RendererContext, RendererCreationError};
    use futures::executor::block_on;
    use std::panic::{self, AssertUnwindSafe};
    use std::sync::{mpsc, Arc};
    use std::thread;
    use std::time::Duration;

    fn create_texture_manager() -> Option<TextureManager> {
        match block_on(RendererContext::try_new()) {
            Ok(context) => Some(context.inner.texture_manager.clone()),
            Err(RendererCreationError::AdapterNotAvailable(_)) => {
                println!("Skipping test: no suitable GPU adapter available.");
                None
            }
            Err(error) => panic!("Failed to create renderer context: {error}"),
        }
    }

    fn assert_storage_is_locked_before_cache(
        operation: impl FnOnce(&TextureManager) + Send + 'static,
    ) {
        let Some(manager) = create_texture_manager() else {
            return;
        };
        let storage = Arc::clone(&manager.texture_storage);
        assert!(thread::spawn(move || {
            let _storage = storage.write().unwrap();
            panic!("poison storage to observe which lock the operation acquires first");
        })
        .join()
        .is_err());

        // Storage-first operations reach the poisoned lock while the cache is held.
        // Cache-first operations cannot finish until the cache is released below.
        let cache = manager.shape_bind_group_cache.write().unwrap();
        let worker_manager = manager.clone();
        let (completed, completion) = mpsc::channel();
        let worker = thread::spawn(move || {
            let result = panic::catch_unwind(AssertUnwindSafe(|| operation(&worker_manager)));
            completed.send(result.is_err()).unwrap();
        });
        let result = completion.recv_timeout(Duration::from_secs(5));
        drop(cache);
        worker.join().unwrap();

        assert_eq!(
            result,
            Ok(true),
            "operation waited for the cache before acquiring texture storage",
        );
    }

    #[test]
    fn texture_replacement_locks_storage_before_cache() {
        assert_storage_is_locked_before_cache(|manager| manager.allocate_texture(7, (1, 1)));
    }

    #[test]
    fn texture_removal_locks_storage_before_cache() {
        assert_storage_is_locked_before_cache(|manager| manager.remove_texture(7));
    }
}
