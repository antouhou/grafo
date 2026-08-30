use std::collections::HashMap;
use std::sync::{Arc, RwLock};

#[derive(Debug, thiserror::Error)]
pub enum TextureManagerError {
    #[error("Texture {0} not found")]
    TextureNotFound(u64),
    #[error("Texture {0} dimensions or data length are invalid")]
    InvalidTextureData(u64),
}

fn validate_texture_region(
    texture_id: u64,
    texture_dimensions: (u32, u32),
    region_origin: (u32, u32),
    region_dimensions: (u32, u32),
    byte_length: usize,
) -> Result<(), TextureManagerError> {
    let expected_byte_length = region_dimensions
        .0
        .try_into()
        .ok()
        .and_then(|width: usize| {
            usize::try_from(region_dimensions.1)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .and_then(|pixels| pixels.checked_mul(4));
    let region_fits = region_dimensions.0 > 0
        && region_dimensions.1 > 0
        && region_origin
            .0
            .checked_add(region_dimensions.0)
            .is_some_and(|right| right <= texture_dimensions.0)
        && region_origin
            .1
            .checked_add(region_dimensions.1)
            .is_some_and(|bottom| bottom <= texture_dimensions.1);
    if !region_fits || expected_byte_length != Some(byte_length) {
        return Err(TextureManagerError::InvalidTextureData(texture_id));
    }
    Ok(())
}

/// A manager for textures providing granular control over texture handling.
///
/// This manager allows for:
/// - Loading textures from different threads while keeping usage safe in the rendering thread.
/// - Allocating textures and subsequently loading image data into them.
/// - Updating the data in an existing texture.
///
/// # Examples
///
/// Allocate a texture and then load data into it:
///
/// ```rust,no_run
/// # use std::sync::Arc;
/// # use futures::executor::block_on;
/// # use winit::application::ApplicationHandler;
/// # use winit::event_loop::{ActiveEventLoop, EventLoop};
/// # use winit::window::Window;
/// # use grafo::Renderer;
/// # use grafo::Shape;
/// # use grafo::Color;
/// # use grafo::Stroke;
/// #
/// # struct App;
/// # impl ApplicationHandler for App {
/// #     fn resumed(&mut self, event_loop: &ActiveEventLoop) {
/// #         let window_surface = Arc::new(
/// #             event_loop.create_window(Window::default_attributes()).unwrap()
/// #         );
/// #         let physical_size = (800, 600);
/// #         let scale_factor = 1.0;
/// #         let mut renderer = block_on(Renderer::new(window_surface, physical_size, scale_factor, true, false, 1));
/// #
/// let texture_manager = renderer.texture_manager();
/// let texture_id = 42;
/// let texture_dimensions = (256, 256);
/// let data = vec![255u8; 256 * 256 * 4];
///
/// // Allocate texture and load data
/// texture_manager.allocate_texture_with_data(texture_id, texture_dimensions, &data);
/// // Update data in the texture
/// texture_manager.load_data_into_texture(texture_id, texture_dimensions, &data).unwrap();
/// // Check if the texture is loaded
/// assert!(texture_manager.is_texture_loaded(texture_id));
/// // Clone the texture manager to pass to another thread
/// let texture_manager_clone = texture_manager.clone();
/// #     }
/// #
/// #     fn window_event(&mut self, _: &ActiveEventLoop, _: winit::window::WindowId, _: winit::event::WindowEvent) {
/// #         // Handle window events (stub for doc test)
/// #     }
/// # }
/// ```
///
/// The texture manager internally uses an `Arc<RwLock<_>>` to manage its bind group layout and texture storage.
#[derive(Clone)]
pub struct TextureManager {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    sampler: Arc<wgpu::Sampler>,
    // Operations that need both locks must acquire texture_storage before shape_bind_group_cache.
    /// Textures is raw image data, without any screen position information
    texture_storage: Arc<RwLock<HashMap<u64, wgpu::Texture>>>,
    /// Each entry retains its layout until that layout is explicitly retired.
    shape_bind_group_cache: Arc<RwLock<BindGroupCache>>,
}

type BindGroupCache = HashMap<(u64, wgpu::BindGroupLayout), Arc<wgpu::BindGroup>>;

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

    pub fn clear(&self) {
        let mut texture_storage = self.texture_storage.write().unwrap();
        let mut bind_group_cache = self.shape_bind_group_cache.write().unwrap();
        texture_storage.clear();
        bind_group_cache.clear();
    }

    pub fn size(&self) -> (usize, usize) {
        (
            self.texture_storage.read().unwrap().len(),
            self.shape_bind_group_cache.read().unwrap().len(),
        )
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

    /// Allocates a new RGBA8 texture with the given dimensions without providing any data.
    ///
    /// If you want to allocate and load data into the texture at the same time, use
    /// [`TextureManager::allocate_texture_with_data`] instead.
    /// You can then load data into the texture later using [`TextureManager::load_data_into_texture`].
    ///
    /// # Parameters
    /// - `texture_id`: Unique identifier for the texture.
    /// - `texture_dimensions`: A tuple `(width, height)` representing the dimensions of the texture.
    pub fn allocate_texture(&self, texture_id: u64, texture_dimensions: (u32, u32)) {
        self.replace_texture(texture_id, texture_dimensions);
    }

    fn get_or_replace_texture(
        &self,
        texture_id: u64,
        texture_dimensions: (u32, u32),
    ) -> wgpu::Texture {
        let existing_texture = self
            .texture_storage
            .read()
            .unwrap()
            .get(&texture_id)
            .filter(|texture| {
                texture.width() == texture_dimensions.0 && texture.height() == texture_dimensions.1
            })
            .cloned();
        existing_texture.unwrap_or_else(|| self.replace_texture(texture_id, texture_dimensions))
    }

    fn replace_texture(&self, texture_id: u64, texture_dimensions: (u32, u32)) -> wgpu::Texture {
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

        self.replace_stored_texture(texture_id, texture.clone());
        texture
    }

    fn replace_stored_texture(
        &self,
        texture_id: u64,
        texture: wgpu::Texture,
    ) {
        let mut texture_storage = self.texture_storage.write().unwrap();
        let mut bind_group_cache = self.shape_bind_group_cache.write().unwrap();
        bind_group_cache
            .retain(|(cached_texture_id, _layout), _bind_group| *cached_texture_id != texture_id);
        texture_storage.insert(texture_id, texture);
    }

    /// Allocates a texture and writes its image data to the GPU queue.
    ///
    /// This function creates the texture when needed, then writes the provided data.
    ///
    /// If you are seeing fringes when sampling/minifying near transparent edges, ensure that your
    /// texture data is in a premultiplied alpha format. You can use the
    /// `premultiply_rgba8_srgb_inplace` helper function provided in this crate to convert your
    /// RGBA8 sRGB data to premultiplied alpha.
    ///
    /// # Parameters
    /// - `texture_id`: Unique identifier for the texture.
    /// - `texture_dimensions`: A tuple `(width, height)` representing the dimensions of the texture.
    /// - `texture_data`: Tightly packed RGBA8 sRGB data with premultiplied alpha. Its length must
    ///   match the texture dimensions.
    pub fn allocate_texture_with_data(
        &self,
        texture_id: u64,
        texture_dimensions: (u32, u32),
        texture_data: &[u8],
    ) {
        let texture = self.get_or_replace_texture(texture_id, texture_dimensions);
        self.write_texture_region(
            texture_id,
            &texture,
            (0, 0),
            texture_dimensions,
            texture_data,
        )
        .expect("allocated texture dimensions match the provided data");
    }

    /// Writes image data into an already allocated texture. If you are seeing fringes when
    /// sampling/minifying near transparent edges, ensure that your texture data is in a
    /// premultiplied alpha format. You can use the `premultiply_rgba8_srgb_inplace` helper
    /// function provided in this crate to convert your RGBA8 sRGB data to premultiplied alpha.
    ///
    /// # Parameters
    /// - `texture_id`: Unique identifier for the texture.
    /// - `texture_dimensions`: A tuple `(width, height)` representing the texture dimensions.
    /// - `texture_data`: Tightly packed RGBA8 sRGB data with premultiplied alpha.
    ///   If your texture isn't premultiplied, consider using a `premultiply_rgba8_srgb_inplace` helper
    ///   function provided in this crate. This is needed to avoid fringes when sampling/minifying near transparent edges.
    ///
    /// # Returns
    /// - `Ok(())` if the operation succeeds.
    /// - `Err(TextureManagerError::TextureNotFound(texture_id))` if the texture does not exist.
    /// - `Err(TextureManagerError::InvalidTextureData(texture_id))` if the supplied dimensions do
    ///   not match the texture or its byte length.
    pub fn load_data_into_texture(
        &self,
        texture_id: u64,
        texture_dimensions: (u32, u32),
        texture_data: &[u8],
    ) -> Result<(), TextureManagerError> {
        let texture = self
            .texture_storage
            .read()
            .unwrap()
            .get(&texture_id)
            .cloned()
            .ok_or(TextureManagerError::TextureNotFound(texture_id))?;
        if (texture.width(), texture.height()) != texture_dimensions {
            return Err(TextureManagerError::InvalidTextureData(texture_id));
        }
        self.write_texture_region(
            texture_id,
            &texture,
            (0, 0),
            texture_dimensions,
            texture_data,
        )
    }

    /// Writes a texture region to the GPU queue.
    ///
    /// WGPU copies `texture_data` before this function returns. The transfer runs before the next
    /// queue submission, independently of any prepared scene.
    ///
    /// # Parameters
    /// - `texture_id`: Unique identifier for the texture.
    /// - `region_origin`: Top-left `(x, y)` coordinate of the region within the texture.
    /// - `region_dimensions`: Width and height of the region.
    /// - `texture_data`: Tightly packed RGBA8 sRGB data with premultiplied alpha for the region.
    pub fn load_data_into_texture_region(
        &self,
        texture_id: u64,
        region_origin: (u32, u32),
        region_dimensions: (u32, u32),
        texture_data: &[u8],
    ) -> Result<(), TextureManagerError> {
        let texture = self
            .texture_storage
            .read()
            .unwrap()
            .get(&texture_id)
            .cloned()
            .ok_or(TextureManagerError::TextureNotFound(texture_id))?;
        self.write_texture_region(
            texture_id,
            &texture,
            region_origin,
            region_dimensions,
            texture_data,
        )
    }

    /// Removes the texture identified by `texture_id` from the manager.
    pub fn remove_texture(&self, texture_id: u64) {
        let mut texture_storage = self.texture_storage.write().unwrap();
        let mut bind_group_cache = self.shape_bind_group_cache.write().unwrap();
        bind_group_cache
            .retain(|(cached_texture_id, _layout), _bind_group| *cached_texture_id != texture_id);
        texture_storage.remove(&texture_id);
    }

    fn write_texture_region(
        &self,
        texture_id: u64,
        texture: &wgpu::Texture,
        region_origin: (u32, u32),
        region_dimensions: (u32, u32),
        texture_data: &[u8],
    ) -> Result<(), TextureManagerError> {
        validate_texture_region(
            texture_id,
            (texture.width(), texture.height()),
            region_origin,
            region_dimensions,
            texture_data.len(),
        )?;
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: region_origin.0,
                    y: region_origin.1,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            texture_data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(region_dimensions.0 * 4),
                rows_per_image: Some(region_dimensions.1),
            },
            wgpu::Extent3d {
                width: region_dimensions.0,
                height: region_dimensions.1,
                depth_or_array_layers: 1,
            },
        );
        Ok(())
    }

    /// Creates a bind group for the provided `layout` using the stored sampler and
    /// the texture identified by `texture_id`.
    ///
    /// Returns a cached bind group for this layout and texture,
    /// creating and caching it if necessary. This avoids per-frame bind group creation
    /// when binding textures for shapes.
    pub(crate) fn get_or_create_shape_bind_group(
        &self,
        layout: &wgpu::BindGroupLayout,
        texture_id: u64,
    ) -> Result<Arc<wgpu::BindGroup>, TextureManagerError> {
        // Fast path: check cache
        if let Some(bg) = self
            .shape_bind_group_cache
            .read()
            .unwrap()
            .get(&(texture_id, layout.clone()))
            .cloned()
        {
            return Ok(bg);
        }

        // Create bind group
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

        // Insert into cache
        self.shape_bind_group_cache
            .write()
            .unwrap()
            .insert((texture_id, layout.clone()), bind_group.clone());

        Ok(bind_group)
    }

    pub(crate) fn retire_shape_bind_group_layout(&self, layout: &wgpu::BindGroupLayout) {
        self.shape_bind_group_cache
            .write()
            .unwrap()
            .retain(|(_, cached_layout), _| cached_layout != layout);
    }

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

// Converts an RGBA8 sRGB image in-place to premultiplied alpha.
// This operates in linear space for correct results:
// 1) convert sRGB to linear
// 2) multiply RGB by A
// 3) convert back to sRGB
// Alpha remains unchanged numerically in 0..1 mapped to 0..255.
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
        // keep alpha as-is
    }
}
