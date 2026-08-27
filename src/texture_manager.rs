use std::collections::HashMap;
use std::sync::{Arc, RwLock};

#[derive(Debug, thiserror::Error)]
pub enum TextureManagerError {
    #[error("Texture {0} not found")]
    TextureNotFound(u64),
    #[error("Texture {0} upload dimensions or byte length are invalid")]
    InvalidUpload(u64),
}

fn validate_upload(
    texture_id: u64,
    texture_dimensions: (u32, u32),
    upload_dimensions: (u32, u32),
    byte_length: usize,
    reset: bool,
) -> Result<(), TextureManagerError> {
    let expected_byte_length = upload_dimensions
        .0
        .try_into()
        .ok()
        .and_then(|width: usize| {
            usize::try_from(upload_dimensions.1)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .and_then(|pixels| pixels.checked_mul(4));
    if upload_dimensions != texture_dimensions
        || (!reset && expected_byte_length != Some(byte_length))
    {
        return Err(TextureManagerError::InvalidUpload(texture_id));
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
    /// Textures is raw image data, without any screen position information
    texture_storage: Arc<RwLock<HashMap<u64, wgpu::Texture>>>,
    /// Each entry retains its layout until that layout is explicitly retired.
    shape_bind_group_cache: Arc<RwLock<BindGroupCache>>,
    uploads: Arc<RwLock<Vec<TextureUpload>>>,
}

struct TextureUpload {
    texture_id: u64,
    texture: Option<wgpu::Texture>,
    bytes: Vec<u8>,
    bytes_per_row: u32,
    buffer: Option<wgpu::Buffer>,
    pending: bool,
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
            uploads: Arc::new(RwLock::new(Vec::new())),
        }
    }

    pub fn clear(&self) {
        self.texture_storage.write().unwrap().clear();
        self.shape_bind_group_cache.write().unwrap().clear();
        for upload in self.uploads.write().unwrap().iter_mut() {
            upload.texture = None;
            upload.bytes.clear();
            upload.pending = false;
        }
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
        let existing = self
            .texture_storage
            .read()
            .unwrap()
            .get(&texture_id)
            .cloned();
        if let Some(texture) = existing.filter(|texture| {
            texture.width() == texture_dimensions.0 && texture.height() == texture_dimensions.1
        }) {
            self.stage_upload(texture_id, &texture, texture_dimensions, &[], true)
                .expect("valid allocated texture");
            return;
        }
        let mut bind_group_cache = self.shape_bind_group_cache.write().unwrap();
        // If the binding cache contains entries for this texture_id, remove them
        // as the texture is being re-allocated, and the old bind groups are no longer valid.
        bind_group_cache
            .retain(|(cached_texture_id, _shape_id), _bind_group| *cached_texture_id != texture_id);

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

        self.texture_storage
            .write()
            .unwrap()
            .insert(texture_id, texture.clone());
        self.stage_upload(texture_id, &texture, texture_dimensions, &[], true)
            .expect("valid allocated texture");
    }

    /// Allocates a texture and stages image data for the next committed scene.
    ///
    /// This function will first allocate the texture, then attempt to load the provided data.
    ///
    /// If you are seeing fringes when sampling/minifying near transparent edges, ensure that your
    /// texture data is in a premultiplied alpha format. You can use the
    /// `premultiply_rgba8_srgb_inplace` helper function provided in this crate to convert your
    /// RGBA8 sRGB data to premultiplied alpha.
    ///
    /// # Parameters
    /// - `texture_id`: Unique identifier for the texture.
    /// - `texture_dimensions`: A tuple `(width, height)` representing the dimensions of the texture.
    /// - `texture_data`: A byte slice containing the image data. The data length is expected to
    ///   match the texture dimensions and pixel format (RGBA8 with premultiplied alpha).
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

    /// Stages image data for an already allocated texture. If you are seeing fringes when
    /// sampling/minifying near transparent edges, ensure that your texture data is in a
    /// premultiplied alpha format. You can use the `premultiply_rgba8_srgb_inplace` helper
    /// function provided in this crate to convert your RGBA8 sRGB data to premultiplied alpha.
    ///
    /// # Parameters
    /// - `texture_id`: Unique identifier for the texture.
    /// - `texture_dimensions`: A tuple `(width, height)` representing the dimensions of the texture.
    /// - `texture_data`: A byte slice containing the image data in an RGBA8 format with premultiplied alpha.
    ///   If your texture isn't premultiplied, consider using a `premultiply_rgba8_srgb_inplace` helper
    ///   function provided in this crate. This is needed to avoid fringes when sampling/minifying near transparent edges.
    ///
    /// # Returns
    /// - `Ok(())` if the operation succeeds.
    /// - `Err(TextureManagerError::TextureNotFound(texture_id))` if the texture does not exist.
    /// - `Err(TextureManagerError::InvalidUpload(texture_id))` if dimensions or byte length do not
    ///   exactly match the allocated RGBA8 texture.
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

        self.stage_upload(texture_id, texture, texture_dimensions, texture_data, false)
    }

    /// Removes the texture identified by `texture_id` from the manager.
    pub fn remove_texture(&self, texture_id: u64) {
        let mut bind_group_cache = self.shape_bind_group_cache.write().unwrap();
        // If the binding cache contains entries for this texture_id, remove them
        // as the texture is being removed, and the old bind groups are no longer valid.
        bind_group_cache
            .retain(|(cached_texture_id, _shape_id), _bind_group| *cached_texture_id != texture_id);

        self.texture_storage.write().unwrap().remove(&texture_id);
        if let Some(upload) = self
            .uploads
            .write()
            .unwrap()
            .iter_mut()
            .find(|upload| upload.texture_id == texture_id)
        {
            upload.texture = None;
            upload.bytes.clear();
            upload.pending = false;
        }
    }

    fn stage_upload(
        &self,
        texture_id: u64,
        texture: &wgpu::Texture,
        dimensions: (u32, u32),
        bytes: &[u8],
        reset: bool,
    ) -> Result<(), TextureManagerError> {
        validate_upload(
            texture_id,
            (texture.width(), texture.height()),
            dimensions,
            bytes.len(),
            reset,
        )?;
        let source_row_length = dimensions.0 as usize * 4;
        let mut uploads = self.uploads.write().unwrap();
        let index = uploads
            .iter()
            .position(|upload| upload.texture_id == texture_id)
            .or_else(|| uploads.iter().position(|upload| upload.texture.is_none()))
            .unwrap_or_else(|| {
                uploads.push(TextureUpload {
                    texture_id,
                    texture: None,
                    bytes: Vec::new(),
                    bytes_per_row: 0,
                    buffer: None,
                    pending: false,
                });
                uploads.len() - 1
            });
        let upload = &mut uploads[index];
        upload.texture_id = texture_id;
        let bytes_per_row = (texture.width() * 4).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        upload
            .bytes
            .resize(bytes_per_row as usize * texture.height() as usize, 0);
        if reset {
            upload.bytes.fill(0);
        } else {
            for row in 0..dimensions.1 as usize {
                let destination_start = row * bytes_per_row as usize;
                upload.bytes[destination_start..destination_start + source_row_length]
                    .copy_from_slice(
                        &bytes[row * source_row_length..(row + 1) * source_row_length],
                    );
            }
        }
        upload.texture = Some(texture.clone());
        upload.bytes_per_row = bytes_per_row;
        upload.pending = true;
        Ok(())
    }

    pub(crate) fn restore_pending_uploads(&self) {
        for upload in self.uploads.write().unwrap().iter_mut() {
            upload.pending = upload.texture.is_some();
        }
    }

    /// Encodes asset writes only into the selected scene's command buffer.
    pub(crate) fn encode_uploads(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut uploads = self.uploads.write().unwrap();
        for upload in uploads.iter_mut().filter(|upload| upload.pending) {
            let Some(texture) = &upload.texture else {
                continue;
            };
            let size = upload.bytes.len() as u64;
            if upload
                .buffer
                .as_ref()
                .is_none_or(|buffer| buffer.size() < size)
            {
                upload.buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("texture upload storage"),
                    size,
                    usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
            }
            let buffer = upload
                .buffer
                .as_ref()
                .expect("texture upload storage allocated");
            self.queue.write_buffer(buffer, 0, &upload.bytes);
            encoder.copy_buffer_to_texture(
                wgpu::TexelCopyBufferInfo {
                    buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(upload.bytes_per_row),
                        rows_per_image: Some(texture.height()),
                    },
                },
                wgpu::TexelCopyTextureInfo {
                    texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                texture.size(),
            );
            upload.pending = false;
        }
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

#[cfg(test)]
mod tests {
    use super::{validate_upload, TextureManagerError};

    #[test]
    fn texture_upload_requires_exact_dimensions_and_byte_length() {
        assert!(validate_upload(7, (4, 3), (4, 3), 48, false).is_ok());
        assert!(matches!(
            validate_upload(7, (4, 3), (3, 3), 36, false),
            Err(TextureManagerError::InvalidUpload(7))
        ));
        assert!(matches!(
            validate_upload(7, (4, 3), (4, 3), 47, false),
            Err(TextureManagerError::InvalidUpload(7))
        ));
        assert!(matches!(
            validate_upload(7, (4, 3), (4, 3), 49, false),
            Err(TextureManagerError::InvalidUpload(7))
        ));
        assert!(validate_upload(7, (4, 3), (4, 3), 0, true).is_ok());
        assert!(matches!(
            validate_upload(7, (4, 3), (3, 3), 0, true),
            Err(TextureManagerError::InvalidUpload(7))
        ));
    }
}
