use wgpu::{
    Device, Extent3d, Texture, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
    TextureView, TextureViewDescriptor,
};
/// A pooled offscreen texture with color, optional depth/stencil, and optional MSAA resolve
/// resources.
pub(in crate::renderer) struct PooledTexture {
    pub texture_id: u64,
    pub color_texture: Texture,
    pub color_view: TextureView,
    pub depth_stencil_view: Option<TextureView>,
    pub resolve_texture: Option<Texture>,
    pub resolve_view: Option<TextureView>,
    pub width: u32,
    pub height: u32,
    pub sample_count: u32,
}

/// Pool of reusable offscreen textures for effect compositing.
/// Textures return to the pool after render submission.
pub(in crate::renderer) struct OffscreenTexturePool {
    available: Vec<PooledTexture>,
    next_texture_id: u64,
}

/// Maximum number of textures to keep in the pool.
const MAX_POOL_SIZE: usize = 8;

impl OffscreenTexturePool {
    pub fn new() -> Self {
        Self {
            available: Vec::new(),
            next_texture_id: 1,
        }
    }

    /// Return textures to the pool and discard entries beyond `MAX_POOL_SIZE`.
    pub fn recycle(&mut self, textures: &mut Vec<PooledTexture>) {
        self.available.append(textures);
        self.available.truncate(MAX_POOL_SIZE);
    }

    /// Retain textures matching the dimensions and sample count, capped at `MAX_POOL_SIZE`.
    pub fn trim(&mut self, width: u32, height: u32, sample_count: u32) {
        self.available.retain(|texture| {
            texture.width == width
                && texture.height == height
                && texture.sample_count == sample_count
        });
        if self.available.len() > MAX_POOL_SIZE {
            self.available.truncate(MAX_POOL_SIZE);
        }
    }

    /// Acquire a texture matching the given dimensions and sample count, plus a depth/stencil
    /// attachment for render passes that write depth or stencil.
    pub fn acquire_with_depth(
        &mut self,
        device: &Device,
        width: u32,
        height: u32,
        format: TextureFormat,
        sample_count: u32,
    ) -> PooledTexture {
        self.acquire(device, width, height, format, sample_count, true)
    }

    /// Acquire a color-only texture matching the given dimensions and sample count.
    pub fn acquire_color_only(
        &mut self,
        device: &Device,
        width: u32,
        height: u32,
        format: TextureFormat,
        sample_count: u32,
    ) -> PooledTexture {
        self.acquire(device, width, height, format, sample_count, false)
    }

    fn acquire(
        &mut self,
        device: &Device,
        width: u32,
        height: u32,
        format: TextureFormat,
        sample_count: u32,
        with_depth: bool,
    ) -> PooledTexture {
        let found = self.available.iter().position(|texture| {
            texture.width == width
                && texture.height == height
                && texture.sample_count == sample_count
                && texture.depth_stencil_view.is_some() == with_depth
        });

        if let Some(texture_index) = found {
            self.available.swap_remove(texture_index)
        } else {
            self.create_pooled_texture(device, width, height, format, sample_count, with_depth)
        }
    }

    fn create_pooled_texture(
        &mut self,
        device: &Device,
        width: u32,
        height: u32,
        format: TextureFormat,
        sample_count: u32,
        with_depth: bool,
    ) -> PooledTexture {
        let texture_id = self.next_texture_id;
        self.next_texture_id += 1;

        let color_texture = device.create_texture(&TextureDescriptor {
            label: Some("effect_offscreen_color"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count,
            dimension: TextureDimension::D2,
            format,
            usage: TextureUsages::RENDER_ATTACHMENT
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_SRC
                | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let color_view = color_texture.create_view(&TextureViewDescriptor::default());

        let depth_stencil_view = with_depth.then(|| {
            let depth_stencil_texture = device.create_texture(&TextureDescriptor {
                label: Some("effect_offscreen_depth_stencil"),
                size: Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count,
                dimension: TextureDimension::D2,
                format: TextureFormat::Depth24PlusStencil8,
                usage: TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            depth_stencil_texture.create_view(&TextureViewDescriptor::default())
        });

        // Effect shaders sample the resolved texture when MSAA is enabled.
        let (resolve_texture, resolve_view) = if sample_count > 1 {
            let resolve_texture = device.create_texture(&TextureDescriptor {
                label: Some("effect_offscreen_resolve"),
                size: Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format,
                usage: TextureUsages::RENDER_ATTACHMENT
                    | TextureUsages::TEXTURE_BINDING
                    | TextureUsages::COPY_SRC
                    | TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let resolve_view = resolve_texture.create_view(&TextureViewDescriptor::default());
            (Some(resolve_texture), Some(resolve_view))
        } else {
            (None, None)
        };

        PooledTexture {
            texture_id,
            color_texture,
            color_view,
            depth_stencil_view,
            resolve_texture,
            resolve_view,
            width,
            height,
            sample_count,
        }
    }
}
