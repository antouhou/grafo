use super::bindings::create_texture_sample_bind_group;
use ahash::{HashMap, HashMapExt};
use bucket::TextureBucket;
use wgpu::{
    BindGroup, BindGroupLayout, Device, Extent3d, Sampler, Texture, TextureDescriptor,
    TextureDimension, TextureFormat, TextureUsages, TextureView, TextureViewDescriptor,
};
mod bucket;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct TextureDescriptorKey {
    width: u32,
    height: u32,
    format: TextureFormat,
    sample_count: u32,
    with_depth: bool,
}

struct TextureSampleBinding {
    layout: BindGroupLayout,
    sampler: Sampler,
    bind_group: BindGroup,
}

fn prepare_sample_binding<'a>(
    binding: &'a mut Option<TextureSampleBinding>,
    device: &Device,
    layout: &BindGroupLayout,
    view: &TextureView,
    sampler: &Sampler,
) -> &'a BindGroup {
    if binding
        .as_ref()
        .is_none_or(|binding| binding.layout != *layout || binding.sampler != *sampler)
    {
        *binding = Some(TextureSampleBinding {
            layout: layout.clone(),
            sampler: sampler.clone(),
            bind_group: create_texture_sample_bind_group(
                device,
                layout,
                view,
                sampler,
                Some("pooled_texture_sample"),
            ),
        });
    }
    &binding
        .as_ref()
        .expect("texture sampling binding was prepared")
        .bind_group
}

/// A pooled offscreen texture with color, optional depth/stencil, and optional MSAA resolve
/// resources.
pub(crate) struct PooledTexture {
    pub texture_id: u64,
    descriptor: TextureDescriptorKey,
    input_binding: Option<TextureSampleBinding>,
    composite_binding: Option<TextureSampleBinding>,
    pub color_texture: Texture,
    pub color_view: TextureView,
    pub depth_stencil_view: Option<TextureView>,
    pub resolve_texture: Option<Texture>,
    pub resolve_view: Option<TextureView>,
    pub sample_count: u32,
}

impl PooledTexture {
    pub(crate) fn input_bind_group(
        &mut self,
        device: &Device,
        layout: &BindGroupLayout,
        sampler: &Sampler,
    ) -> &BindGroup {
        prepare_sample_binding(
            &mut self.input_binding,
            device,
            layout,
            self.resolve_view.as_ref().unwrap_or(&self.color_view),
            sampler,
        )
    }

    pub(crate) fn composite_bind_group(
        &mut self,
        device: &Device,
        layout: &BindGroupLayout,
        sampler: &Sampler,
    ) -> &BindGroup {
        prepare_sample_binding(
            &mut self.composite_binding,
            device,
            layout,
            &self.color_view,
            sampler,
        )
    }
}

/// Textures return after submission; descriptor buckets avoid a full texture scan.
pub(crate) struct OffscreenTexturePool {
    available: HashMap<TextureDescriptorKey, TextureBucket<PooledTexture>>,
    next_texture_id: u64,
}

impl OffscreenTexturePool {
    pub fn new() -> Self {
        Self {
            available: HashMap::new(),
            next_texture_id: 1,
        }
    }

    /// Retain the submitted working set and release surplus textures from earlier renders.
    pub fn recycle(&mut self, textures: &mut Vec<PooledTexture>) {
        for bucket in self.available.values_mut() {
            bucket.discard_unused();
        }
        for texture in textures.drain(..) {
            self.available
                .entry(texture.descriptor)
                .or_default()
                .recycle(texture.texture_id, texture);
        }
        self.available.retain(|_, bucket| !bucket.is_empty());
    }

    /// Viewport or MSAA changes invalidate transient target sizes and bindings.
    pub fn clear(&mut self) {
        self.available.clear();
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
        let descriptor = TextureDescriptorKey {
            width,
            height,
            format,
            sample_count,
            with_depth,
        };
        if let Some(texture) = self
            .available
            .get_mut(&descriptor)
            .and_then(TextureBucket::acquire)
        {
            texture
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
            descriptor: TextureDescriptorKey {
                width,
                height,
                format,
                sample_count,
                with_depth,
            },
            input_binding: None,
            composite_binding: None,
            color_texture,
            color_view,
            depth_stencil_view,
            resolve_texture,
            resolve_view,
            sample_count,
        }
    }
}

#[cfg(test)]
mod tests;
