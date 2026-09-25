use super::super::effects::PooledTexture;
use crate::core::cache::{CachedTessellation, FrameCache};
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use wgpu::TextureFormat;

/// Identifies a mask by geometry and rasterization settings, allowing reuse across
/// effects and parameter changes.
#[derive(Clone)]
pub(crate) struct ShapeEffectMaskCacheKey {
    pub tessellation: Arc<CachedTessellation>,
    /// Shape-local raster origin in physical pixels.
    /// Node transforms do not invalidate this entry.
    pub local_raster_origin: [i32; 2],
    pub raster_size: [u32; 2],
    pub scale_factor_bits: u64,
    pub fringe_width_bits: u32,
    pub downsample_bits: u32,
    pub texture_format: TextureFormat,
}

impl PartialEq for ShapeEffectMaskCacheKey {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.tessellation, &other.tessellation)
            && self.local_raster_origin == other.local_raster_origin
            && self.raster_size == other.raster_size
            && self.scale_factor_bits == other.scale_factor_bits
            && self.fringe_width_bits == other.fringe_width_bits
            && self.downsample_bits == other.downsample_bits
            && self.texture_format == other.texture_format
    }
}

impl Eq for ShapeEffectMaskCacheKey {}

impl Hash for ShapeEffectMaskCacheKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (Arc::as_ptr(&self.tessellation) as usize).hash(state);
        self.local_raster_origin.hash(state);
        self.raster_size.hash(state);
        self.scale_factor_bits.hash(state);
        self.fringe_width_bits.hash(state);
        self.downsample_bits.hash(state);
        self.texture_format.hash(state);
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) struct ShapeEffectCacheKey {
    pub mask_key: ShapeEffectMaskCacheKey,
    pub effect_id: u64,
    /// Content hash computed when parameter bytes are stored in the render plan.
    pub parameters_hash: u64,
}

pub(crate) struct CachedShapeEffectMask {
    pub texture: PooledTexture,
}

pub(crate) type ShapeEffectMaskCache = FrameCache<ShapeEffectMaskCacheKey, CachedShapeEffectMask>;
