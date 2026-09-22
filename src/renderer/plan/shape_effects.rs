use crate::shape::CachedShapeDrawData;

#[derive(Copy, Clone, Debug, PartialEq)]
pub(in crate::renderer) struct ShapeEffectRasterRect {
    /// Shape-local origin scaled to physical pixels. Node transforms apply later,
    /// so moving the shape on screen does not change this value.
    pub local_physical_origin: [i32; 2],
    /// Mask/effect texture size in texels. Smaller than the full-resolution
    /// physical extent when the effect config downsamples the rasterization.
    pub texture_size: [u32; 2],
    pub local_bounds: [(f32, f32); 2],
}

pub(in crate::renderer) struct PreparedShapeEffectLeaf {
    pub(in crate::renderer) draw_data: CachedShapeDrawData,
    pub(in crate::renderer) raster_rect: ShapeEffectRasterRect,
}
