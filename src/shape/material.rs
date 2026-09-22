use super::ShapeTextureBinding;
use crate::gradient::types::Fill;
use crate::PhysicalRect;

/// Coordinates used to sample a shape material's texture.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) enum TextureSampling {
    #[default]
    ShapeUv,
    /// Full-resolution physical bounds, independent of the texture's resolution.
    TargetPixels(PhysicalRect),
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ShapeTextureLayer {
    pub(crate) texture: ShapeTextureBinding,
    pub(crate) sampling: TextureSampling,
}

/// Borrows scene materials and adds generated textures for one draw.
#[derive(Clone, Copy)]
pub(crate) struct ShapeDrawMaterial<'a> {
    pub(crate) fill: Option<&'a Fill>,
    pub(crate) texture_bindings: &'a [ShapeTextureBinding; 2],
    pub(crate) under_fill_texture: Option<ShapeTextureLayer>,
}

impl ShapeDrawMaterial<'_> {
    pub(crate) fn has_gradient_fill(self) -> bool {
        matches!(self.fill, Some(Fill::Gradient(_)))
    }
}
