use super::ShapeTextureBinding;
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

/// Value parameters and texture IDs for one draw. Material resources stay in execution.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ShapeDrawMaterial {
    pub(crate) has_gradient_fill: bool,
    pub(crate) texture_bindings: [ShapeTextureBinding; 2],
    pub(crate) under_fill_texture: Option<ShapeTextureLayer>,
}

impl ShapeDrawMaterial {
    pub(crate) fn has_gradient_fill(self) -> bool {
        self.has_gradient_fill
    }
}
