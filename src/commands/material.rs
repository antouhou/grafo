use super::IntermediateTextureId;
use crate::core::PhysicalRect;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) enum ShapeTextureBinding {
    #[default]
    None,
    Managed(u64),
    Intermediate(IntermediateTextureId),
}

impl ShapeTextureBinding {
    pub(crate) fn is_present(&self) -> bool {
        !matches!(self, Self::None)
    }

    pub(crate) fn managed_texture_id(&self) -> Option<u64> {
        match self {
            Self::Managed(texture_id) => Some(*texture_id),
            Self::None | Self::Intermediate(_) => None,
        }
    }
}

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
