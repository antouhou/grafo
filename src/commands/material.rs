use super::IntermediateTextureId;
use crate::core::PhysicalRect;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ShapeTextureBinding {
    #[default]
    None,
    Managed(u64),
    Intermediate(IntermediateTextureId),
}

impl ShapeTextureBinding {
    pub fn is_present(&self) -> bool {
        !matches!(self, Self::None)
    }

    pub fn managed_texture_id(&self) -> Option<u64> {
        match self {
            Self::Managed(texture_id) => Some(*texture_id),
            Self::None | Self::Intermediate(_) => None,
        }
    }
}

/// Coordinates used to sample a shape material's texture.
#[derive(Clone, Copy, Debug, Default)]
pub enum TextureSampling {
    #[default]
    ShapeUv,
    /// Full-resolution physical bounds, independent of the texture's resolution.
    TargetPixels(PhysicalRect),
}

#[derive(Clone, Copy, Debug)]
pub struct ShapeTextureLayer {
    pub texture: ShapeTextureBinding,
    pub sampling: TextureSampling,
}

/// Value parameters and texture IDs for one draw. Material resources stay in execution.
#[derive(Clone, Copy, Debug)]
pub struct ShapeDrawMaterial {
    pub has_gradient_fill: bool,
    pub texture_bindings: [ShapeTextureBinding; 2],
    pub under_fill_texture: Option<ShapeTextureLayer>,
}

impl ShapeDrawMaterial {
    pub fn has_gradient_fill(self) -> bool {
        self.has_gradient_fill
    }
}
