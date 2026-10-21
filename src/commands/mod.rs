//! A flat command stream passed from planning to execution.
//!
//! Commands carry resource IDs, borrowed parameters, and resolved clips.

use crate::core::effect::CaptureRegion;
use crate::core::vertex::{InstanceTransform, TextureUvTransform};
use crate::core::{Size, UnsignedPhysicalRect};
pub use material::{ShapeDrawMaterial, ShapeTextureBinding, ShapeTextureLayer, TextureSampling};
pub use textures::IntermediateTextureId;

mod material;
mod textures;

/// Identifies an uploaded shape instance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShapeDrawId(pub usize);

/// Physical scissor bounds and stencil reference for one draw.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DrawClip {
    pub scissor: UnsignedPhysicalRect,
    pub stencil_reference: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct ShapeDraw {
    pub id: ShapeDrawId,
    pub material: ShapeDrawMaterial,
}

#[derive(Clone, Copy, Debug)]
pub enum RenderOperation<'scene> {
    DrawShape(ShapeDraw),
    DrawShapeAndIncrementStencil(ShapeDraw),
    DecrementStencil(ShapeDraw),
    CompositeTexture(TextureComposite),
    BeginTarget(Target),
    EndTarget,
    DrawShapeMask(ShapeMaskDraw),
    Capture(CaptureConfig),
    ApplyEffect(EffectApplication<'scene>),
}

#[derive(Clone, Copy, Debug)]
pub struct RenderCommand<'scene> {
    pub operation: RenderOperation<'scene>,
    pub clip: DrawClip,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CaptureSource {
    /// Pixels already rendered into the active target
    Target,
    /// The active target composited over a separate base texture
    Layered { base: IntermediateTextureId },
}

#[derive(Clone, Copy, Debug)]
pub struct CaptureConfig {
    pub source: CaptureSource,
    pub region: CaptureRegion,
    pub output: IntermediateTextureId,
    pub sampling_size: Size,
}

/// Borrowed effect bytes, with their hash computed when attached to the scene.
#[derive(Clone, Copy, Debug)]
pub struct EffectParameters<'scene> {
    pub bytes: &'scene [u8],
    pub hash: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct EffectApplication<'scene> {
    pub effect_id: u64,
    pub parameters: EffectParameters<'scene>,
    pub input: IntermediateTextureId,
    pub output: IntermediateTextureId,
}

#[derive(Clone, Copy, Debug)]
pub enum TexturePlacement {
    Target,
    Local {
        transform: InstanceTransform,
        sampling: TextureUvTransform,
    },
}

#[derive(Clone, Copy, Debug)]
pub struct TextureComposite {
    pub texture: IntermediateTextureId,
    pub placement: TexturePlacement,
}

/// Transparent, linear premultiplied coverage mask at the requested sampling size.
#[derive(Clone, Copy, Debug)]
pub struct MaskTarget {
    pub texture: IntermediateTextureId,
    pub size: [u32; 2],
}

/// BeginTarget clears intermediate targets. The surface retains pixels outside the root scissor.
/// EndTarget restores its parent without clearing it.
#[derive(Clone, Copy, Debug)]
pub enum Target {
    Surface,
    Texture {
        texture: IntermediateTextureId,
        size: Size,
    },
    Mask(MaskTarget),
}

#[derive(Clone, Copy, Debug)]
pub struct ShapeMaskDraw {
    pub shape: ShapeDrawId,
    pub clip: DrawClip,
    pub local_physical_origin: [i32; 2],
    pub local_bounds: [(f32, f32); 2],
    pub scale_factor: f64,
    pub downsample: f32,
}
