//! A flat command stream passed from planning to execution.
//!
//! Commands carry resource IDs, borrowed parameters, and resolved clips.

use crate::core::vertex::{InstanceTransform, TextureUvTransform};
use crate::core::{BlendMode, MathRect, PhysicalRect, Size, UnsignedPhysicalRect};
pub use material::{ShapeDrawMaterial, ShapeTextureBinding, ShapeTextureLayer, TextureSampling};
pub use textures::IntermediateTextureId;

mod material;
mod textures;

/// Identifies an uploaded shape instance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShapeDrawId(pub usize);

/// Scissor bounds in physical screen coordinates and a stencil reference for one draw.
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

/// Geometry that clips subsequent draws until the matching stencil decrement.
#[derive(Clone, Copy, Debug)]
pub enum ClipGeometry {
    Shape(ShapeDraw),
    /// Axis-aligned bounds in logical screen coordinates, with no visible paint.
    Rectangle(MathRect),
}

#[derive(Clone, Copy, Debug)]
pub struct ClipDraw {
    pub geometry: ClipGeometry,
    /// Descendant draws read or modify the existing destination color.
    pub requires_destination: bool,
    /// Descendants require a shared clip layer for effects or perspective rasterization.
    pub requires_layer: bool,
}

#[derive(Clone, Copy, Debug)]
pub enum RenderOperation<'scene> {
    DrawShape(ShapeDraw),
    DrawShapeAndIncrementStencil(ClipDraw),
    DecrementStencil(ClipDraw),
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
    /// The active target composited over a base texture at its physical screen origin.
    Layered {
        base: IntermediateTextureId,
        origin: [i32; 2],
    },
}

#[derive(Clone, Copy, Debug)]
pub struct CaptureConfig {
    pub source: CaptureSource,
    /// Base-texture pixels to copy in physical screen coordinates. `None` skips the copy.
    pub source_rect: Option<UnsignedPhysicalRect>,
    /// Physical screen rectangle represented by the full-resolution destination, including padding.
    pub destination_bounds: PhysicalRect,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlendDestination {
    CurrentTarget,
    // Full-resolution destination pixels in the placement's original UV space.
    Texture(IntermediateTextureId),
}

#[derive(Clone, Copy, Debug)]
pub struct TextureComposite {
    pub source: IntermediateTextureId,
    pub placement: TexturePlacement,
    pub blend_mode: BlendMode,
    pub destination: BlendDestination,
}

impl TextureComposite {
    pub fn requires_destination_texture(&self) -> bool {
        match self.blend_mode {
            BlendMode::Overlay
            | BlendMode::Darken
            | BlendMode::Lighten
            | BlendMode::ColorDodge
            | BlendMode::ColorBurn
            | BlendMode::HardLight
            | BlendMode::SoftLight
            | BlendMode::Difference
            | BlendMode::Multiply
            | BlendMode::Hue
            | BlendMode::Saturation
            | BlendMode::Color
            | BlendMode::Luminosity => true,
            BlendMode::Clear
            | BlendMode::Source
            | BlendMode::Destination
            | BlendMode::SourceOver
            | BlendMode::DestinationOver
            | BlendMode::SourceIn
            | BlendMode::DestinationIn
            | BlendMode::SourceOut
            | BlendMode::DestinationOut
            | BlendMode::SourceAtop
            | BlendMode::DestinationAtop
            | BlendMode::Xor
            | BlendMode::Plus
            | BlendMode::Screen
            | BlendMode::Exclusion => false,
        }
    }
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
        /// Full-resolution physical screen rectangle represented by the target texture.
        bounds: PhysicalRect,
        /// Permitted drawing area in physical screen coordinates. Padding stays transparent.
        draw_scissor: UnsignedPhysicalRect,
        /// Content filters receive the source image before ancestor clipping.
        is_content_effect_input: bool,
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
