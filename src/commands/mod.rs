//! Draw commands passed from planning to execution.
//!
//! Plans store resource IDs, parameters, and resolved clips. Execution looks up
//! uploaded resources by ID after planning has finished.

use crate::core::vertex::{InstanceTransform, TextureUvTransform};
use crate::core::{PhysicalRect, Size, UnsignedPhysicalPoint, UnsignedPhysicalRect};
pub(crate) use material::{
    ShapeDrawMaterial, ShapeTextureBinding, ShapeTextureLayer, TextureSampling,
};
use std::ops::Range;
use std::sync::Arc;
pub(crate) use textures::IntermediateTextureId;

mod material;
mod textures;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) struct BackdropCaptureRegion {
    /// Requested bounds in full-resolution physical pixels, including offscreen padding.
    pub(crate) bounds: PhysicalRect,
    /// Viewport overlap to copy, or None when the requested bounds are fully offscreen.
    pub(crate) source_rect: Option<UnsignedPhysicalRect>,
    pub(crate) copy_destination_origin: UnsignedPhysicalPoint,
}

/// Identifies an uploaded shape instance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ShapeDrawId(pub usize);

/// Physical scissor bounds and stencil reference for one draw.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct DrawClip {
    pub(crate) scissor: UnsignedPhysicalRect,
    pub(crate) stencil_reference: u32,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ShapeDraw {
    pub(crate) id: ShapeDrawId,
    pub(crate) material: ShapeDrawMaterial,
}

pub(crate) type TextureCompositeId = usize;

#[derive(Clone, Copy, Debug)]
pub(crate) enum DrawOperation {
    IncrementStencil(ShapeDrawId),
    DecrementStencil(ShapeDraw),
    DrawShape(ShapeDraw),
    DrawShapeAndIncrementStencil(ShapeDraw),
    CompositeTexture(TextureCompositeId),
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct DrawInstruction {
    pub(crate) operation: DrawOperation,
    pub(crate) clip: DrawClip,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BackdropCaptureSource {
    Target,
    Layered { base: IntermediateTextureId },
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct BackdropCapture {
    pub(crate) source: BackdropCaptureSource,
    pub(crate) region: BackdropCaptureRegion,
    pub(crate) output: IntermediateTextureId,
    pub(crate) sampling_size: Size,
}

#[derive(Clone, Debug)]
pub(crate) struct EffectApplication {
    pub(crate) effect_id: u64,
    pub(crate) parameters: EffectParameters,
    pub(crate) input: IntermediateTextureId,
    pub(crate) output: IntermediateTextureId,
}

/// Parameter storage is owned by the command stream. Shared bytes avoid copying
/// immutable shape-effect parameters on cache hits and queue rebuilds.
#[derive(Clone, Debug)]
pub(crate) enum EffectParameters {
    Bytes(Range<usize>),
    Shared(Arc<[u8]>),
}

impl EffectParameters {
    pub fn bytes<'a>(&'a self, storage: &'a [u8]) -> &'a [u8] {
        match self {
            Self::Bytes(range) => &storage[range.clone()],
            Self::Shared(bytes) => bytes,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum TexturePlacement {
    Target,
    Local {
        transform: InstanceTransform,
        sampling: TextureUvTransform,
    },
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct TextureComposite {
    pub texture: IntermediateTextureId,
    pub placement: TexturePlacement,
}

/// Transparent, linear premultiplied coverage mask at the requested sampling size.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MaskTarget {
    pub texture: IntermediateTextureId,
    pub size: [u32; 2],
}

/// Scene targets use transparent color and stencil zero at the start of each scope.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Target {
    Surface,
    Texture {
        texture: IntermediateTextureId,
        size: Size,
    },
    Mask(MaskTarget),
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ShapeMaskDraw {
    pub shape: ShapeDrawId,
    pub clip: DrawClip,
    pub local_physical_origin: [i32; 2],
    pub local_bounds: [(f32, f32); 2],
    pub scale_factor: f64,
    pub fringe_width: f32,
    pub downsample: f32,
}

pub(crate) enum DrawSegment {
    BeginTarget(Target),
    DrawShapeMask(ShapeMaskDraw),
    EndTarget,
    Draws {
        instructions: Range<usize>,
        texture_materials: Range<usize>,
        composites: Range<usize>,
    },
    CaptureBackdrop(BackdropCapture),
    ApplyEffect(EffectApplication),
}

/// Draw commands and effect parameters, with storage reused across renders.
#[derive(Default)]
pub(crate) struct DrawPlan {
    pub(crate) instructions: Vec<DrawInstruction>,
    pub(crate) segments: Vec<DrawSegment>,
    pub(crate) effect_parameters: Vec<u8>,
    /// Instruction indices requiring texture bindings before their draw pass opens.
    pub(crate) texture_material_draws: Vec<usize>,
    pub(crate) texture_count: usize,
    pub(crate) has_backdrop_captures: bool,
    pub(crate) composite_draws: Vec<usize>,
    pub(crate) composites: Vec<TextureComposite>,
    #[cfg(feature = "render_metrics")]
    pub(crate) scissor_clip_count: u32,
}

impl DrawPlan {
    pub(crate) fn clear(&mut self) {
        self.instructions.clear();
        self.segments.clear();
        self.effect_parameters.clear();
        self.texture_material_draws.clear();
        self.texture_count = 0;
        self.has_backdrop_captures = false;
        self.composite_draws.clear();
        self.composites.clear();
        #[cfg(feature = "render_metrics")]
        {
            self.scissor_clip_count = 0;
        }
    }

    pub(crate) fn allocate_texture(&mut self) -> IntermediateTextureId {
        let texture = IntermediateTextureId::Planned(self.texture_count);
        self.texture_count += 1;
        texture
    }

    pub(crate) fn push_composite(&mut self, composite: TextureComposite, clip: DrawClip) {
        let index = self.composites.len();
        self.composites.push(composite);
        self.push_draw(DrawInstruction {
            operation: DrawOperation::CompositeTexture(index),
            clip,
        });
    }

    pub(crate) fn push_draw(&mut self, instruction: DrawInstruction) {
        let material_start = self.texture_material_draws.len();
        let composite_start = self.composite_draws.len();
        if let DrawOperation::CompositeTexture(index) = instruction.operation {
            if matches!(
                self.composites[index].placement,
                TexturePlacement::Local { .. }
            ) {
                self.composite_draws.push(self.instructions.len());
            }
        }
        if matches!(instruction.operation, DrawOperation::DrawShape(draw)
            | DrawOperation::DrawShapeAndIncrementStencil(draw)
            if draw.material.under_fill_texture.is_some())
        {
            self.texture_material_draws.push(self.instructions.len());
        }
        self.instructions.push(instruction);
        if let Some(DrawSegment::Draws {
            instructions,
            texture_materials,
            composites,
        }) = self.segments.last_mut()
        {
            instructions.end = self.instructions.len();
            texture_materials.end = self.texture_material_draws.len();
            composites.end = self.composite_draws.len();
        } else {
            self.segments.push(DrawSegment::Draws {
                instructions: self.instructions.len() - 1..self.instructions.len(),
                texture_materials: material_start..self.texture_material_draws.len(),
                composites: composite_start..self.composite_draws.len(),
            });
        }
    }
}

/// Completed commands for shape-effect masks followed by the scene's target scopes.
#[derive(Default)]
pub struct RenderPlan {
    pub(crate) shape_effects: DrawPlan,
    pub(crate) scene: DrawPlan,
}
