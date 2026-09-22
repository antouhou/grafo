//! Draw commands passed from planning to execution.
//!
//! Plans store resource IDs, parameters, and resolved clips. Execution looks up
//! uploaded resources by ID after planning has finished.

use crate::shape::ShapeDrawMaterial;
use crate::{PhysicalRect, Size, UnsignedPhysicalPoint, UnsignedPhysicalRect};
use std::ops::Range;
pub(crate) use textures::IntermediateTextureId;

mod textures;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(in crate::renderer) struct BackdropCaptureRegion {
    /// Requested bounds in full-resolution physical pixels, including offscreen padding.
    pub(in crate::renderer) bounds: PhysicalRect,
    /// Viewport overlap to copy, or None when the requested bounds are fully offscreen.
    pub(in crate::renderer) source_rect: Option<UnsignedPhysicalRect>,
    pub(in crate::renderer) copy_destination_origin: UnsignedPhysicalPoint,
}

/// Identifies an uploaded shape instance or prepared effect leaf.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::renderer) enum ShapeDrawId {
    Shape(usize),
    EffectLeaf(usize),
}

/// Physical scissor bounds and stencil reference for one draw.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(in crate::renderer) struct DrawClip {
    pub(in crate::renderer) scissor: UnsignedPhysicalRect,
    pub(in crate::renderer) stencil_reference: u32,
}

#[derive(Clone, Copy, Debug)]
pub(in crate::renderer) struct ShapeDraw {
    pub(in crate::renderer) id: ShapeDrawId,
    pub(in crate::renderer) material: ShapeDrawMaterial,
}

#[derive(Clone, Copy, Debug)]
pub(in crate::renderer) enum DrawOperation {
    IncrementStencil(ShapeDrawId),
    DecrementStencil(ShapeDraw),
    DrawShape(ShapeDraw),
    DrawShapeAndIncrementStencil(ShapeDraw),
    CompositeTexture(IntermediateTextureId),
}

#[derive(Clone, Copy, Debug)]
pub(in crate::renderer) struct DrawInstruction {
    pub(in crate::renderer) operation: DrawOperation,
    pub(in crate::renderer) clip: DrawClip,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::renderer) enum BackdropCaptureSource {
    Target,
    Layered { base: IntermediateTextureId },
}

#[derive(Clone, Copy, Debug)]
pub(in crate::renderer) struct BackdropCapture {
    pub(in crate::renderer) source: BackdropCaptureSource,
    pub(in crate::renderer) region: BackdropCaptureRegion,
    pub(in crate::renderer) output: IntermediateTextureId,
    pub(in crate::renderer) sampling_size: Size,
}

#[derive(Clone, Debug)]
pub(in crate::renderer) struct EffectApplication {
    pub(in crate::renderer) effect_id: u64,
    pub(in crate::renderer) parameters: Range<usize>,
    pub(in crate::renderer) input: IntermediateTextureId,
    pub(in crate::renderer) output: IntermediateTextureId,
}

pub(in crate::renderer) enum DrawSegment {
    Draws {
        instructions: Range<usize>,
        texture_materials: Range<usize>,
    },
    CaptureBackdrop(BackdropCapture),
    ApplyEffect(EffectApplication),
}

/// Draw commands and effect parameters, with storage reused across renders.
#[derive(Default)]
pub(in crate::renderer) struct DrawPlan {
    pub(in crate::renderer) instructions: Vec<DrawInstruction>,
    pub(in crate::renderer) segments: Vec<DrawSegment>,
    pub(in crate::renderer) effect_parameters: Vec<u8>,
    /// Instruction indices requiring texture bindings before their draw pass opens.
    pub(in crate::renderer) texture_material_draws: Vec<usize>,
    pub(in crate::renderer) texture_count: usize,
    #[cfg(feature = "render_metrics")]
    pub(in crate::renderer) scissor_clip_count: u32,
}

impl DrawPlan {
    pub(in crate::renderer) fn clear(&mut self) {
        self.instructions.clear();
        self.segments.clear();
        self.effect_parameters.clear();
        self.texture_material_draws.clear();
        self.texture_count = 0;
        #[cfg(feature = "render_metrics")]
        {
            self.scissor_clip_count = 0;
        }
    }

    pub(in crate::renderer) fn allocate_texture(&mut self) -> IntermediateTextureId {
        let texture = IntermediateTextureId::Planned(self.texture_count);
        self.texture_count += 1;
        texture
    }

    pub(in crate::renderer) fn push_draw(&mut self, instruction: DrawInstruction) {
        let material_start = self.texture_material_draws.len();
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
        }) = self.segments.last_mut()
        {
            instructions.end = self.instructions.len();
            texture_materials.end = self.texture_material_draws.len();
        } else {
            self.segments.push(DrawSegment::Draws {
                instructions: self.instructions.len() - 1..self.instructions.len(),
                texture_materials: material_start..self.texture_material_draws.len(),
            });
        }
    }
}
