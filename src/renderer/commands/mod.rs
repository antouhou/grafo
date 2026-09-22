//! Draw commands passed from planning to execution.
//!
//! Plans store resource IDs, parameters, and resolved clips. Execution looks up
//! uploaded resources by ID after planning has finished.

use crate::shape::ShapeDrawMaterial;
use crate::{PhysicalRect, UnsignedPhysicalPoint, UnsignedPhysicalRect};
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
    DrawShape(ShapeDraw),
    DrawShapeAndIncrementStencil(ShapeDraw),
    DecrementStencil(ShapeDraw),
    CompositeTexture(IntermediateTextureId),
}

#[derive(Clone, Copy, Debug)]
pub(in crate::renderer) struct DrawInstruction {
    pub(in crate::renderer) operation: DrawOperation,
    pub(in crate::renderer) clip: DrawClip,
}

/// Capture, effect, and shape operands resolved at one backdrop boundary.
#[derive(Clone, Copy)]
pub(in crate::renderer) struct BackdropDraw {
    pub(in crate::renderer) draw: ShapeDraw,
    pub(in crate::renderer) effect_id: u64,
    pub(in crate::renderer) parameter_start: usize,
    pub(in crate::renderer) parameter_end: usize,
    pub(in crate::renderer) downsample: f32,
    pub(in crate::renderer) capture: Option<BackdropCaptureRegion>,
    pub(in crate::renderer) parent_clip: DrawClip,
    pub(in crate::renderer) shape_clip: DrawClip,
    pub(in crate::renderer) decrements_stencil: bool,
}

pub(in crate::renderer) enum DrawSegment {
    Draws(Range<usize>),
    Backdrop(BackdropDraw),
}

/// Draw commands and effect parameters, with storage reused across renders.
#[derive(Default)]
pub(in crate::renderer) struct DrawPlan {
    pub(in crate::renderer) instructions: Vec<DrawInstruction>,
    pub(in crate::renderer) segments: Vec<DrawSegment>,
    pub(in crate::renderer) effect_parameters: Vec<u8>,
    #[cfg(feature = "render_metrics")]
    pub(in crate::renderer) scissor_clip_count: u32,
}
