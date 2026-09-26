//! A flat command stream passed from planning to execution.
//!
//! Plans store resource IDs, parameters, and resolved clips. Execution looks up
//! uploaded resources by ID after planning has finished.

use crate::core::effect::BackdropCaptureRegion;
use crate::core::vertex::{InstanceTransform, TextureUvTransform};
use crate::core::{Size, UnsignedPhysicalRect};
use ahash::RandomState;
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
pub enum RenderOperation {
    IncrementStencil(ShapeDrawId),
    DecrementStencil(ShapeDraw),
    DrawShape(ShapeDraw),
    DrawShapeAndIncrementStencil(ShapeDraw),
    CompositeTexture(TextureComposite),
    BeginTarget(Target),
    EndTarget,
    DrawShapeMask(ShapeMaskDraw),
    CaptureBackdrop(BackdropCapture),
    ApplyEffect(EffectApplication),
}

#[derive(Clone, Copy, Debug)]
pub struct RenderCommand {
    pub operation: RenderOperation,
    pub clip: DrawClip,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackdropCaptureSource {
    Target,
    Layered { base: IntermediateTextureId },
}

#[derive(Clone, Copy, Debug)]
pub struct BackdropCapture {
    pub source: BackdropCaptureSource,
    pub region: BackdropCaptureRegion,
    pub output: IntermediateTextureId,
    pub sampling_size: Size,
}

/// A byte range in the command stream's effect parameter storage.
#[derive(Clone, Copy, Debug)]
pub struct EffectParameterRange {
    pub start: usize,
    pub end: usize,
}

/// Parameters stored in the render plan, with content hashed when written.
#[derive(Clone, Copy, Debug)]
pub struct EffectParameters {
    pub range: EffectParameterRange,
    pub hash: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct EffectApplication {
    pub effect_id: u64,
    pub parameters: EffectParameters,
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

/// BeginTarget clears a new target. EndTarget restores its parent without clearing it.
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
    pub fringe_width: f32,
    pub downsample: f32,
}

/// One ordered command stream, with storage reused when the draw queue is rebuilt.
#[derive(Default)]
pub struct RenderPlan {
    pub instructions: Vec<RenderCommand>,
    pub effect_parameters: Vec<u8>,
    /// Local composite commands whose instance data must be uploaded.
    pub composite_draws: Vec<usize>,
    pub texture_count: usize,
    pub has_backdrop_captures: bool,
    #[cfg(feature = "render_metrics")]
    pub scissor_clip_count: u32,
    parameter_hasher: RandomState,
}

impl RenderPlan {
    pub fn clear(&mut self) {
        self.clear_commands();
        self.effect_parameters.clear();
    }

    /// Rebuild commands while preserving parameters written during queuing.
    pub(crate) fn clear_commands(&mut self) {
        self.instructions.clear();
        self.composite_draws.clear();
        self.texture_count = 0;
        self.has_backdrop_captures = false;
        #[cfg(feature = "render_metrics")]
        {
            self.scissor_clip_count = 0;
        }
    }

    pub fn store_parameters(&mut self, parameters: &[u8]) -> EffectParameters {
        let start = self.effect_parameters.len();
        self.effect_parameters.extend_from_slice(parameters);
        EffectParameters {
            range: EffectParameterRange {
                start,
                end: self.effect_parameters.len(),
            },
            hash: self.parameter_hasher.hash_one(parameters),
        }
    }

    pub(crate) fn update_parameters(
        &mut self,
        stored: EffectParameters,
        parameters: &[u8],
    ) -> EffectParameters {
        let range = stored.range;
        if range.end - range.start != parameters.len() {
            return self.store_parameters(parameters);
        }
        self.effect_parameters[range.start..range.end].copy_from_slice(parameters);
        EffectParameters {
            range,
            hash: self.parameter_hasher.hash_one(parameters),
        }
    }

    pub fn parameters(&self, parameters: EffectParameters) -> &[u8] {
        &self.effect_parameters[parameters.range.start..parameters.range.end]
    }

    pub fn allocate_texture(&mut self) -> IntermediateTextureId {
        let texture = IntermediateTextureId::Planned(self.texture_count);
        self.texture_count += 1;
        texture
    }

    pub fn push(&mut self, operation: RenderOperation) {
        self.push_command(RenderCommand {
            operation,
            clip: DrawClip::default(),
        });
    }

    pub fn push_composite(&mut self, composite: TextureComposite, clip: DrawClip) {
        self.push_command(RenderCommand {
            operation: RenderOperation::CompositeTexture(composite),
            clip,
        });
    }

    pub fn push_command(&mut self, instruction: RenderCommand) {
        self.has_backdrop_captures |=
            matches!(instruction.operation, RenderOperation::CaptureBackdrop(_));
        if matches!(
            instruction.operation,
            RenderOperation::CompositeTexture(TextureComposite {
                placement: TexturePlacement::Local { .. },
                ..
            })
        ) {
            self.composite_draws.push(self.instructions.len());
        }
        self.instructions.push(instruction);
    }
}
