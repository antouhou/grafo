//! A flat command stream passed from planning to execution.
//!
//! Plans store resource IDs, parameters, and resolved clips. Execution looks up
//! uploaded resources by ID after planning has finished.

use crate::core::effect::BackdropCaptureRegion;
use crate::core::vertex::{InstanceTransform, TextureUvTransform};
use crate::core::{Size, UnsignedPhysicalRect};
pub use material::{ShapeDrawMaterial, ShapeTextureBinding, ShapeTextureLayer, TextureSampling};
use std::sync::Arc;
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

#[derive(Clone, Copy, Debug)]
pub struct EffectApplication {
    pub effect_id: u64,
    pub parameters: EffectParameters,
    pub input: IntermediateTextureId,
    pub output: IntermediateTextureId,
}

/// Parameter storage is owned by the command stream. Shared bytes avoid copying
/// immutable shape-effect parameters on cache hits and queue rebuilds.
#[derive(Clone, Copy, Debug)]
pub enum EffectParameters {
    Bytes { start: usize, end: usize },
    Shared(usize),
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
    /// Keep shared ownership outside Copy commands so clearing draws is constant time.
    pub shared_effect_parameters: Vec<Arc<[u8]>>,
    /// Local composite commands whose instance data must be uploaded.
    pub composite_draws: Vec<usize>,
    pub texture_count: usize,
    pub has_backdrop_captures: bool,
    #[cfg(feature = "render_metrics")]
    pub scissor_clip_count: u32,
}

impl RenderPlan {
    pub fn clear(&mut self) {
        self.instructions.clear();
        self.effect_parameters.clear();
        self.shared_effect_parameters.clear();
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
        EffectParameters::Bytes {
            start,
            end: self.effect_parameters.len(),
        }
    }

    pub fn share_parameters(&mut self, parameters: &Arc<[u8]>) -> EffectParameters {
        let index = self.shared_effect_parameters.len();
        self.shared_effect_parameters.push(Arc::clone(parameters));
        EffectParameters::Shared(index)
    }

    pub fn parameters(&self, parameters: EffectParameters) -> &[u8] {
        match parameters {
            EffectParameters::Bytes { start, end } => &self.effect_parameters[start..end],
            EffectParameters::Shared(index) => &self.shared_effect_parameters[index],
        }
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
