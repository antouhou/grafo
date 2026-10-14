use crate::commands::{
    CaptureConfig, CaptureSource, EffectApplication, IntermediateTextureId, RenderOperation,
    RenderPlan, TextureComposite, TexturePlacement,
};
use crate::core::effect::GroupEffectBounds;
use crate::core::geometry;
use crate::core::vertex::TextureUvTransform;
use crate::planner::TextureIdAllocator;
use crate::scene::effects::{EffectInstance, GroupEffectInstance};

#[derive(Clone, Copy)]
pub(crate) struct GroupCapture {
    effect: EffectInstance,
    bounds: GroupEffectBounds,
}

impl GroupCapture {
    pub(crate) fn new(effect: GroupEffectInstance) -> Option<Self> {
        Some(Self {
            effect: EffectInstance {
                effect_id: effect.effect_id,
                parameters: effect.parameters,
            },
            bounds: effect.bounds?,
        })
    }

    pub(crate) fn append_capture(
        self,
        output: &mut RenderPlan,
        texture_ids: &mut TextureIdAllocator,
    ) -> IntermediateTextureId {
        let capture_texture = texture_ids.allocate();
        output.push(RenderOperation::Capture(CaptureConfig {
            source: CaptureSource::Target,
            region: self.bounds.capture_region,
            output: capture_texture,
            sampling_size: self.bounds.sampling_size,
        }));
        capture_texture
    }

    pub(crate) fn append_effect(
        self,
        input_texture: IntermediateTextureId,
        output: &mut RenderPlan,
        texture_ids: &mut TextureIdAllocator,
    ) -> TextureComposite {
        let output_texture = texture_ids.allocate();
        output.push(RenderOperation::ApplyEffect(EffectApplication {
            effect_id: self.effect.effect_id,
            parameters: self.effect.parameters,
            input: input_texture,
            output: output_texture,
        }));
        TextureComposite {
            texture: output_texture,
            placement: TexturePlacement::Local {
                transform: geometry::unit_quad_transform(self.bounds.logical_output_bounds, None),
                sampling: TextureUvTransform::IDENTITY,
            },
        }
    }
}
