use crate::commands::{
    CaptureConfig, CaptureSource, EffectApplication, IntermediateTextureId, RenderOperation,
    TextureComposite, TexturePlacement,
};
use crate::core::effect::ContentEffectBounds;
use crate::core::geometry;
use crate::core::vertex::TextureUvTransform;
use crate::planner::commands::TraversalCommands;
use crate::planner::TextureIdAllocator;
use crate::scene_state::effects::parameters::EffectParameterStorage;
use crate::scene_state::effects::{ContentEffectInstance, EffectInstance};

#[derive(Clone, Copy)]
pub(crate) struct ContentCapture {
    effect: EffectInstance,
    bounds: ContentEffectBounds,
}

impl ContentCapture {
    pub(crate) fn new(effect: ContentEffectInstance) -> Option<Self> {
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
        output: &mut TraversalCommands<'_>,
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

    pub(crate) fn append_effect<'scene>(
        self,
        input_texture: IntermediateTextureId,
        parameters: &'scene EffectParameterStorage,
        output: &mut TraversalCommands<'scene>,
        texture_ids: &mut TextureIdAllocator,
    ) -> TextureComposite {
        let output_texture = texture_ids.allocate();
        output.push(RenderOperation::ApplyEffect(EffectApplication {
            effect_id: self.effect.effect_id,
            parameters: parameters.get(self.effect.parameters),
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
