use crate::commands::{
    CaptureConfig, CaptureSource, EffectApplication, IntermediateTextureId, RenderOperation,
    Target, TextureComposite, TexturePlacement,
};
use crate::core::effect::{CaptureRegion, ContentEffectBounds};
use crate::core::geometry;
use crate::core::vertex::TextureUvTransform;
use crate::core::UnsignedPhysicalRect;
use crate::planner::commands::TraversalCommands;
use crate::planner::TextureIdAllocator;
use crate::scene_state::effects::parameters::EffectParameterStorage;
use crate::scene_state::effects::{ContentEffectInstance, EffectInstance};

#[derive(Clone, Copy)]
pub(crate) struct ContentCapture {
    effect: EffectInstance,
    bounds: ContentEffectBounds,
    texture: IntermediateTextureId,
    region: CaptureRegion,
}

impl ContentCapture {
    pub(crate) fn new(
        effect: ContentEffectInstance,
        backdrop_region: Option<UnsignedPhysicalRect>,
        texture_ids: &mut TextureIdAllocator,
    ) -> Option<Self> {
        let bounds = effect.bounds?;
        let mut region = bounds.capture_region;
        if let Some(backdrop) = backdrop_region.filter(|backdrop| {
            region
                .source_rect
                .and_then(|input| input.intersection(backdrop))
                != Some(*backdrop)
        }) {
            let input = region
                .source_rect
                .map_or(backdrop, |input| input.union(&backdrop));
            // Keep all pixels needed by internal captures until the content is cropped.
            region = CaptureRegion {
                bounds: input.to_i32(),
                source_rect: Some(input),
            };
        }
        Some(Self {
            effect: EffectInstance {
                effect_id: effect.effect_id,
                parameters: effect.parameters,
            },
            bounds,
            texture: texture_ids.allocate(),
            region,
        })
    }

    pub(crate) fn append_begin(self, output: &mut TraversalCommands<'_>) {
        output.push(RenderOperation::BeginTarget(Target::Texture {
            texture: self.texture,
            region: self.region,
        }));
    }

    pub(crate) fn append_end(
        self,
        output: &mut TraversalCommands<'_>,
        texture_ids: &mut TextureIdAllocator,
    ) -> IntermediateTextureId {
        let mut texture = self.texture;
        if self.region != self.bounds.capture_region
            || self.bounds.sampling_size != self.region.bounds.size().to_u32()
        {
            texture = texture_ids.allocate();
            output.push(RenderOperation::Capture(CaptureConfig {
                source: CaptureSource::Target,
                region: self.bounds.capture_region,
                output: texture,
                sampling_size: self.bounds.sampling_size,
            }));
        }
        output.push(RenderOperation::EndTarget);
        texture
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
