use crate::commands::{
    BlendDestination, CaptureConfig, CaptureSource, EffectApplication, IntermediateTextureId,
    RenderOperation, Target, TextureComposite, TexturePlacement,
};
use crate::core::geometry;
use crate::core::vertex::TextureUvTransform;
use crate::core::{BlendMode, PhysicalRect, Size, UnsignedPhysicalRect};
use crate::planner::commands::TraversalCommands;
use crate::planner::TextureIdAllocator;
use crate::scene_state::content_groups::ContentGroup;
use crate::scene_state::effects::parameters::EffectParameterStorage;
use crate::scene_state::effects::EffectInstance;

#[derive(Clone, Copy)]
pub(crate) struct ContentCapture {
    effect: Option<EffectInstance>,
    texture: IntermediateTextureId,
    render_bounds: PhysicalRect,
    draw_scissor: UnsignedPhysicalRect,
    source_rect: Option<UnsignedPhysicalRect>,
    output_bounds: PhysicalRect,
    sampling_size: Size,
    logical_output_bounds: [(f32, f32); 2],
}

impl ContentCapture {
    pub(crate) fn draw_region(self) -> Option<UnsignedPhysicalRect> {
        (!self.draw_scissor.is_empty()).then_some(self.draw_scissor)
    }

    pub(crate) fn new(
        group: ContentGroup,
        scale_factor: f64,
        backdrop_region: Option<UnsignedPhysicalRect>,
        texture_ids: &mut TextureIdAllocator,
    ) -> Option<Self> {
        let output_bounds = group.output_bounds?;
        let output_size = output_bounds.size().to_u32();
        let source_rect = group.input_bounds;
        let mut render_bounds = output_bounds;
        let mut draw_scissor = source_rect.unwrap_or_else(UnsignedPhysicalRect::zero);
        if let Some(backdrop) = backdrop_region.filter(|backdrop| {
            source_rect.and_then(|input| input.intersection(backdrop)) != Some(*backdrop)
        }) {
            let input = source_rect.map_or(backdrop, |input| input.union(&backdrop));
            // Keep all pixels needed by internal captures until the content is cropped.
            render_bounds = input.to_i32();
            draw_scissor = input;
        }
        let logical_output_bounds = output_bounds.to_f32() / scale_factor as f32;
        Some(Self {
            effect: group.effect.map(|effect| EffectInstance {
                effect_id: effect.effect_id,
                parameters: effect.parameters,
            }),
            texture: texture_ids.allocate(),
            render_bounds,
            draw_scissor,
            source_rect,
            output_bounds,
            sampling_size: group.effect.map_or(output_size, |effect| {
                geometry::compute_downsampled_dimensions(output_size, effect.config.downsample)
            }),
            logical_output_bounds: [
                logical_output_bounds.min.to_tuple(),
                logical_output_bounds.max.to_tuple(),
            ],
        })
    }

    pub(crate) fn append_begin(self, output: &mut TraversalCommands<'_>) {
        output.push(RenderOperation::BeginTarget(Target::Texture {
            texture: self.texture,
            bounds: self.render_bounds,
            draw_scissor: self.draw_scissor,
        }));
    }

    pub(crate) fn append_end(
        self,
        output: &mut TraversalCommands<'_>,
        texture_ids: &mut TextureIdAllocator,
    ) -> IntermediateTextureId {
        let mut texture = self.texture;
        if self.render_bounds != self.output_bounds
            || self.draw_region() != self.source_rect
            || self.sampling_size != self.render_bounds.size().to_u32()
        {
            texture = texture_ids.allocate();
            output.push(RenderOperation::Capture(CaptureConfig {
                source: CaptureSource::Target,
                source_rect: self.source_rect,
                destination_bounds: self.output_bounds,
                output: texture,
                sampling_size: self.sampling_size,
            }));
        }
        output.push(RenderOperation::EndTarget);
        texture
    }

    pub(crate) fn prepare_composite<'scene>(
        self,
        input_texture: IntermediateTextureId,
        blend_mode: BlendMode,
        viewport_size: Size,
        parameters: &'scene EffectParameterStorage,
        output: &mut TraversalCommands<'scene>,
        texture_ids: &mut TextureIdAllocator,
    ) -> TextureComposite {
        let texture = if let Some(effect) = self.effect {
            let output_texture = texture_ids.allocate();
            output.push(RenderOperation::ApplyEffect(EffectApplication {
                effect_id: effect.effect_id,
                parameters: parameters.get(effect.parameters),
                input: input_texture,
                output: output_texture,
            }));
            output_texture
        } else {
            input_texture
        };
        let mut composite = TextureComposite {
            source: texture,
            blend_mode,
            destination: BlendDestination::FinalTarget,
            placement: TexturePlacement::Local {
                transform: geometry::unit_quad_transform(self.logical_output_bounds, None),
                sampling: TextureUvTransform::IDENTITY,
            },
        };
        if composite.requires_destination_texture() {
            let destination = texture_ids.allocate();
            let source_rect = self
                .output_bounds
                .to_i64()
                .intersection(&UnsignedPhysicalRect::from_size(viewport_size).to_i64())
                .map(|bounds| bounds.to_u32());
            output.push(RenderOperation::Capture(CaptureConfig {
                source: CaptureSource::Target,
                source_rect,
                destination_bounds: self.output_bounds,
                output: destination,
                sampling_size: self.output_bounds.size().to_u32(),
            }));
            composite.destination = BlendDestination::Texture(destination);
        }
        composite
    }
}
