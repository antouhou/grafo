use super::{DrawPlanner, DrawPlanningInput};
use crate::commands::{
    CaptureConfig, EffectApplication, RenderOperation, ShapeTextureBinding, ShapeTextureLayer,
    TextureSampling,
};
use crate::core::{geometry, UnsignedPhysicalRect};
use crate::planner::commands::TraversalCommands;
use crate::planner::TextureIdAllocator;

impl DrawPlanner {
    /// Produces a texture containing the filtered preceding color.
    pub(super) fn plan_backdrop_texture<'scene>(
        &self,
        node_id: usize,
        input: &DrawPlanningInput<'scene, '_>,
        output: &mut TraversalCommands<'scene>,
        texture_ids: &mut TextureIdAllocator,
    ) -> Option<ShapeTextureLayer> {
        let source = self.current.backdrop_source;
        let effect = input.backdrop_effects.get(&node_id)?;
        let bounds = effect.capture_bounds?;
        let source_rect = bounds
            .to_i64()
            .intersection(&UnsignedPhysicalRect::from_size(input.physical_size).to_i64())
            .map(|bounds| bounds.to_u32());
        let capture = texture_ids.allocate();
        let filtered = texture_ids.allocate();
        output.push(RenderOperation::Capture(CaptureConfig {
            source,
            source_rect,
            destination_bounds: bounds,
            output: capture,
            sampling_size: geometry::compute_downsampled_dimensions(
                bounds.size().to_u32(),
                effect.config.downsample,
            ),
        }));
        let parameters = input.parameters.get(effect.effect.parameters);
        output.push(RenderOperation::ApplyEffect(EffectApplication {
            effect_id: effect.effect.effect_id,
            parameters,
            input: capture,
            output: filtered,
        }));
        Some(ShapeTextureLayer {
            texture: ShapeTextureBinding::Intermediate(filtered),
            sampling: TextureSampling::TargetPixels(bounds),
        })
    }
}
