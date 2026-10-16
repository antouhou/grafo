use super::{DrawPlanner, DrawPlanningInput};
use crate::commands::{
    CaptureConfig, EffectApplication, RenderOperation, ShapeTextureBinding, ShapeTextureLayer,
    TextureSampling,
};
use crate::core::geometry;
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
        let source = input.backdrop_source?;
        let effect = input.backdrop_effects.get(&node_id)?;
        let region = effect.capture_region?;
        let capture = texture_ids.allocate();
        let filtered = texture_ids.allocate();
        output.push(RenderOperation::Capture(CaptureConfig {
            source,
            region,
            output: capture,
            sampling_size: geometry::compute_downsampled_dimensions(
                region.bounds.size().to_u32(),
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
            sampling: TextureSampling::TargetPixels(region.bounds),
        })
    }
}
