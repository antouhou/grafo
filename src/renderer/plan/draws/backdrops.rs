use super::{has_geometry, DrawPlanner, DrawPlanningInput};
use crate::renderer::commands::{
    BackdropCapture, DrawClip, DrawInstruction, DrawOperation, DrawPlan, DrawSegment,
    EffectApplication, ShapeDraw, ShapeDrawId,
};
use crate::renderer::plan::backdrops::compute_backdrop_capture_region;
use crate::renderer::rect_utils::compute_downsampled_dimensions;
use crate::renderer::types::{DrawTreeNode, TraversalEvent};
use crate::shape::{ShapeTextureBinding, ShapeTextureLayer, TextureSampling};
use crate::MathRect;

impl DrawPlanner {
    /// Captures preceding color before entering the shape's stencil coverage.
    pub(super) fn plan_backdrop(
        &mut self,
        event: TraversalEvent,
        input: &DrawPlanningInput<'_>,
        output: &mut DrawPlan,
    ) -> bool {
        let TraversalEvent::Pre(node_id) = event else {
            return false;
        };
        let (Some(max_dimension), Some(source)) =
            (input.max_capture_dimension, input.backdrop_source)
        else {
            return false;
        };
        if input.effect_results.contains_key(&node_id) {
            return false;
        }
        let (Some(effect), Some(node)) = (
            input.backdrop_effects.get(&node_id),
            input.tree.get(node_id),
        ) else {
            return false;
        };
        let DrawTreeNode::CachedShape(description) = node else {
            return false;
        };
        if !has_geometry(description) {
            return false;
        }
        let mut draw = ShapeDraw {
            id: ShapeDrawId::Shape(node_id),
            material: description.material(),
        };
        let bounds = node.local_bounds();
        if let Some(region) = compute_backdrop_capture_region(
            MathRect::new(bounds[0].into(), bounds[1].into()),
            node.transform(),
            effect.config,
            input.scale_factor,
            input.physical_size,
            max_dimension,
        ) {
            let capture = output.allocate_texture();
            let filtered = output.allocate_texture();
            output
                .segments
                .push(DrawSegment::CaptureBackdrop(BackdropCapture {
                    source,
                    region,
                    output: capture,
                    sampling_size: compute_downsampled_dimensions(
                        region.bounds.size().to_u32(),
                        effect.config.downsample,
                    ),
                }));
            let parameter_start = output.effect_parameters.len();
            output
                .effect_parameters
                .extend_from_slice(&effect.effect.params);
            output
                .segments
                .push(DrawSegment::ApplyEffect(EffectApplication {
                    effect_id: effect.effect.effect_id,
                    parameters: parameter_start..output.effect_parameters.len(),
                    input: capture,
                    output: filtered,
                }));
            draw.material.under_fill_texture = Some(ShapeTextureLayer {
                texture: ShapeTextureBinding::Intermediate(filtered),
                sampling: TextureSampling::TargetPixels(region.bounds),
            });
        }
        self.draw_backdrop(node, draw, output);
        true
    }

    fn draw_backdrop(&mut self, node: &DrawTreeNode, draw: ShapeDraw, output: &mut DrawPlan) {
        let parent_clip = self.current.clip;
        let shape_clip = DrawClip {
            stencil_reference: parent_clip.stencil_reference + 1,
            ..parent_clip
        };
        output.push_draw(DrawInstruction {
            operation: DrawOperation::IncrementStencil(draw.id),
            clip: parent_clip,
        });
        output.push_draw(DrawInstruction {
            operation: DrawOperation::DrawShape(draw),
            clip: shape_clip,
        });
        if node.is_leaf() || !node.clips_children() {
            output.push_draw(DrawInstruction {
                operation: DrawOperation::DecrementStencil(draw),
                clip: shape_clip,
            });
        }
        if !node.is_leaf() {
            self.parents.push(self.current);
            self.current.decrements_stencil = node.clips_children();
            if node.clips_children() {
                self.current.clip = shape_clip;
            }
        }
    }
}
