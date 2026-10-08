use super::{DrawPlanner, DrawPlanningInput};
use crate::commands::{
    BackdropCapture, DrawClip, EffectApplication, RenderCommand, RenderOperation, RenderPlan,
    ShapeDraw, ShapeDrawId, ShapeTextureBinding, ShapeTextureLayer, TextureSampling,
};
use crate::core::geometry;
use crate::planner::draws;
use crate::planner::TextureIdAllocator;
use crate::scene::types::DrawTreeNode;

impl DrawPlanner {
    /// Captures preceding color before entering the shape's stencil coverage.
    pub(super) fn plan_backdrop(
        &mut self,
        node_id: usize,
        node: &DrawTreeNode,
        input: &DrawPlanningInput<'_>,
        output: &mut RenderPlan,
        texture_ids: &mut TextureIdAllocator,
    ) -> bool {
        let Some(source) = input.backdrop_source else {
            return false;
        };
        let Some(effect) = input.backdrop_effects.get(&node_id) else {
            return false;
        };
        let DrawTreeNode::CachedShape(description) = node else {
            return false;
        };
        if !description.has_geometry() {
            return false;
        }
        let mut draw = ShapeDraw {
            id: ShapeDrawId(node_id),
            material: draws::shape_material(&description.instance),
        };
        if let Some(region) = effect.capture_region {
            let capture = texture_ids.allocate();
            let filtered = texture_ids.allocate();
            output.push(RenderOperation::CaptureBackdrop(BackdropCapture {
                source,
                region,
                output: capture,
                sampling_size: geometry::compute_downsampled_dimensions(
                    region.bounds.size().to_u32(),
                    effect.config.downsample,
                ),
            }));
            let parameters = effect.effect.parameters;
            output.push(RenderOperation::ApplyEffect(EffectApplication {
                effect_id: effect.effect.effect_id,
                parameters,
                input: capture,
                output: filtered,
            }));
            draw.material.under_fill_texture = Some(ShapeTextureLayer {
                texture: ShapeTextureBinding::Intermediate(filtered),
                sampling: TextureSampling::TargetPixels(region.bounds),
            });
        }
        let has_children = !input.tree.children(node_id).is_empty();
        self.draw_backdrop(node, draw, has_children, output);
        true
    }

    fn draw_backdrop(
        &mut self,
        node: &DrawTreeNode,
        draw: ShapeDraw,
        has_children: bool,
        output: &mut RenderPlan,
    ) {
        let parent_clip = self.current.clip;
        let shape_clip = DrawClip {
            stencil_reference: parent_clip.stencil_reference + 1,
            ..parent_clip
        };
        output.push_command(RenderCommand {
            operation: RenderOperation::IncrementStencil(draw.id),
            clip: parent_clip,
        });
        output.push_command(RenderCommand {
            operation: RenderOperation::DrawShape(draw),
            clip: shape_clip,
        });
        if !has_children || !node.clips_children() {
            output.push_command(RenderCommand {
                operation: RenderOperation::DecrementStencil(draw),
                clip: shape_clip,
            });
        }
        if has_children {
            self.current.decrements_stencil = node.clips_children();
            if node.clips_children() {
                self.current.clip = shape_clip;
            }
        }
    }
}
