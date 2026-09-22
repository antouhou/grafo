use crate::effect::{BackdropEffectInstance, EffectInstance};
use crate::renderer::commands::{
    BackdropCaptureSource, DrawClip, DrawInstruction, DrawOperation, DrawPlan,
    IntermediateTextureId, ShapeDraw, ShapeDrawId,
};
use crate::renderer::plan::shape_effects::PreparedShapeEffectLeaf;
use crate::renderer::rect_utils::{should_skip_visible_rect_draw, try_scissor_for_rect};
use crate::renderer::types::{DrawTreeNode, TraversalEvent};
use crate::shape::CachedShapeDrawData;
use crate::{Size, UnsignedPhysicalRect};
use ahash::HashMap;
use easy_tree::Tree;

mod backdrops;

fn has_geometry(shape: &CachedShapeDrawData) -> bool {
    let geometry = shape.cached_shape.vertex_buffers();
    !geometry.vertices.is_empty() && !geometry.indices.is_empty()
}

#[derive(Clone, Copy, Default)]
struct ClipState {
    clip: DrawClip,
    decrements_stencil: bool,
}

pub(in crate::renderer) struct DrawPlanningInput<'a> {
    pub(in crate::renderer) tree: &'a Tree<DrawTreeNode>,
    pub(in crate::renderer) effect_results: &'a HashMap<usize, IntermediateTextureId>,
    pub(in crate::renderer) effect_leaves: &'a HashMap<usize, PreparedShapeEffectLeaf>,
    pub(in crate::renderer) group_effects: &'a HashMap<usize, EffectInstance>,
    pub(in crate::renderer) backdrop_effects: &'a HashMap<usize, BackdropEffectInstance>,
    pub(in crate::renderer) scale_factor: f64,
    pub(in crate::renderer) physical_size: Size,
    pub(in crate::renderer) backdrop_source: Option<BackdropCaptureSource>,
    /// None disables captures when rendering a group's backdrop source.
    pub(in crate::renderer) max_capture_dimension: Option<u32>,
}

/// Resolves traversal events into draw commands and clip operands.
#[derive(Default)]
pub(in crate::renderer) struct DrawPlanner {
    parents: Vec<ClipState>,
    current: ClipState,
}

impl DrawPlanner {
    /// Replaces the output commands while reusing their storage.
    pub(in crate::renderer) fn plan(
        &mut self,
        events: &[TraversalEvent],
        input: DrawPlanningInput<'_>,
        output: &mut DrawPlan,
    ) {
        self.parents.clear();
        self.current = ClipState {
            clip: DrawClip {
                scissor: UnsignedPhysicalRect::from_size(input.physical_size),
                stencil_reference: 0,
            },
            decrements_stencil: false,
        };
        output.clear();
        for &event in events {
            if self.plan_backdrop(event, &input, output) {
                continue;
            }
            if let Some(instruction) = self.plan_event(event, &input, output) {
                output.push_draw(instruction);
            }
        }
        debug_assert!(
            self.parents.is_empty(),
            "draw traversal must balance parent clips"
        );
    }

    fn plan_event(
        &mut self,
        event: TraversalEvent,
        input: &DrawPlanningInput<'_>,
        output: &mut DrawPlan,
    ) -> Option<DrawInstruction> {
        let node_id = match event {
            TraversalEvent::PreparedLeaf(node_id) => {
                return input
                    .effect_leaves
                    .get(&node_id)
                    .filter(|leaf| has_geometry(&leaf.draw_data))
                    .map(|leaf| {
                        self.draw_shape(ShapeDraw {
                            id: ShapeDrawId::EffectLeaf(node_id),
                            material: leaf.draw_data.material(),
                        })
                    });
            }
            TraversalEvent::Pre(node_id) | TraversalEvent::Post(node_id) => node_id,
        };
        if let Some(&texture) = input.effect_results.get(&node_id) {
            return matches!(event, TraversalEvent::Pre(_)).then_some(DrawInstruction {
                operation: DrawOperation::CompositeTexture(texture),
                clip: self.current.clip,
            });
        }
        let node = input.tree.get(node_id)?;
        let draw = match node {
            DrawTreeNode::CachedShape(description) if has_geometry(description) => {
                Some(ShapeDraw {
                    id: ShapeDrawId::Shape(node_id),
                    material: description.material(),
                })
            }
            _ => None,
        };
        match event {
            TraversalEvent::Pre(_) => self.enter_node(node_id, node, draw, input, output),
            TraversalEvent::Post(_) if !node.is_leaf() => self.leave_node(draw),
            _ => None,
        }
    }

    fn draw_shape(&self, draw: ShapeDraw) -> DrawInstruction {
        DrawInstruction {
            operation: DrawOperation::DrawShape(draw),
            clip: self.current.clip,
        }
    }

    fn enter_node(
        &mut self,
        node_id: usize,
        node: &DrawTreeNode,
        draw: Option<ShapeDraw>,
        input: &DrawPlanningInput<'_>,
        _output: &mut DrawPlan,
    ) -> Option<DrawInstruction> {
        let should_draw = !should_skip_visible_rect_draw(
            node_id,
            node,
            input.group_effects,
            input.backdrop_effects,
        );
        let visible_draw = draw.filter(|_| should_draw);
        if node.is_leaf() {
            return visible_draw.map(|draw| self.draw_shape(draw));
        }
        self.parents.push(self.current);
        self.current.decrements_stencil = false;
        if !node.clips_children() {
            return visible_draw.map(|draw| self.draw_shape(draw));
        }
        if let Some(scissor) = try_scissor_for_rect(node, input.scale_factor, input.physical_size) {
            self.current.clip.scissor = self
                .current
                .clip
                .scissor
                .intersection(&scissor)
                .unwrap_or_else(UnsignedPhysicalRect::zero);
            #[cfg(feature = "render_metrics")]
            {
                _output.scissor_clip_count += 1;
            }
            return visible_draw.map(|draw| self.draw_shape(draw));
        }
        let draw = draw?;
        let clip = self.current.clip;
        self.current.decrements_stencil = true;
        self.current.clip.stencil_reference += 1;
        Some(DrawInstruction {
            operation: DrawOperation::DrawShapeAndIncrementStencil(draw),
            clip,
        })
    }

    fn leave_node(&mut self, draw: Option<ShapeDraw>) -> Option<DrawInstruction> {
        let instruction = self.current.decrements_stencil.then(|| DrawInstruction {
            operation: DrawOperation::DecrementStencil(
                draw.expect("stencil clips have shape geometry"),
            ),
            clip: self.current.clip,
        });
        self.current = self.parents.pop().expect("parent clip is balanced");
        instruction
    }
}

#[cfg(test)]
mod tests;
