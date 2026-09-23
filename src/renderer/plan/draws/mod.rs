use crate::effect::{BackdropEffectInstance, EffectInstance};
use crate::renderer::commands::{
    BackdropCaptureSource, DrawClip, DrawInstruction, DrawOperation, DrawPlan,
    IntermediateTextureId, ShapeDraw, ShapeDrawId, TextureComposite, TexturePlacement,
};
use crate::renderer::rect_utils::{should_skip_visible_rect_draw, try_scissor_for_rect};
use crate::renderer::types::DrawTreeNode;
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

struct ParentDrawState {
    node_id: usize,
    next_child: usize,
    clip_state: ClipState,
}

#[derive(Clone, Copy, Default)]
pub(in crate::renderer) struct DrawTreeSelection {
    /// None selects the scene root. Ancestor clips are omitted for a selected subtree.
    pub(in crate::renderer) subtree_root: Option<usize>,
    pub(in crate::renderer) excluded_subtree: Option<usize>,
}

pub(in crate::renderer) struct DrawPlanningInput<'a> {
    pub(in crate::renderer) tree: &'a Tree<DrawTreeNode>,
    pub(in crate::renderer) selection: DrawTreeSelection,
    pub(in crate::renderer) effect_results: &'a HashMap<usize, IntermediateTextureId>,
    pub(in crate::renderer) shape_effects: &'a HashMap<usize, TextureComposite>,
    pub(in crate::renderer) group_effects: &'a HashMap<usize, EffectInstance>,
    pub(in crate::renderer) backdrop_effects: &'a HashMap<usize, BackdropEffectInstance>,
    pub(in crate::renderer) scale_factor: f64,
    pub(in crate::renderer) physical_size: Size,
    pub(in crate::renderer) backdrop_source: Option<BackdropCaptureSource>,
    /// None disables captures when rendering a group's backdrop source.
    pub(in crate::renderer) max_capture_dimension: Option<u32>,
}

/// Walks the CPU tree to emit draw commands and resolved clip operands.
#[derive(Default)]
pub(in crate::renderer) struct DrawPlanner {
    parents: Vec<ParentDrawState>,
    current: ClipState,
}

impl DrawPlanner {
    /// Appends a selected tree with its own resolved clip state.
    pub(in crate::renderer) fn append(
        &mut self,
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
        let mut next_node = Some(input.selection.subtree_root.unwrap_or(0));
        while let Some(node_id) = next_node {
            self.plan_node(node_id, &input, output);
            next_node = self.next_node(&input, output);
        }
        debug_assert!(
            self.parents.is_empty(),
            "draw traversal must balance parent clips"
        );
    }

    fn plan_node(&mut self, node_id: usize, input: &DrawPlanningInput<'_>, output: &mut DrawPlan) {
        if input.selection.excluded_subtree == Some(node_id) {
            return;
        }
        let Some(node) = input.tree.get(node_id) else {
            return;
        };
        if let Some(&texture) = input.effect_results.get(&node_id) {
            output.push_composite(
                TextureComposite {
                    texture,
                    placement: TexturePlacement::Target,
                },
                self.current.clip,
            );
            return;
        }
        if let Some(&composite) = input.shape_effects.get(&node_id) {
            output.push_composite(composite, self.current.clip);
        }
        if !node.is_leaf() {
            self.parents.push(ParentDrawState {
                node_id,
                next_child: 0,
                clip_state: self.current,
            });
            self.current.decrements_stencil = false;
        }
        if self.plan_backdrop(node_id, node, input, output) {
            return;
        }
        let draw = match node {
            DrawTreeNode::CachedShape(description) if has_geometry(description) => {
                Some(ShapeDraw {
                    id: ShapeDrawId(node_id),
                    material: description.material(),
                })
            }
            _ => None,
        };
        if let Some(instruction) = self.enter_node(node_id, node, draw, input, output) {
            output.push_draw(instruction);
        }
    }

    /// Advances through siblings and closes each completed parent's clip.
    fn next_node(&mut self, input: &DrawPlanningInput<'_>, output: &mut DrawPlan) -> Option<usize> {
        while let Some(parent) = self.parents.last_mut() {
            if let Some(&child) = input.tree.children(parent.node_id).get(parent.next_child) {
                parent.next_child += 1;
                return Some(child);
            }
            let node_id = parent.node_id;
            if self.current.decrements_stencil {
                let Some(DrawTreeNode::CachedShape(description)) = input.tree.get(node_id) else {
                    unreachable!("stencil clips have shape geometry");
                };
                output.push_draw(DrawInstruction {
                    operation: DrawOperation::DecrementStencil(ShapeDraw {
                        id: ShapeDrawId(node_id),
                        material: description.material(),
                    }),
                    clip: self.current.clip,
                });
            }
            self.current = self
                .parents
                .pop()
                .expect("parent clip is balanced")
                .clip_state;
        }
        None
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
}

#[cfg(test)]
mod tests;
