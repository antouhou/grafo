use crate::commands::{
    CaptureSource, DrawClip, RenderCommand, RenderOperation, ShapeDraw, ShapeDrawId,
    ShapeDrawMaterial, ShapeTextureBinding, Target, TextureComposite,
};
use crate::core::shape::ShapeInstance;
use crate::planner::commands::TraversalCommands;
use crate::planner::content_effects::ContentCapture;
use crate::planner::TextureIdAllocator;
use crate::scene::effects::parameters::EffectParameterStorage;
use crate::scene::effects::{BackdropEffectInstance, ContentEffectInstance};
use crate::scene::types::DrawTreeNode;
use crate::{Size, UnsignedPhysicalRect};
use ahash::HashMap;
use easy_tree::Tree;

mod backdrops;
mod rectangles;

fn shape_material(instance: &ShapeInstance) -> ShapeDrawMaterial {
    ShapeDrawMaterial {
        has_gradient_fill: instance.has_gradient_fill(),
        texture_bindings: instance.textures.map(|texture| {
            texture
                .texture_id
                .map_or(ShapeTextureBinding::None, ShapeTextureBinding::Managed)
        }),
        under_fill_texture: None,
    }
}

#[derive(Clone, Copy, Default)]
struct ClipState {
    clip: DrawClip,
    decrements_stencil: bool,
}

impl ClipState {
    fn for_viewport(size: Size) -> Self {
        Self {
            clip: DrawClip {
                scissor: UnsignedPhysicalRect::from_size(size),
                stencil_reference: 0,
            },
            decrements_stencil: false,
        }
    }
}

struct ParentDrawState {
    node_id: usize,
    next_child: usize,
    clip_state: ClipState,
    group_capture: Option<ContentCapture>,
}

#[derive(Clone, Copy, Default)]
pub(crate) struct DrawTreeSelection {
    /// None selects the scene root. Ancestor clips are omitted for a selected subtree.
    pub(crate) subtree_root: Option<usize>,
    pub(crate) excluded_subtree: Option<usize>,
}

pub(crate) struct DrawPlanningInput<'scene, 'planning> {
    pub(crate) tree: &'scene Tree<DrawTreeNode>,
    pub(crate) parameters: &'scene EffectParameterStorage,
    pub(crate) selection: DrawTreeSelection,
    pub(crate) effect_results: &'planning HashMap<usize, TextureComposite>,
    pub(crate) shape_mask_effects: &'planning HashMap<usize, TextureComposite>,
    pub(crate) content_effects: &'scene HashMap<usize, ContentEffectInstance>,
    pub(crate) backdrop_effects: &'scene HashMap<usize, BackdropEffectInstance>,
    pub(crate) scale_factor: f64,
    pub(crate) physical_size: Size,
    /// None disables captures when rendering a group's backdrop source.
    pub(crate) backdrop_source: Option<CaptureSource>,
}

impl DrawPlanningInput<'_, '_> {
    fn resolve_group_capture(&self, node_id: usize) -> Option<ContentCapture> {
        let can_capture_inline =
            self.backdrop_effects.is_empty() && self.selection.subtree_root != Some(node_id);
        if !can_capture_inline {
            return None;
        }
        let effect = self.content_effects.get(&node_id)?;
        ContentCapture::new(*effect)
    }
}

/// Walks the CPU tree to emit draw commands and resolved clip operands.
#[derive(Default)]
pub(crate) struct DrawPlanner {
    parents: Vec<ParentDrawState>,
    current: ClipState,
    next_node: Option<usize>,
}

impl DrawPlanner {
    pub(crate) fn start(&mut self, selection: DrawTreeSelection, physical_size: Size) {
        self.parents.clear();
        self.current = ClipState::for_viewport(physical_size);
        self.next_node = Some(selection.subtree_root.unwrap_or(0));
    }

    /// Advances one node or closes one parent, keeping each command group bounded.
    pub(crate) fn advance<'scene>(
        &mut self,
        input: &DrawPlanningInput<'scene, '_>,
        output: &mut TraversalCommands<'scene>,
        texture_ids: &mut TextureIdAllocator,
    ) -> bool {
        if let Some(node_id) = self.next_node.take() {
            self.plan_node(node_id, input, output, texture_ids);
            return true;
        }
        let Some(parent) = self.parents.last_mut() else {
            return false;
        };
        if let Some(&child) = input.tree.children(parent.node_id).get(parent.next_child) {
            parent.next_child += 1;
            self.next_node = Some(child);
            return true;
        }
        self.close_parent(input, output, texture_ids);
        true
    }

    fn plan_node<'scene>(
        &mut self,
        node_id: usize,
        input: &DrawPlanningInput<'scene, '_>,
        output: &mut TraversalCommands<'scene>,
        texture_ids: &mut TextureIdAllocator,
    ) {
        if input.selection.excluded_subtree == Some(node_id) {
            return;
        }
        let Some(node) = input.tree.get(node_id) else {
            return;
        };
        if let Some(&composite) = input.effect_results.get(&node_id) {
            output.push_composite(composite, self.current.clip);
            return;
        }
        let parent_clip = self.current;
        let group_capture = input.resolve_group_capture(node_id);
        if group_capture.is_some() {
            let texture = texture_ids.allocate();
            output.push(RenderOperation::BeginTarget(Target::Texture {
                texture,
                size: input.physical_size,
            }));
            self.current = ClipState::for_viewport(input.physical_size);
        }
        if let Some(&composite) = input.shape_mask_effects.get(&node_id) {
            output.push_composite(composite, self.current.clip);
        }
        let has_children = !input.tree.children(node_id).is_empty();
        if has_children || group_capture.is_some() {
            self.parents.push(ParentDrawState {
                node_id,
                next_child: 0,
                clip_state: parent_clip,
                group_capture,
            });
            self.current.decrements_stencil = false;
        }
        let draw = match node {
            DrawTreeNode::CachedShape(description) if description.has_geometry() => {
                let mut material = shape_material(&description.instance);
                // If there's a backdrop filter attached to this shape, plan it first
                // and set filter output texture as this shape underfill
                material.under_fill_texture =
                    self.plan_backdrop_texture(node_id, input, output, texture_ids);
                Some(ShapeDraw {
                    id: ShapeDrawId(node_id),
                    material,
                })
            }
            _ => None,
        };
        if let Some(instruction) = self.enter_node(node_id, node, draw, input) {
            output.push_command(instruction);
        }
    }

    fn close_parent<'scene>(
        &mut self,
        input: &DrawPlanningInput<'scene, '_>,
        output: &mut TraversalCommands<'scene>,
        texture_ids: &mut TextureIdAllocator,
    ) {
        if let Some(parent) = self.parents.last() {
            let node_id = parent.node_id;
            if self.current.decrements_stencil {
                let Some(DrawTreeNode::CachedShape(description)) = input.tree.get(node_id) else {
                    unreachable!("stencil clips have shape geometry");
                };
                output.push_command(RenderCommand {
                    operation: RenderOperation::DecrementStencil(ShapeDraw {
                        id: ShapeDrawId(node_id),
                        material: shape_material(&description.instance),
                    }),
                    clip: self.current.clip,
                });
            }
            let parent = self.parents.pop().expect("parent clip is balanced");
            self.current = parent.clip_state;
            if let Some(capture) = parent.group_capture {
                let input_texture = capture.append_capture(output, texture_ids);
                output.push(RenderOperation::EndTarget);
                let composite =
                    capture.append_effect(input_texture, input.parameters, output, texture_ids);
                output.push_composite(composite, self.current.clip);
            }
        }
    }

    fn draw_shape<'scene>(&self, draw: ShapeDraw) -> RenderCommand<'scene> {
        RenderCommand {
            operation: RenderOperation::DrawShape(draw),
            clip: self.current.clip,
        }
    }

    fn enter_node<'scene>(
        &mut self,
        node_id: usize,
        node: &DrawTreeNode,
        draw: Option<ShapeDraw>,
        input: &DrawPlanningInput<'scene, '_>,
    ) -> Option<RenderCommand<'scene>> {
        let should_draw = !rectangles::should_skip_visible_rect_draw(
            node_id,
            node,
            draw.map(|draw| draw.material),
            input.content_effects,
        );
        let visible_draw = draw.filter(|_| should_draw);
        if input.tree.children(node_id).is_empty() {
            return visible_draw.map(|draw| self.draw_shape(draw));
        }
        if !node.clips_children() {
            return visible_draw.map(|draw| self.draw_shape(draw));
        }
        if let Some(scissor) =
            rectangles::try_scissor_for_rect(node, input.scale_factor, input.physical_size)
        {
            self.current.clip.scissor = self
                .current
                .clip
                .scissor
                .intersection(&scissor)
                .unwrap_or_else(UnsignedPhysicalRect::zero);
            return visible_draw.map(|draw| self.draw_shape(draw));
        }
        let draw = draw?;
        let clip = self.current.clip;
        self.current.decrements_stencil = true;
        self.current.clip.stencil_reference += 1;
        Some(RenderCommand {
            operation: RenderOperation::DrawShapeAndIncrementStencil(draw),
            clip,
        })
    }
}

#[cfg(test)]
mod tests;
