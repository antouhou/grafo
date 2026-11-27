use crate::commands::{
    CaptureSource, ClipDraw, ClipGeometry, DrawClip, RenderCommand, RenderOperation, ShapeDraw,
    ShapeDrawId, ShapeDrawMaterial, ShapeTextureBinding, TextureComposite,
};
use crate::core::shape::ShapeInstance;
use crate::core::{BlendMode, MathRect};
use crate::planner::commands::TraversalCommands;
use crate::planner::content_effects::{content_intersects_damage, ContentCapture};
use crate::planner::TextureIdAllocator;
use crate::scene_state::content_groups::ContentGroup;
use crate::scene_state::effects::parameters::EffectParameterStorage;
use crate::scene_state::effects::BackdropEffectInstance;
use crate::scene_state::types::DrawTreeNode;
use crate::{Size, UnsignedPhysicalRect};
use ahash::{HashMap, HashSet};
use easy_tree::Tree;

mod backdrops;
mod content_effects;
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

#[derive(Clone, Copy)]
struct DrawTargetState {
    clip: DrawClip,
    decrements_stencil: bool,
    backdrop_source: CaptureSource,
    damaged_region: Option<MathRect>,
    is_content_capture: bool,
}

impl Default for DrawTargetState {
    fn default() -> Self {
        Self {
            clip: DrawClip::default(),
            decrements_stencil: false,
            backdrop_source: CaptureSource::Target,
            damaged_region: None,
            is_content_capture: false,
        }
    }
}

impl DrawTargetState {
    fn for_viewport(size: Size, backdrop_source: CaptureSource) -> Self {
        Self {
            clip: DrawClip {
                scissor: UnsignedPhysicalRect::from_size(size),
                stencil_reference: 0,
            },
            decrements_stencil: false,
            backdrop_source,
            damaged_region: None,
            is_content_capture: false,
        }
    }
}

struct ParentDrawState {
    node_id: usize,
    next_child: usize,
    target_state: DrawTargetState,
    group_capture: Option<ContentCapture>,
}

pub(crate) struct DrawPlanningInput<'scene, 'planning> {
    pub(crate) tree: &'scene Tree<DrawTreeNode>,
    pub(crate) parameters: &'scene EffectParameterStorage,
    pub(crate) shape_mask_effects: &'planning HashMap<usize, TextureComposite>,
    pub(crate) content_groups: &'scene HashMap<usize, ContentGroup>,
    pub(crate) backdrop_effects: &'scene HashMap<usize, BackdropEffectInstance>,
    pub(crate) scale_factor: f64,
    pub(crate) physical_size: Size,
    pub(crate) damaged_region: Option<MathRect>,
}

/// Walks the CPU tree to emit draw commands and resolved clip operands.
#[derive(Default)]
pub(crate) struct DrawPlanner {
    pub(super) subtree_backdrop_regions: HashMap<usize, UnsignedPhysicalRect>,
    pub(super) node_order: Vec<usize>,
    subtree_bounds: HashMap<usize, Option<MathRect>>,
    destination_dependent_subtrees: HashSet<usize>,
    layered_subtrees: HashSet<usize>,
    parents: Vec<ParentDrawState>,
    current: DrawTargetState,
    next_node: Option<usize>,
}

impl DrawPlanner {
    fn prepare_node_order(&mut self, tree: &Tree<DrawTreeNode>) {
        self.node_order.clear();
        if tree.get(0).is_none() {
            return;
        }
        self.node_order.push(0);
        let mut next_node_index = 0;
        while let Some(&node_id) = self.node_order.get(next_node_index) {
            self.node_order.extend_from_slice(tree.children(node_id));
            next_node_index += 1;
        }
    }

    pub(crate) fn clear_caches(&mut self) {
        *self = Self::default();
    }

    pub(crate) fn start(&mut self, input: &DrawPlanningInput<'_, '_>) {
        self.parents.clear();
        self.current = DrawTargetState::for_viewport(input.physical_size, CaptureSource::Target);
        self.current.damaged_region = input.damaged_region;
        self.next_node = input.damaged_region.is_some().then_some(0);
        if self.next_node.is_some() {
            self.prepare_node_order(input.tree);
            self.resolve_backdrop_regions(input);
            self.resolve_subtree_bounds(input);
        }
    }

    pub(super) fn next_capture_node_damage(&self) -> Option<(usize, MathRect)> {
        if !self.current.is_content_capture {
            return None;
        }
        Some((self.next_node?, self.current.damaged_region?))
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
        let Some(node) = input.tree.get(node_id) else {
            return;
        };
        let content_group = input.content_groups.get(&node_id);
        let intersects_damage = self.current.damaged_region.is_some_and(|damage| {
            node.logical_screen_bounds().intersects(&damage)
                || input.shape_mask_effects.contains_key(&node_id)
                || content_intersects_damage(content_group, damage, input.scale_factor)
        });
        if !intersects_damage {
            let has_content_capture =
                content_group.is_some_and(|group| group.output_bounds.is_some());
            if !node.has_child_clip()
                && !has_content_capture
                && !input.tree.children(node_id).is_empty()
            {
                self.parents.push(ParentDrawState {
                    node_id,
                    next_child: 0,
                    target_state: self.current,
                    group_capture: None,
                });
                self.current.decrements_stencil = false;
            }
            return;
        }
        let parent_state = self.current;
        let group_capture = self.begin_content_capture(node_id, input, output, texture_ids);
        if let Some(&composite) = input.shape_mask_effects.get(&node_id) {
            output.push_composite(composite, self.current.clip);
        }
        let has_children = !input.tree.children(node_id).is_empty();
        if has_children || group_capture.is_some() {
            self.parents.push(ParentDrawState {
                node_id,
                next_child: 0,
                target_state: parent_state,
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
                let clip_draw = match input.tree.get(node_id) {
                    Some(DrawTreeNode::CachedShape(description)) => {
                        ClipGeometry::Shape(ShapeDraw {
                            id: ShapeDrawId(node_id),
                            material: shape_material(&description.instance),
                        })
                    }
                    Some(DrawTreeNode::ClipRect(rectangle)) => {
                        ClipGeometry::Rectangle(rectangle.logical_screen_bounds)
                    }
                    _ => unreachable!("clip scope requires a clipping node"),
                };
                output.push_command(RenderCommand {
                    operation: RenderOperation::DecrementStencil(ClipDraw {
                        geometry: clip_draw,
                        requires_destination: false,
                        requires_layer: false,
                    }),
                    clip: self.current.clip,
                });
            }
            let parent = self.parents.pop().expect("parent clip is balanced");
            self.current = parent.target_state;
            if let Some(capture) = parent.group_capture {
                let blend_mode = match input.tree.get(node_id) {
                    Some(DrawTreeNode::CachedShape(shape)) => shape.blend_mode,
                    _ => BlendMode::SourceOver,
                };
                let input_texture = capture.append_end(output, texture_ids);
                let composite = capture.prepare_composite(
                    input_texture,
                    blend_mode,
                    input.physical_size,
                    input.parameters,
                    output,
                    texture_ids,
                );
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
        let should_draw = !rectangles::should_skip_visible_draw(
            node_id,
            node,
            draw.map(|draw| draw.material),
            input.content_groups,
        );
        let visible_draw = draw.filter(|_| should_draw);
        if input.tree.children(node_id).is_empty() {
            return visible_draw.map(|draw| self.draw_shape(draw));
        }
        if !node.clips_children() {
            return visible_draw.map(|draw| self.draw_shape(draw));
        }
        if let Some(scissor) = rectangles::try_scissor_for_rect(
            node,
            self.child_bounds(node_id, input),
            input.scale_factor,
            input.physical_size,
        ) {
            self.current.clip.scissor = self
                .current
                .clip
                .scissor
                .intersection(&scissor)
                .unwrap_or_else(UnsignedPhysicalRect::zero);
            return visible_draw.map(|draw| self.draw_shape(draw));
        }
        if self.children_fit_inside_clip(node_id, node, input) {
            return visible_draw.map(|draw| self.draw_shape(draw));
        }
        let draw = match (draw, node) {
            (Some(draw), _) => ClipGeometry::Shape(draw),
            (None, DrawTreeNode::ClipRect(rectangle)) => {
                ClipGeometry::Rectangle(rectangle.logical_screen_bounds)
            }
            _ => return None,
        };
        let clip = self.current.clip;
        self.current.decrements_stencil = true;
        self.current.clip.stencil_reference += 1;
        Some(RenderCommand {
            operation: RenderOperation::DrawShapeAndIncrementStencil(ClipDraw {
                geometry: draw,
                requires_destination: input
                    .tree
                    .children(node_id)
                    .iter()
                    .any(|child| self.destination_dependent_subtrees.contains(child)),
                requires_layer: input
                    .tree
                    .children(node_id)
                    .iter()
                    .any(|child| self.layered_subtrees.contains(child)),
            }),
            clip,
        })
    }
}

#[cfg(test)]
mod tests;
