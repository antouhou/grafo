use crate::commands::{
    BackdropCaptureSource, DrawClip, EffectApplication, IntermediateTextureId, RenderCommand,
    RenderOperation, RenderPlan, ShapeDraw, ShapeDrawId, ShapeDrawMaterial, ShapeTextureBinding,
    Target, TextureComposite, TexturePlacement,
};
use crate::core::shape::ShapeInstance;
use crate::scene::effects::{BackdropEffectInstance, EffectInstance};
use crate::scene::types::CachedShapeDrawData;
use crate::scene::types::DrawTreeNode;
use crate::{Size, UnsignedPhysicalRect};
use ahash::HashMap;
use easy_tree::Tree;

mod backdrops;
mod rectangles;

fn has_geometry(shape: &CachedShapeDrawData) -> bool {
    let geometry = shape.instance.cached_shape.vertex_buffers();
    !geometry.vertices.is_empty() && !geometry.indices.is_empty()
}

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

struct ParentDrawState {
    node_id: usize,
    next_child: usize,
    clip_state: ClipState,
    group_target: Option<IntermediateTextureId>,
}

#[derive(Clone, Copy, Default)]
pub(crate) struct DrawTreeSelection {
    /// None selects the scene root. Ancestor clips are omitted for a selected subtree.
    pub(crate) subtree_root: Option<usize>,
    pub(crate) excluded_subtree: Option<usize>,
}

pub(crate) struct DrawPlanningInput<'a> {
    pub(crate) tree: &'a Tree<DrawTreeNode>,
    pub(crate) selection: DrawTreeSelection,
    pub(crate) effect_results: &'a HashMap<usize, IntermediateTextureId>,
    pub(crate) shape_effects: &'a HashMap<usize, TextureComposite>,
    pub(crate) group_effects: &'a HashMap<usize, EffectInstance>,
    pub(crate) backdrop_effects: &'a HashMap<usize, BackdropEffectInstance>,
    pub(crate) scale_factor: f64,
    pub(crate) physical_size: Size,
    pub(crate) backdrop_source: Option<BackdropCaptureSource>,
    /// None disables captures when rendering a group's backdrop source.
    pub(crate) max_capture_dimension: Option<u32>,
}

/// Walks the CPU tree to emit draw commands and resolved clip operands.
#[derive(Default)]
pub(crate) struct DrawPlanner {
    parents: Vec<ParentDrawState>,
    current: ClipState,
}

impl DrawPlanner {
    /// Appends a selected tree with its own resolved clip state.
    pub(crate) fn append(&mut self, input: DrawPlanningInput<'_>, output: &mut RenderPlan) {
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

    fn plan_node(
        &mut self,
        node_id: usize,
        input: &DrawPlanningInput<'_>,
        output: &mut RenderPlan,
    ) {
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
        let parent_clip = self.current;
        let group_target = if input.backdrop_effects.is_empty()
            && input.selection.subtree_root != Some(node_id)
            && input.group_effects.contains_key(&node_id)
        {
            let texture = output.allocate_texture();
            output.push(RenderOperation::BeginTarget(Target::Texture {
                texture,
                size: input.physical_size,
            }));
            self.current = ClipState {
                clip: DrawClip {
                    scissor: UnsignedPhysicalRect::from_size(input.physical_size),
                    stencil_reference: 0,
                },
                decrements_stencil: false,
            };
            Some(texture)
        } else {
            None
        };
        if let Some(&composite) = input.shape_effects.get(&node_id) {
            output.push_composite(composite, self.current.clip);
        }
        if !node.is_leaf() || group_target.is_some() {
            self.parents.push(ParentDrawState {
                node_id,
                next_child: 0,
                clip_state: parent_clip,
                group_target,
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
                    material: shape_material(&description.instance),
                })
            }
            _ => None,
        };
        if let Some(instruction) = self.enter_node(node_id, node, draw, input, output) {
            output.push_command(instruction);
        }
    }

    /// Advances through siblings and closes each completed parent's clip.
    fn next_node(
        &mut self,
        input: &DrawPlanningInput<'_>,
        output: &mut RenderPlan,
    ) -> Option<usize> {
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
            if let Some(input_texture) = parent.group_target {
                output.push(RenderOperation::EndTarget);
                let effect = &input.group_effects[&node_id];
                let parameters = output.store_parameters(&effect.params);
                let texture = output.allocate_texture();
                output.push(RenderOperation::ApplyEffect(EffectApplication {
                    effect_id: effect.effect_id,
                    parameters,
                    input: input_texture,
                    output: texture,
                }));
                output.push_composite(
                    TextureComposite {
                        texture,
                        placement: TexturePlacement::Target,
                    },
                    self.current.clip,
                );
            }
        }
        None
    }

    fn draw_shape(&self, draw: ShapeDraw) -> RenderCommand {
        RenderCommand {
            operation: RenderOperation::DrawShape(draw),
            clip: self.current.clip,
        }
    }

    fn enter_node(
        &mut self,
        node_id: usize,
        node: &DrawTreeNode,
        draw: Option<ShapeDraw>,
        input: &DrawPlanningInput<'_>,
        _output: &mut RenderPlan,
    ) -> Option<RenderCommand> {
        let should_draw = !rectangles::should_skip_visible_rect_draw(
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
        if let Some(scissor) =
            rectangles::try_scissor_for_rect(node, input.scale_factor, input.physical_size)
        {
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
        Some(RenderCommand {
            operation: RenderOperation::DrawShapeAndIncrementStencil(draw),
            clip,
        })
    }
}

#[cfg(test)]
mod tests;
