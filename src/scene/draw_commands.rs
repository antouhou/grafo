use super::damage::{mark_dirty, mark_physical_dirty, mark_shape_mask_effect_dirty};
use super::{DrawCommandError, Scene};
use crate::commands::ShapeDrawId;
use crate::core::effect::ShapeMaskEffectBounds;
use crate::core::shape::{Shape, ShapeDrawCommandOptions};
use crate::core::vertex::InstanceTransform;
use crate::render_backend::RenderBackend;
use crate::scene_state::effects::EffectConfig;
use crate::scene_state::types::{CachedShapeDrawData, DrawTreeNode};
use crate::scene_state::{SceneError, SceneState};

#[derive(Debug)]
pub(super) enum DrawCommand<'shape> {
    Shape {
        shape: &'shape Shape,
        geometry_id: Option<u64>,
        options: ShapeDrawCommandOptions,
    },
    CachedShape {
        cache_key: u64,
        options: ShapeDrawCommandOptions,
    },
    ClippingRect {
        rect_bounds: [(f32, f32); 2],
        transform: Option<InstanceTransform>,
        clips_children: bool,
    },
}

pub(super) enum NodeTarget {
    Insert { parent: Option<usize> },
    Replace { node_id: usize },
}

pub(super) fn has_child_clip(node: &DrawTreeNode) -> bool {
    node.clips_children()
        && match node {
            DrawTreeNode::Group(_) => false,
            DrawTreeNode::CachedShape(shape) => shape.has_geometry() || node.is_axis_aligned_rect(),
            DrawTreeNode::ClipRect(_) => true,
        }
}

impl<B: RenderBackend> Scene<B> {
    fn prepare_draw_command(
        &mut self,
        command: DrawCommand<'_>,
    ) -> Result<DrawTreeNode, SceneError> {
        Ok(match command {
            DrawCommand::Shape {
                shape,
                geometry_id,
                options,
            } => DrawTreeNode::CachedShape(CachedShapeDrawData::new(
                self.state.tessellate(shape, geometry_id),
                options,
            )),
            DrawCommand::CachedShape { cache_key, options } => DrawTreeNode::CachedShape(
                CachedShapeDrawData::new(self.state.loaded_shape(cache_key)?, options),
            ),
            DrawCommand::ClippingRect {
                rect_bounds,
                transform,
                clips_children,
            } => SceneState::prepare_clipping_rect(rect_bounds, transform, clips_children)?,
        })
    }

    pub(super) fn replace_node(
        &mut self,
        node_id: usize,
        node: DrawTreeNode,
        shape_mask_effect_bounds: Option<ShapeMaskEffectBounds>,
    ) {
        let previous_group_output = self.state.content_effect_output(node_id);
        let old_effect_bounds = self
            .state
            .shape_mask_effects
            .get(&node_id)
            .map(|effect| effect.bounds);
        let new_bounds = node.logical_screen_bounds();
        let clips_children = has_child_clip(&node);
        let previous = self.state.replace_node(
            node_id,
            node,
            shape_mask_effect_bounds,
            self.backend.maximum_texture_dimension(),
        );
        for bounds in [previous.logical_screen_bounds(), new_bounds] {
            mark_dirty(&mut self.dirty_bounds, bounds, self.state.viewport());
        }
        for bounds in old_effect_bounds
            .into_iter()
            .chain(shape_mask_effect_bounds)
        {
            mark_shape_mask_effect_dirty(&mut self.dirty_bounds, bounds, self.state.viewport());
        }
        mark_physical_dirty(&mut self.dirty_bounds, previous_group_output);
        mark_physical_dirty(
            &mut self.dirty_bounds,
            self.state.content_effect_output(node_id),
        );
        if has_child_clip(&previous) != clips_children
            && !self.state.draw_tree.children(node_id).is_empty()
        {
            self.pending_damage.insert(node_id);
        }
    }

    pub(super) fn queue_draw_command(
        &mut self,
        target: NodeTarget,
        command: DrawCommand<'_>,
    ) -> Result<usize, DrawCommandError<B::Error>> {
        let node_id = match target {
            NodeTarget::Insert { parent } => {
                self.state.validate_parent(parent)?;
                self.state.next_node_id()
            }
            NodeTarget::Replace { node_id } => {
                self.state
                    .draw_tree
                    .get(node_id)
                    .ok_or(SceneError::NodeNotFound(node_id))?;
                node_id
            }
        };
        let previous_dirty = self.dirty_bounds;
        let node = self.prepare_draw_command(command)?;
        let replacement_effects = match (&target, &node) {
            (NodeTarget::Replace { node_id }, DrawTreeNode::CachedShape(_))
                if !matches!(
                    self.state.draw_tree.get_unchecked(*node_id),
                    DrawTreeNode::CachedShape(_)
                ) =>
            {
                let mut effects = self.state.node_effects(*node_id);
                effects.inherit_from(self.state.inherited_effects(*node_id));
                Some(effects)
            }
            _ => None,
        };
        if let (Some(effects), DrawTreeNode::CachedShape(shape)) = (replacement_effects, &node) {
            for effect in effects.iter() {
                self.backend
                    .validate_effect_params(
                        effect.instance.effect_id,
                        self.state
                            .effect_parameters
                            .get(effect.instance.parameters)
                            .bytes,
                    )
                    .map_err(DrawCommandError::Backend)?;
                if let EffectConfig::ShapeMask(config) = effect.config {
                    ShapeMaskEffectBounds::new(
                        shape.instance.cached_shape.local_bounds(),
                        config,
                        shape.instance.transform,
                        self.state.viewport().scale_factor,
                    )
                    .ok_or(SceneError::InvalidShapeMaskEffectBounds(node_id))?;
                }
            }
        }
        let shape_mask_effect_bounds = match (&node, self.state.shape_mask_effects.get(&node_id)) {
            (DrawTreeNode::CachedShape(shape), Some(effect)) => Some(
                ShapeMaskEffectBounds::new(
                    shape.instance.cached_shape.local_bounds(),
                    effect.config,
                    shape.instance.transform,
                    self.state.viewport().scale_factor,
                )
                .ok_or(SceneError::InvalidShapeMaskEffectBounds(node_id))?,
            ),
            _ => None,
        };
        let member_effects = if matches!(target, NodeTarget::Replace { .. })
            && matches!(node, DrawTreeNode::ClipRect(_))
            && matches!(
                self.state.draw_tree.get_unchecked(node_id),
                DrawTreeNode::CachedShape(_)
            ) {
            Some(
                self.prepare_inherited_member_effects(
                    node_id,
                    self.state.inherited_effects(node_id),
                )?,
            )
        } else {
            None
        };
        if let DrawTreeNode::CachedShape(shape) = &node {
            self.backend
                .register_shape(ShapeDrawId(node_id), &shape.instance)
                .map_err(DrawCommandError::Backend)?;
            if shape_mask_effect_bounds.is_some() {
                self.backend.set_shape_mask_effect_geometry(
                    ShapeDrawId(node_id),
                    &shape.instance.cached_shape,
                );
            }
        } else if matches!(target, NodeTarget::Replace { .. }) {
            self.backend.unregister_shapes(&[ShapeDrawId(node_id)]);
        }
        match target {
            NodeTarget::Insert { parent } => {
                let shape_bounds = match &node {
                    DrawTreeNode::CachedShape(shape) => Some(shape.logical_screen_bounds),
                    DrawTreeNode::Group(_) | DrawTreeNode::ClipRect(_) => None,
                };
                let inserted_id = self.state.insert_node(node, parent);
                debug_assert_eq!(inserted_id, node_id);
                if let Some(bounds) = shape_bounds {
                    mark_dirty(&mut self.dirty_bounds, bounds, self.state.viewport());
                }
                if shape_bounds.is_some() {
                    let effects = self.state.inherited_effects(node_id);
                    if let Err(error) = self.apply_effects(node_id, effects) {
                        self.remove_subtrees([node_id], |_| {});
                        self.dirty_bounds = previous_dirty;
                        return Err(error);
                    }
                }
            }
            NodeTarget::Replace { node_id } => {
                self.remove_text_glyphs(node_id);
                if matches!(
                    (self.state.draw_tree.get_unchecked(node_id), &node),
                    (DrawTreeNode::Group(_), DrawTreeNode::ClipRect(_))
                ) {
                    self.remove_content_effect(node_id);
                    self.remove_backdrop_effect(node_id);
                    self.remove_shape_mask_effect(node_id);
                }
                self.replace_node(node_id, node, shape_mask_effect_bounds);
                if let Some(effects) = replacement_effects {
                    self.remove_inherited_member_effects(node_id);
                    self.apply_effects(node_id, effects)?;
                }
                if let Some(effects) = member_effects {
                    self.apply_member_effects(effects)?;
                }
            }
        }
        Ok(node_id)
    }
}
