use super::damage::{mark_dirty, mark_shape_effect_dirty};
use super::types::DrawCommandError;
use super::{RenderBackend, Renderer};
use crate::commands::ShapeDrawId;
use crate::core::effect::ShapeEffectBounds;
use crate::core::shape::{Shape, ShapeDrawCommandOptions};
use crate::core::vertex::InstanceTransform;
use crate::scene::types::{CachedShapeDrawData, DrawTreeNode};
use crate::scene::{Scene, SceneError};

/// The command to put at an existing node, keeping its ID, parent and children.
#[derive(Debug)]
pub enum DrawCommandReplacement<'shape> {
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

enum NodeTarget {
    Insert { parent: Option<usize> },
    Replace { node_id: usize },
}

fn has_child_clip(node: &DrawTreeNode) -> bool {
    node.clips_children()
        && match node {
            DrawTreeNode::CachedShape(shape) => shape.has_geometry() || node.is_axis_aligned_rect(),
            DrawTreeNode::ClipRect(_) => true,
        }
}

impl<B: RenderBackend> Renderer<B> {
    /// Tessellates into the shared CPU cache. Geometry IDs let identical shapes share uploads.
    pub fn load_shape(
        &mut self,
        shape: impl AsRef<Shape>,
        cache_key: u64,
        geometry_id: Option<u64>,
    ) {
        self.scene.load_shape(shape, cache_key, geometry_id);
    }

    pub fn remove_shape(&mut self, cache_key: u64) {
        self.scene.remove_shape(cache_key);
    }

    /// Queues a loaded shape and prepares backend resources for this instance.
    pub fn add_cached_shape(
        &mut self,
        cache_key: u64,
        parent_shape_id: Option<usize>,
        options: ShapeDrawCommandOptions,
    ) -> Result<usize, DrawCommandError<B::Error>> {
        self.queue_draw_command(
            NodeTarget::Insert {
                parent: parent_shape_id,
            },
            DrawCommandReplacement::CachedShape { cache_key, options },
        )
    }

    /// Queues a shape without retaining it in the loaded-shape cache.
    /// Children inherit clipping unless their parent uses `clips_children(false)`.
    pub fn add_shape(
        &mut self,
        shape: impl AsRef<Shape>,
        parent_shape_id: Option<usize>,
        geometry_id: Option<u64>,
        options: ShapeDrawCommandOptions,
    ) -> Result<usize, DrawCommandError<B::Error>> {
        self.queue_draw_command(
            NodeTarget::Insert {
                parent: parent_shape_id,
            },
            DrawCommandReplacement::Shape {
                shape: shape.as_ref(),
                geometry_id,
                options,
            },
        )
    }

    /// Adds a scissor clip without geometry. Rotation, skew and perspective are rejected.
    pub fn add_clipping_rect(
        &mut self,
        rect_bounds: [(f32, f32); 2],
        parent_shape_id: Option<usize>,
        transform: Option<impl Into<InstanceTransform>>,
        clips_children: bool,
    ) -> Result<usize, DrawCommandError<B::Error>> {
        self.queue_draw_command(
            NodeTarget::Insert {
                parent: parent_shape_id,
            },
            DrawCommandReplacement::ClippingRect {
                rect_bounds,
                transform: transform.map(Into::into),
                clips_children,
            },
        )
    }

    /// Replaces one command, preserving its ID, parent, children and attached effects.
    pub fn replace_shape(
        &mut self,
        node_id: usize,
        shape: impl AsRef<Shape>,
        geometry_id: Option<u64>,
        options: ShapeDrawCommandOptions,
    ) -> Result<(), DrawCommandError<B::Error>> {
        self.replace_draw_commands([(
            node_id,
            DrawCommandReplacement::Shape {
                shape: shape.as_ref(),
                geometry_id,
                options,
            },
        )])
    }

    /// Replaces one command with a loaded shape, retaining its children and effects.
    pub fn replace_cached_shape(
        &mut self,
        node_id: usize,
        cache_key: u64,
        options: ShapeDrawCommandOptions,
    ) -> Result<(), DrawCommandError<B::Error>> {
        self.replace_draw_commands([(
            node_id,
            DrawCommandReplacement::CachedShape { cache_key, options },
        )])
    }

    /// Replaces one command with a clip rectangle, retaining its children.
    /// Effects attached to this node are removed because they require a shape.
    pub fn replace_clipping_rect(
        &mut self,
        node_id: usize,
        rect_bounds: [(f32, f32); 2],
        transform: Option<impl Into<InstanceTransform>>,
        clips_children: bool,
    ) -> Result<(), DrawCommandError<B::Error>> {
        self.replace_draw_commands([(
            node_id,
            DrawCommandReplacement::ClippingRect {
                rect_bounds,
                transform: transform.map(Into::into),
                clips_children,
            },
        )])
    }

    /// Replaces commands in order. Shape replacements retain
    /// effects; clip rectangles remove effects attached to the replaced node.
    ///
    /// Stops at the first error. Earlier replacements remain applied, while the
    /// failing command and later entries remain unchanged. Repeated IDs are allowed.
    pub fn replace_draw_commands<'shape>(
        &mut self,
        replacements: impl IntoIterator<Item = (usize, DrawCommandReplacement<'shape>)>,
    ) -> Result<(), DrawCommandError<B::Error>> {
        let mut should_compact_parameters = false;
        let mut result = Ok(());
        for (node_id, command) in replacements {
            let removes_effects = matches!(command, DrawCommandReplacement::ClippingRect { .. })
                && (self.scene.group_effects.contains_key(&node_id)
                    || self.scene.backdrop_effects.contains_key(&node_id)
                    || self.scene.shape_effects.contains_key(&node_id));
            result = self
                .queue_draw_command(NodeTarget::Replace { node_id }, command)
                .map(|_| ());
            if result.is_err() {
                break;
            }
            should_compact_parameters = should_compact_parameters || removes_effects;
        }
        if should_compact_parameters {
            self.planner.compact_effect_parameters(&mut self.scene);
        }
        result
    }

    fn prepare_draw_command(
        &mut self,
        command: DrawCommandReplacement<'_>,
    ) -> Result<DrawTreeNode, SceneError> {
        Ok(match command {
            DrawCommandReplacement::Shape {
                shape,
                geometry_id,
                options,
            } => DrawTreeNode::CachedShape(CachedShapeDrawData::new(
                self.scene.tessellate(shape, geometry_id),
                options,
            )),
            DrawCommandReplacement::CachedShape { cache_key, options } => {
                DrawTreeNode::CachedShape(CachedShapeDrawData::new(
                    self.scene.loaded_shape(cache_key)?,
                    options,
                ))
            }
            DrawCommandReplacement::ClippingRect {
                rect_bounds,
                transform,
                clips_children,
            } => Scene::prepare_clipping_rect(rect_bounds, transform, clips_children)?,
        })
    }

    fn queue_draw_command(
        &mut self,
        target: NodeTarget,
        command: DrawCommandReplacement<'_>,
    ) -> Result<usize, DrawCommandError<B::Error>> {
        let node_id = match target {
            NodeTarget::Insert { parent } => {
                self.scene.validate_parent(parent)?;
                self.scene.next_node_id()
            }
            NodeTarget::Replace { node_id } => {
                self.scene
                    .draw_tree
                    .get(node_id)
                    .ok_or(SceneError::NodeNotFound(node_id))?;
                node_id
            }
        };
        let node = self.prepare_draw_command(command)?;
        let shape_effect_bounds = match (&node, self.scene.shape_effects.get(&node_id)) {
            (DrawTreeNode::CachedShape(shape), Some(effect)) => Some(
                ShapeEffectBounds::new(
                    shape.instance.cached_shape.local_bounds(),
                    effect.config,
                    shape.instance.transform,
                    self.viewport.scale_factor,
                    self.fringe_width,
                )
                .ok_or(SceneError::InvalidShapeEffectBounds(node_id))?,
            ),
            _ => None,
        };
        if let DrawTreeNode::CachedShape(shape) = &node {
            self.backend
                .register_shape(ShapeDrawId(node_id), &shape.instance)
                .map_err(DrawCommandError::Backend)?;
            if shape_effect_bounds.is_some() {
                self.backend
                    .set_shape_effect_geometry(ShapeDrawId(node_id), &shape.instance.cached_shape);
            }
        } else if matches!(target, NodeTarget::Replace { .. }) {
            self.backend.unregister_shapes(&[ShapeDrawId(node_id)]);
        }
        match target {
            NodeTarget::Insert { parent } => {
                let shape_bounds = match &node {
                    DrawTreeNode::CachedShape(shape) => Some(shape.logical_screen_bounds),
                    DrawTreeNode::ClipRect(_) => None,
                };
                let inserted_id = self.scene.insert_node(node, parent)?;
                debug_assert_eq!(inserted_id, node_id);
                if let Some(bounds) = shape_bounds {
                    mark_dirty(
                        &mut self.dirty_bounds,
                        bounds,
                        self.viewport,
                        self.fringe_width,
                    );
                }
            }
            NodeTarget::Replace { node_id } => {
                self.replace_node(node_id, node, shape_effect_bounds);
            }
        }
        Ok(node_id)
    }

    fn replace_node(
        &mut self,
        node_id: usize,
        node: DrawTreeNode,
        shape_effect_bounds: Option<ShapeEffectBounds>,
    ) {
        let old_effect_bounds = self
            .scene
            .shape_effects
            .get(&node_id)
            .map(|effect| effect.bounds);
        let new_bounds = node.logical_screen_bounds();
        let clips_children = has_child_clip(&node);
        let previous = self.scene.replace_node(
            node_id,
            node,
            shape_effect_bounds,
            self.viewport,
            self.fringe_width,
            self.backend.maximum_texture_dimension(),
        );
        for bounds in [previous.logical_screen_bounds(), new_bounds] {
            mark_dirty(
                &mut self.dirty_bounds,
                bounds,
                self.viewport,
                self.fringe_width,
            );
        }
        for bounds in old_effect_bounds.into_iter().chain(shape_effect_bounds) {
            mark_shape_effect_dirty(&mut self.dirty_bounds, bounds, self.viewport);
        }
        if has_child_clip(&previous) != clips_children
            && !self.scene.draw_tree.children(node_id).is_empty()
        {
            self.pending_clip_damage.insert(node_id);
        }
    }

    pub fn texture_manager(&self) -> &B::TextureManager {
        self.backend.texture_manager()
    }

    /// Removes queued nodes and their descendants.
    ///
    /// Calls `removed` once per removed ID.
    /// Remaining nodes keep their IDs and effects. New nodes may reuse removed IDs.
    /// Loaded shapes and textures remain available.
    pub fn remove_subtrees(
        &mut self,
        node_ids: impl IntoIterator<Item = usize>,
        mut removed: impl FnMut(usize),
    ) {
        self.removed_shape_ids.clear();
        let mut has_removed_nodes = false;
        self.scene
            .remove_subtrees_with(node_ids, |id, node, effect| {
                if matches!(node, DrawTreeNode::CachedShape(_)) {
                    mark_dirty(
                        &mut self.dirty_bounds,
                        node.logical_screen_bounds(),
                        self.viewport,
                        self.fringe_width,
                    );
                }
                if let Some(effect) = effect {
                    mark_shape_effect_dirty(&mut self.dirty_bounds, effect.bounds, self.viewport);
                }
                has_removed_nodes = true;
                if matches!(node, DrawTreeNode::CachedShape(_)) {
                    self.removed_shape_ids.push(ShapeDrawId(id));
                }
                removed(id);
            });
        if !has_removed_nodes {
            return;
        }
        if self.scene.draw_tree.is_empty() {
            self.planner.clear();
            self.backend.clear_draw_queue();
        } else {
            if !self.removed_shape_ids.is_empty() {
                self.backend.unregister_shapes(&self.removed_shape_ids);
            }
            self.planner.compact_effect_parameters(&mut self.scene);
        }
        self.removed_shape_ids.clear();
    }

    pub fn clear_draw_queue(&mut self) {
        self.remove_subtrees([0], |_| {});
    }
}
