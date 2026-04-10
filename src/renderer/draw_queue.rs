use super::prepared_scene::PreparedGeometryKey;
use super::*;
use crate::gradient::types::Fill;

impl<'a> Renderer<'a> {
    fn should_mark_existing_instance_dirty(
        prepared_scene: &super::prepared_scene::PreparedScene,
        draw_command: &DrawCommand,
    ) -> bool {
        !prepared_scene.topology_dirty || draw_command.instance_index().is_some()
    }

    pub fn add_shape(
        &mut self,
        shape: impl Into<Shape>,
        clip_to_shape: Option<usize>,
        cache_key: Option<u64>,
    ) -> usize {
        self.add_draw_command(
            DrawCommand::Shape(ShapeDrawData::new(shape, cache_key)),
            clip_to_shape,
        )
    }

    pub fn load_shape(
        &mut self,
        shape: impl AsRef<Shape>,
        cache_key: u64,
        tessellation_cache_key: Option<u64>,
    ) {
        let cached_shape = CachedShape::new(
            shape.as_ref(),
            &mut self.tessellator,
            &mut self.buffers_pool_manager,
            tessellation_cache_key,
        );
        self.shape_cache.insert(cache_key, cached_shape);
        self.prepared_scene
            .mark_geometry_key_dirty(PreparedGeometryKey::CachedShape(cache_key));

        let cached_shape = self.shape_cache.get(&cache_key);
        for (_node_id, draw_command) in self.draw_tree.iter_mut() {
            let DrawCommand::CachedShape(cached_shape_draw_data) = draw_command else {
                continue;
            };

            if cached_shape_draw_data.id == cache_key {
                match cached_shape {
                    Some(cached_shape) => {
                        cached_shape_draw_data.is_rect = cached_shape.is_rect;
                        cached_shape_draw_data.rect_bounds = cached_shape.rect_bounds;
                    }
                    None => {
                        cached_shape_draw_data.is_rect = false;
                        cached_shape_draw_data.rect_bounds = None;
                    }
                }
            }
        }
    }

    pub fn add_cached_shape_to_the_render_queue(
        &mut self,
        cache_key: u64,
        clip_to_shape: Option<usize>,
    ) -> usize {
        let draw_data = if let Some(cached) = self.shape_cache.get(&cache_key) {
            if cached.is_rect {
                if let Some(bounds) = cached.rect_bounds {
                    CachedShapeDrawData::new_rect(cache_key, bounds)
                } else {
                    CachedShapeDrawData::new(cache_key)
                }
            } else {
                CachedShapeDrawData::new(cache_key)
            }
        } else {
            CachedShapeDrawData::new(cache_key)
        };
        self.add_draw_command(DrawCommand::CachedShape(draw_data), clip_to_shape)
    }

    pub fn texture_manager(&self) -> &TextureManager {
        &self.texture_manager
    }

    pub fn clear_draw_queue(&mut self) {
        self.draw_tree.clear();
        self.metadata_to_clips.clear();
        self.group_effects.clear();
        self.backdrop_effects.clear();
        self.prepared_scene.clear();
        // Keep scratch storage bounded even if queue contents fluctuate frame-to-frame.
        self.trim_scratch_on_resize_or_policy();
    }

    pub fn remove_subtree(&mut self, node_id: usize) {
        if self.draw_tree.get(node_id).is_none() {
            return;
        }

        let parent_node_id = self.draw_tree.parent_index_unchecked(node_id);
        if self.draw_tree.children(node_id).is_empty() {
            self.draw_tree.remove_subtree(node_id);
            self.metadata_to_clips.remove(&node_id);
            self.group_effects.remove(&node_id);
            self.backdrop_effects.remove(&node_id);
            self.prepared_scene.remove_node(node_id);

            if let Some(parent_node_id) = parent_node_id {
                self.sync_leaf_state(parent_node_id);
            }
            return;
        }

        let mut removed_node_ids =
            std::mem::take(&mut self.prepared_scene.removed_node_ids_scratch);
        let mut removed_node_id_set =
            std::mem::take(&mut self.prepared_scene.removed_node_id_set_scratch);
        let mut traversal_stack = std::mem::take(&mut self.prepared_scene.traversal_stack_scratch);
        collect_subtree_node_ids_into(
            &self.draw_tree,
            node_id,
            &mut removed_node_ids,
            &mut traversal_stack,
        );
        removed_node_id_set.clear();
        removed_node_id_set.extend(removed_node_ids.iter().copied());

        self.draw_tree.remove_subtree(node_id);
        self.remove_metadata_and_effects_for_removed_nodes(&removed_node_id_set);
        self.prepared_scene.remove_nodes(&removed_node_ids);

        removed_node_ids.clear();
        removed_node_id_set.clear();
        traversal_stack.clear();
        self.prepared_scene.removed_node_ids_scratch = removed_node_ids;
        self.prepared_scene.removed_node_id_set_scratch = removed_node_id_set;
        self.prepared_scene.traversal_stack_scratch = traversal_stack;

        if let Some(parent_node_id) = parent_node_id {
            self.sync_leaf_state(parent_node_id);
        }
    }

    fn add_draw_command(
        &mut self,
        draw_command: DrawCommand,
        clip_to_shape: Option<usize>,
    ) -> usize {
        let node_id = if self.draw_tree.is_empty() {
            self.draw_tree.add_node(draw_command)
        } else if let Some(clip_to_shape) = clip_to_shape {
            // Mark the parent as non-leaf since it now has a child.
            if let Some(parent) = self.draw_tree.get_mut(clip_to_shape) {
                parent.set_not_leaf();
            }
            self.draw_tree.add_child(clip_to_shape, draw_command)
        } else {
            // Adding to root — mark root as non-leaf.
            if let Some(root) = self.draw_tree.get_mut(0) {
                root.set_not_leaf();
            }
            self.draw_tree.add_child_to_root(draw_command)
        };

        self.prepared_scene.mark_topology_changed();
        self.prepared_scene.mark_node_instance_dirty(node_id);
        node_id
    }

    fn remove_metadata_and_effects_for_removed_nodes(
        &mut self,
        removed_node_id_set: &ahash::HashSet<usize>,
    ) {
        self.metadata_to_clips.retain(|node_id, clip_node_id| {
            !removed_node_id_set.contains(node_id) && !removed_node_id_set.contains(clip_node_id)
        });
        self.group_effects
            .retain(|node_id, _| !removed_node_id_set.contains(node_id));
        self.backdrop_effects
            .retain(|node_id, _| !removed_node_id_set.contains(node_id));
    }

    fn sync_leaf_state(&mut self, node_id: usize) {
        if self.draw_tree.get(node_id).is_none() {
            return;
        }

        let node_has_children = !self.draw_tree.children(node_id).is_empty();
        if let Some(draw_command) = self.draw_tree.get_mut(node_id) {
            draw_command.set_leaf_state(!node_has_children);
        }
    }

    pub fn set_shape_transform_cols(&mut self, node_id: usize, cols: [[f32; 4]; 4]) {
        let transform = InstanceTransform {
            col0: cols[0],
            col1: cols[1],
            col2: cols[2],
            col3: cols[3],
        };

        let should_mark_dirty = {
            let Some(draw_command) = self.draw_tree.get_mut(node_id) else {
                return;
            };
            let should_mark_dirty =
                Self::should_mark_existing_instance_dirty(&self.prepared_scene, draw_command);
            draw_command.set_transform(transform);
            should_mark_dirty
        };

        if should_mark_dirty {
            self.prepared_scene.mark_node_instance_dirty(node_id);
        }
    }

    pub fn set_shape_transform(&mut self, node_id: usize, transform: impl Into<InstanceTransform>) {
        let transform = transform.into();
        let should_mark_dirty = {
            let Some(draw_command) = self.draw_tree.get_mut(node_id) else {
                return;
            };
            let should_mark_dirty =
                Self::should_mark_existing_instance_dirty(&self.prepared_scene, draw_command);
            draw_command.set_transform(transform);
            should_mark_dirty
        };

        if should_mark_dirty {
            self.prepared_scene.mark_node_instance_dirty(node_id);
        }
    }

    pub fn set_shape_texture(&mut self, node_id: usize, texture_id: Option<u64>) {
        self.set_shape_texture_layer(node_id, 0, texture_id);
    }

    pub fn set_shape_texture_layer(
        &mut self,
        node_id: usize,
        layer: usize,
        texture_id: Option<u64>,
    ) {
        if layer > 1 {
            return;
        }

        let should_mark_dirty = {
            let Some(draw_command) = self.draw_tree.get_mut(node_id) else {
                return;
            };
            let should_mark_dirty =
                Self::should_mark_existing_instance_dirty(&self.prepared_scene, draw_command);
            draw_command.set_texture_id(layer, texture_id);
            should_mark_dirty
        };

        if should_mark_dirty {
            self.prepared_scene.mark_node_instance_dirty(node_id);
        }
    }

    pub fn set_shape_texture_on(
        &mut self,
        node_id: usize,
        layer: TextureLayer,
        texture_id: Option<u64>,
    ) {
        self.set_shape_texture_layer(node_id, layer.into(), texture_id);
    }

    pub fn set_shape_color(&mut self, node_id: usize, color: Option<Color>) {
        let normalized_color = color.map(|value| value.normalize());
        let should_mark_dirty = {
            let Some(draw_command) = self.draw_tree.get_mut(node_id) else {
                return;
            };
            let should_mark_dirty =
                Self::should_mark_existing_instance_dirty(&self.prepared_scene, draw_command);
            draw_command.set_instance_color_override(normalized_color);
            // set_shape_color is sugar for Fill::Solid / None
            let fill = color.map(Fill::Solid);
            draw_command.set_fill(fill);
            draw_command.refresh_gradient_bind_group(
                &mut self.buffers_pool_manager.gradient_cache,
                &self.device,
                &self.queue,
                &self.gradient_bind_group_layout,
                &self.gradient_ramp_sampler,
                self.gradient_bind_group_layout_epoch,
            );
            should_mark_dirty
        };

        if should_mark_dirty {
            self.prepared_scene.mark_node_instance_dirty(node_id);
        }
    }

    pub fn set_shape_fill(&mut self, node_id: usize, fill: Option<Fill>) {
        let should_mark_dirty = {
            let Some(draw_command) = self.draw_tree.get_mut(node_id) else {
                return;
            };
            let should_mark_dirty =
                Self::should_mark_existing_instance_dirty(&self.prepared_scene, draw_command);
            // Derive color_override from fill for the solid fast path
            let color_override = match &fill {
                Some(Fill::Solid(color)) => Some(color.normalize()),
                _ => None,
            };
            draw_command.set_instance_color_override(color_override);
            draw_command.set_fill(fill);
            draw_command.refresh_gradient_bind_group(
                &mut self.buffers_pool_manager.gradient_cache,
                &self.device,
                &self.queue,
                &self.gradient_bind_group_layout,
                &self.gradient_ramp_sampler,
                self.gradient_bind_group_layout_epoch,
            );
            should_mark_dirty
        };

        if should_mark_dirty {
            self.prepared_scene.mark_node_instance_dirty(node_id);
        }
    }
}

fn collect_subtree_node_ids_into(
    draw_tree: &easy_tree::Tree<DrawCommand>,
    root_node_id: usize,
    subtree_node_ids: &mut Vec<usize>,
    traversal_stack: &mut Vec<usize>,
) {
    if draw_tree.get(root_node_id).is_none() {
        subtree_node_ids.clear();
        traversal_stack.clear();
        return;
    }

    subtree_node_ids.clear();
    traversal_stack.clear();
    traversal_stack.push(root_node_id);

    while let Some(node_id) = traversal_stack.pop() {
        if draw_tree.get(node_id).is_none() {
            continue;
        }

        subtree_node_ids.push(node_id);
        traversal_stack.extend(draw_tree.children(node_id).iter().copied());
    }
}
