use super::types::DrawCommandError;
use super::*;
use crate::gradient::types::Fill;

impl<'a> Renderer<'a> {
    pub fn add_shape(
        &mut self,
        shape: impl Into<Shape>,
        clip_to_shape: Option<usize>,
        cache_key: Option<u64>,
    ) -> Result<usize, DrawCommandError> {
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
    }

    pub fn add_cached_shape_to_the_render_queue(
        &mut self,
        cache_key: u64,
        clip_to_shape: Option<usize>,
    ) -> Result<usize, DrawCommandError> {
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
        // Keep scratch storage bounded even if queue contents fluctuate frame-to-frame.
        self.trim_scratch_on_resize_or_policy();
    }

    fn add_draw_command(
        &mut self,
        draw_command: DrawCommand,
        clip_to_shape: Option<usize>,
    ) -> Result<usize, DrawCommandError> {
        if self.draw_tree.is_empty() {
            Ok(self.draw_tree.add_node(draw_command))
        } else if let Some(clip_to_shape) = clip_to_shape {
            // Mark the parent as non-leaf since it now has a child.
            if let Some(parent) = self.draw_tree.get_mut(clip_to_shape) {
                parent.set_not_leaf();
                Ok(self.draw_tree.add_child(clip_to_shape, draw_command))
            } else {
                Err(DrawCommandError::InvalidShapeId(clip_to_shape))
            }
        } else {
            // Adding to root — mark root as non-leaf.
            if let Some(root) = self.draw_tree.get_mut(0) {
                root.set_not_leaf();
            }
            Ok(self.draw_tree.add_child_to_root(draw_command))
        }
    }

    pub fn set_shape_transform_cols(&mut self, node_id: usize, cols: [[f32; 4]; 4]) {
        let transform = InstanceTransform {
            col0: cols[0],
            col1: cols[1],
            col2: cols[2],
            col3: cols[3],
        };

        let Some(draw_command) = self.draw_tree.get_mut(node_id) else {
            return;
        };
        draw_command.set_transform(transform);
    }

    pub fn set_shape_transform(&mut self, node_id: usize, transform: impl Into<InstanceTransform>) {
        let transform = transform.into();
        let Some(draw_command) = self.draw_tree.get_mut(node_id) else {
            return;
        };
        draw_command.set_transform(transform);
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

        let Some(draw_command) = self.draw_tree.get_mut(node_id) else {
            return;
        };
        draw_command.set_texture_id(layer, texture_id);
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
        let Some(draw_command) = self.draw_tree.get_mut(node_id) else {
            return;
        };
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
    }

    pub fn set_shape_fill(&mut self, node_id: usize, fill: Option<Fill>) {
        let Some(draw_command) = self.draw_tree.get_mut(node_id) else {
            return;
        };
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
    }
}
