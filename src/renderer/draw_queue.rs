use super::*;

impl<'a> Renderer<'a> {
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
    }

    pub fn add_cached_shape_to_the_render_queue(
        &mut self,
        cache_key: u64,
        clip_to_shape: Option<usize>,
    ) -> usize {
        self.add_draw_command(
            DrawCommand::CachedShape(CachedShapeDrawData::new(cache_key)),
            clip_to_shape,
        )
    }

    pub fn texture_manager(&self) -> &TextureManager {
        &self.texture_manager
    }

    pub fn clear_draw_queue(&mut self) {
        self.draw_tree.clear();
        self.metadata_to_clips.clear();
        self.group_effects.clear();
        self.backdrop_effects.clear();
        self.trim_scratch_on_resize_or_policy();
    }

    fn add_draw_command(
        &mut self,
        draw_command: DrawCommand,
        clip_to_shape: Option<usize>,
    ) -> usize {
        if self.draw_tree.is_empty() {
            self.draw_tree.add_node(draw_command)
        } else if let Some(clip_to_shape) = clip_to_shape {
            self.draw_tree.add_child(clip_to_shape, draw_command)
        } else {
            self.draw_tree.add_child_to_root(draw_command)
        }
    }

    fn mutate_draw_command(&mut self, node_id: usize, mutator: impl FnOnce(&mut DrawCommand)) {
        if let Some(draw_command) = self.draw_tree.get_mut(node_id) {
            mutator(draw_command);
        }
    }

    pub fn set_shape_transform_rows(&mut self, node_id: usize, rows: [[f32; 4]; 4]) {
        let transform = InstanceTransform {
            row0: rows[0],
            row1: rows[1],
            row2: rows[2],
            row3: rows[3],
        };

        self.mutate_draw_command(node_id, |draw_command| {
            draw_command.set_transform(transform)
        });
    }

    pub fn set_shape_transform(&mut self, node_id: usize, transform: impl Into<InstanceTransform>) {
        let transform = transform.into();
        self.mutate_draw_command(node_id, |draw_command| {
            draw_command.set_transform(transform)
        });
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

        self.mutate_draw_command(node_id, |draw_command| {
            draw_command.set_texture_id(layer, texture_id)
        });
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
        self.mutate_draw_command(node_id, |draw_command| {
            draw_command.set_instance_color_override(normalized_color)
        });
    }
}
