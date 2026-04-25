use super::types::{ClipRectDrawData, DrawCommandError};
use super::*;
use crate::gradient::types::Fill;

fn clip_rect_supports_transform(transform: InstanceTransform) -> bool {
    super::rect_utils::extract_axis_aligned_rect_transform(Some(transform)).is_some()
}

impl<'a> Renderer<'a> {
    /// Adds a shape to the draw tree.
    ///
    /// When `parent_shape_id` is `Some`, the new shape is attached as a child of that node.
    /// Children are clipped to their parent unless the parent uses [`ShapeOverflow::Visible`].
    pub fn add_shape(
        &mut self,
        shape: impl AsRef<Shape>,
        parent_shape_id: Option<usize>,
        geometry_id: Option<u64>,
    ) -> Result<usize, DrawCommandError> {
        let cached_shape = CachedShapeHandle::new(
            shape.as_ref(),
            &mut self.tessellator,
            &mut self.buffers_pool_manager,
            geometry_id,
        );
        let draw_data = CachedShapeDrawData::new(cached_shape);
        self.add_draw_command(DrawCommand::CachedShape(draw_data), parent_shape_id)
    }

    /// Adds an axis-aligned scissor clipping rectangle without preparing geometry.
    ///
    /// This node clips its children like a transparent rect parent by default when its
    /// transform preserves axis alignment. Rotated, skewed, or perspective transforms are
    /// rejected by the transform setters because this node intentionally has no geometry
    /// for stencil fallback. Use [`Renderer::set_shape_overflow`] with
    /// [`ShapeOverflow::Visible`] when the node should only group descendants.
    ///
    /// When `parent_shape_id` is `Some`, the clipping rectangle is attached as a child of
    /// that node and inherits ancestor clips.
    pub fn add_clipping_rect(
        &mut self,
        rect_bounds: [(f32, f32); 2],
        parent_shape_id: Option<usize>,
    ) -> Result<usize, DrawCommandError> {
        self.add_draw_command(
            DrawCommand::ClipRect(ClipRectDrawData::new(rect_bounds)),
            parent_shape_id,
        )
    }

    pub fn load_shape(
        &mut self,
        shape: impl AsRef<Shape>,
        cache_key: u64,
        // id to identify the geometry for that shape; Geometry will be deduped by this id when
        //  the geometry is loaded to the GPU.
        geometry_id: Option<u64>,
    ) {
        let cached_shape = CachedShapeHandle::new(
            shape.as_ref(),
            &mut self.tessellator,
            &mut self.buffers_pool_manager,
            geometry_id,
        );
        self.shape_cache.insert(cache_key, cached_shape);
    }

    pub fn load_shape_2(
        &mut self,
        shape: impl AsRef<Shape>,
        // id to identify the geometry for that shape; Geometry will be deduped by this id when
        //  the geometry is loaded to the GPU.
        geometry_id: Option<u64>,
    ) -> CachedShapeHandle {
        let cached_shape = CachedShapeHandle::new(
            shape.as_ref(),
            &mut self.tessellator,
            &mut self.buffers_pool_manager,
            geometry_id,
        );
        cached_shape
    }

    /// Adds a previously loaded cached shape to the draw tree.
    ///
    /// When `parent_shape_id` is `Some`, the cached shape is attached as a child of that node.
    /// Children are clipped to their parent unless the parent uses [`ShapeOverflow::Visible`].
    pub fn add_cached_shape_to_the_render_queue(
        &mut self,
        cache_key: u64,
        parent_shape_id: Option<usize>,
    ) -> Result<usize, DrawCommandError> {
        let draw_data = if let Some(cached_shape_handle) = self.shape_cache.get(&cache_key) {
            CachedShapeDrawData::new(cached_shape_handle.clone())
        } else {
            return Err(DrawCommandError::ShapeNotLoaded(cache_key));
        };
        self.add_draw_command(DrawCommand::CachedShape(draw_data), parent_shape_id)
    }

    pub fn add_cached_shape_to_the_render_queue_by_handle(
        &mut self,
        cached_shape_handle: &CachedShapeHandle,
        parent_shape_id: Option<usize>,
    ) -> Result<usize, DrawCommandError> {
        let draw_data = CachedShapeDrawData::new(cached_shape_handle.clone());
        self.add_draw_command(DrawCommand::CachedShape(draw_data), parent_shape_id)
    }

    pub fn texture_manager(&self) -> &TextureManager {
        &self.texture_manager
    }

    pub fn clear_draw_queue(&mut self) {
        self.draw_tree.clear();
        self.geometry_node_ids.clear();
        self.metadata_to_clips.clear();
        self.group_effects.clear();
        self.backdrop_effects.clear();
        // Keep scratch storage bounded even if queue contents fluctuate frame-to-frame.
        self.trim_scratch_on_resize_or_policy();
        // Clear memory buffers that are used for GPU upload
        self.clear_buffers();
    }

    fn refresh_geometry_cache(&mut self, cached_shape_data: &CachedShapeDrawData) {
        if let Some(geometry_id) = cached_shape_data.cached_shape.geometry_id {
            self.buffers_pool_manager
                .tessellation_cache
                .refresh_vertex_buffers(
                    geometry_id,
                    &cached_shape_data.cached_shape.vertex_buffers,
                );
        }
    }

    fn add_draw_command(
        &mut self,
        mut draw_command: DrawCommand,
        parent_shape_id: Option<usize>,
    ) -> Result<usize, DrawCommandError> {
        let shape_geometry = draw_command.as_shape_draw_data_mut();
        let has_geometry = shape_geometry.is_some();
        if let Some(cached_shape_data) = shape_geometry {
            self.refresh_geometry_cache(cached_shape_data);
            let index_range = preparation::append_aggregated_geometry_for_shape(
                &cached_shape_data,
                &mut self.temp_vertices,
                &mut self.temp_indices,
                &mut self.geometry_dedup_map,
            );
            if let Some((index_start, index_count)) = index_range {
                cached_shape_data.index_buffer_range = Some((index_start, index_count));
                cached_shape_data.is_empty = false;
            } else {
                cached_shape_data.is_empty = true;
            }
        }

        if self.draw_tree.is_empty() {
            let node_id = self.draw_tree.add_node(draw_command);
            if has_geometry {
                self.geometry_node_ids.push(node_id);
            }
            Ok(node_id)
        } else if let Some(parent_shape_id) = parent_shape_id {
            // Mark the parent as non-leaf since it now has a child.
            if let Some(parent) = self.draw_tree.get_mut(parent_shape_id) {
                parent.set_not_leaf();
                let node_id = self.draw_tree.add_child(parent_shape_id, draw_command);
                if has_geometry {
                    self.geometry_node_ids.push(node_id);
                }
                Ok(node_id)
            } else {
                Err(DrawCommandError::InvalidShapeId(parent_shape_id))
            }
        } else {
            // Adding to root — mark root as non-leaf.
            if let Some(root) = self.draw_tree.get_mut(0) {
                root.set_not_leaf();
            }
            let node_id = self.draw_tree.add_child_to_root(draw_command);
            if has_geometry {
                self.geometry_node_ids.push(node_id);
            }
            Ok(node_id)
        }
    }

    pub fn set_shape_transform_cols(
        &mut self,
        node_id: usize,
        cols: [[f32; 4]; 4],
    ) -> Result<(), DrawCommandError> {
        let transform = InstanceTransform {
            col0: cols[0],
            col1: cols[1],
            col2: cols[2],
            col3: cols[3],
        };

        self.set_shape_transform(node_id, transform)
    }

    pub fn set_shape_transform(
        &mut self,
        node_id: usize,
        transform: impl Into<InstanceTransform>,
    ) -> Result<(), DrawCommandError> {
        let transform = transform.into();
        let draw_command = self
            .draw_tree
            .get_mut(node_id)
            .ok_or(DrawCommandError::InvalidShapeId(node_id))?;
        if draw_command.is_clip_rect() && !clip_rect_supports_transform(transform) {
            return Err(DrawCommandError::UnsupportedClipRectTransform(node_id));
        }
        draw_command.set_transform(transform);
        Ok(())
    }

    /// Sets whether a shape or clipping rectangle clips child nodes attached to it.
    ///
    /// [`ShapeOverflow::Hidden`] clips descendants to this shape, while
    /// [`ShapeOverflow::Visible`] lets descendants render outside this shape and inherit the
    /// nearest ancestor clip instead.
    pub fn set_shape_overflow(
        &mut self,
        node_id: usize,
        overflow: ShapeOverflow,
    ) -> Result<(), DrawCommandError> {
        let draw_command = self
            .draw_tree
            .get_mut(node_id)
            .ok_or(DrawCommandError::InvalidShapeId(node_id))?;
        draw_command.set_clips_children(overflow.clips_children());
        Ok(())
    }

    /// Same as for set_shape_overflow, but you can pass the boolean directly
    pub fn set_clips_children(
        &mut self,
        node_id: usize,
        clips_children: bool,
    ) -> Result<(), DrawCommandError> {
        let draw_command = self
            .draw_tree
            .get_mut(node_id)
            .ok_or(DrawCommandError::InvalidShapeId(node_id))?;
        draw_command.set_clips_children(clips_children);
        Ok(())
    }

    pub fn set_shape_texture(
        &mut self,
        node_id: usize,
        texture_id: Option<u64>,
    ) -> Result<(), DrawCommandError> {
        self.set_shape_texture_layer(node_id, 0, texture_id)
    }

    pub fn set_shape_texture_layer(
        &mut self,
        node_id: usize,
        layer: usize,
        texture_id: Option<u64>,
    ) -> Result<(), DrawCommandError> {
        if layer > 1 {
            return Err(DrawCommandError::InvalidTextureLayer(layer));
        }

        let draw_command = self
            .draw_tree
            .get_mut(node_id)
            .ok_or(DrawCommandError::InvalidShapeId(node_id))?;
        if draw_command.is_clip_rect() {
            return if texture_id.is_some() {
                Err(DrawCommandError::UnsupportedClipRectOperation(
                    node_id, "textures",
                ))
            } else {
                Ok(())
            };
        }
        draw_command.set_texture_id(layer, texture_id);
        Ok(())
    }

    pub fn set_shape_texture_on(
        &mut self,
        node_id: usize,
        layer: TextureLayer,
        texture_id: Option<u64>,
    ) -> Result<(), DrawCommandError> {
        self.set_shape_texture_layer(node_id, layer.into(), texture_id)
    }

    pub fn set_shape_color(
        &mut self,
        node_id: usize,
        color: Option<Color>,
    ) -> Result<(), DrawCommandError> {
        let normalized_color = color.map(|value| value.normalize());
        let draw_command = self
            .draw_tree
            .get_mut(node_id)
            .ok_or(DrawCommandError::InvalidShapeId(node_id))?;
        if draw_command.is_clip_rect() {
            return Err(DrawCommandError::UnsupportedClipRectOperation(
                node_id, "color",
            ));
        }
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
        Ok(())
    }

    pub fn set_shape_fill(
        &mut self,
        node_id: usize,
        fill: Option<Fill>,
    ) -> Result<(), DrawCommandError> {
        let draw_command = self
            .draw_tree
            .get_mut(node_id)
            .ok_or(DrawCommandError::InvalidShapeId(node_id))?;
        if draw_command.is_clip_rect() {
            return Err(DrawCommandError::UnsupportedClipRectOperation(
                node_id, "fill",
            ));
        }
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
        Ok(())
    }
}
