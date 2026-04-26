use super::types::{ClipRectDrawData, DrawCommandError};
use super::*;
use crate::ShapeDrawCommandOptions;

fn clip_rect_supports_transform(transform: InstanceTransform) -> bool {
    rect_utils::extract_axis_aligned_rect_transform(Some(transform)).is_some()
}

impl<'a> Renderer<'a> {
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
        options: ShapeDrawCommandOptions,
    ) -> Result<usize, DrawCommandError> {
        let draw_data = if let Some(cached_shape_handle) = self.shape_cache.get(&cache_key) {
            CachedShapeDrawData::new(cached_shape_handle.clone(), options)
        } else {
            return Err(DrawCommandError::ShapeNotLoaded(cache_key));
        };
        self.add_draw_command(DrawCommand::CachedShape(draw_data), parent_shape_id)
    }

    /// Adds a shape to the draw tree.
    ///
    /// When `parent_shape_id` is `Some`, the new shape is attached as a child of that node.
    /// Children are clipped to their parent unless the parent uses [`ShapeOverflow::Visible`].
    pub fn add_shape(
        &mut self,
        shape: impl AsRef<Shape>,
        parent_shape_id: Option<usize>,
        geometry_id: Option<u64>,
        options: ShapeDrawCommandOptions,
    ) -> Result<usize, DrawCommandError> {
        let cached_shape = CachedShapeHandle::new(
            shape.as_ref(),
            &mut self.tessellator,
            &mut self.buffers_pool_manager,
            geometry_id,
        );
        let draw_data = CachedShapeDrawData::new(cached_shape, options);

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
        transform: Option<impl Into<InstanceTransform>>,
        clips_children: bool,
    ) -> Result<usize, DrawCommandError> {
        let transform = transform.map(Into::into);
        if let Some(transform) = transform {
            if !clip_rect_supports_transform(transform) {
                return Err(DrawCommandError::UnsupportedClipRectTransform);
            }
        }
        self.add_draw_command(
            DrawCommand::ClipRect(ClipRectDrawData::new(
                rect_bounds,
                transform,
                clips_children,
            )),
            parent_shape_id,
        )
    }

    fn add_draw_command(
        &mut self,
        mut draw_command: DrawCommand,
        parent_shape_id: Option<usize>,
    ) -> Result<usize, DrawCommandError> {
        let shape_geometry = draw_command.as_shape_draw_data_mut();
        if let Some(cached_shape_data) = shape_geometry {
            self.refresh_geometry_cache(cached_shape_data);
            cached_shape_data.refresh_gradient_bind_group(
                &mut self.buffers_pool_manager.gradient_cache,
                &self.device,
                &self.queue,
                &self.gradient_bind_group_layout,
                &self.gradient_ramp_sampler,
                self.gradient_bind_group_layout_epoch,
            );
            let index_range = preparation::append_aggregated_geometry_for_shape(
                cached_shape_data,
                &mut self.temp_vertices,
                &mut self.temp_indices,
                &mut self.geometry_dedup_map,
            );
            if let Some((index_start, index_count)) = index_range {
                cached_shape_data.index_buffer_range = Some((index_start, index_count));
                cached_shape_data.is_empty = false;
                let instance_index = preparation::append_instance_data(
                    &mut self.temp_instance_transforms,
                    &mut self.temp_instance_colors,
                    &mut self.temp_instance_metadata,
                    cached_shape_data.transform(),
                    cached_shape_data.instance_color_override(),
                    cached_shape_data.texture_ids,
                );
                *cached_shape_data.instance_index_mut() = Some(instance_index);
            } else {
                cached_shape_data.is_empty = true;
            }
        }

        if self.draw_tree.is_empty() {
            let node_id = self.draw_tree.add_node(draw_command);
            Ok(node_id)
        } else if let Some(parent_shape_id) = parent_shape_id {
            // Mark the parent as non-leaf since it now has a child.
            if let Some(parent) = self.draw_tree.get_mut(parent_shape_id) {
                parent.set_not_leaf();
                let node_id = self.draw_tree.add_child(parent_shape_id, draw_command);
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
            Ok(node_id)
        }
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
        // Clear memory buffers that are used for GPU upload
        self.clear_buffers();
    }
}
