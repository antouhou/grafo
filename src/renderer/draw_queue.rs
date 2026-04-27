use super::types::{ClipRectDrawData, DrawCommandError};
use super::*;
use crate::ShapeDrawCommandOptions;
use crate::ShapeTextureFitMode;

fn clip_rect_supports_transform(transform: InstanceTransform) -> bool {
    rect_utils::extract_axis_aligned_rect_transform(Some(transform)).is_some()
}

impl<'a> Renderer<'a> {
    /// Tessellates the shape and stores the tessellated result in a cache, so it can be accessed
    /// later with the provided it. Accepts optional `geometry_id` to dedupe geometry and avoid
    /// loading the same geometry multiple times. `geometry_id` should be a stable id describing
    /// that particular shape path. Pass `None` if you're not sure what that means. Or use a hash
    /// of the points in the path if you're sure that you're going to draw a lot of the same shapes.
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

    /// Adds a previously loaded cached shape to the draw tree.
    ///
    /// When `parent_shape_id` is `Some`, the cached shape is attached as a child of that node.
    /// Children are clipped to their parent unless the parent was queued with
    /// [`ShapeDrawCommandOptions::clips_children(false)`].
    pub fn add_cached_shape_to_the_render_queue(
        &mut self,
        cache_key: u64,
        parent_shape_id: Option<usize>,
        options: ShapeDrawCommandOptions,
    ) -> Result<usize, DrawCommandError> {
        let mut draw_data = if let Some(cached_shape_handle) = self.shape_cache.get(&cache_key) {
            CachedShapeDrawData::new(cached_shape_handle.clone(), &options)
        } else {
            return Err(DrawCommandError::ShapeNotLoaded(cache_key));
        };
        self.append_buffers_for_shape(&mut draw_data, &options);
        self.add_draw_command(DrawCommand::CachedShape(draw_data), parent_shape_id)
    }

    /// Adds a shape to the draw tree.
    ///
    /// When `parent_shape_id` is `Some`, the new shape is attached as a child of that node.
    /// Children are clipped to their parent unless the parent was queued with
    /// [`ShapeDrawCommandOptions::clips_children(false)`].
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
        let mut draw_data = CachedShapeDrawData::new(cached_shape, &options);

        self.append_buffers_for_shape(&mut draw_data, &options);
        self.add_draw_command(DrawCommand::CachedShape(draw_data), parent_shape_id)
    }

    /// Adds an axis-aligned scissor clipping rectangle without preparing geometry.
    ///
    /// This node clips its children like a transparent rect parent by default when its
    /// transform preserves axis alignment. Rotated, skewed, or perspective transforms are
    /// rejected by the transform setters because this node intentionally has no geometry
    /// for stencil fallback. To let children overflow from a shape parent, queue that parent with
    /// [`ShapeDrawCommandOptions::clips_children(false)`] instead of relying on the older
    /// overflow API wording.
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

    fn append_buffers_for_shape(
        &mut self,
        cached_shape_data: &mut CachedShapeDrawData,
        draw_options: &ShapeDrawCommandOptions,
    ) {
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
            let texture_uv_scales = self.compute_texture_uv_scales(
                cached_shape_data.cached_shape.texture_mapping_size,
                draw_options,
            );
            let instance_index = preparation::append_instance_data(
                &mut self.temp_instance_transforms,
                &mut self.temp_instance_colors,
                &mut self.temp_instance_metadata,
                draw_options.transform,
                match &draw_options.fill {
                    None => None,
                    Some(fill) => fill.to_normalized_solid(),
                },
                preparation::InstanceTextureData {
                    texture_ids: cached_shape_data.texture_ids,
                    texture_uv_scales,
                },
            );
            *cached_shape_data.instance_index_mut() = Some(instance_index);
        } else {
            cached_shape_data.is_empty = true;
        }
    }

    fn add_draw_command(
        &mut self,
        draw_command: DrawCommand,
        parent_shape_id: Option<usize>,
    ) -> Result<usize, DrawCommandError> {
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

    fn compute_texture_uv_scales(
        &self,
        texture_mapping_size: [f32; 2],
        draw_options: &ShapeDrawCommandOptions,
    ) -> [[f32; 2]; 2] {
        [
            self.compute_texture_uv_scale_for_layer(
                draw_options.background_texture.texture_id,
                draw_options.background_texture.fit_mode,
                texture_mapping_size,
            ),
            self.compute_texture_uv_scale_for_layer(
                draw_options.foreground_texture.texture_id,
                draw_options.foreground_texture.fit_mode,
                texture_mapping_size,
            ),
        ]
    }

    fn compute_texture_uv_scale_for_layer(
        &self,
        texture_id: Option<u64>,
        texture_fit_mode: ShapeTextureFitMode,
        texture_mapping_size: [f32; 2],
    ) -> [f32; 2] {
        if texture_fit_mode != ShapeTextureFitMode::OriginalSize {
            return [1.0, 1.0];
        }

        let Some(texture_id) = texture_id else {
            return [1.0, 1.0];
        };

        let Some((texture_width, texture_height)) =
            self.texture_manager.texture_dimensions(texture_id)
        else {
            return [1.0, 1.0];
        };

        [
            texture_mapping_size[0] * self.scale_factor as f32 / texture_width.max(1) as f32,
            texture_mapping_size[1] * self.scale_factor as f32 / texture_height.max(1) as f32,
        ]
    }
}
