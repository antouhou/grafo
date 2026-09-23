use super::types::{ClipRectDrawData, DrawCommandError};
use super::*;
use crate::ShapeDrawCommandOptions;

fn clip_rect_supports_transform(transform: InstanceTransform) -> bool {
    rect_utils::extract_axis_aligned_rect_transform(Some(transform)).is_some()
}

impl<'a> Renderer<'a> {
    /// Tessellates the shape and stores it under `cache_key` in the shared context.
    /// Renderers using that context can reuse the cached shape.
    ///
    /// Use a content-derived `geometry_id` to reuse identical geometry during tessellation
    /// and GPU upload. Pass `None` if you cannot supply a reliable geometry key.
    pub fn load_shape(
        &mut self,
        shape: impl AsRef<Shape>,
        cache_key: u64,
        geometry_id: Option<u64>,
    ) {
        let cached_shape = CachedShapeHandle::new(
            shape.as_ref(),
            &mut self.tessellator,
            &mut self.state.shape_resources,
            geometry_id,
        );
        self.context
            .inner
            .shape_cache
            .write()
            .expect("shared shape cache lock poisoned")
            .insert(cache_key, cached_shape);
    }

    /// Removes a loaded shape from the cache.
    pub fn remove_shape(&mut self, cache_key: u64) {
        self.context
            .inner
            .shape_cache
            .write()
            .expect("shared shape cache lock poisoned")
            .remove(&cache_key);
    }

    /// Adds a previously loaded cached shape to the draw tree.
    ///
    /// When `parent_shape_id` is `Some`, the cached shape is attached as a child of that node.
    /// Children are clipped to their parent unless the parent was queued with
    /// [`ShapeDrawCommandOptions::clips_children(false)`].
    pub fn add_cached_shape(
        &mut self,
        cache_key: u64,
        parent_shape_id: Option<usize>,
        options: ShapeDrawCommandOptions,
    ) -> Result<usize, DrawCommandError> {
        let mut draw_data = if let Some(cached_shape_handle) = self
            .context
            .inner
            .shape_cache
            .read()
            .expect("shared shape cache lock poisoned")
            .get(&cache_key)
        {
            CachedShapeDrawData::new(cached_shape_handle.clone(), &options)
        } else {
            return Err(DrawCommandError::ShapeNotLoaded(cache_key));
        };
        let resources = self.append_shape_resources(&mut draw_data)?;
        let node_id =
            self.add_draw_tree_node(DrawTreeNode::CachedShape(draw_data), parent_shape_id)?;
        self.state.shape_execution.draws.insert(node_id, resources);
        Ok(node_id)
    }

    /// Adds a shape to the draw tree without retaining it in the loaded-shape cache.
    /// To reuse a loaded shape, call [`load_shape`] and [`add_cached_shape`].
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
            &mut self.state.shape_resources,
            geometry_id,
        );
        let mut draw_data = CachedShapeDrawData::new(cached_shape, &options);

        let resources = self.append_shape_resources(&mut draw_data)?;
        let node_id =
            self.add_draw_tree_node(DrawTreeNode::CachedShape(draw_data), parent_shape_id)?;
        self.state.shape_execution.draws.insert(node_id, resources);
        Ok(node_id)
    }

    /// Adds an axis-aligned scissor clipping rectangle without preparing geometry.
    ///
    /// This node clips its children when `clips_children` is true.
    /// This method rejects rotation, skew, and perspective because the node
    /// has no geometry for stencil clipping.
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
        self.add_draw_tree_node(
            DrawTreeNode::ClipRect(ClipRectDrawData::new(
                rect_bounds,
                transform,
                clips_children,
            )),
            parent_shape_id,
        )
    }

    fn add_draw_tree_node(
        &mut self,
        draw_tree_node: DrawTreeNode,
        parent_shape_id: Option<usize>,
    ) -> Result<usize, DrawCommandError> {
        if self.state.draw_tree.is_empty() {
            let node_id = self.state.draw_tree.add_node(draw_tree_node);
            Ok(node_id)
        } else if let Some(parent_shape_id) = parent_shape_id {
            if let Some(parent) = self.state.draw_tree.get_mut(parent_shape_id) {
                parent.set_not_leaf();
                let node_id = self
                    .state
                    .draw_tree
                    .add_child(parent_shape_id, draw_tree_node);
                Ok(node_id)
            } else {
                Err(DrawCommandError::InvalidShapeId(parent_shape_id))
            }
        } else {
            if let Some(root) = self.state.draw_tree.get_mut(0) {
                root.set_not_leaf();
            }
            let node_id = self.state.draw_tree.add_child_to_root(draw_tree_node);
            Ok(node_id)
        }
    }

    pub(super) fn refresh_geometry_cache(&mut self, cached_shape_data: &CachedShapeDrawData) {
        if let Some(geometry_id) = cached_shape_data.cached_shape.geometry_id {
            self.state
                .shape_resources
                .tessellation_cache
                .refresh_tessellation(geometry_id, &cached_shape_data.cached_shape.tessellation);
        }
    }

    pub fn texture_manager(&self) -> &TextureManager {
        &self.pipeline_resources.shapes.texture_manager
    }

    pub fn clear_draw_queue(&mut self) {
        self.state.draw_tree.clear();
        self.state.group_effects.clear();
        self.state.backdrop_effects.clear();
        self.state.shape_effects.clear();
        self.state.shape_execution.clear_draw_queue();
    }
}
