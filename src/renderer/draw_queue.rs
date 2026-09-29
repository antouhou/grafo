use super::types::DrawCommandError;
use super::{RenderBackend, Renderer};
use crate::commands::ShapeDrawId;
use crate::core::shape::{Shape, ShapeDrawCommandOptions, ShapeInstance};
use crate::core::vertex::InstanceTransform;
use crate::scene::types::DrawTreeNode;

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
        self.scene.validate_parent(parent_shape_id)?;
        let shape = self.scene.loaded_shape(cache_key)?;
        let clips_children = options.clips_children;
        self.queue_shape(
            ShapeInstance::new(shape, options),
            parent_shape_id,
            clips_children,
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
        self.scene.validate_parent(parent_shape_id)?;
        let shape = self.scene.tessellate(shape.as_ref(), geometry_id);
        let clips_children = options.clips_children;
        self.queue_shape(
            ShapeInstance::new(shape, options),
            parent_shape_id,
            clips_children,
        )
    }

    fn queue_shape(
        &mut self,
        instance: ShapeInstance,
        parent: Option<usize>,
        clips_children: bool,
    ) -> Result<usize, DrawCommandError<B::Error>> {
        let id = self.scene.next_node_id();
        self.backend
            .register_shape(ShapeDrawId(id), &instance)
            .map_err(DrawCommandError::Backend)?;
        let inserted_id = self.scene.insert_shape(instance, parent, clips_children)?;
        debug_assert_eq!(inserted_id, id);
        Ok(id)
    }

    /// Adds a scissor clip without geometry. Rotation, skew and perspective are rejected.
    pub fn add_clipping_rect(
        &mut self,
        rect_bounds: [(f32, f32); 2],
        parent_shape_id: Option<usize>,
        transform: Option<impl Into<InstanceTransform>>,
        clips_children: bool,
    ) -> Result<usize, DrawCommandError<B::Error>> {
        Ok(self.scene.add_clipping_rect(
            rect_bounds,
            parent_shape_id,
            transform.map(Into::into),
            clips_children,
        )?)
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
        let mut has_removed_root = false;
        self.scene.remove_subtrees_with(node_ids, |id, node| {
            has_removed_nodes = true;
            has_removed_root = has_removed_root || id == 0;
            if !has_removed_root && matches!(node, DrawTreeNode::CachedShape(_)) {
                self.removed_shape_ids.push(ShapeDrawId(id));
            }
            removed(id);
        });
        if !has_removed_nodes {
            return;
        }
        if has_removed_root {
            self.planner.clear();
            self.backend.clear_draw_queue();
        } else {
            if !self.removed_shape_ids.is_empty() {
                self.backend.unregister_shapes(&self.removed_shape_ids);
            }
            self.planner.retain_effect_parameters(&mut self.scene);
        }
        self.removed_shape_ids.clear();
    }

    pub fn clear_draw_queue(&mut self) {
        self.scene.clear();
        self.planner.clear();
        self.backend.clear_draw_queue();
    }
}
