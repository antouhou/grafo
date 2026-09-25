//! CPU scene descriptions, tessellation caches and effect attachments.
use self::effects::{BackdropEffectInstance, EffectInstance, ShapeEffectInstance};
pub use self::errors::SceneError;
use self::types::{CachedShapeDrawData, ClipRectDrawData, DrawTreeNode};
use crate::core::geometry;
use crate::core::shape::{CachedShapeHandle, Shape, ShapeDrawCommandOptions, ShapeInstance};
use crate::core::util::ShapeResources;
use crate::core::vertex::InstanceTransform;
use ahash::{HashMap, HashMapExt};
use easy_tree::Tree;
use lyon::tessellation::FillTessellator;
use std::sync::{Arc, RwLock};
pub(crate) mod effects;
mod errors;
pub(crate) mod types;

/// Loaded CPU geometry shared by scenes. Keys identify shape content.
#[derive(Clone, Default)]
pub struct SceneContext {
    loaded_shapes: Arc<RwLock<HashMap<u64, CachedShapeHandle>>>,
}

/// Owns CPU descriptions. Clearing removes queued nodes while retaining caches and capacity.
pub struct Scene {
    pub(crate) draw_tree: Tree<DrawTreeNode>,
    context: SceneContext,
    tessellator: FillTessellator,
    shape_resources: ShapeResources,
    pub(crate) group_effects: HashMap<usize, EffectInstance>,
    pub(crate) backdrop_effects: HashMap<usize, BackdropEffectInstance>,
    pub(crate) shape_effects: HashMap<usize, ShapeEffectInstance>,
}

impl Default for Scene {
    fn default() -> Self {
        Self::new(SceneContext::default())
    }
}

impl Scene {
    pub fn new(context: SceneContext) -> Self {
        Self {
            context,
            draw_tree: Tree::new(),
            tessellator: FillTessellator::new(),
            shape_resources: ShapeResources::new(),
            group_effects: HashMap::new(),
            backdrop_effects: HashMap::new(),
            shape_effects: HashMap::new(),
        }
    }

    pub fn load_shape(
        &mut self,
        shape: impl AsRef<Shape>,
        cache_key: u64,
        geometry_id: Option<u64>,
    ) {
        let shape = self.tessellate(shape.as_ref(), geometry_id);
        self.context
            .loaded_shapes
            .write()
            .expect("shared shape cache lock poisoned")
            .insert(cache_key, shape);
    }

    pub fn remove_shape(&mut self, cache_key: u64) {
        self.context
            .loaded_shapes
            .write()
            .expect("shared shape cache lock poisoned")
            .remove(&cache_key);
    }

    pub fn tessellate(&mut self, shape: &Shape, geometry_id: Option<u64>) -> CachedShapeHandle {
        CachedShapeHandle::new(
            shape,
            &mut self.tessellator,
            &mut self.shape_resources,
            geometry_id,
        )
    }

    pub fn loaded_shape(&self, cache_key: u64) -> Result<CachedShapeHandle, SceneError> {
        self.context
            .loaded_shapes
            .read()
            .expect("shared shape cache lock poisoned")
            .get(&cache_key)
            .cloned()
            .ok_or(SceneError::ShapeNotLoaded(cache_key))
    }

    pub fn validate_parent(&self, parent: Option<usize>) -> Result<(), SceneError> {
        if let Some(parent) = parent {
            if self.draw_tree.get(parent).is_none() {
                return Err(SceneError::InvalidShapeId(parent));
            }
        }
        Ok(())
    }

    /// Nodes are append-only until the queue is cleared.
    pub(crate) fn next_node_id(&self) -> usize {
        self.draw_tree.len()
    }

    pub fn add_shape(
        &mut self,
        shape: CachedShapeHandle,
        parent: Option<usize>,
        options: ShapeDrawCommandOptions,
    ) -> Result<usize, SceneError> {
        self.insert_shape_data(CachedShapeDrawData::new(shape, options), parent)
    }

    pub fn insert_shape(
        &mut self,
        instance: ShapeInstance,
        parent: Option<usize>,
        clips_children: bool,
    ) -> Result<usize, SceneError> {
        self.insert_shape_data(
            CachedShapeDrawData {
                instance,
                is_leaf: true,
                clips_children,
            },
            parent,
        )
    }

    fn insert_shape_data(
        &mut self,
        shape: CachedShapeDrawData,
        parent: Option<usize>,
    ) -> Result<usize, SceneError> {
        self.validate_parent(parent)?;
        if let Some(geometry_id) = shape.instance.cached_shape.geometry_id {
            self.shape_resources
                .tessellation_cache
                .refresh_tessellation(geometry_id, &shape.instance.cached_shape.tessellation);
        }
        Ok(self.insert_node(DrawTreeNode::CachedShape(shape), parent))
    }

    pub fn add_clipping_rect(
        &mut self,
        rect_bounds: [(f32, f32); 2],
        parent: Option<usize>,
        transform: Option<InstanceTransform>,
        clips_children: bool,
    ) -> Result<usize, SceneError> {
        self.validate_parent(parent)?;
        if geometry::extract_axis_aligned_rect_transform(transform).is_none() {
            return Err(SceneError::UnsupportedClipRectTransform);
        }
        Ok(self.insert_node(
            DrawTreeNode::ClipRect(ClipRectDrawData::new(
                rect_bounds,
                transform,
                clips_children,
            )),
            parent,
        ))
    }

    fn insert_node(&mut self, node: DrawTreeNode, parent: Option<usize>) -> usize {
        if self.draw_tree.is_empty() {
            return self.draw_tree.add_node(node);
        }
        let parent = parent.unwrap_or(0);
        self.draw_tree
            .get_mut(parent)
            .expect("validated parent")
            .set_not_leaf();
        self.draw_tree.add_child(parent, node)
    }

    pub fn shape(&self, node_id: usize) -> Result<&ShapeInstance, SceneError> {
        match self
            .draw_tree
            .get(node_id)
            .ok_or(SceneError::NodeNotFound(node_id))?
        {
            DrawTreeNode::CachedShape(shape) => Ok(&shape.instance),
            DrawTreeNode::ClipRect(_) => {
                Err(SceneError::UnsupportedClipRectOperation(node_id, "effects"))
            }
        }
    }

    pub fn clear(&mut self) {
        self.draw_tree.clear();
        self.group_effects.clear();
        self.backdrop_effects.clear();
        self.shape_effects.clear();
    }

    pub(crate) fn finish_preparation(&mut self) {
        self.shape_resources.tessellation_cache.end_frame();
    }

    #[cfg(feature = "render_metrics")]
    pub fn print_memory_usage_info(&self) {
        println!(
            "Cached shapes: {}",
            self.context
                .loaded_shapes
                .read()
                .expect("shared shape cache lock poisoned")
                .len()
        );
        println!("Draw tree size: {}", self.draw_tree.len());
        self.shape_resources.print_sizes();
    }
}
