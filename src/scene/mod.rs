//! CPU scene descriptions, tessellation caches and effect attachments.
use self::backdrop_damage::BackdropDamage;
use self::effects::{
    resolve_backdrop_effect, BackdropEffectInstance, EffectInstance, ShapeEffectInstance,
};
pub use self::errors::SceneError;
use self::types::{CachedShapeDrawData, ClipRectDrawData, DrawTreeNode};
use crate::core::effect::ShapeEffectBounds;
use crate::core::shape::{CachedShapeHandle, Shape, ShapeDrawCommandOptions, ShapeInstance};
use crate::core::util::ShapeResources;
use crate::core::vertex::InstanceTransform;
use crate::core::{geometry, Size, UnsignedPhysicalRect, Viewport};
use ahash::{HashMap, HashMapExt};
use easy_tree::Tree;
use lyon::tessellation::FillTessellator;
use std::sync::{Arc, RwLock};
mod backdrop_damage;
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
    backdrop_damage: BackdropDamage,
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
            backdrop_damage: BackdropDamage::default(),
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

    pub(crate) fn validate_parent(&self, parent: Option<usize>) -> Result<(), SceneError> {
        if let Some(parent) = parent {
            if self.draw_tree.get(parent).is_none() {
                return Err(SceneError::InvalidShapeId(parent));
            }
        }
        Ok(())
    }

    /// Removed node slots can be reused by the next insertion.
    pub(crate) fn next_node_id(&self) -> usize {
        self.draw_tree.next_node_id()
    }

    pub fn add_shape(
        &mut self,
        shape: CachedShapeHandle,
        parent: Option<usize>,
        options: ShapeDrawCommandOptions,
    ) -> Result<usize, SceneError> {
        self.validate_parent(parent)?;
        Ok(self.insert_node(
            DrawTreeNode::CachedShape(CachedShapeDrawData::new(shape, options)),
            parent,
        ))
    }

    pub fn add_clipping_rect(
        &mut self,
        rect_bounds: [(f32, f32); 2],
        parent: Option<usize>,
        transform: Option<InstanceTransform>,
        clips_children: bool,
    ) -> Result<usize, SceneError> {
        self.validate_parent(parent)?;
        Ok(self.insert_node(
            Self::prepare_clipping_rect(rect_bounds, transform, clips_children)?,
            parent,
        ))
    }

    pub(crate) fn prepare_clipping_rect(
        rect_bounds: [(f32, f32); 2],
        transform: Option<InstanceTransform>,
        clips_children: bool,
    ) -> Result<DrawTreeNode, SceneError> {
        if !geometry::is_axis_aligned_rect_transform(transform) {
            return Err(SceneError::UnsupportedClipRectTransform);
        }
        Ok(DrawTreeNode::ClipRect(ClipRectDrawData::new(
            rect_bounds,
            transform,
            clips_children,
        )))
    }

    /// The parent must be validated before insertion.
    pub(crate) fn insert_node(&mut self, node: DrawTreeNode, parent: Option<usize>) -> usize {
        self.refresh_tessellation_cache(&node);
        if self.draw_tree.is_empty() {
            return self.draw_tree.add_node(node);
        }
        self.draw_tree.add_child(parent.unwrap_or(0), node)
    }

    pub(crate) fn replace_node(
        &mut self,
        node_id: usize,
        node: DrawTreeNode,
        shape_effect_bounds: Option<ShapeEffectBounds>,
        viewport: Viewport,
        fringe_width: f32,
        maximum_texture_dimension: u32,
    ) -> DrawTreeNode {
        self.refresh_tessellation_cache(&node);
        if matches!(node, DrawTreeNode::CachedShape(_)) {
            if let Some(bounds) = shape_effect_bounds {
                self.shape_effects
                    .get_mut(&node_id)
                    .expect("replacement retains the validated attachment")
                    .bounds = bounds;
            }
        } else {
            self.group_effects.remove(&node_id);
            self.shape_effects.remove(&node_id);
            self.remove_backdrop_effect(node_id);
        }
        let refreshed_backdrop = self.backdrop_effects.get(&node_id).map(|instance| {
            resolve_backdrop_effect(
                node_id,
                &node,
                instance.effect,
                instance.config,
                viewport,
                fringe_width,
                maximum_texture_dimension,
            )
        });
        let previous = self
            .draw_tree
            .replace(node_id, node)
            .expect("replacement node was validated before resource preparation");
        if let Some((instance, entry)) = refreshed_backdrop {
            self.replace_backdrop_effect(node_id, instance, entry);
        }
        previous
    }

    fn refresh_tessellation_cache(&mut self, node: &DrawTreeNode) {
        if let DrawTreeNode::CachedShape(shape) = node {
            if let Some(geometry_id) = shape.instance.cached_shape.geometry_id {
                self.shape_resources
                    .tessellation_cache
                    .refresh_tessellation(geometry_id, &shape.instance.cached_shape.tessellation);
            }
        }
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

    /// Removes nodes, their descendants and attached effects.
    /// Calls `removed` once per removed node. New nodes may reuse removed IDs.
    pub(crate) fn remove_subtrees_with(
        &mut self,
        node_ids: impl IntoIterator<Item = usize>,
        mut removed: impl FnMut(usize, DrawTreeNode, Option<ShapeEffectInstance>),
    ) {
        self.draw_tree.remove_subtrees_with(node_ids, |id, node| {
            self.group_effects.remove(&id);
            self.backdrop_effects.remove(&id);
            self.backdrop_damage.remove(id);
            let shape_effect = self.shape_effects.remove(&id);
            removed(id, node, shape_effect);
        });
    }

    pub fn clear(&mut self) {
        self.draw_tree.clear();
        self.group_effects.clear();
        self.backdrop_effects.clear();
        self.backdrop_damage.clear();
        self.shape_effects.clear();
    }

    pub(crate) fn expand_backdrop_damage(
        &mut self,
        physical_size: Size,
        dirty_bounds: Option<UnsignedPhysicalRect>,
    ) -> Option<UnsignedPhysicalRect> {
        self.backdrop_damage.expand(physical_size, dirty_bounds)
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
