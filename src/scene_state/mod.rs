//! CPU scene descriptions, tessellation caches and effect attachments.
use crate::core::effect::ShapeMaskEffectBounds;
use crate::core::shape::{CachedShapeHandle, Shape, ShapeInstance};
use crate::core::util::ShapeResources;
use crate::core::vertex::InstanceTransform;
use crate::core::{geometry, UnsignedPhysicalRect, Viewport};
use crate::render_backend::texture_manager::{TextureUpdatePublisher, TextureUpdateSubscription};
use ahash::{HashMap, HashMapExt, HashSet};
use easy_tree::Tree;
use effect_damage::{EffectDamage, EffectDamageKind};
use effects::parameters::EffectParameterStorage;
use effects::{
    resolve_backdrop_effect, BackdropEffectInstance, ContentEffectInstance, ShapeMaskEffectInstance,
};
pub use errors::SceneError;
use lyon::tessellation::FillTessellator;
use std::sync::{Arc, RwLock};
use texture_index::TextureIndex;
use types::{CachedShapeDrawData, ClipRectDrawData, DrawTreeNode};

mod effect_damage;
pub(crate) mod effects;
mod errors;
mod texture_index;
pub(crate) mod types;

/// Loaded CPU geometry shared by scenes. Keys identify shape content.
#[derive(Clone, Default)]
pub(crate) struct SceneStateContext {
    loaded_shapes: Arc<RwLock<HashMap<u64, CachedShapeHandle>>>,
    texture_updates: TextureUpdatePublisher,
}

impl SceneStateContext {
    pub(crate) fn new(texture_updates: TextureUpdatePublisher) -> Self {
        Self {
            loaded_shapes: Arc::new(RwLock::new(HashMap::new())),
            texture_updates,
        }
    }
}

/// Owns CPU descriptions and rasterization settings.
pub(crate) struct SceneState {
    pub(crate) draw_tree: Tree<DrawTreeNode>,
    texture_index: TextureIndex,
    texture_updates: TextureUpdateSubscription,
    updated_texture_ids: Vec<u64>,
    updated_shape_ids: HashSet<usize>,
    context: SceneStateContext,
    viewport: Viewport,
    tessellator: FillTessellator,
    shape_resources: ShapeResources,
    pub(crate) effect_parameters: EffectParameterStorage,
    pub(crate) content_effects: HashMap<usize, ContentEffectInstance>,
    pub(crate) backdrop_effects: HashMap<usize, BackdropEffectInstance>,
    effect_damage: EffectDamage,
    pub(crate) shape_mask_effects: HashMap<usize, ShapeMaskEffectInstance>,
}

impl SceneState {
    /// Creates an empty scene for the supplied output dimensions and rasterization settings.
    pub(crate) fn new(context: SceneStateContext, viewport: Viewport) -> Self {
        Self {
            texture_updates: context.texture_updates.subscribe(),
            updated_texture_ids: Vec::new(),
            updated_shape_ids: HashSet::default(),
            context,
            viewport,
            draw_tree: Tree::new(),
            texture_index: TextureIndex::default(),
            tessellator: FillTessellator::new(),
            shape_resources: ShapeResources::new(),
            effect_parameters: EffectParameterStorage::default(),
            content_effects: HashMap::new(),
            backdrop_effects: HashMap::new(),
            effect_damage: EffectDamage::default(),
            shape_mask_effects: HashMap::new(),
        }
    }

    /// Returns output dimensions and the logical-to-physical scale.
    pub(crate) fn viewport(&self) -> Viewport {
        self.viewport
    }

    /// Updates effect bounds and backdrop captures for the new scale.
    /// Preserves settings and attachments if any shape mask effect cannot use the new values.
    pub(crate) fn change_scale_factor(
        &mut self,
        scale_factor: f64,
        maximum_texture_dimension: u32,
    ) -> Result<(), SceneError> {
        if let Err(error) = self.refresh_shape_mask_effect_bounds(scale_factor) {
            self.refresh_shape_mask_effect_bounds(self.viewport.scale_factor)
                .expect(
                    "failed to restore shape mask effect bounds with previous rasterization settings",
                );
            return Err(error);
        }
        self.viewport.scale_factor = scale_factor;
        self.refresh_backdrop_capture_regions(maximum_texture_dimension);
        self.refresh_all_content_effect_bounds(maximum_texture_dimension);
        Ok(())
    }

    /// Updates output dimensions and refreshes backdrop captures.
    pub(crate) fn resize(&mut self, physical_size: (u32, u32), maximum_texture_dimension: u32) {
        self.viewport.physical_size = physical_size;
        self.refresh_backdrop_capture_regions(maximum_texture_dimension);
        self.refresh_all_content_effect_bounds(maximum_texture_dimension);
    }

    pub(crate) fn load_shape(
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

    pub(crate) fn remove_shape(&mut self, cache_key: u64) {
        self.context
            .loaded_shapes
            .write()
            .expect("shared shape cache lock poisoned")
            .remove(&cache_key);
    }

    pub(crate) fn tessellate(
        &mut self,
        shape: &Shape,
        geometry_id: Option<u64>,
    ) -> CachedShapeHandle {
        CachedShapeHandle::new(
            shape,
            &mut self.tessellator,
            &mut self.shape_resources,
            geometry_id,
        )
    }

    pub(crate) fn loaded_shape(&self, cache_key: u64) -> Result<CachedShapeHandle, SceneError> {
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
        self.texture_index.replace(self.next_node_id(), &node);
        if self.draw_tree.is_empty() {
            return self.draw_tree.add_node(node);
        }
        self.draw_tree.add_child(parent.unwrap_or(0), node)
    }

    pub(crate) fn replace_node(
        &mut self,
        node_id: usize,
        node: DrawTreeNode,
        shape_mask_effect_bounds: Option<ShapeMaskEffectBounds>,
        maximum_texture_dimension: u32,
    ) -> DrawTreeNode {
        self.refresh_tessellation_cache(&node);
        self.texture_index.replace(node_id, &node);
        if matches!(node, DrawTreeNode::CachedShape(_)) {
            if let Some(bounds) = shape_mask_effect_bounds {
                self.shape_mask_effects
                    .get_mut(&node_id)
                    .expect("replacement retains the validated attachment")
                    .bounds = bounds;
            }
        } else {
            self.remove_content_effect(node_id);
            if self.shape_mask_effects.remove(&node_id).is_some() {
                self.effect_parameters.request_compaction();
            }
            self.remove_backdrop_effect(node_id);
        }
        let refreshed_backdrop = self.backdrop_effects.get(&node_id).map(|instance| {
            resolve_backdrop_effect(
                node_id,
                &node,
                instance.effect,
                instance.config,
                self.viewport,
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
        self.refresh_content_effect_bounds(node_id, maximum_texture_dimension);
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

    pub(crate) fn shape(&self, node_id: usize) -> Result<&ShapeInstance, SceneError> {
        match self
            .draw_tree
            .get(node_id)
            .ok_or(SceneError::NodeNotFound(node_id))?
        {
            DrawTreeNode::CachedShape(shape) => Ok(&shape.instance),
            DrawTreeNode::Group | DrawTreeNode::ClipRect(_) => {
                Err(SceneError::UnsupportedNodeOperation(node_id, "effects"))
            }
        }
    }

    /// Removes nodes, their descendants and attached effects.
    /// Calls `removed` once per removed node. New nodes may reuse removed IDs.
    pub(crate) fn remove_subtrees_with(
        &mut self,
        node_ids: impl IntoIterator<Item = usize>,
        mut removed: impl FnMut(
            usize,
            DrawTreeNode,
            Option<ShapeMaskEffectInstance>,
            Option<UnsignedPhysicalRect>,
        ),
    ) {
        self.draw_tree.remove_subtrees_with(node_ids, |id, node| {
            self.texture_index.remove(id);
            let group_output = self
                .effect_damage
                .output_region(EffectDamageKind::Content, id);
            if self.content_effects.remove(&id).is_some() {
                self.effect_parameters.request_compaction();
            }
            self.effect_damage.remove(EffectDamageKind::Content, id);
            if self.backdrop_effects.remove(&id).is_some() {
                self.effect_parameters.request_compaction();
            }
            self.effect_damage.remove(EffectDamageKind::Backdrop, id);
            let shape_mask_effect = self.shape_mask_effects.remove(&id);
            if shape_mask_effect.is_some() {
                self.effect_parameters.request_compaction();
            }
            removed(id, node, shape_mask_effect, group_output);
        });
        if self.draw_tree.is_empty() {
            self.effect_parameters.clear();
        }
    }

    /// Consumes pending texture updates and yields each affected shape once.
    pub(crate) fn shapes_with_updated_textures(
        &mut self,
    ) -> impl Iterator<Item = (usize, &CachedShapeDrawData)> {
        self.texture_updates
            .drain_into(&mut self.updated_texture_ids);
        self.updated_shape_ids.clear();
        let texture_index = &self.texture_index;
        let draw_tree = &self.draw_tree;
        let updated_shape_ids = &mut self.updated_shape_ids;
        self.updated_texture_ids
            .iter()
            .flat_map(move |&texture_id| texture_index.shapes_using(texture_id))
            .filter(move |&shape_id| updated_shape_ids.insert(shape_id))
            .map(move |shape_id| {
                let DrawTreeNode::CachedShape(shape) = draw_tree.get_unchecked(shape_id) else {
                    unreachable!("texture index contains only shape nodes");
                };
                (shape_id, shape)
            })
    }

    pub(crate) fn expand_effect_damage(
        &mut self,
        dirty_bounds: Option<UnsignedPhysicalRect>,
    ) -> Option<UnsignedPhysicalRect> {
        self.effect_damage
            .expand(self.viewport.physical_size.into(), dirty_bounds)
    }

    pub(crate) fn finish_preparation(&mut self) {
        self.shape_resources.tessellation_cache.end_frame();
    }
}
