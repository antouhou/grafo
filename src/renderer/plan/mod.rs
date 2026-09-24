use super::types::DrawTreeNode;
use crate::commands::{RenderOperation, RenderPlan, Target, TextureComposite};
use crate::core::effect::{BackdropEffectInstance, EffectInstance, ShapeEffectInstance};
use crate::core::util::ShapeResources;
use crate::CachedShapeHandle;
use ahash::{HashMap, HashMapExt};
use easy_tree::Tree;
use groups::{GroupPlanningInput, SceneTraversal};
use lyon::tessellation::FillTessellator;
use shape_effects::append_shape_effects;
use std::sync::{Arc, RwLock};

pub(super) mod backdrops;
pub(super) mod draws;
pub(super) mod groups;
mod scene;
pub(super) mod shape_effects;

/// Physical output dimensions and logical-to-physical coordinate scale.
#[derive(Clone, Copy, Debug)]
pub struct Viewport {
    pub physical_size: (u32, u32),
    pub scale_factor: f64,
}

/// Scene descriptions and reusable command storage, independent of the backend.
pub(super) struct Planner {
    pub(super) draw_tree: Tree<DrawTreeNode>,
    pub(super) loaded_shapes: Arc<RwLock<HashMap<u64, CachedShapeHandle>>>,
    pub(super) tessellator: FillTessellator,
    pub(super) shape_resources: ShapeResources,
    pub(super) group_effects: HashMap<usize, EffectInstance>,
    pub(super) backdrop_effects: HashMap<usize, BackdropEffectInstance>,
    pub(super) shape_effects: HashMap<usize, ShapeEffectInstance>,
    pub(super) fringe_width: f32,
    maximum_texture_dimension: u32,
    shape_composites: HashMap<usize, TextureComposite>,
    traversal: SceneTraversal,
    commands: RenderPlan,
}

impl Planner {
    pub(super) fn new(
        loaded_shapes: Arc<RwLock<HashMap<u64, CachedShapeHandle>>>,
        maximum_texture_dimension: u32,
        fringe_width: f32,
    ) -> Self {
        Self {
            draw_tree: Tree::new(),
            loaded_shapes,
            tessellator: FillTessellator::new(),
            shape_resources: ShapeResources::new(),
            group_effects: HashMap::new(),
            backdrop_effects: HashMap::new(),
            shape_effects: HashMap::new(),
            fringe_width,
            maximum_texture_dimension,
            shape_composites: HashMap::new(),
            traversal: SceneTraversal::default(),
            commands: RenderPlan::default(),
        }
    }

    pub(super) fn plan(&mut self, viewport: Viewport) -> &RenderPlan {
        self.commands.clear();
        self.commands
            .push(RenderOperation::BeginTarget(Target::Surface));
        append_shape_effects(
            &mut self.commands,
            &mut self.shape_composites,
            &self.draw_tree,
            &self.shape_effects,
            viewport,
            self.fringe_width,
            self.maximum_texture_dimension,
        );
        self.traversal.plan(
            GroupPlanningInput {
                tree: &self.draw_tree,
                group_effects: &self.group_effects,
                backdrop_effects: &self.backdrop_effects,
                shape_effects: &self.shape_composites,
                scale_factor: viewport.scale_factor,
                physical_size: viewport.physical_size.into(),
                max_capture_dimension: self.maximum_texture_dimension,
            },
            &mut self.commands,
        );
        self.commands.push(RenderOperation::EndTarget);
        self.shape_resources.tessellation_cache.end_frame();
        &self.commands
    }

    pub(super) fn clear_draw_queue(&mut self) {
        self.draw_tree.clear();
        self.group_effects.clear();
        self.backdrop_effects.clear();
        self.shape_effects.clear();
        self.commands.clear();
        self.shape_composites.clear();
    }
}
