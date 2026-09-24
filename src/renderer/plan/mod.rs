use super::types::DrawTreeNode;
use crate::commands::RenderPlan;
use crate::core::effect::{BackdropEffectInstance, EffectInstance, ShapeEffectInstance};
use crate::core::util::ShapeResources;
use crate::CachedShapeHandle;
use ahash::{HashMap, HashMapExt};
use easy_tree::Tree;
use groups::{GroupPlanner, GroupPlanningInput};
use lyon::tessellation::FillTessellator;
use shape_effects::ShapeEffectPlan;
use std::mem;
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
    shape_effect_plan: ShapeEffectPlan,
    group_planner: GroupPlanner,
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
            shape_effect_plan: ShapeEffectPlan::new(),
            group_planner: GroupPlanner::default(),
            commands: RenderPlan::default(),
        }
    }

    pub(super) fn plan(&mut self, viewport: Viewport) -> &RenderPlan {
        // Return the previous mask storage to its compiler before rebuilding it.
        mem::swap(
            &mut self.commands.shape_effects,
            &mut self.shape_effect_plan.commands,
        );
        self.shape_effect_plan.plan(
            &self.draw_tree,
            &self.shape_effects,
            viewport.scale_factor,
            self.fringe_width,
            viewport.physical_size.into(),
            self.maximum_texture_dimension,
        );
        self.group_planner.plan(
            GroupPlanningInput {
                tree: &self.draw_tree,
                group_effects: &self.group_effects,
                backdrop_effects: &self.backdrop_effects,
                shape_effects: &self.shape_effect_plan.composites,
                scale_factor: viewport.scale_factor,
                physical_size: viewport.physical_size.into(),
                max_capture_dimension: self.maximum_texture_dimension,
            },
            &mut self.commands.scene,
        );
        mem::swap(
            &mut self.commands.shape_effects,
            &mut self.shape_effect_plan.commands,
        );
        self.shape_resources.tessellation_cache.end_frame();
        &self.commands
    }

    pub(super) fn clear_draw_queue(&mut self) {
        self.draw_tree.clear();
        self.group_effects.clear();
        self.backdrop_effects.clear();
        self.shape_effects.clear();
        self.commands.shape_effects.clear();
        self.commands.scene.clear();
        self.shape_effect_plan.clear();
    }
}
