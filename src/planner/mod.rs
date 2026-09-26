use crate::commands::{EffectParameters, RenderOperation, RenderPlan, Target, TextureComposite};
use crate::core::Viewport;
use crate::scene::Scene;
use ahash::HashMap;
use groups::{GroupPlanningInput, SceneTraversal};
use shape_effects::append_shape_effects;

pub(super) mod backdrops;
pub(super) mod draws;
pub(super) mod groups;
pub(super) mod shape_effects;

/// Reusable planning scratch and flat command storage.
#[derive(Default)]
pub(crate) struct Planner {
    shape_composites: HashMap<usize, TextureComposite>,
    traversal: SceneTraversal,
    commands: RenderPlan,
}

impl Planner {
    pub(crate) fn store_effect_parameters(&mut self, parameters: &[u8]) -> EffectParameters {
        self.commands.store_parameters(parameters)
    }

    pub(crate) fn update_effect_parameters(
        &mut self,
        stored: EffectParameters,
        parameters: &[u8],
    ) -> EffectParameters {
        self.commands.update_parameters(stored, parameters)
    }

    pub(crate) fn plan(
        &mut self,
        scene: &Scene,
        viewport: Viewport,
        fringe_width: f32,
        maximum_texture_dimension: u32,
    ) -> &RenderPlan {
        self.commands.clear_commands();
        self.commands
            .push(RenderOperation::BeginTarget(Target::Surface));
        append_shape_effects(
            &mut self.commands,
            &mut self.shape_composites,
            &scene.draw_tree,
            &scene.shape_effects,
            viewport,
            fringe_width,
            maximum_texture_dimension,
        );
        self.traversal.plan(
            GroupPlanningInput {
                tree: &scene.draw_tree,
                group_effects: &scene.group_effects,
                backdrop_effects: &scene.backdrop_effects,
                shape_effects: &self.shape_composites,
                scale_factor: viewport.scale_factor,
                physical_size: viewport.physical_size.into(),
                max_capture_dimension: maximum_texture_dimension,
            },
            &mut self.commands,
        );
        self.commands.push(RenderOperation::EndTarget);
        &self.commands
    }

    pub(crate) fn clear(&mut self) {
        self.commands.clear();
        self.shape_composites.clear();
    }
}
