use crate::commands::{
    EffectParameters, IntermediateTextureId, RenderOperation, RenderPlan, Target, TextureComposite,
};
use crate::core::UnsignedPhysicalRect;
use crate::scene::Scene;
use ahash::HashMap;
use groups::{GroupPlanningInput, SceneTraversal};
use shape_effects::append_shape_effects;

pub(super) mod draws;
pub(super) mod groups;
pub(super) mod shape_effects;

#[derive(Default)]
pub(super) struct TextureIdAllocator {
    next_id: usize,
}

impl TextureIdAllocator {
    fn allocate(&mut self) -> IntermediateTextureId {
        let texture = IntermediateTextureId::Planned(self.next_id);
        self.next_id += 1;
        texture
    }
}

/// Reusable planning scratch and flat command storage.
#[derive(Default)]
pub(crate) struct Planner {
    shape_composites: HashMap<usize, TextureComposite>,
    traversal: SceneTraversal,
    commands: RenderPlan,
    compacted_parameters: Vec<u8>,
}

impl Planner {
    /// Compacts parameter storage and updates the scene's attachment ranges.
    pub(crate) fn compact_effect_parameters(&mut self, scene: &mut Scene) {
        scene.compact_effect_parameters(
            &mut self.commands.effect_parameters,
            &mut self.compacted_parameters,
        );
    }

    pub(crate) fn store_effect_parameters(&mut self, parameters: &[u8]) -> EffectParameters {
        self.commands.store_parameters(parameters)
    }

    /// Rebuilds commands and composites
    pub(crate) fn plan(
        &mut self,
        scene: &Scene,
        maximum_texture_dimension: u32,
        root_scissor: Option<UnsignedPhysicalRect>,
    ) -> &RenderPlan {
        self.commands.clear_commands();
        let mut texture_ids = TextureIdAllocator::default();
        self.commands.root_scissor = root_scissor;
        self.commands
            .push(RenderOperation::BeginTarget(Target::Surface));
        append_shape_effects(
            &mut self.commands,
            &mut texture_ids,
            &mut self.shape_composites,
            &scene.draw_tree,
            &scene.shape_effects,
            scene.viewport(),
            scene.fringe_width(),
            maximum_texture_dimension,
        );
        self.traversal.plan(
            GroupPlanningInput {
                tree: &scene.draw_tree,
                group_effects: &scene.group_effects,
                backdrop_effects: &scene.backdrop_effects,
                shape_effects: &self.shape_composites,
                scale_factor: scene.viewport().scale_factor,
                physical_size: scene.viewport().physical_size.into(),
            },
            &mut self.commands,
            &mut texture_ids,
        );
        self.commands.push(RenderOperation::EndTarget);
        &self.commands
    }

    pub(crate) fn clear(&mut self) {
        self.commands.clear();
        self.shape_composites.clear();
    }
}

#[cfg(test)]
mod tests;
