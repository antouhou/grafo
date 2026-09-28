use crate::commands::{
    EffectParameterRange, EffectParameters, RenderOperation, RenderPlan, Target, TextureComposite,
    TexturePlacement,
};
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
    retained_parameters: Vec<u8>,
    parameter_relocations: HashMap<EffectParameterRange, EffectParameters>,
}

impl Planner {
    /// Drops effect commands whose parameter storage was discarded and reindexes composites.
    fn remap_effect_commands(&mut self) {
        self.commands.composite_draws.clear();
        let mut retained_index = 0;
        self.commands.instructions.retain_mut(|command| {
            if let RenderOperation::ApplyEffect(effect) = &mut command.operation {
                let Some(parameters) = self.parameter_relocations.get(&effect.parameters.range)
                else {
                    return false;
                };
                effect.parameters = *parameters;
            }
            if matches!(
                command.operation,
                RenderOperation::CompositeTexture(TextureComposite {
                    placement: TexturePlacement::Local { .. },
                    ..
                })
            ) {
                self.commands.composite_draws.push(retained_index);
            }
            retained_index += 1;
            true
        });
    }

    /// Compacts parameters and remaps attachments and recorded effect commands.
    /// Structural scene changes still require planning before execution.
    pub(crate) fn retain_effect_parameters(&mut self, scene: &mut Scene) {
        self.parameter_relocations.clear();
        scene.retain_effect_parameters(
            &mut self.commands.effect_parameters,
            &mut self.retained_parameters,
            |previous_range, retained_parameters| {
                self.parameter_relocations
                    .insert(previous_range, retained_parameters);
            },
        );
        self.remap_effect_commands();
        self.parameter_relocations.clear();
        self.shape_composites
            .retain(|node_id, _| scene.shape_effects.contains_key(node_id));
    }

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

#[cfg(test)]
mod tests;
