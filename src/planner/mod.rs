use crate::commands::{
    IntermediateTextureId, RenderCommand, RenderOperation, Target, TextureComposite,
};
use crate::scene::effects::ShapeMaskEffectInstance;
use crate::scene::Scene;
use ahash::HashMap;
use commands::{PendingCommands, TraversalCommands};
use content_effects::{ContentPlanningInput, SceneTraversal};
use shape_mask_effects::append_shape_mask_effect;
use std::collections::hash_map;
use std::iter::FusedIterator;

mod commands;
pub(super) mod content_effects;
pub(super) mod draws;
pub(super) mod shape_mask_effects;

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

enum PlanningPhase {
    Beginning,
    PlanningShapeMaskEffects,
    PlanningMainContent,
    Ending,
    Finished,
}

/// Planning scratch
#[derive(Default)]
pub(crate) struct Planner {
    shape_composites: HashMap<usize, TextureComposite>,
    traversal: SceneTraversal,
}

pub(crate) struct CommandIterator<'scene, 'planner> {
    scene: &'scene Scene,
    planner: &'planner mut Planner,
    shape_effects: hash_map::Iter<'scene, usize, ShapeMaskEffectInstance>,
    pending_commands: PendingCommands<'scene>,
    texture_ids: TextureIdAllocator,
    phase: PlanningPhase,
    maximum_texture_dimension: u32,
}

impl Planner {
    /// Starts a fresh traversal, including after an earlier iterator was dropped.
    pub(crate) fn plan<'scene, 'planner>(
        &'planner mut self,
        scene: &'scene Scene,
        maximum_texture_dimension: u32,
    ) -> CommandIterator<'scene, 'planner> {
        self.shape_composites.clear();
        self.traversal.start(&ContentPlanningInput {
            tree: &scene.draw_tree,
            parameters: &scene.effect_parameters,
            content_effects: &scene.content_effects,
            backdrop_effects: &scene.backdrop_effects,
            shape_mask_effects: &self.shape_composites,
            scale_factor: scene.viewport().scale_factor,
            physical_size: scene.viewport().physical_size.into(),
        });
        CommandIterator {
            scene,
            planner: self,
            shape_effects: scene.shape_mask_effects.iter(),
            pending_commands: TraversalCommands::default().into_iter(),
            texture_ids: TextureIdAllocator::default(),
            phase: PlanningPhase::Beginning,
            maximum_texture_dimension,
        }
    }

    pub(crate) fn clear(&mut self) {
        self.shape_composites.clear();
    }
}

impl<'scene> Iterator for CommandIterator<'scene, '_> {
    type Item = RenderCommand<'scene>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(command) = self.pending_commands.next() {
                return Some(command);
            }
            let mut commands = TraversalCommands::default();
            match self.phase {
                PlanningPhase::Beginning => {
                    commands.push(RenderOperation::BeginTarget(Target::Surface));
                    self.phase = PlanningPhase::PlanningShapeMaskEffects;
                }
                PlanningPhase::PlanningShapeMaskEffects => {
                    if let Some((&node_id, effect)) = self.shape_effects.next() {
                        append_shape_mask_effect(
                            &mut commands,
                            &mut self.texture_ids,
                            &mut self.planner.shape_composites,
                            &self.scene.draw_tree,
                            node_id,
                            effect,
                            &self.scene.effect_parameters,
                            self.scene.viewport(),
                            self.scene.fringe_width(),
                            self.maximum_texture_dimension,
                        );
                    } else {
                        self.phase = PlanningPhase::PlanningMainContent;
                    }
                }
                PlanningPhase::PlanningMainContent => {
                    let progressed = self.planner.traversal.advance(
                        &ContentPlanningInput {
                            tree: &self.scene.draw_tree,
                            parameters: &self.scene.effect_parameters,
                            content_effects: &self.scene.content_effects,
                            backdrop_effects: &self.scene.backdrop_effects,
                            shape_mask_effects: &self.planner.shape_composites,
                            scale_factor: self.scene.viewport().scale_factor,
                            physical_size: self.scene.viewport().physical_size.into(),
                        },
                        &mut commands,
                        &mut self.texture_ids,
                    );
                    if !progressed {
                        self.phase = PlanningPhase::Ending;
                    }
                }
                PlanningPhase::Ending => {
                    commands.push(RenderOperation::EndTarget);
                    self.phase = PlanningPhase::Finished;
                }
                PlanningPhase::Finished => return None,
            }
            self.pending_commands = commands.into_iter();
        }
    }
}

impl FusedIterator for CommandIterator<'_, '_> {}

#[cfg(test)]
mod tests;
