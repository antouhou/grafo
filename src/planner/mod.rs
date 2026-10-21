use crate::commands::{
    IntermediateTextureId, RenderCommand, RenderOperation, Target, TextureComposite,
};
use crate::scene_state::effects::ShapeMaskEffectInstance;
use crate::scene_state::SceneState;
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

pub(crate) struct CommandIterator<'state, 'planner> {
    state: &'state SceneState,
    planner: &'planner mut Planner,
    shape_effects: hash_map::Iter<'state, usize, ShapeMaskEffectInstance>,
    pending_commands: PendingCommands<'state>,
    texture_ids: TextureIdAllocator,
    phase: PlanningPhase,
    maximum_texture_dimension: u32,
}

impl Planner {
    /// Starts a fresh traversal, including after an earlier iterator was dropped.
    pub(crate) fn plan<'state, 'planner>(
        &'planner mut self,
        state: &'state SceneState,
        maximum_texture_dimension: u32,
    ) -> CommandIterator<'state, 'planner> {
        self.shape_composites.clear();
        self.traversal.start(&ContentPlanningInput {
            tree: &state.draw_tree,
            parameters: &state.effect_parameters,
            content_effects: &state.content_effects,
            backdrop_effects: &state.backdrop_effects,
            shape_mask_effects: &self.shape_composites,
            scale_factor: state.viewport().scale_factor,
            physical_size: state.viewport().physical_size.into(),
        });
        CommandIterator {
            state,
            planner: self,
            shape_effects: state.shape_mask_effects.iter(),
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

impl<'state> Iterator for CommandIterator<'state, '_> {
    type Item = RenderCommand<'state>;

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
                            &self.state.draw_tree,
                            node_id,
                            effect,
                            &self.state.effect_parameters,
                            self.state.viewport(),
                            self.state.fringe_width(),
                            self.maximum_texture_dimension,
                        );
                    } else {
                        self.phase = PlanningPhase::PlanningMainContent;
                    }
                }
                PlanningPhase::PlanningMainContent => {
                    let progressed = self.planner.traversal.advance(
                        &ContentPlanningInput {
                            tree: &self.state.draw_tree,
                            parameters: &self.state.effect_parameters,
                            content_effects: &self.state.content_effects,
                            backdrop_effects: &self.state.backdrop_effects,
                            shape_mask_effects: &self.planner.shape_composites,
                            scale_factor: self.state.viewport().scale_factor,
                            physical_size: self.state.viewport().physical_size.into(),
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
