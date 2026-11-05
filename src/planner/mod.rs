use crate::commands::{
    IntermediateTextureId, RenderCommand, RenderOperation, Target, TextureComposite,
};
use crate::core::{geometry, UnsignedPhysicalRect};
use crate::scene_state::SceneState;
use ahash::HashMap;
use commands::{PendingCommands, TraversalCommands};
use draws::{DrawPlanner, DrawPlanningInput};
use shape_mask_effects::append_shape_mask_effect;
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
    PlanningMainContent,
    Ending,
    Finished,
}

/// Planning scratch
#[derive(Default)]
pub(crate) struct Planner {
    shape_composites: HashMap<usize, TextureComposite>,
    traversal: DrawPlanner,
}

pub(crate) struct CommandIterator<'state, 'planner> {
    state: &'state SceneState,
    planner: &'planner mut Planner,
    pending_commands: PendingCommands<'state>,
    texture_ids: TextureIdAllocator,
    phase: PlanningPhase,
    maximum_texture_dimension: u32,
    damaged_region: Option<UnsignedPhysicalRect>,
}

impl Planner {
    /// Starts a fresh traversal, including after an earlier iterator was dropped.
    pub(crate) fn plan<'state, 'planner>(
        &'planner mut self,
        state: &'state SceneState,
        maximum_texture_dimension: u32,
        damaged_region: Option<UnsignedPhysicalRect>,
    ) -> CommandIterator<'state, 'planner> {
        self.shape_composites.clear();
        self.traversal.start(&DrawPlanningInput {
            tree: &state.draw_tree,
            parameters: &state.effect_parameters,
            content_effects: &state.content_effects,
            backdrop_effects: &state.backdrop_effects,
            shape_mask_effects: &self.shape_composites,
            scale_factor: state.viewport().scale_factor,
            physical_size: state.viewport().physical_size.into(),
            damaged_region,
        });
        CommandIterator {
            state,
            planner: self,
            pending_commands: TraversalCommands::default().into_iter(),
            texture_ids: TextureIdAllocator::default(),
            phase: if damaged_region.is_some_and(|region| !region.is_empty()) {
                PlanningPhase::Beginning
            } else {
                PlanningPhase::Finished
            },
            maximum_texture_dimension,
            damaged_region,
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
                    self.phase = PlanningPhase::PlanningMainContent;
                }
                PlanningPhase::PlanningMainContent => {
                    self.plan_shape_mask_effect(&mut commands);
                    let progressed = self.planner.traversal.advance(
                        &DrawPlanningInput {
                            tree: &self.state.draw_tree,
                            parameters: &self.state.effect_parameters,
                            content_effects: &self.state.content_effects,
                            backdrop_effects: &self.state.backdrop_effects,
                            shape_mask_effects: &self.planner.shape_composites,
                            scale_factor: self.state.viewport().scale_factor,
                            physical_size: self.state.viewport().physical_size.into(),
                            damaged_region: self.damaged_region,
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

impl<'state> CommandIterator<'state, '_> {
    fn plan_shape_mask_effect(&mut self, commands: &mut TraversalCommands<'state>) {
        let Some((node_id, damage)) = self.planner.traversal.next_node_damage() else {
            return;
        };
        let Some(effect) = self.state.shape_mask_effects.get(&node_id) else {
            return;
        };
        let mask_intersects_damage = geometry::logical_bounds_to_viewport_rect(
            effect.bounds.logical_screen_bounds,
            self.state.viewport(),
        )
        .is_some_and(|bounds| bounds.intersects(&damage));
        let content_intersects_damage = self
            .state
            .content_effect_output(node_id)
            .is_some_and(|bounds| bounds.intersects(&damage));
        if !mask_intersects_damage && !content_intersects_damage {
            return;
        }
        append_shape_mask_effect(
            commands,
            &mut self.texture_ids,
            &mut self.planner.shape_composites,
            &self.state.draw_tree,
            node_id,
            effect,
            &self.state.effect_parameters,
            self.state.viewport(),
            self.maximum_texture_dimension,
        );
    }
}

impl FusedIterator for CommandIterator<'_, '_> {}

#[cfg(test)]
mod tests;
