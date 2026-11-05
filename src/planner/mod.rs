use crate::commands::{
    IntermediateTextureId, RenderCommand, RenderOperation, Target, TextureComposite,
};
use crate::core::{geometry, MathRect, UnsignedPhysicalRect};
use crate::scene_state::effects::{ContentEffectInstance, ShapeMaskEffectInstance};
use crate::scene_state::SceneState;
use ahash::HashMap;
use commands::{PendingCommands, TraversalCommands};
use draws::{DrawPlanner, DrawPlanningInput};
use shape_mask_effects::append_shape_mask_effect;
use std::collections::hash_map;
use std::iter::FusedIterator;

mod commands;
pub(super) mod content_effects;
pub(super) mod draws;
pub(super) mod shape_mask_effects;

fn content_intersects_damage(effect: Option<&ContentEffectInstance>, damage: MathRect) -> bool {
    effect
        .and_then(|effect| effect.bounds)
        .is_some_and(|bounds| {
            MathRect::new(
                bounds.logical_output_bounds[0].into(),
                bounds.logical_output_bounds[1].into(),
            )
            .intersects(&damage)
        })
}

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
    traversal: DrawPlanner,
}

pub(crate) struct CommandIterator<'state, 'planner> {
    state: &'state SceneState,
    planner: &'planner mut Planner,
    shape_effects: hash_map::Iter<'state, usize, ShapeMaskEffectInstance>,
    pending_commands: PendingCommands<'state>,
    texture_ids: TextureIdAllocator,
    phase: PlanningPhase,
    maximum_texture_dimension: u32,
    damaged_region: Option<MathRect>,
    has_deferred_shape_masks: bool,
}

impl Planner {
    /// Starts a fresh traversal, including after an earlier iterator was dropped.
    pub(crate) fn plan<'state, 'planner>(
        &'planner mut self,
        state: &'state SceneState,
        maximum_texture_dimension: u32,
        damaged_region: Option<UnsignedPhysicalRect>,
    ) -> CommandIterator<'state, 'planner> {
        let damaged_region = damaged_region
            .filter(|region| !region.is_empty())
            .map(|region| {
                geometry::physical_bounds_to_logical_screen_rect(
                    region,
                    state.viewport().scale_factor,
                )
            });
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
            shape_effects: state.shape_mask_effects.iter(),
            pending_commands: TraversalCommands::default().into_iter(),
            texture_ids: TextureIdAllocator::default(),
            phase: if damaged_region.is_some() {
                PlanningPhase::Beginning
            } else {
                PlanningPhase::Finished
            },
            maximum_texture_dimension,
            damaged_region,
            has_deferred_shape_masks: false,
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
                        if self.damaged_region.is_some_and(|damage| {
                            self.mask_intersects_damage(node_id, effect, damage)
                        }) {
                            self.append_shape_mask_effect(node_id, effect, &mut commands);
                        } else {
                            self.has_deferred_shape_masks = !self.state.content_effects.is_empty();
                        }
                    } else {
                        self.phase = PlanningPhase::PlanningMainContent;
                    }
                }
                PlanningPhase::PlanningMainContent => {
                    if self.has_deferred_shape_masks
                        && self.plan_deferred_shape_mask_effect(&mut commands)
                    {
                        self.pending_commands = commands.into_iter();
                        continue;
                    }
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
    fn mask_intersects_damage(
        &self,
        node_id: usize,
        effect: &ShapeMaskEffectInstance,
        damage: MathRect,
    ) -> bool {
        effect.bounds.logical_screen_bounds.intersects(&damage)
            || content_intersects_damage(self.state.content_effects.get(&node_id), damage)
    }

    fn plan_deferred_shape_mask_effect(
        &mut self,
        commands: &mut TraversalCommands<'state>,
    ) -> bool {
        let Some((node_id, damage)) = self.planner.traversal.next_capture_node_damage() else {
            return false;
        };
        if self.planner.shape_composites.contains_key(&node_id) {
            return false;
        }
        let Some(effect) = self.state.shape_mask_effects.get(&node_id) else {
            return false;
        };
        if !self.mask_intersects_damage(node_id, effect, damage) {
            return false;
        }
        // A selected content capture can need masks outside the surface damage.
        self.append_shape_mask_effect(node_id, effect, commands);
        self.planner.shape_composites.contains_key(&node_id)
    }

    fn append_shape_mask_effect(
        &mut self,
        node_id: usize,
        effect: &ShapeMaskEffectInstance,
        commands: &mut TraversalCommands<'state>,
    ) {
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
