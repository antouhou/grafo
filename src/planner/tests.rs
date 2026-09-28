use super::Planner;
use crate::commands::{
    DrawClip, IntermediateTextureId, RenderOperation, RenderPlan, TextureComposite,
    TexturePlacement,
};
use crate::core::{
    BackdropEffectConfig, Color, Shape, ShapeDrawCommandOptions, ShapeEffectConfig, Viewport,
};
use crate::scene::Scene;

#[derive(Debug, PartialEq, Eq)]
struct EffectSnapshot {
    effect_id: u64,
    input: IntermediateTextureId,
    output: IntermediateTextureId,
    parameters: Vec<u8>,
    parameter_hash: u64,
    clip: DrawClip,
}

fn effect_snapshots(plan: &RenderPlan) -> Vec<EffectSnapshot> {
    plan.instructions
        .iter()
        .filter_map(|command| {
            let RenderOperation::ApplyEffect(effect) = command.operation else {
                return None;
            };
            Some(EffectSnapshot {
                effect_id: effect.effect_id,
                input: effect.input,
                output: effect.output,
                parameters: plan.parameters(effect.parameters).to_vec(),
                parameter_hash: effect.parameters.hash,
                clip: command.clip,
            })
        })
        .collect()
}

fn other_commands(plan: &RenderPlan) -> Vec<String> {
    plan.instructions
        .iter()
        .filter(|command| !matches!(command.operation, RenderOperation::ApplyEffect(_)))
        .map(|command| format!("{command:?}"))
        .collect()
}

fn add_shape(scene: &mut Scene, parent: Option<usize>) -> usize {
    let shape = scene.tessellate(&Shape::rect([(0.0, 0.0), (32.0, 32.0)]), Some(1));
    scene
        .add_shape(
            shape,
            parent,
            ShapeDrawCommandOptions::new().color(Color::WHITE),
        )
        .unwrap()
}

fn plan_scene(planner: &mut Planner, scene: &Scene) {
    planner.plan(
        scene,
        Viewport {
            physical_size: (32, 32),
            scale_factor: 1.0,
        },
        0.75,
        4096,
    );
}

#[test]
fn parameter_compaction_remaps_recorded_effects_without_clearing_other_commands() {
    let mut scene = Scene::default();
    let mut planner = Planner::default();
    let root = add_shape(&mut scene, None);
    let removed = add_shape(&mut scene, Some(root));
    planner.store_effect_parameters(&[99; 4]);
    // Compaction visits groups, backdrops, then shapes, so destinations overlap old ranges.
    let shape_parameters = planner.store_effect_parameters(&[1; 4]);
    let group_parameters = planner.store_effect_parameters(&[2; 4]);
    let backdrop_parameters = planner.store_effect_parameters(&[3; 4]);
    let removed_parameters = planner.store_effect_parameters(&[4; 4]);
    scene
        .set_shape_effect(root, 17, shape_parameters, ShapeEffectConfig::default())
        .unwrap();
    scene.set_group_effect(root, 17, group_parameters).unwrap();
    scene
        .set_shape_backdrop_effect(
            root,
            17,
            backdrop_parameters,
            BackdropEffectConfig::default(),
        )
        .unwrap();
    scene
        .set_group_effect(removed, 17, removed_parameters)
        .unwrap();
    scene
        .set_shape_effect(
            removed,
            17,
            removed_parameters,
            ShapeEffectConfig::default(),
        )
        .unwrap();
    plan_scene(&mut planner, &scene);

    let expected_effects: Vec<_> = effect_snapshots(&planner.commands)
        .into_iter()
        .filter(|effect| effect.parameters != [4; 4])
        .collect();
    for parameters in [[1; 4], [2; 4], [3; 4]] {
        assert!(expected_effects
            .iter()
            .any(|effect| effect.parameters == parameters));
    }
    let expected_other_commands = other_commands(&planner.commands);
    let instructions_address = planner.commands.instructions.as_ptr();
    let texture_count = planner.commands.texture_count;
    assert!(planner.commands.has_backdrop_captures);
    assert!(planner.shape_composites.contains_key(&removed));

    scene.remove_subtree(removed, |_| {});
    for _ in 0..2 {
        planner.retain_effect_parameters(&mut scene);
        assert_eq!(effect_snapshots(&planner.commands), expected_effects);
        assert_eq!(other_commands(&planner.commands), expected_other_commands);
        assert_eq!(planner.commands.instructions.as_ptr(), instructions_address);
        assert_eq!(planner.commands.effect_parameters.len(), 12);
        assert_eq!(planner.commands.texture_count, texture_count);
        assert!(planner.commands.has_backdrop_captures);
        assert!(planner.shape_composites.contains_key(&root));
        assert!(!planner.shape_composites.contains_key(&removed));
        let expected_composites: Vec<_> = planner
            .commands
            .instructions
            .iter()
            .enumerate()
            .filter_map(|(index, command)| {
                matches!(
                    command.operation,
                    RenderOperation::CompositeTexture(TextureComposite {
                        placement: TexturePlacement::Local { .. },
                        ..
                    })
                )
                .then_some(index)
            })
            .collect();
        assert!(!expected_composites.is_empty());
        assert_eq!(planner.commands.composite_draws, expected_composites);
    }
}

#[test]
fn parameter_compaction_remaps_shared_empty_ranges() {
    let mut scene = Scene::default();
    let mut planner = Planner::default();
    let root = add_shape(&mut scene, None);
    planner.store_effect_parameters(&[99; 4]);
    let parameters = planner.store_effect_parameters(&[]);
    scene.set_group_effect(root, 17, parameters).unwrap();
    scene
        .set_shape_effect(root, 18, parameters, ShapeEffectConfig::default())
        .unwrap();
    scene
        .set_shape_backdrop_effect(root, 19, parameters, BackdropEffectConfig::default())
        .unwrap();
    plan_scene(&mut planner, &scene);
    let expected_effects = effect_snapshots(&planner.commands);
    assert!(!expected_effects.is_empty());
    planner.retain_effect_parameters(&mut scene);
    assert!(planner.commands.effect_parameters.is_empty());
    assert_eq!(effect_snapshots(&planner.commands), expected_effects);
}
