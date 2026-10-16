use super::Planner;
use crate::commands::{DrawClip, IntermediateTextureId, RenderCommand, RenderOperation};
use crate::core::{
    BackdropEffectConfig, Color, ContentEffectConfig, Shape, ShapeDrawCommandOptions,
    ShapeMaskEffectConfig, Viewport,
};
use crate::scene::types::{CachedShapeDrawData, DrawTreeNode};
use crate::scene::{Scene, SceneContext};

#[derive(Default)]
pub(super) struct RecordedCommands<'scene> {
    pub(super) instructions: Vec<RenderCommand<'scene>>,
}

impl<'scene> RecordedCommands<'scene> {
    pub(super) fn clear_commands(&mut self) {
        self.instructions.clear();
    }
    pub(super) fn push(&mut self, operation: RenderOperation<'scene>) {
        self.instructions.push(RenderCommand {
            operation,
            clip: DrawClip::default(),
        });
    }
}

#[derive(Debug, PartialEq, Eq)]
struct EffectSnapshot {
    effect_id: u64,
    input: IntermediateTextureId,
    output: IntermediateTextureId,
    parameters: Vec<u8>,
    parameter_hash: u64,
    clip: DrawClip,
}

fn effect_snapshots(commands: &[RenderCommand<'_>]) -> Vec<EffectSnapshot> {
    commands
        .iter()
        .filter_map(|command| {
            let RenderOperation::ApplyEffect(effect) = command.operation else {
                return None;
            };
            Some(EffectSnapshot {
                effect_id: effect.effect_id,
                input: effect.input,
                output: effect.output,
                parameters: effect.parameters.bytes.to_vec(),
                parameter_hash: effect.parameters.hash,
                clip: command.clip,
            })
        })
        .collect()
}

fn add_shape(scene: &mut Scene, parent: Option<usize>) -> usize {
    let shape = scene.tessellate(&Shape::rect([(0.0, 0.0), (32.0, 32.0)]), Some(1));
    scene.validate_parent(parent).unwrap();
    scene.insert_node(
        DrawTreeNode::CachedShape(CachedShapeDrawData::new(
            shape,
            ShapeDrawCommandOptions::new().color(Color::WHITE),
        )),
        parent,
    )
}

fn scene() -> Scene {
    Scene::new(
        SceneContext::default(),
        Viewport {
            physical_size: (32, 32),
            scale_factor: 1.0,
        },
        0.75,
    )
}

fn attach_effects(scene: &mut Scene, node: usize, parameters: &[u8]) {
    scene
        .set_shape_mask_effect(node, 17, parameters, ShapeMaskEffectConfig::default(), 4096)
        .unwrap();
    scene
        .set_content_effect(node, 18, parameters, ContentEffectConfig::default(), 4096)
        .unwrap();
    scene
        .set_backdrop_effect(node, 19, parameters, BackdropEffectConfig::default(), 4096)
        .unwrap();
}

#[test]
fn parameter_compaction_preserves_replanned_effects() {
    let mut scene = scene();
    let mut planner = Planner::default();
    let root = add_shape(&mut scene, None);
    let removed = add_shape(&mut scene, Some(root));
    attach_effects(&mut scene, root, &[1; 4]);
    attach_effects(&mut scene, removed, &[4; 4]);
    planner.plan(&scene, 4096).for_each(drop);
    assert!(planner.shape_composites.contains_key(&removed));
    scene.remove_subtrees_with([removed], |_, _, _, _| {});
    scene
        .set_content_effect(root, 18, &[2; 4], ContentEffectConfig::default(), 4096)
        .unwrap();
    scene
        .set_backdrop_effect(root, 19, &[3; 4], BackdropEffectConfig::default(), 4096)
        .unwrap();
    let expected = effect_snapshots(&planner.plan(&scene, 4096).collect::<Vec<_>>());
    let expected_commands: Vec<_> = planner
        .plan(&scene, 4096)
        .map(|command| format!("{command:?}"))
        .collect();
    for _ in 0..2 {
        scene.compact_effect_parameters();
        assert_eq!(scene.effect_parameters.bytes.len(), 12);
        let commands: Vec<_> = planner.plan(&scene, 4096).collect();
        assert_eq!(effect_snapshots(&commands), expected);
        assert_eq!(
            commands
                .iter()
                .map(|command| format!("{command:?}"))
                .collect::<Vec<_>>(),
            expected_commands
        );
        assert!(planner.shape_composites.contains_key(&root));
        assert!(!planner.shape_composites.contains_key(&removed));
    }
}

#[test]
fn parameter_compaction_preserves_replanned_effects_with_empty_parameters() {
    let mut scene = scene();
    let mut planner = Planner::default();
    let root = add_shape(&mut scene, None);
    attach_effects(&mut scene, root, &[99; 4]);
    attach_effects(&mut scene, root, &[]);
    let expected = effect_snapshots(&planner.plan(&scene, 4096).collect::<Vec<_>>());
    scene.compact_effect_parameters();
    assert!(scene.effect_parameters.bytes.is_empty());
    assert_eq!(
        effect_snapshots(&planner.plan(&scene, 4096).collect::<Vec<_>>()),
        expected
    );
}

#[test]
fn dropping_each_partial_traversal_preserves_the_next_complete_stream() {
    let mut scene = scene();
    let root = add_shape(&mut scene, None);
    let child = add_shape(&mut scene, Some(root));
    let descendant = add_shape(&mut scene, Some(child));
    attach_effects(&mut scene, root, &[1; 4]);
    attach_effects(&mut scene, child, &[2; 4]);
    attach_effects(&mut scene, descendant, &[3; 4]);
    let mut planner = Planner::default();
    let expected: Vec<_> = planner
        .plan(&scene, 4096)
        .map(|command| format!("{command:?}"))
        .collect();
    for prefix_length in 0..=expected.len() {
        planner
            .plan(&scene, 4096)
            .take(prefix_length)
            .for_each(drop);
        let actual: Vec<_> = planner
            .plan(&scene, 4096)
            .map(|command| format!("{command:?}"))
            .collect();
        assert_eq!(actual, expected, "prefix length {prefix_length}");
    }
}

#[test]
fn yielded_parameters_borrow_the_scene_independently_of_planner_scratch() {
    let mut scene = scene();
    let root = add_shape(&mut scene, None);
    attach_effects(&mut scene, root, &[1, 2, 3, 4]);
    let mut planner = Planner::default();
    let parameters = planner
        .plan(&scene, 4096)
        .find_map(|command| match command.operation {
            RenderOperation::ApplyEffect(effect) => Some(effect.parameters),
            _ => None,
        })
        .unwrap();
    planner.plan(&scene, 4096).for_each(drop);
    let stored = scene.shape_mask_effect(root).unwrap().parameters;
    assert_eq!(
        parameters.bytes.as_ptr(),
        scene.effect_parameters.get(stored).bytes.as_ptr()
    );
    assert_eq!(parameters.bytes, &[1, 2, 3, 4]);
    assert_eq!(parameters.hash, stored.hash);
}

#[test]
fn partially_consumed_effects_do_not_survive_planning_another_scene() {
    let mut previous = scene();
    let root = add_shape(&mut previous, None);
    attach_effects(&mut previous, root, &[1; 4]);
    let mut planner = Planner::default();
    planner.plan(&previous, 4096).take(3).for_each(drop);
    let mut current = scene();
    add_shape(&mut current, None);
    let commands: Vec<_> = planner.plan(&current, 4096).collect();
    assert!(commands.iter().all(|command| !matches!(
        command.operation,
        RenderOperation::ApplyEffect(_) | RenderOperation::DrawShapeMask(_)
    )));
    assert!(planner.shape_composites.is_empty());
}
