use super::Planner;
use crate::commands::{
    DrawClip, IntermediateTextureId, RenderCommand, RenderOperation, ShapeDrawId,
};
use crate::core::{
    BackdropEffectConfig, Color, ContentEffectConfig, Shape, ShapeDrawCommandOptions,
    ShapeMaskEffectConfig, UnsignedPhysicalRect, Viewport,
};
use crate::scene_state::types::{CachedShapeDrawData, DrawTreeNode};
use crate::scene_state::{SceneState, SceneStateContext};

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

fn add_shape(scene: &mut SceneState, parent: Option<usize>) -> usize {
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

fn scene() -> SceneState {
    SceneState::new(
        SceneStateContext::default(),
        Viewport {
            physical_size: (32, 32),
            scale_factor: 1.0,
        },
    )
}

fn full_damage(scene: &SceneState) -> Option<UnsignedPhysicalRect> {
    Some(UnsignedPhysicalRect::from_size(
        scene.viewport().physical_size.into(),
    ))
}

fn add_rectangle(
    scene: &mut SceneState,
    parent: Option<usize>,
    bounds: [(f32, f32); 2],
    clips_children: bool,
) -> usize {
    let shape = scene.tessellate(&Shape::rect(bounds), None);
    scene.validate_parent(parent).unwrap();
    scene.insert_node(
        DrawTreeNode::CachedShape(CachedShapeDrawData::new(
            shape,
            ShapeDrawCommandOptions::new()
                .color(Color::WHITE)
                .clips_children(clips_children),
        )),
        parent,
    )
}

fn attach_effects(scene: &mut SceneState, node: usize, parameters: &[u8]) {
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
    planner
        .plan(&scene, 4096, full_damage(&scene))
        .for_each(drop);
    assert!(planner.shape_composites.contains_key(&removed));
    scene.remove_subtrees_with([removed], |_, _, _, _| {});
    scene
        .set_content_effect(root, 18, &[2; 4], ContentEffectConfig::default(), 4096)
        .unwrap();
    scene
        .set_backdrop_effect(root, 19, &[3; 4], BackdropEffectConfig::default(), 4096)
        .unwrap();
    let expected = effect_snapshots(
        &planner
            .plan(&scene, 4096, full_damage(&scene))
            .collect::<Vec<_>>(),
    );
    let expected_commands: Vec<_> = planner
        .plan(&scene, 4096, full_damage(&scene))
        .map(|command| format!("{command:?}"))
        .collect();
    for _ in 0..2 {
        scene.compact_effect_parameters();
        assert_eq!(scene.effect_parameters.bytes.len(), 12);
        let commands: Vec<_> = planner.plan(&scene, 4096, full_damage(&scene)).collect();
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
    let expected = effect_snapshots(
        &planner
            .plan(&scene, 4096, full_damage(&scene))
            .collect::<Vec<_>>(),
    );
    scene.compact_effect_parameters();
    assert!(scene.effect_parameters.bytes.is_empty());
    assert_eq!(
        effect_snapshots(
            &planner
                .plan(&scene, 4096, full_damage(&scene))
                .collect::<Vec<_>>()
        ),
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
        .plan(&scene, 4096, full_damage(&scene))
        .map(|command| format!("{command:?}"))
        .collect();
    for prefix_length in 0..=expected.len() {
        planner
            .plan(&scene, 4096, full_damage(&scene))
            .take(prefix_length)
            .for_each(drop);
        let actual: Vec<_> = planner
            .plan(&scene, 4096, full_damage(&scene))
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
        .plan(&scene, 4096, full_damage(&scene))
        .find_map(|command| match command.operation {
            RenderOperation::ApplyEffect(effect) => Some(effect.parameters),
            _ => None,
        })
        .unwrap();
    planner
        .plan(&scene, 4096, full_damage(&scene))
        .for_each(drop);
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
    planner
        .plan(&previous, 4096, full_damage(&previous))
        .take(3)
        .for_each(drop);
    let mut current = scene();
    add_shape(&mut current, None);
    let commands: Vec<_> = planner
        .plan(&current, 4096, full_damage(&current))
        .collect();
    assert!(commands.iter().all(|command| !matches!(
        command.operation,
        RenderOperation::ApplyEffect(_) | RenderOperation::DrawShapeMask(_)
    )));
    assert!(planner.shape_composites.is_empty());
}

#[test]
fn damage_filters_draws_without_skipping_overflowing_children() {
    let mut scene = scene();
    scene.insert_node(DrawTreeNode::Group(Default::default()), None);
    let visible = add_rectangle(&mut scene, None, [(2.0, 2.0), (8.0, 8.0)], true);
    add_rectangle(&mut scene, None, [(20.0, 20.0), (28.0, 28.0)], true);
    let parent = add_rectangle(&mut scene, None, [(22.0, 22.0), (28.0, 28.0)], false);
    let child = add_rectangle(&mut scene, Some(parent), [(4.0, 4.0), (6.0, 6.0)], true);
    let damage = Some(UnsignedPhysicalRect::new((3, 3).into(), (7, 7).into()));
    let mut planner = Planner::default();
    let draws: Vec<_> = planner
        .plan(&scene, 4096, damage)
        .filter_map(|command| match command.operation {
            RenderOperation::DrawShape(draw)
            | RenderOperation::DrawShapeAndIncrementStencil(draw) => Some(draw.id),
            _ => None,
        })
        .collect();
    assert_eq!(draws, [ShapeDrawId(visible), ShapeDrawId(child)]);
}

#[test]
fn damage_excludes_disjoint_shapes_and_all_their_effect_commands() {
    let mut scene = scene();
    scene.insert_node(DrawTreeNode::Group(Default::default()), None);
    let visible = add_rectangle(&mut scene, None, [(2.0, 2.0), (8.0, 8.0)], true);
    let outside = add_rectangle(&mut scene, None, [(22.0, 22.0), (28.0, 28.0)], true);
    attach_effects(&mut scene, outside, &[1; 4]);
    let damage = Some(UnsignedPhysicalRect::new((3, 3).into(), (7, 7).into()));
    let mut planner = Planner::default();
    let commands: Vec<_> = planner.plan(&scene, 4096, damage).collect();
    assert_eq!(commands.len(), 3);
    assert!(matches!(
        commands[1].operation,
        RenderOperation::DrawShape(draw) if draw.id == ShapeDrawId(visible)
    ));
    assert!(planner.shape_composites.is_empty());
}

#[test]
fn damage_in_effect_padding_retains_mask_and_content_outputs() {
    for is_content_effect in [false, true] {
        let mut scene = scene();
        scene.insert_node(DrawTreeNode::Group(Default::default()), None);
        let node = add_rectangle(&mut scene, None, [(12.0, 12.0), (16.0, 16.0)], true);
        let child = if is_content_effect {
            scene
                .set_shape_mask_effect(node, 22, &[], ShapeMaskEffectConfig::default(), 4096)
                .unwrap();
            scene
                .set_content_effect(node, 17, &[], ContentEffectConfig::new().outset(4.0), 4096)
                .unwrap();
            let child = add_rectangle(&mut scene, Some(node), [(13.0, 13.0), (15.0, 15.0)], true);
            scene
                .set_shape_mask_effect(
                    child,
                    23,
                    &[],
                    ShapeMaskEffectConfig::new().outset(0.5),
                    4096,
                )
                .unwrap();
            Some(child)
        } else {
            scene
                .set_shape_mask_effect(
                    node,
                    17,
                    &[],
                    ShapeMaskEffectConfig::new().outset(4.0),
                    4096,
                )
                .unwrap();
            None
        };
        let outside = add_rectangle(&mut scene, None, [(24.0, 24.0), (28.0, 28.0)], true);
        attach_effects(&mut scene, outside, &[1; 4]);
        let damage = UnsignedPhysicalRect::new((8, 13).into(), (10, 15).into());
        if is_content_effect {
            assert!(!scene
                .shape_mask_effect(node)
                .unwrap()
                .bounds
                .logical_screen_bounds
                .intersects(&damage.to_f32()));
        }
        let mut planner = Planner::default();
        let commands: Vec<_> = planner.plan(&scene, 4096, Some(damage)).collect();
        let effect_ids: Vec<_> = effect_snapshots(&commands)
            .into_iter()
            .map(|effect| effect.effect_id)
            .collect();
        let expected_effect_ids: &[u64] = if is_content_effect {
            &[22, 23, 17]
        } else {
            &[17]
        };
        assert_eq!(effect_ids, expected_effect_ids);
        if let Some(child) = child {
            assert!(commands.iter().any(|command| matches!(
                command.operation,
                RenderOperation::DrawShapeMask(mask) if mask.shape == ShapeDrawId(node)
            )));
            assert!(commands.iter().any(|command| matches!(
                command.operation,
                RenderOperation::DrawShape(draw) if draw.id == ShapeDrawId(child)
            )));
            assert!(commands.iter().any(|command| matches!(
                command.operation,
                RenderOperation::DrawShapeMask(mask) if mask.shape == ShapeDrawId(child)
            )));
        }
        assert!(commands
            .iter()
            .any(|command| matches!(command.operation, RenderOperation::CompositeTexture(_))));
        let begins = commands
            .iter()
            .filter(|command| matches!(command.operation, RenderOperation::BeginTarget(_)))
            .count();
        let ends = commands
            .iter()
            .filter(|command| matches!(command.operation, RenderOperation::EndTarget))
            .count();
        assert_eq!(begins, ends);
    }
}

#[test]
fn absent_damage_yields_nothing_and_does_not_affect_the_next_plan() {
    let mut scene = scene();
    let node = add_shape(&mut scene, None);
    attach_effects(&mut scene, node, &[1; 4]);
    let mut planner = Planner::default();
    let expected: Vec<_> = planner
        .plan(&scene, 4096, full_damage(&scene))
        .map(|command| format!("{command:?}"))
        .collect();
    planner
        .plan(&scene, 4096, full_damage(&scene))
        .take(3)
        .for_each(drop);
    {
        let mut commands = planner.plan(&scene, 4096, None);
        assert!(commands.next().is_none());
        assert!(commands.next().is_none());
    }
    assert!(planner.shape_composites.is_empty());
    let actual: Vec<_> = planner
        .plan(&scene, 4096, full_damage(&scene))
        .map(|command| format!("{command:?}"))
        .collect();
    assert_eq!(actual, expected);
}

#[test]
fn visible_masks_are_prepared_before_main_draws() {
    let mut scene = scene();
    let root = add_rectangle(&mut scene, None, [(0.0, 0.0), (32.0, 32.0)], false);
    for offset in [2.0, 10.0, 18.0] {
        let node = add_rectangle(
            &mut scene,
            Some(root),
            [(offset, offset), (offset + 4.0, offset + 4.0)],
            true,
        );
        scene
            .set_shape_mask_effect(node, 17, &[], ShapeMaskEffectConfig::default(), 4096)
            .unwrap();
    }
    let mut planner = Planner::default();
    let commands: Vec<_> = planner.plan(&scene, 4096, full_damage(&scene)).collect();
    let first_draw = commands
        .iter()
        .position(|command| {
            matches!(
                command.operation,
                RenderOperation::DrawShape(_)
                    | RenderOperation::DrawShapeAndIncrementStencil(_)
                    | RenderOperation::CompositeTexture(_)
            )
        })
        .unwrap();
    let mask_positions: Vec<_> = commands
        .iter()
        .enumerate()
        .filter_map(|(index, command)| {
            matches!(command.operation, RenderOperation::DrawShapeMask(_)).then_some(index)
        })
        .collect();
    assert_eq!(mask_positions.len(), 3);
    assert!(mask_positions.iter().all(|&index| index < first_draw));
}

#[test]
fn scaled_damage_keeps_fractional_edge_pixels_and_excludes_touching_bounds() {
    let mut scene = SceneState::new(
        SceneStateContext::default(),
        Viewport {
            physical_size: (32, 32),
            scale_factor: 1.25,
        },
    );
    scene.insert_node(DrawTreeNode::Group(Default::default()), None);
    let fractional = add_rectangle(&mut scene, None, [(8.1, 6.5), (9.7, 7.1)], true);
    add_rectangle(&mut scene, None, [(6.0, 6.5), (8.0, 7.1)], true);
    let mut planner = Planner::default();
    let damage = Some(UnsignedPhysicalRect::new((10, 8).into(), (11, 9).into()));
    let draws: Vec<_> = planner
        .plan(&scene, 4096, damage)
        .filter_map(|command| match command.operation {
            RenderOperation::DrawShape(draw) => Some(draw.id.0),
            _ => None,
        })
        .collect();
    assert_eq!(draws, [fractional]);
}
