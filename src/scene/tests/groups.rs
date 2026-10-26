use super::{scene, surface};
use crate::commands::{RenderOperation, Target};
use crate::core::{
    BackdropEffectConfig, Color, ContentEffectConfig, Shape, ShapeDrawCommandOptions,
    ShapeMaskEffectConfig, UnsignedPhysicalRect,
};
use crate::scene::EffectError;
use crate::scene_state::types::DrawTreeNode;
use crate::scene_state::SceneError;

#[test]
fn nested_groups_draw_in_order_and_removal_cleans_up_only_their_descendants() {
    let mut scene = scene();
    let root = scene.add_group(None).unwrap();
    let group = scene.add_group(Some(root)).unwrap();
    let nested = scene.add_group(Some(group)).unwrap();
    let empty = scene.add_group(Some(nested)).unwrap();
    let mut shapes = Vec::new();
    for (parent, position) in [(group, 4.0), (nested, 16.0), (root, 24.0)] {
        shapes.push(
            scene
                .add_shape(
                    Shape::rect([(position, position), (position + 4.0, position + 4.0)]),
                    Some(parent),
                    None,
                    ShapeDrawCommandOptions::new().color(Color::WHITE),
                )
                .unwrap(),
        );
    }
    let mut surface = surface();
    scene.render(&mut surface).unwrap();
    assert_eq!(surface.resource().draws, [shapes[1], shapes[0], shapes[2]]);
    assert_eq!(scene.backend.registered_shapes, shapes);
    assert!(scene
        .planner
        .plan(&scene.state, 4096)
        .all(|command| matches!(
            command.operation,
            RenderOperation::BeginTarget(Target::Surface)
                | RenderOperation::DrawShape(_)
                | RenderOperation::EndTarget
        )));

    let mut removed = Vec::new();
    scene.remove_subtrees([nested, group, group], |id| removed.push(id));
    removed.sort_unstable();
    assert_eq!(removed, [group, nested, empty, shapes[0], shapes[1]]);
    assert_eq!(scene.backend.registered_shapes, [shapes[2]]);
    scene.render(&mut surface).unwrap();
    assert_eq!(surface.resource().draws, [shapes[2]]);
    assert_eq!(
        scene.backend.root_scissor,
        Some(UnsignedPhysicalRect::new((4, 4).into(), (20, 20).into()))
    );

    scene
        .replace_with_shape(
            root,
            Shape::rect([(8.0, 8.0), (16.0, 16.0)]),
            None,
            ShapeDrawCommandOptions::new().color(Color::BLACK),
        )
        .unwrap();
    assert_eq!(scene.state.draw_tree.children(root), &[shapes[2]]);
    scene.render(&mut surface).unwrap();
    assert_eq!(
        scene.backend.root_scissor,
        Some(UnsignedPhysicalRect::new((8, 8).into(), (28, 28).into()))
    );
    scene.clear();
    scene.render(&mut surface).unwrap();
    assert!(surface.resource().draws.is_empty());
    assert!(scene.backend.registered_shapes.is_empty());
}

#[test]
fn groups_validate_parents_and_effect_configs_without_changing_the_scene() {
    let mut scene = scene();
    let group = scene.add_group(None).unwrap();
    let mut surface = surface();
    scene.render(&mut surface).unwrap();
    assert!(matches!(
        scene.add_group(Some(usize::MAX)),
        Err(SceneError::InvalidShapeId(usize::MAX))
    ));
    for result in [
        scene.set_content_effect(
            group,
            1,
            &[1, 2, 3, 4],
            ContentEffectConfig::default().downsample(0.0),
        ),
        scene.set_backdrop_effect(
            group,
            1,
            &[1, 2, 3, 4],
            BackdropEffectConfig::default().downsample(0.0),
        ),
        scene.set_shape_mask_effect(
            group,
            1,
            &[1, 2, 3, 4],
            ShapeMaskEffectConfig::default().downsample(0.0),
        ),
    ] {
        assert!(matches!(
            result,
            Err(EffectError::Scene(SceneError::InvalidParams(_)))
        ));
    }
    assert!(scene.state.draw_tree.children(group).is_empty());
    assert!(scene.state.effect_parameters.bytes.is_empty());
    assert!(scene.state.content_effects.is_empty());
    assert!(scene.state.backdrop_effects.is_empty());
    assert!(scene.state.shape_mask_effects.is_empty());
    assert!(scene.dirty_bounds.is_none());
    assert!(scene.backend.registered_shapes.is_empty());
}

#[test]
fn group_effects_apply_to_existing_and_new_members() {
    let mut scene = scene();
    let root = scene.add_group(None).unwrap();
    let group = scene.add_group(Some(root)).unwrap();
    let first = scene
        .add_shape(
            Shape::rect([(2.0, 2.0), (8.0, 8.0)]),
            Some(group),
            None,
            ShapeDrawCommandOptions::new().color(Color::WHITE),
        )
        .unwrap();
    scene
        .set_content_effect(group, 8, &[1, 2, 3, 4], ContentEffectConfig::default())
        .unwrap();
    scene
        .set_shape_mask_effect(group, 7, &[1, 2, 3, 4], ShapeMaskEffectConfig::default())
        .unwrap();
    let nested = scene.add_group(Some(group)).unwrap();
    let second = scene
        .add_shape(
            Shape::rect([(12.0, 12.0), (18.0, 18.0)]),
            Some(nested),
            None,
            ShapeDrawCommandOptions::new().color(Color::WHITE),
        )
        .unwrap();
    let mut surface = surface();
    scene.render(&mut surface).unwrap();
    for id in [first, second] {
        assert!(scene.state.content_effects.contains_key(&id));
        assert!(scene.state.shape_mask_effects.contains_key(&id));
    }
    assert_eq!(surface.resource().shape_masks, 2);
    scene.remove_shape_mask_effect(group);
    scene.unload_effect(8);
    scene.render(&mut surface).unwrap();
    assert!(surface.resource().effects.is_empty());
    assert!(scene.state.shape_mask_effects.is_empty());
    assert!(scene.state.content_effects.is_empty());
    scene.clear();
    assert!(scene.state.effect_parameters.bytes.is_empty());
}

#[test]
fn group_effects_survive_replacement_and_nested_overrides() {
    let mut scene = scene();
    let root = scene.add_group(None).unwrap();
    let nested = scene.add_group(Some(root)).unwrap();
    scene
        .set_content_effect(root, 8, &[1, 2, 3, 4], ContentEffectConfig::default())
        .unwrap();
    scene
        .set_content_effect(nested, 9, &[1, 2, 3, 4], ContentEffectConfig::default())
        .unwrap();
    let shape = scene
        .add_shape(
            Shape::rect([(2.0, 2.0), (8.0, 8.0)]),
            Some(nested),
            None,
            ShapeDrawCommandOptions::new().color(Color::WHITE),
        )
        .unwrap();
    let mut surface = surface();
    scene.render(&mut surface).unwrap();
    scene
        .set_content_effect(root, 10, &[1, 2, 3, 4], ContentEffectConfig::default())
        .unwrap();
    assert_eq!(scene.state.content_effects[&shape].effect_id, 9);
    scene.remove_content_effect(nested);
    assert_eq!(scene.state.content_effects[&shape].effect_id, 10);
    scene.render(&mut surface).unwrap();
    scene
        .set_content_effect(nested, 9, &[1, 2, 3, 4], ContentEffectConfig::default())
        .unwrap();
    scene.unload_effect(9);
    assert_eq!(scene.state.content_effects[&shape].effect_id, 10);
    let replacement = scene.add_group(Some(root)).unwrap();
    scene
        .replace_with_shape(
            replacement,
            Shape::rect([(12.0, 12.0), (18.0, 18.0)]),
            None,
            ShapeDrawCommandOptions::new().color(Color::WHITE),
        )
        .unwrap();
    assert_eq!(scene.state.content_effects[&replacement].effect_id, 10);
    scene
        .replace_with_shape(
            root,
            Shape::rect([(0.0, 0.0), (32.0, 32.0)]),
            None,
            ShapeDrawCommandOptions::new().color(Color::WHITE),
        )
        .unwrap();
    assert_eq!(scene.state.content_effects[&root].effect_id, 10);
    scene.render(&mut surface).unwrap();
    scene.unload_effect(10);
    scene.render(&mut surface).unwrap();
    assert!(surface.resource().effects.is_empty());
}

#[test]
fn inherited_mask_failure_preserves_the_group_and_existing_shapes() {
    let mut scene = scene();
    let group = scene.add_group(None).unwrap();
    scene
        .set_shape_mask_effect(group, 7, &[1, 2, 3, 4], ShapeMaskEffectConfig::default())
        .unwrap();
    let mut surface = surface();
    scene.render(&mut surface).unwrap();
    let options = ShapeDrawCommandOptions::new();
    assert!(scene
        .add_shape(
            Shape::rect([(0.0, 0.0), (f32::MAX, f32::MAX)]),
            Some(group),
            None,
            options.clone(),
        )
        .is_err());
    assert!(scene.state.draw_tree.children(group).is_empty());
    assert!(scene.dirty_bounds.is_none());
    assert!(scene.backend.registered_shapes.is_empty());
    assert!(scene
        .replace_with_shape(
            group,
            Shape::rect([(0.0, 0.0), (f32::MAX, f32::MAX)]),
            None,
            options,
        )
        .is_err());
    assert!(matches!(
        scene.state.draw_tree.get(group),
        Some(DrawTreeNode::Group(_))
    ));
}
