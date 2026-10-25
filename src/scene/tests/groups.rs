use super::{scene, surface};
use crate::commands::{RenderOperation, Target};
use crate::core::{
    BackdropEffectConfig, Color, ContentEffectConfig, Shape, ShapeDrawCommandOptions,
    ShapeMaskEffectConfig, UnsignedPhysicalRect,
};
use crate::scene::EffectError;
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
fn groups_validate_parents_and_reject_effects_without_changing_the_scene() {
    let mut scene = scene();
    let group = scene.add_group(None).unwrap();
    let mut surface = surface();
    scene.render(&mut surface).unwrap();
    assert!(matches!(
        scene.add_group(Some(usize::MAX)),
        Err(SceneError::InvalidShapeId(usize::MAX))
    ));
    for result in [
        scene.set_content_effect(group, 1, &[], ContentEffectConfig::default()),
        scene.set_backdrop_effect(group, 1, &[], BackdropEffectConfig::default()),
        scene.set_shape_mask_effect(group, 1, &[], ShapeMaskEffectConfig::default()),
    ] {
        assert!(matches!(
            result,
            Err(EffectError::Scene(SceneError::UnsupportedNodeOperation(id, "effects")))
                if id == group
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
