use super::{queue_shape, scene, surface, TestBackend};
use crate::core::vertex::InstanceTransform;
use crate::core::{
    BackdropEffectConfig, Color, ContentEffectConfig, Shape, ShapeDrawCommandOptions,
    ShapeMaskEffectConfig,
};
use crate::scene::{DrawCommandError, Scene};
use crate::scene_state::SceneError;

fn attach_effects(scene: &mut Scene<TestBackend>, node: usize) {
    scene
        .set_content_effect(node, 8, &[1, 2, 3, 4], ContentEffectConfig::default())
        .unwrap();
    scene
        .set_shape_mask_effect(node, 7, &[1, 2, 3, 4], ShapeMaskEffectConfig::default())
        .unwrap();
    scene
        .set_backdrop_effect(node, 9, &[1, 2, 3, 4], BackdropEffectConfig::default())
        .unwrap();
}

#[test]
fn overlapping_subtree_removals_keep_sibling_effects() {
    let mut scene = scene();
    let mut surface = surface();
    let root = queue_shape(&mut scene, false);
    let removed = queue_shape(&mut scene, true);
    let descendant = scene
        .add_cached_shape(
            1,
            Some(removed),
            ShapeDrawCommandOptions::new().color(Color::WHITE),
        )
        .unwrap();
    attach_effects(&mut scene, descendant);
    let second_descendant = scene
        .add_cached_shape(
            1,
            Some(removed),
            ShapeDrawCommandOptions::new().color(Color::WHITE),
        )
        .unwrap();
    attach_effects(&mut scene, second_descendant);
    let removed_clip = scene
        .add_clipping_rect(
            [(0.0, 0.0), (16.0, 16.0)],
            Some(root),
            None::<InstanceTransform>,
            true,
        )
        .unwrap();
    let clipped_descendant = scene
        .add_cached_shape(
            1,
            Some(removed_clip),
            ShapeDrawCommandOptions::new().color(Color::WHITE),
        )
        .unwrap();
    attach_effects(&mut scene, clipped_descendant);
    let survivor = queue_shape(&mut scene, false);
    attach_effects(&mut scene, survivor);
    scene.render(&mut surface).unwrap();

    let mut removed_ids = Vec::new();
    let removed_roots = [
        descendant,
        removed,
        removed_clip,
        descendant,
        clipped_descendant,
        usize::MAX,
        removed,
    ];
    scene.remove_subtrees(removed_roots.iter().copied(), |id| removed_ids.push(id));
    removed_ids.sort_unstable();
    let expected_removed_ids = [
        removed,
        descendant,
        second_descendant,
        removed_clip,
        clipped_descendant,
    ];
    assert_eq!(removed_ids, expected_removed_ids);
    assert_eq!(scene.backend.registered_shapes, [root, survivor]);
    for node in expected_removed_ids {
        assert!(scene.state.draw_tree.get(node).is_none());
        assert!(!scene.state.content_effects.contains_key(&node));
        assert!(!scene.state.backdrop_effects.contains_key(&node));
        assert!(!scene.state.shape_mask_effects.contains_key(&node));
    }
    assert_eq!(scene.state.draw_tree.children(root), [survivor]);
    assert!(matches!(
        scene.add_cached_shape(1, Some(removed), ShapeDrawCommandOptions::new()),
        Err(DrawCommandError::Scene(SceneError::InvalidShapeId(id))) if id == removed
    ));
    scene.remove_subtrees([removed], |_| panic!("removed ID triggered a callback"));
    scene.remove_subtrees([usize::MAX], |_| panic!("missing ID triggered a callback"));
    scene.remove_subtrees([], |_| panic!("empty batch triggered a callback"));
    scene.render(&mut surface).unwrap();
    assert_eq!(surface.resource().shape_masks, 1);

    scene.backend.should_fail = true;
    assert!(matches!(
        scene.add_cached_shape(1, None, ShapeDrawCommandOptions::new()),
        Err(DrawCommandError::Backend(_))
    ));
    assert_eq!(scene.backend.registered_shapes, [root, survivor]);
    scene.backend.should_fail = false;
    let replacement = queue_shape(&mut scene, false);
    assert!(expected_removed_ids.contains(&replacement));
    assert!(!scene.state.content_effects.contains_key(&replacement));
    assert!(!scene.state.backdrop_effects.contains_key(&replacement));
    assert!(!scene.state.shape_mask_effects.contains_key(&replacement));
    scene.render(&mut surface).unwrap();
    assert!(surface.resource().draws.contains(&replacement));
    assert_eq!(surface.resource().shape_masks, 1);

    scene.remove_subtrees([survivor, replacement], |_| {});
    assert!(scene.state.draw_tree.children(root).is_empty());
    scene.render(&mut surface).unwrap();
    assert_eq!(surface.resource().draws, [root]);
    assert!(surface.resource().effects.is_empty());
}

#[test]
fn removing_clip_subtrees_and_the_root_preserves_loaded_shapes() {
    let mut scene = scene();
    let mut surface = surface();
    let root = scene
        .add_clipping_rect(
            [(0.0, 0.0), (32.0, 32.0)],
            None,
            None::<InstanceTransform>,
            true,
        )
        .unwrap();
    let clip = scene
        .add_clipping_rect(
            [(0.0, 0.0), (16.0, 16.0)],
            Some(root),
            None::<InstanceTransform>,
            true,
        )
        .unwrap();
    scene.load_shape(Shape::rect([(0.0, 0.0), (16.0, 16.0)]), 1, Some(1));
    let child = scene
        .add_cached_shape(
            1,
            Some(clip),
            ShapeDrawCommandOptions::new().color(Color::WHITE),
        )
        .unwrap();
    let mut removed_ids = Vec::new();
    scene.remove_subtrees([child], |id| removed_ids.push(id));
    assert_eq!(removed_ids, [child]);
    assert!(scene.state.draw_tree.children(clip).is_empty());
    let replacement_child = scene
        .add_cached_shape(
            1,
            Some(clip),
            ShapeDrawCommandOptions::new().color(Color::WHITE),
        )
        .unwrap();
    removed_ids.clear();
    scene.remove_subtrees([clip], |id| removed_ids.push(id));
    removed_ids.sort_unstable();
    assert_eq!(removed_ids, [clip, replacement_child]);
    assert!(scene.state.draw_tree.children(root).is_empty());
    assert!(scene.backend.registered_shapes.is_empty());
    let remaining_shape = queue_shape(&mut scene, true);
    removed_ids.clear();
    scene.remove_subtrees([remaining_shape, root, remaining_shape, root], |id| {
        removed_ids.push(id)
    });
    removed_ids.sort_unstable();
    assert_eq!(removed_ids, [root, remaining_shape]);
    scene.render(&mut surface).unwrap();
    assert!(surface.resource().draws.is_empty());
    assert!(surface.resource().effects.is_empty());
    assert!(scene.state.draw_tree.is_empty());
    assert!(scene.backend.registered_shapes.is_empty());
    scene.remove_subtrees([root], |_| panic!("empty queue triggered a callback"));
    assert_eq!(
        scene
            .add_cached_shape(1, None, ShapeDrawCommandOptions::new().color(Color::WHITE))
            .unwrap(),
        0
    );
    scene.render(&mut surface).unwrap();
    assert_eq!(surface.resource().draws, [0]);
}

#[test]
fn removed_parents_are_rejected_before_resource_and_clip_preparation() {
    let mut scene = scene();
    let root = queue_shape(&mut scene, false);
    let removed = queue_shape(&mut scene, false);
    scene.remove_subtrees([removed], |_| {});
    let bounds = [(0.0, 0.0), (16.0, 16.0)];
    let transform = InstanceTransform::rotation_z_deg(45.0);
    let dirty = scene.dirty_bounds;
    let next_node = scene.state.next_node_id();
    scene.backend.should_fail = true;

    for result in [
        scene.add_shape(
            Shape::rect(bounds),
            Some(removed),
            None,
            ShapeDrawCommandOptions::new(),
        ),
        scene.add_cached_shape(999, Some(removed), ShapeDrawCommandOptions::new()),
        scene.add_clipping_rect(bounds, Some(removed), Some(transform), true),
    ] {
        assert!(matches!(
            result,
            Err(DrawCommandError::Scene(SceneError::InvalidShapeId(id))) if id == removed
        ));
    }

    assert!(matches!(
        scene.add_shape(
            Shape::rect(bounds),
            Some(root),
            None,
            ShapeDrawCommandOptions::new(),
        ),
        Err(DrawCommandError::Backend(_))
    ));
    assert!(matches!(
        scene.add_cached_shape(999, Some(root), ShapeDrawCommandOptions::new()),
        Err(DrawCommandError::Scene(SceneError::ShapeNotLoaded(999)))
    ));
    assert!(matches!(
        scene.add_clipping_rect(bounds, Some(root), Some(transform), true),
        Err(DrawCommandError::Scene(
            SceneError::UnsupportedClipRectTransform
        ))
    ));
    assert_eq!(scene.state.next_node_id(), next_node);
    assert!(scene.state.draw_tree.get(removed).is_none());
    assert!(scene.state.draw_tree.children(root).is_empty());
    assert_eq!(scene.backend.registered_shapes, [root]);
    assert_eq!(scene.dirty_bounds, dirty);
}

#[test]
fn replacements_and_subtree_removals_defer_compaction_and_preserve_surviving_parameters() {
    let mut scene = scene();
    let mut surface = surface();
    queue_shape(&mut scene, false);
    let mut branch = queue_shape(&mut scene, false);
    attach_effects(&mut scene, branch);
    let survivor = queue_shape(&mut scene, false);
    attach_effects(&mut scene, survivor);
    for _ in 0..20 {
        let converted = queue_shape(&mut scene, false);
        attach_effects(&mut scene, converted);
        scene
            .set_content_effect(survivor, 8, &[5; 4], ContentEffectConfig::default())
            .unwrap();
        scene
            .set_backdrop_effect(survivor, 9, &[6; 4], BackdropEffectConfig::default())
            .unwrap();
        scene
            .set_shape_mask_effect(survivor, 7, &[7; 4], ShapeMaskEffectConfig::default())
            .unwrap();
        let group_parameters = scene.state.content_effects[&survivor].parameters;
        let backdrop_parameters = scene.state.backdrop_effects[&survivor].effect.parameters;
        let shape_parameters = scene.state.shape_mask_effect(survivor).unwrap().parameters;
        scene.remove_subtrees([branch], |_| {});
        scene
            .replace_with_clipping_rect(
                converted,
                [(0.0, 0.0), (16.0, 16.0)],
                None::<InstanceTransform>,
                true,
            )
            .unwrap();
        scene.remove_subtrees([converted], |_| {});
        assert_eq!(scene.state.effect_parameters.bytes.len(), 48);
        assert_eq!(
            scene.state.effect_parameters.get(group_parameters).bytes,
            &[5; 4]
        );
        assert_eq!(
            scene.state.effect_parameters.get(backdrop_parameters).bytes,
            &[6; 4]
        );
        assert_eq!(
            scene.state.effect_parameters.get(shape_parameters).bytes,
            &[7; 4]
        );
        scene
            .set_content_effect(survivor, 8, &[1, 2, 3, 4], ContentEffectConfig::default())
            .unwrap();
        scene
            .set_backdrop_effect(survivor, 9, &[1, 2, 3, 4], BackdropEffectConfig::default())
            .unwrap();
        scene
            .set_shape_mask_effect(survivor, 7, &[1, 2, 3, 4], ShapeMaskEffectConfig::default())
            .unwrap();
        scene.render(&mut surface).unwrap();
        assert_eq!(scene.state.effect_parameters.bytes.len(), 12);
        branch = queue_shape(&mut scene, false);
        attach_effects(&mut scene, branch);
        scene.render(&mut surface).unwrap();
        assert_eq!(surface.resource().shape_masks, 2);
    }
}
