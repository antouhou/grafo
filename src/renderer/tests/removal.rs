use super::{queue_shape, renderer, surface, TestBackend};
use crate::core::vertex::InstanceTransform;
use crate::core::{BackdropEffectConfig, Color, Shape, ShapeDrawCommandOptions, ShapeEffectConfig};
use crate::renderer::{DrawCommandError, Renderer};
use crate::scene::SceneError;

fn attach_effects(renderer: &mut Renderer<TestBackend>, node: usize) {
    renderer.set_group_effect(node, 8, &[1, 2, 3, 4]).unwrap();
    renderer
        .set_shape_effect(node, 7, &[1, 2, 3, 4], ShapeEffectConfig::default())
        .unwrap();
    renderer
        .set_shape_backdrop_effect(node, 9, &[1, 2, 3, 4], BackdropEffectConfig::default())
        .unwrap();
}

#[test]
fn overlapping_subtree_removals_keep_sibling_effects() {
    let mut renderer = renderer();
    let mut surface = surface();
    let root = queue_shape(&mut renderer, false);
    let removed = queue_shape(&mut renderer, true);
    let descendant = renderer
        .add_cached_shape(
            1,
            Some(removed),
            ShapeDrawCommandOptions::new().color(Color::WHITE),
        )
        .unwrap();
    attach_effects(&mut renderer, descendant);
    let second_descendant = renderer
        .add_cached_shape(
            1,
            Some(removed),
            ShapeDrawCommandOptions::new().color(Color::WHITE),
        )
        .unwrap();
    attach_effects(&mut renderer, second_descendant);
    let removed_clip = renderer
        .add_clipping_rect(
            [(0.0, 0.0), (16.0, 16.0)],
            Some(root),
            None::<InstanceTransform>,
            true,
        )
        .unwrap();
    let clipped_descendant = renderer
        .add_cached_shape(
            1,
            Some(removed_clip),
            ShapeDrawCommandOptions::new().color(Color::WHITE),
        )
        .unwrap();
    attach_effects(&mut renderer, clipped_descendant);
    let survivor = queue_shape(&mut renderer, false);
    attach_effects(&mut renderer, survivor);
    renderer.render(&mut surface).unwrap();

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
    renderer.remove_subtrees(removed_roots.iter().copied(), |id| removed_ids.push(id));
    removed_ids.sort_unstable();
    let expected_removed_ids = [
        removed,
        descendant,
        second_descendant,
        removed_clip,
        clipped_descendant,
    ];
    assert_eq!(removed_ids, expected_removed_ids);
    assert_eq!(renderer.backend.registered_shapes, [root, survivor]);
    for node in expected_removed_ids {
        assert!(renderer.scene.draw_tree.get(node).is_none());
        assert!(!renderer.scene.group_effects.contains_key(&node));
        assert!(!renderer.scene.backdrop_effects.contains_key(&node));
        assert!(!renderer.scene.shape_effects.contains_key(&node));
    }
    assert_eq!(renderer.scene.draw_tree.children(root), [survivor]);
    assert!(matches!(
        renderer.add_cached_shape(1, Some(removed), ShapeDrawCommandOptions::new()),
        Err(DrawCommandError::Scene(SceneError::InvalidShapeId(id))) if id == removed
    ));
    renderer.remove_subtrees([removed], |_| panic!("removed ID triggered a callback"));
    renderer.remove_subtrees([usize::MAX], |_| panic!("missing ID triggered a callback"));
    renderer.remove_subtrees([], |_| panic!("empty batch triggered a callback"));
    renderer.render(&mut surface).unwrap();
    assert_eq!(surface.resource().shape_masks, 1);

    renderer.backend.should_fail = true;
    assert!(matches!(
        renderer.add_cached_shape(1, None, ShapeDrawCommandOptions::new()),
        Err(DrawCommandError::Backend(_))
    ));
    assert_eq!(renderer.backend.registered_shapes, [root, survivor]);
    renderer.backend.should_fail = false;
    let replacement = queue_shape(&mut renderer, false);
    assert!(expected_removed_ids.contains(&replacement));
    assert!(!renderer.scene.group_effects.contains_key(&replacement));
    assert!(!renderer.scene.backdrop_effects.contains_key(&replacement));
    assert!(!renderer.scene.shape_effects.contains_key(&replacement));
    renderer.render(&mut surface).unwrap();
    assert!(surface.resource().draws.contains(&replacement));
    assert_eq!(surface.resource().shape_masks, 1);

    renderer.remove_subtrees([survivor, replacement], |_| {});
    assert!(renderer.scene.draw_tree.children(root).is_empty());
    renderer.render(&mut surface).unwrap();
    assert_eq!(surface.resource().draws, [root]);
    assert!(surface.resource().effects.is_empty());
}

#[test]
fn removing_clip_subtrees_and_the_root_preserves_loaded_shapes() {
    let mut renderer = renderer();
    let mut surface = surface();
    let root = renderer
        .add_clipping_rect(
            [(0.0, 0.0), (32.0, 32.0)],
            None,
            None::<InstanceTransform>,
            true,
        )
        .unwrap();
    let clip = renderer
        .add_clipping_rect(
            [(0.0, 0.0), (16.0, 16.0)],
            Some(root),
            None::<InstanceTransform>,
            true,
        )
        .unwrap();
    renderer.load_shape(Shape::rect([(0.0, 0.0), (16.0, 16.0)]), 1, Some(1));
    let child = renderer
        .add_cached_shape(
            1,
            Some(clip),
            ShapeDrawCommandOptions::new().color(Color::WHITE),
        )
        .unwrap();
    let mut removed_ids = Vec::new();
    renderer.remove_subtrees([child], |id| removed_ids.push(id));
    assert_eq!(removed_ids, [child]);
    assert!(renderer.scene.draw_tree.children(clip).is_empty());
    let replacement_child = renderer
        .add_cached_shape(
            1,
            Some(clip),
            ShapeDrawCommandOptions::new().color(Color::WHITE),
        )
        .unwrap();
    removed_ids.clear();
    renderer.remove_subtrees([clip], |id| removed_ids.push(id));
    removed_ids.sort_unstable();
    assert_eq!(removed_ids, [clip, replacement_child]);
    assert!(renderer.scene.draw_tree.children(root).is_empty());
    assert!(renderer.backend.registered_shapes.is_empty());
    let remaining_shape = queue_shape(&mut renderer, true);
    removed_ids.clear();
    renderer.remove_subtrees([remaining_shape, root, remaining_shape, root], |id| {
        removed_ids.push(id)
    });
    removed_ids.sort_unstable();
    assert_eq!(removed_ids, [root, remaining_shape]);
    renderer.render(&mut surface).unwrap();
    assert!(surface.resource().draws.is_empty());
    assert!(surface.resource().effects.is_empty());
    assert!(renderer.scene.draw_tree.is_empty());
    assert!(renderer.backend.registered_shapes.is_empty());
    renderer.remove_subtrees([root], |_| panic!("empty queue triggered a callback"));
    assert_eq!(
        renderer
            .add_cached_shape(1, None, ShapeDrawCommandOptions::new().color(Color::WHITE))
            .unwrap(),
        0
    );
    renderer.render(&mut surface).unwrap();
    assert_eq!(surface.resource().draws, [0]);
}

#[test]
fn removed_parents_are_rejected_before_resource_and_clip_preparation() {
    let mut renderer = renderer();
    let root = queue_shape(&mut renderer, false);
    let removed = queue_shape(&mut renderer, false);
    renderer.remove_subtrees([removed], |_| {});
    let bounds = [(0.0, 0.0), (16.0, 16.0)];
    let transform = InstanceTransform::rotation_z_deg(45.0);
    let dirty = renderer.dirty_bounds;
    let next_node = renderer.scene.next_node_id();
    renderer.backend.should_fail = true;

    for result in [
        renderer.add_shape(
            Shape::rect(bounds),
            Some(removed),
            None,
            ShapeDrawCommandOptions::new(),
        ),
        renderer.add_cached_shape(999, Some(removed), ShapeDrawCommandOptions::new()),
        renderer.add_clipping_rect(bounds, Some(removed), Some(transform), true),
    ] {
        assert!(matches!(
            result,
            Err(DrawCommandError::Scene(SceneError::InvalidShapeId(id))) if id == removed
        ));
    }

    let cached_shape = renderer.scene.loaded_shape(1).unwrap();
    for result in [
        renderer
            .scene
            .add_shape(cached_shape, Some(removed), ShapeDrawCommandOptions::new()),
        renderer
            .scene
            .add_clipping_rect(bounds, Some(removed), Some(transform), true),
    ] {
        assert!(matches!(result, Err(SceneError::InvalidShapeId(id)) if id == removed));
    }

    assert!(matches!(
        renderer.add_shape(
            Shape::rect(bounds),
            Some(root),
            None,
            ShapeDrawCommandOptions::new(),
        ),
        Err(DrawCommandError::Backend(_))
    ));
    assert!(matches!(
        renderer.add_cached_shape(999, Some(root), ShapeDrawCommandOptions::new()),
        Err(DrawCommandError::Scene(SceneError::ShapeNotLoaded(999)))
    ));
    assert!(matches!(
        renderer.add_clipping_rect(bounds, Some(root), Some(transform), true),
        Err(DrawCommandError::Scene(
            SceneError::UnsupportedClipRectTransform
        ))
    ));
    assert!(matches!(
        renderer
            .scene
            .add_clipping_rect(bounds, Some(root), Some(transform), true),
        Err(SceneError::UnsupportedClipRectTransform)
    ));
    assert_eq!(renderer.scene.next_node_id(), next_node);
    assert!(renderer.scene.draw_tree.get(removed).is_none());
    assert!(renderer.scene.draw_tree.children(root).is_empty());
    assert_eq!(renderer.backend.registered_shapes, [root]);
    assert_eq!(renderer.dirty_bounds, dirty);
}

#[test]
fn replacements_and_subtree_removals_defer_compaction_and_preserve_surviving_parameters() {
    let mut renderer = renderer();
    let mut surface = surface();
    queue_shape(&mut renderer, false);
    let mut branch = queue_shape(&mut renderer, false);
    attach_effects(&mut renderer, branch);
    let survivor = queue_shape(&mut renderer, false);
    attach_effects(&mut renderer, survivor);
    for _ in 0..20 {
        let converted = queue_shape(&mut renderer, false);
        attach_effects(&mut renderer, converted);
        renderer
            .update_group_effect_params(survivor, &[5; 4])
            .unwrap();
        renderer
            .update_backdrop_effect_params(survivor, &[6; 4])
            .unwrap();
        renderer
            .update_shape_effect_params(survivor, &[7; 4])
            .unwrap();
        let group_parameters = renderer.scene.group_effect(survivor).unwrap().parameters;
        let backdrop_parameters = renderer.scene.backdrop_effect(survivor).unwrap().parameters;
        let shape_parameters = renderer.scene.shape_effect(survivor).unwrap().parameters;
        renderer.remove_subtrees([branch], |_| {});
        renderer
            .replace_with_clipping_rect(
                converted,
                [(0.0, 0.0), (16.0, 16.0)],
                None::<InstanceTransform>,
                true,
            )
            .unwrap();
        renderer.remove_subtrees([converted], |_| {});
        let plan = renderer.planner.plan(&renderer.scene, 4096, None);
        assert_eq!(plan.effect_parameters.len(), 36);
        assert_eq!(plan.parameters(group_parameters), &[5; 4]);
        assert_eq!(plan.parameters(backdrop_parameters), &[6; 4]);
        assert_eq!(plan.parameters(shape_parameters), &[7; 4]);
        renderer
            .update_group_effect_params(survivor, &[1, 2, 3, 4])
            .unwrap();
        renderer
            .update_backdrop_effect_params(survivor, &[1, 2, 3, 4])
            .unwrap();
        renderer
            .update_shape_effect_params(survivor, &[1, 2, 3, 4])
            .unwrap();
        renderer.render(&mut surface).unwrap();
        assert_eq!(
            renderer
                .planner
                .plan(&renderer.scene, 4096, None)
                .effect_parameters
                .len(),
            12
        );
        branch = queue_shape(&mut renderer, false);
        attach_effects(&mut renderer, branch);
        renderer.render(&mut surface).unwrap();
        assert_eq!(surface.resource().shape_masks, 2);
    }
}
