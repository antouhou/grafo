use super::{queue_shape, scene, surface, DrawCommandError, Scene, TestBackend, TestBackendError};
use crate::core::vertex::InstanceTransform;
use crate::core::{
    BackdropCaptureArea, BackdropEffectConfig, ContentEffectConfig, Shape, ShapeDrawCommandOptions,
    ShapeMaskEffectConfig, UnsignedPhysicalRect,
};
use crate::scene_state::effects::EffectConfig;
use crate::scene_state::types::DrawTreeNode;
use crate::scene_state::SceneError;

fn shape(scene: &mut Scene<TestBackend>, bounds: [(f32, f32); 2], parent: usize) -> usize {
    scene
        .add_shape(
            Shape::rect(bounds),
            Some(parent),
            None,
            ShapeDrawCommandOptions::new(),
        )
        .unwrap()
}

fn clip(scene: &mut Scene<TestBackend>, node_id: usize, clips_children: bool) {
    scene
        .replace_with_clipping_rect(
            node_id,
            [(10.0, 10.0), (18.0, 18.0)],
            None::<InstanceTransform>,
            clips_children,
        )
        .unwrap();
}

fn rect(min: (u32, u32), max: (u32, u32)) -> Option<UnsignedPhysicalRect> {
    Some(UnsignedPhysicalRect::new(min.into(), max.into()))
}

#[test]
fn shape_to_clip_replacement_restores_inheritance_and_preserves_overrides() {
    for config in [
        EffectConfig::Content(ContentEffectConfig::default()),
        EffectConfig::Backdrop(BackdropEffectConfig::default()),
        EffectConfig::ShapeMask(ShapeMaskEffectConfig::default()),
    ] {
        for has_own_effect in [false, true] {
            let mut scene = scene();
            let ancestor = scene.add_group(None).unwrap();
            scene
                .set_effect(ancestor, 8, &[1, 2, 3, 4], config)
                .unwrap();
            let bounds = [(2.0, 2.0), (14.0, 14.0)];
            let parent = shape(&mut scene, bounds, ancestor);
            if has_own_effect {
                scene.set_effect(parent, 9, &[1, 2, 3, 4], config).unwrap();
            }
            let member = shape(&mut scene, bounds, parent);
            let descendant = shape(&mut scene, bounds, member);
            let nested = scene.add_group(Some(parent)).unwrap();
            let nested_clip = scene
                .add_clipping_rect(bounds, Some(nested), None::<InstanceTransform>, false)
                .unwrap();
            let nested_member = shape(&mut scene, bounds, nested_clip);
            let explicit_member = shape(&mut scene, bounds, parent);
            scene
                .set_effect(explicit_member, 11, &[1, 2, 3, 4], config)
                .unwrap();
            let overriding_group = scene.add_group(Some(parent)).unwrap();
            scene
                .set_effect(overriding_group, 10, &[1, 2, 3, 4], config)
                .unwrap();
            let overridden_member = shape(&mut scene, bounds, overriding_group);
            let mut surface = surface();
            scene.render(&mut surface).unwrap();

            clip(&mut scene, parent, false);
            let new_member = shape(&mut scene, bounds, nested_clip);
            let inherited_members = [member, nested_member, new_member];
            for &node_id in &inherited_members {
                let effect = scene.state.node_effects(node_id).get(config.kind());
                assert_eq!(
                    effect.map(|effect| (effect.instance.effect_id, effect.source_group_id)),
                    Some((8, Some(ancestor))),
                    "{config:?}, node {node_id}, own effect: {has_own_effect}"
                );
            }
            scene.render(&mut surface).unwrap();
            scene
                .set_effect(ancestor, 12, &[1, 2, 3, 4], config)
                .unwrap();
            scene.render(&mut surface).unwrap();
            for (node_id, expected) in inherited_members
                .into_iter()
                .map(|node_id| (node_id, Some((12, Some(ancestor)))))
                .chain([
                    (descendant, None),
                    (explicit_member, Some((11, None))),
                    (overridden_member, Some((10, Some(overriding_group)))),
                ])
            {
                let effect = scene.state.node_effects(node_id).get(config.kind());
                assert_eq!(
                    effect.map(|effect| (effect.instance.effect_id, effect.source_group_id)),
                    expected,
                    "{config:?}, node {node_id}, own effect: {has_own_effect}"
                );
            }
        }
    }
}

#[test]
fn shape_to_clip_replacement_rejects_invalid_inherited_masks_before_mutation() {
    let mut scene = scene();
    let ancestor = scene.add_group(None).unwrap();
    scene
        .set_shape_mask_effect(ancestor, 7, &[1, 2, 3, 4], ShapeMaskEffectConfig::default())
        .unwrap();
    let parent = shape(&mut scene, [(2.0, 2.0), (14.0, 14.0)], ancestor);
    let child = shape(&mut scene, [(0.0, 0.0), (f32::MAX, f32::MAX)], parent);
    let mut surface = surface();
    scene.render(&mut surface).unwrap();
    let registered_shapes = scene.backend.registered_shapes.clone();
    let parameters = scene.state.effect_parameters.bytes.clone();

    let result = scene.replace_with_clipping_rect(
        parent,
        [(2.0, 2.0), (14.0, 14.0)],
        None::<InstanceTransform>,
        true,
    );

    assert!(matches!(
        result,
        Err(DrawCommandError::Scene(SceneError::InvalidShapeMaskEffectBounds(id))) if id == child
    ));
    assert!(matches!(
        scene.state.draw_tree.get(parent),
        Some(DrawTreeNode::CachedShape(_))
    ));
    assert_eq!(scene.state.draw_tree.children(parent), &[child]);
    assert_eq!(scene.backend.registered_shapes, registered_shapes);
    assert_eq!(scene.state.effect_parameters.bytes, parameters);
    assert!(scene.state.shape_mask_effects.contains_key(&parent));
    assert!(!scene.state.shape_mask_effects.contains_key(&child));
    assert!(!scene.state.has_text_glyphs());
    assert!(scene.dirty_bounds.is_none());
}

#[test]
fn group_to_clip_replacement_rejects_invalid_inherited_masks_before_mutation() {
    let mut scene = scene();
    let ancestor = scene.add_group(None).unwrap();
    let group = scene.add_group(Some(ancestor)).unwrap();
    let attachments = [
        (8, EffectConfig::Content(ContentEffectConfig::default())),
        (9, EffectConfig::Backdrop(BackdropEffectConfig::default())),
        (7, EffectConfig::ShapeMask(ShapeMaskEffectConfig::default())),
    ];
    for (effect_id, config) in attachments {
        scene
            .set_effect(group, effect_id, &[1, 2, 3, 4], config)
            .unwrap();
    }
    let member = shape(&mut scene, [(11.0, 11.0), (12.0, 12.0)], group);
    scene
        .set_shape_mask_effect(
            ancestor,
            10,
            &[1, 2, 3, 4],
            ShapeMaskEffectConfig::new().outset(f32::MAX),
        )
        .unwrap();
    let mut surface = surface();
    scene.render(&mut surface).unwrap();
    let registered_shapes = scene.backend.registered_shapes.clone();
    let parameters = scene.state.effect_parameters.bytes.clone();

    let result = scene.replace_with_clipping_rect(
        group,
        [(10.0, 10.0), (18.0, 18.0)],
        None::<InstanceTransform>,
        true,
    );

    assert!(
        matches!(
            &result,
            Err(DrawCommandError::Scene(SceneError::InvalidShapeMaskEffectBounds(id))) if *id == member
        ),
        "{result:?}"
    );
    assert!(matches!(
        scene.state.draw_tree.get(group),
        Some(DrawTreeNode::Group(_))
    ));
    assert_eq!(scene.state.draw_tree.children(group), &[member]);
    assert_eq!(scene.backend.registered_shapes, registered_shapes);
    assert_eq!(scene.state.effect_parameters.bytes, parameters);
    assert!(scene.dirty_bounds.is_none());
    for (effect_id, config) in attachments {
        for (node_id, source_group_id) in [(group, None), (member, Some(group))] {
            let effect = scene
                .state
                .node_effects(node_id)
                .get(config.kind())
                .unwrap();
            assert_eq!(effect.instance.effect_id, effect_id);
            assert_eq!(effect.source_group_id, source_group_id);
        }
    }

    scene
        .set_shape_mask_effect(
            ancestor,
            10,
            &[1, 2, 3, 4],
            ShapeMaskEffectConfig::default(),
        )
        .unwrap();
    clip(&mut scene, group, true);
    let effect = scene.state.node_effects(member).shape_mask.unwrap();
    assert_eq!(effect.instance.effect_id, 10);
    assert_eq!(effect.source_group_id, Some(ancestor));
    scene.render(&mut surface).unwrap();
    assert_eq!(surface.resource().effects, [10]);
}

#[test]
fn replacing_a_group_with_a_clip_restores_member_effects() {
    for config in [
        EffectConfig::Content(ContentEffectConfig::default()),
        EffectConfig::Backdrop(BackdropEffectConfig::default()),
        EffectConfig::ShapeMask(ShapeMaskEffectConfig::default()),
    ] {
        for has_parent_effect in [false, true] {
            let mut scene = scene();
            let parent = scene.add_group(None).unwrap();
            if has_parent_effect {
                scene.set_effect(parent, 8, &[1, 2, 3, 4], config).unwrap();
            }
            let group = scene.add_group(Some(parent)).unwrap();
            scene.set_effect(group, 9, &[1, 2, 3, 4], config).unwrap();
            let bounds = [(11.0, 11.0), (12.0, 12.0)];
            let member = shape(&mut scene, bounds, group);
            let nested = scene.add_group(Some(group)).unwrap();
            let nested_clip = scene
                .add_clipping_rect(bounds, Some(nested), None::<InstanceTransform>, false)
                .unwrap();
            let nested_member = shape(&mut scene, bounds, nested_clip);
            let explicit_member = shape(&mut scene, bounds, group);
            scene
                .set_effect(explicit_member, 9, &[1, 2, 3, 4], config)
                .unwrap();
            let overriding_group = scene.add_group(Some(group)).unwrap();
            scene
                .set_effect(overriding_group, 10, &[1, 2, 3, 4], config)
                .unwrap();
            let overridden_member = shape(&mut scene, bounds, overriding_group);
            let mut surface = surface();
            scene.render(&mut surface).unwrap();

            assert!(matches!(
                scene.replace_with_clipping_rect(
                    group,
                    bounds,
                    Some(InstanceTransform::rotation_z_deg(45.0)),
                    true,
                ),
                Err(DrawCommandError::Scene(
                    SceneError::UnsupportedClipRectTransform
                ))
            ));
            assert_eq!(
                scene
                    .state
                    .node_effects(member)
                    .get(config.kind())
                    .unwrap()
                    .instance
                    .effect_id,
                9
            );
            assert!(scene.dirty_bounds.is_none());

            clip(&mut scene, group, true);
            let new_member = shape(&mut scene, bounds, group);
            let inherited_effect = has_parent_effect.then_some((8, Some(parent)));
            for (node_id, expected) in [
                (member, inherited_effect),
                (nested_member, inherited_effect),
                (new_member, inherited_effect),
                (explicit_member, Some((9, None))),
                (overridden_member, Some((10, Some(overriding_group)))),
            ] {
                let effect = scene.state.node_effects(node_id).get(config.kind());
                assert_eq!(
                    effect.map(|effect| (effect.instance.effect_id, effect.source_group_id)),
                    expected,
                    "{config:?}, node {node_id}, parent effect: {has_parent_effect}"
                );
            }
            scene.render(&mut surface).unwrap();

            scene.set_effect(parent, 11, &[1, 2, 3, 4], config).unwrap();
            scene.render(&mut surface).unwrap();
            let mut effect_ids = surface.resource().effects.clone();
            effect_ids.sort_unstable();
            assert_eq!(effect_ids, [9, 10, 11, 11, 11]);
            scene.unload_effect(9);
            scene.render(&mut surface).unwrap();
            let mut effect_ids = surface.resource().effects.clone();
            effect_ids.sort_unstable();
            assert_eq!(effect_ids, [10, 11, 11, 11, 11]);
        }
    }
}

#[test]
fn replacements_preserve_topology_and_refresh_or_remove_only_the_nodes_effects() {
    let mut scene = scene();
    let root = queue_shape(&mut scene, false);
    let parent = queue_shape(&mut scene, true);
    let first = shape(&mut scene, [(0.0, 0.0), (4.0, 4.0)], parent);
    let second = shape(&mut scene, [(4.0, 4.0), (8.0, 8.0)], parent);
    let sibling = queue_shape(&mut scene, false);
    scene
        .set_content_effect(first, 8, &[1, 2, 3, 4], ContentEffectConfig::default())
        .unwrap();
    scene
        .set_backdrop_effect(parent, 7, &[1, 2, 3, 4], BackdropEffectConfig::default())
        .unwrap();
    let old_effect = *scene.state.shape_mask_effect(parent).unwrap();
    let old_capture = scene.state.backdrop_effects[&parent].capture_region;
    let replacement = Shape::rect([(4.0, 5.0), (12.0, 13.0)]);
    scene
        .replace_with_shape(parent, &replacement, None, ShapeDrawCommandOptions::new())
        .unwrap();
    scene
        .replace_with_cached_shape(first, 1, ShapeDrawCommandOptions::new())
        .unwrap();
    scene
        .replace_with_cached_shape(first, 1, ShapeDrawCommandOptions::new())
        .unwrap();
    assert_eq!(scene.state.draw_tree.children(root), &[parent, sibling]);
    assert_eq!(scene.state.draw_tree.children(parent), &[first, second]);
    assert_eq!(
        scene.state.draw_tree.parent_index_unchecked(first),
        Some(parent)
    );
    let effect = scene.state.shape_mask_effect(parent).unwrap();
    assert_ne!(effect.bounds, old_effect.bounds);
    assert_eq!(effect.parameters.hash, old_effect.parameters.hash);
    assert_ne!(
        scene.state.backdrop_effects[&parent].capture_region,
        old_capture
    );
    let mut surface = surface();
    scene.render(&mut surface).unwrap();
    assert_eq!(surface.resource().shape_masks, 1);
    assert!(surface.resource().effects.contains(&8));

    clip(&mut scene, parent, true);
    assert!(!scene.backend.registered_shapes.contains(&parent));
    assert!(!scene.state.shape_mask_effects.contains_key(&parent));
    assert!(!scene.state.backdrop_effects.contains_key(&parent));
    assert!(!scene.state.content_effects.contains_key(&parent));
    assert!(scene.state.content_effects.contains_key(&first));
    assert_eq!(scene.state.effect_parameters.bytes.len(), 16);
    scene
        .replace_with_cached_shape(parent, 1, ShapeDrawCommandOptions::new())
        .unwrap();
    assert_eq!(scene.state.draw_tree.children(parent), &[first, second]);
    scene.render(&mut surface).unwrap();
    assert_eq!(surface.resource().shape_masks, 0);
    assert!(surface.resource().effects.contains(&8));
    assert_eq!(scene.state.effect_parameters.bytes.len(), 4);
}

#[test]
fn replacement_errors_preserve_the_failing_node_and_prior_replacements() {
    let mut scene = scene();
    queue_shape(&mut scene, false);
    let first = queue_shape(&mut scene, false);
    let second = queue_shape(&mut scene, true);
    let last = queue_shape(&mut scene, false);
    scene.load_shape(Shape::rect([(2.0, 3.0), (6.0, 7.0)]), 2, Some(2));
    let old_effect = *scene.state.shape_mask_effect(second).unwrap();
    scene.backend.registration_failure = Some(second);
    scene
        .replace_with_cached_shape(first, 2, ShapeDrawCommandOptions::new())
        .unwrap();
    assert!(matches!(
        scene.replace_with_cached_shape(second, 2, ShapeDrawCommandOptions::new()),
        Err(DrawCommandError::Backend(TestBackendError))
    ));
    for (node_id, expected) in [(first, Some(2)), (second, Some(1)), (last, Some(1))] {
        assert_eq!(
            scene.state.shape(node_id).unwrap().cached_shape.geometry_id,
            expected
        );
    }
    let dirty = scene.dirty_bounds;
    for (node_id, cache_key) in [(usize::MAX, 1), (first, 999)] {
        assert!(scene
            .replace_with_cached_shape(node_id, cache_key, ShapeDrawCommandOptions::new())
            .is_err());
        assert_eq!(scene.dirty_bounds, dirty);
    }
    assert!(matches!(
        scene.replace_with_shape(second, Shape::rect([(0.0, 0.0), (f32::MAX, f32::MAX)]), None, ShapeDrawCommandOptions::new()),
        Err(DrawCommandError::Scene(SceneError::InvalidShapeMaskEffectBounds(id))) if id == second
    ));
    assert_eq!(
        scene.state.shape_mask_effect(second).unwrap().bounds,
        old_effect.bounds
    );
    assert_eq!(scene.dirty_bounds, dirty);
    scene.backend.registration_failure = None;
    let mut surface = surface();
    scene.render(&mut surface).unwrap();
    assert_eq!(surface.resource().shape_masks, 1);
    assert_eq!(scene.backend.registered_shapes.len(), 4);
}

#[test]
fn clip_damage_is_bounded_and_deferred_before_backdrop_expansion_and_planning() {
    let mut scene = scene();
    let mut surface = surface();
    scene
        .add_clipping_rect(
            [(0.0, 0.0), (32.0, 32.0)],
            None,
            None::<InstanceTransform>,
            false,
        )
        .unwrap();
    let parent = scene
        .add_clipping_rect(
            [(8.0, 8.0), (16.0, 16.0)],
            Some(0),
            None::<InstanceTransform>,
            true,
        )
        .unwrap();
    shape(&mut scene, [(20.0, 20.0), (24.0, 24.0)], parent);
    shape(&mut scene, [(28.0, 28.0), (30.0, 30.0)], 0);
    scene.render(&mut surface).unwrap();
    clip(&mut scene, parent, true);
    scene.render(&mut surface).unwrap();
    assert_eq!(scene.backend.root_scissor, rect((8, 8), (18, 18)));

    scene.load_effect(7, &["backdrop"]).unwrap();
    let backdrop = shape(&mut scene, [(2.0, 20.0), (4.0, 24.0)], 0);
    scene
        .set_backdrop_effect(
            backdrop,
            7,
            &[1, 2, 3, 4],
            BackdropEffectConfig::new().capture_area(BackdropCaptureArea::ScreenRect([
                (20.0, 20.0),
                (24.0, 24.0),
            ])),
        )
        .unwrap();
    scene.render(&mut surface).unwrap();
    for clips_children in [false, true, false] {
        clip(&mut scene, parent, clips_children);
    }
    assert_eq!(scene.dirty_bounds, rect((10, 10), (18, 18)));
    scene.backend.should_fail = true;
    assert!(scene.render(&mut surface).is_err());
    scene.backend.should_fail = false;
    scene.render(&mut surface).unwrap();
    assert_eq!(scene.backend.root_scissor, rect((2, 10), (24, 24)));
    scene.render(&mut surface).unwrap();
    assert_eq!(scene.backend.root_scissor, None);
}
