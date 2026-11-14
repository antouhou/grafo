use super::{scene, surface, Scene, TestBackend};
use crate::core::{
    BlendMode, Color, ContentEffectConfig, PhysicalRect, Shape, ShapeDrawOptions,
    ShapeMaskEffectConfig, TransformInstance, UnsignedPhysicalRect,
};
use crate::scene_state::types::DrawTreeNode;

#[derive(Clone, Copy)]
enum EffectRemoval {
    Direct,
    Unload,
    Reload,
}

fn remove_effect(scene: &mut Scene<TestBackend>, node_id: usize, removal: EffectRemoval) {
    match removal {
        EffectRemoval::Direct => scene.remove_content_effect(node_id),
        EffectRemoval::Unload => scene.unload_effect(8),
        EffectRemoval::Reload => scene.load_effect(8, &["replacement"]).unwrap(),
    }
}

fn add_shape(
    scene: &mut Scene<TestBackend>,
    parent: usize,
    bounds: [(f32, f32); 2],
    blend_mode: BlendMode,
) -> usize {
    scene
        .add_shape(
            Shape::rect(bounds),
            Some(parent),
            None,
            ShapeDrawOptions::new()
                .color(Color::WHITE)
                .blend_mode(blend_mode)
                .clips_children(false),
        )
        .unwrap()
}

fn assert_blend_mode(scene: &Scene<TestBackend>, node_id: usize, expected: BlendMode) {
    let DrawTreeNode::CachedShape(shape) = scene.state.draw_tree.get(node_id).unwrap() else {
        panic!("expected a shape node");
    };
    assert_eq!(shape.blend_mode, expected);
}

fn rect(min: (u32, u32), max: (u32, u32)) -> Option<UnsignedPhysicalRect> {
    Some(UnsignedPhysicalRect::new(min.into(), max.into()))
}

#[test]
fn content_effect_removal_preserves_blending_and_surviving_parameters() {
    for mode in [BlendMode::SourceOver, BlendMode::Multiply] {
        for removal in [
            EffectRemoval::Direct,
            EffectRemoval::Unload,
            EffectRemoval::Reload,
        ] {
            let mut scene = scene();
            let mut surface = surface();
            let root = scene.add_group(None, Default::default()).unwrap();
            let node = add_shape(&mut scene, root, [(10.0, 10.0), (18.0, 18.0)], mode);
            let survivor = add_shape(
                &mut scene,
                root,
                [(20.0, 20.0), (28.0, 28.0)],
                BlendMode::SourceOver,
            );
            scene
                .set_content_effect(
                    node,
                    8,
                    &[1, 2, 3, 4],
                    ContentEffectConfig::new().outset(3.0),
                )
                .unwrap();
            scene
                .set_content_effect(survivor, 9, &[1, 2, 3, 4], ContentEffectConfig::default())
                .unwrap();
            scene.render(&mut surface).unwrap();
            let previous_bounds = scene.state.content_groups[&node].output_bounds;

            for (parameters, config) in [
                (&[0; 3][..], ContentEffectConfig::default()),
                (&[0; 4][..], ContentEffectConfig::new().outset(-1.0)),
            ] {
                assert!(scene
                    .set_content_effect(node, 8, parameters, config)
                    .is_err());
                let group = scene.state.content_groups[&node];
                assert_eq!(group.output_bounds, previous_bounds);
                assert_eq!(
                    scene
                        .state
                        .effect_parameters
                        .get(group.effect.unwrap().parameters)
                        .bytes,
                    &[1, 2, 3, 4]
                );
                assert!(scene.dirty_bounds.is_none());
            }

            remove_effect(&mut scene, node, removal);

            if mode == BlendMode::SourceOver {
                assert!(!scene.state.content_groups.contains_key(&node));
            } else {
                let group = scene.state.content_groups[&node];
                assert!(group.effect.is_none());
                assert_eq!(group.input_bounds, rect((10, 10), (18, 18)));
                assert_eq!(
                    group.output_bounds,
                    Some(PhysicalRect::new((10, 10).into(), (18, 18).into()))
                );
            }
            assert_blend_mode(&scene, node, mode);
            scene.render(&mut surface).unwrap();
            let survivor_effect = scene.state.content_groups[&survivor].effect.unwrap();
            assert_eq!(scene.state.effect_parameters.bytes.len(), 4);
            assert_eq!(
                scene
                    .state
                    .effect_parameters
                    .get(survivor_effect.parameters)
                    .bytes,
                &[1, 2, 3, 4]
            );
            scene.remove_content_effect(node);
            assert!(scene.dirty_bounds.is_none());
            assert_eq!(
                scene.state.content_groups.contains_key(&node),
                mode != BlendMode::SourceOver
            );
        }
    }
}

#[test]
fn shared_effect_unload_and_replacement_refresh_retained_groups() {
    let mut scene = scene();
    let mut surface = surface();
    let root = scene.add_group(None, Default::default()).unwrap();
    scene
        .load_shape(Shape::rect([(10.0, 10.0), (18.0, 18.0)]), 1, None)
        .unwrap();
    let node = scene
        .add_cached_shape(
            1,
            Some(root),
            ShapeDrawOptions::new().blend_mode(BlendMode::Multiply),
        )
        .unwrap();
    scene
        .set_shape_mask_effect(
            node,
            8,
            &[1, 2, 3, 4],
            ShapeMaskEffectConfig::new().outset(2.0),
        )
        .unwrap();
    scene
        .set_content_effect(
            node,
            8,
            &[1, 2, 3, 4],
            ContentEffectConfig::new().outset(2.0),
        )
        .unwrap();
    scene.render(&mut surface).unwrap();

    scene.unload_effect(8);

    assert!(!scene.state.shape_mask_effects.contains_key(&node));
    let group = scene.state.content_groups[&node];
    assert!(group.effect.is_none());
    assert_eq!(group.input_bounds, rect((10, 10), (18, 18)));
    assert_eq!(
        group.output_bounds,
        Some(PhysicalRect::new((10, 10).into(), (18, 18).into()))
    );
    assert_blend_mode(&scene, node, BlendMode::Multiply);
    scene.render(&mut surface).unwrap();
    assert!(scene.state.effect_parameters.bytes.is_empty());

    scene
        .replace_with_cached_shape(node, 1, ShapeDrawOptions::new())
        .unwrap();
    assert_blend_mode(&scene, node, BlendMode::SourceOver);
    assert!(!scene.state.content_groups.contains_key(&node));

    scene
        .set_content_effect(node, 9, &[1, 2, 3, 4], ContentEffectConfig::default())
        .unwrap();
    scene
        .replace_with_cached_shape(
            node,
            1,
            ShapeDrawOptions::new().blend_mode(BlendMode::Multiply),
        )
        .unwrap();
    assert_blend_mode(&scene, node, BlendMode::Multiply);
    assert_eq!(
        scene.state.content_groups[&node].effect.unwrap().effect_id,
        9
    );
    scene.remove_content_effect(node);
    assert!(scene.state.content_groups[&node].effect.is_none());
    scene
        .replace_with_clipping_rect(
            node,
            [(10.0, 10.0), (18.0, 18.0)],
            None::<TransformInstance>,
            false,
        )
        .unwrap();
    assert!(!scene.state.content_groups.contains_key(&node));
}

#[test]
fn removing_effects_restores_mask_and_overflow_damage() {
    for removal in [
        EffectRemoval::Direct,
        EffectRemoval::Unload,
        EffectRemoval::Reload,
    ] {
        let mut scene = scene();
        let mut surface = surface();
        let root = scene.add_group(None, Default::default()).unwrap();
        let node = add_shape(
            &mut scene,
            root,
            [(4.0, 4.0), (8.0, 8.0)],
            BlendMode::Multiply,
        );
        scene
            .set_shape_mask_effect(
                node,
                7,
                &[1, 2, 3, 4],
                ShapeMaskEffectConfig::new().outset(5.0),
            )
            .unwrap();
        scene
            .set_content_effect(
                node,
                8,
                &[1, 2, 3, 4],
                ContentEffectConfig::new().outsets(0.0, 0.0, 400.0, 0.0),
            )
            .unwrap();
        assert!(scene.state.content_groups[&node].output_bounds.is_none());
        scene.render(&mut surface).unwrap();

        remove_effect(&mut scene, node, removal);
        scene.render(&mut surface).unwrap();

        assert_eq!(scene.backend.root_scissor, rect((0, 0), (14, 14)));
        assert!(scene.state.content_groups[&node].effect.is_none());
        let child = add_shape(
            &mut scene,
            node,
            [(20.0, 20.0), (24.0, 24.0)],
            BlendMode::SourceOver,
        );
        scene
            .set_content_effect(
                child,
                9,
                &[1, 2, 3, 4],
                ContentEffectConfig::new().outset(2.0),
            )
            .unwrap();
        scene.render(&mut surface).unwrap();
        scene
            .set_content_effect(node, 8, &[1, 2, 3, 4], ContentEffectConfig::default())
            .unwrap();
        scene.render(&mut surface).unwrap();
        assert_eq!(scene.backend.root_scissor, rect((0, 0), (26, 26)));

        remove_effect(&mut scene, node, removal);
        scene.render(&mut surface).unwrap();

        assert_eq!(scene.backend.root_scissor, rect((0, 0), (26, 26)));
        assert!(surface.resource().draws.contains(&child));
    }
}
