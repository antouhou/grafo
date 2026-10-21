use super::super::{scene, surface, TestBackend};
use crate::commands::RenderOperation;
use crate::core::effect::{ContentEffectConfig, ShapeMaskEffectBounds};
use crate::core::vertex::InstanceTransform;
use crate::core::{
    BackdropEffectConfig, Color, MathRect, Shape, ShapeDrawCommandOptions, ShapeMaskEffectConfig,
    UnsignedPhysicalRect,
};
use crate::scene::Scene;
use crate::scene_state::effects::ShapeMaskEffectInstance;
use crate::scene_state::SceneError;

fn add_effects_with_mixed_coordinate_ranges(scene: &mut Scene<TestBackend>) -> [usize; 2] {
    let nodes = [0, 1].map(|_| {
        scene
            .add_shape(
                Shape::rect([(10.0, 10.0), (14.0, 14.0)]),
                None,
                None,
                ShapeDrawCommandOptions::new().color(Color::WHITE),
            )
            .unwrap()
    });
    for node in nodes {
        scene
            .set_shape_mask_effect(
                node,
                7,
                &[1, 2, 3, 4],
                ShapeMaskEffectConfig::new().outset(3.0),
            )
            .unwrap();
    }
    // Put the failing attachment last so rejection follows a successful bounds update.
    let translated = *scene.state.shape_mask_effects.keys().last().unwrap();
    scene
        .replace_with_shape(
            translated,
            Shape::rect([(1_500_000_000.0, 10.0), (1_500_000_128.0, 14.0)]),
            None,
            ShapeDrawCommandOptions::new()
                .color(Color::WHITE)
                .transform(InstanceTransform::translation(-1_500_000_000.0, 0.0)),
        )
        .unwrap();
    let nearby = nodes.into_iter().find(|&node| node != translated).unwrap();
    [nearby, translated]
}

fn assert_shape_mask_effects_preserved(
    scene: &Scene<TestBackend>,
    previous: [(usize, ShapeMaskEffectInstance); 2],
) {
    assert_eq!(scene.state.shape_mask_effects.len(), previous.len());
    for (node, previous) in previous {
        let current = scene.state.shape_mask_effect(node).unwrap();
        assert_eq!(current.effect_id, previous.effect_id);
        assert_eq!(current.parameters.range, previous.parameters.range);
        assert_eq!(current.parameters.hash, previous.parameters.hash);
        assert_eq!(current.config, previous.config);
        assert_eq!(current.bounds, previous.bounds);
    }
}

fn assert_shape_mask_effect_bounds_match_settings(scene: &Scene<TestBackend>, nodes: [usize; 2]) {
    for node in nodes {
        let shape = scene.state.shape(node).unwrap();
        let effect = scene.state.shape_mask_effect(node).unwrap();
        assert_eq!(
            Some(effect.bounds),
            ShapeMaskEffectBounds::new(
                shape.cached_shape.tessellation.local_bounds,
                effect.config,
                shape.transform,
                scene.scale_factor(),
                scene.fringe_width(),
            )
        );
    }
}

#[test]
fn rejected_scale_change_preserves_all_effects_and_allows_a_later_valid_change() {
    let mut scene = scene();
    let mut surface = surface();
    let nodes = add_effects_with_mixed_coordinate_ranges(&mut scene);
    scene.render(&mut surface).unwrap();
    assert_eq!(surface.resource().shape_masks, 2);
    let previous = nodes.map(|node| (node, *scene.state.shape_mask_effect(node).unwrap()));

    assert!(matches!(
        scene.change_scale_factor(2.0),
        Err(SceneError::InvalidShapeMaskEffectBounds(node)) if node == nodes[1]
    ));
    assert_eq!(scene.scale_factor(), 1.0);
    assert_eq!(scene.fringe_width(), 0.75);
    assert_eq!(scene.size(), (32, 32));
    assert_eq!(scene.backend.size, None);
    assert_eq!(scene.dirty_bounds, None);
    assert_shape_mask_effects_preserved(&scene, previous);
    assert_shape_mask_effect_bounds_match_settings(&scene, nodes);

    scene.change_scale_factor(0.5).unwrap();
    assert_eq!(scene.scale_factor(), 0.5);
    assert_eq!(scene.fringe_width(), 0.75);
    assert_eq!(scene.backend.size, Some((32, 32)));
    assert_shape_mask_effect_bounds_match_settings(&scene, nodes);
    for (node, previous) in previous {
        let current = scene.state.shape_mask_effect(node).unwrap();
        assert_ne!(current.bounds, previous.bounds);
        assert_eq!(current.config, previous.config);
        assert_eq!(current.parameters.range, previous.parameters.range);
        assert_eq!(current.parameters.hash, previous.parameters.hash);
    }
    scene.render(&mut surface).unwrap();
    assert_eq!(surface.resource().shape_masks, 2);
}

#[test]
fn rejected_fringe_change_preserves_all_effects_and_allows_a_later_valid_change() {
    let mut scene = scene();
    let mut surface = surface();
    let nodes = add_effects_with_mixed_coordinate_ranges(&mut scene);
    scene.render(&mut surface).unwrap();
    assert_eq!(surface.resource().shape_masks, 2);
    let previous = nodes.map(|node| (node, *scene.state.shape_mask_effect(node).unwrap()));

    assert!(matches!(
        scene.set_fringe_width(700_000_000.0),
        Err(SceneError::InvalidShapeMaskEffectBounds(node)) if node == nodes[1]
    ));
    assert_eq!(scene.scale_factor(), 1.0);
    assert_eq!(scene.fringe_width(), 0.75);
    assert_eq!(scene.size(), (32, 32));
    assert_eq!(scene.backend.size, None);
    assert_eq!(scene.dirty_bounds, None);
    assert_shape_mask_effects_preserved(&scene, previous);
    assert_shape_mask_effect_bounds_match_settings(&scene, nodes);

    scene.set_fringe_width(2.25).unwrap();
    assert_eq!(scene.scale_factor(), 1.0);
    assert_eq!(scene.fringe_width(), 2.25);
    assert_eq!(scene.backend.size, Some((32, 32)));
    assert_shape_mask_effect_bounds_match_settings(&scene, nodes);
    for (node, previous) in previous {
        let current = scene.state.shape_mask_effect(node).unwrap();
        assert_ne!(current.bounds, previous.bounds);
        assert_eq!(current.config, previous.config);
        assert_eq!(current.parameters.range, previous.parameters.range);
        assert_eq!(current.parameters.hash, previous.parameters.hash);
    }
    scene.render(&mut surface).unwrap();
    assert_eq!(surface.resource().shape_masks, 2);
}

#[test]
fn retained_shape_mask_effect_changes_clear_old_footprints() {
    let mut scene = scene();
    let mut surface = surface();
    let node = scene
        .add_shape(
            Shape::rect([(10.0, 10.0), (14.0, 14.0)]),
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::WHITE),
        )
        .unwrap();
    scene.load_effect(7, &["shape mask effect"]).unwrap();
    scene.render(&mut surface).unwrap();
    let unrepresentable = ShapeMaskEffectConfig::new().outset(f32::MAX);
    assert!(scene
        .set_shape_mask_effect(node, 7, &[1, 2, 3, 4], unrepresentable)
        .is_err());
    assert!(!scene.state.shape_mask_effects.contains_key(&node));
    assert_eq!(scene.dirty_bounds, None);
    scene
        .set_shape_mask_effect(
            node,
            7,
            &[1, 2, 3, 4],
            ShapeMaskEffectConfig::new().outset(3.0),
        )
        .unwrap();
    let large = UnsignedPhysicalRect::new((6, 6).into(), (18, 18).into());
    scene.render(&mut surface).unwrap();
    assert_eq!(scene.backend.root_scissor, Some(large));

    scene
        .set_shape_mask_effect(
            node,
            7,
            &[1, 2, 3, 4],
            ShapeMaskEffectConfig::new().outset(1.0),
        )
        .unwrap();
    scene.render(&mut surface).unwrap();
    assert_eq!(scene.backend.root_scissor, Some(large));
    let small = UnsignedPhysicalRect::new((8, 8).into(), (16, 16).into());
    scene
        .set_shape_mask_effect(
            node,
            7,
            &[1, 2, 3, 4],
            ShapeMaskEffectConfig::new().outset(1.0),
        )
        .unwrap();
    scene.render(&mut surface).unwrap();
    assert_eq!(scene.backend.root_scissor, Some(small));

    let old_effect = *scene.state.shape_mask_effect(node).unwrap();
    assert!(scene
        .set_shape_mask_effect(node, 7, &[3; 4], unrepresentable)
        .is_err());
    let effect = scene.state.shape_mask_effect(node).unwrap();
    assert_eq!(effect.bounds, old_effect.bounds);
    assert_eq!(effect.config, old_effect.config);
    assert_eq!(effect.parameters.range, old_effect.parameters.range);
    assert_eq!(effect.parameters.hash, old_effect.parameters.hash);
    assert!(scene
        .set_shape_mask_effect(node, 7, &[3; 3], old_effect.config)
        .is_err());
    assert!(scene
        .set_shape_mask_effect(node, 7, &[3; 4], ShapeMaskEffectConfig::new().outset(-1.0))
        .is_err());
    scene.render(&mut surface).unwrap();
    assert_eq!(scene.backend.root_scissor, None);

    scene
        .set_shape_mask_effect(
            node,
            7,
            &[1, 2, 3, 4],
            ShapeMaskEffectConfig::new().outsets(5.0, 0.0, 0.0, 4.0),
        )
        .unwrap();
    scene.backend.should_fail = true;
    assert!(scene.render(&mut surface).is_err());
    scene.remove_shape_mask_effect(node);
    scene.backend.should_fail = false;
    scene.render(&mut surface).unwrap();
    assert_eq!(
        scene.backend.root_scissor,
        Some(UnsignedPhysicalRect::new((4, 8).into(), (16, 19).into()))
    );
    scene.remove_shape_mask_effect(node);
    scene.render(&mut surface).unwrap();
    assert_eq!(scene.backend.root_scissor, None);

    for unload in [false, true] {
        scene
            .set_shape_mask_effect(
                node,
                7,
                &[1, 2, 3, 4],
                ShapeMaskEffectConfig::new().outset(3.0),
            )
            .unwrap();
        scene.render(&mut surface).unwrap();
        if unload {
            scene.unload_effect(7);
        } else {
            scene.load_effect(7, &["replacement shader"]).unwrap();
        }
        scene.render(&mut surface).unwrap();
        assert_eq!(scene.backend.root_scissor, Some(large));
        assert!(!scene.state.shape_mask_effects.contains_key(&node));
    }
}

#[test]
fn cached_effect_bounds_follow_rasterization_and_keep_offscreen_coverage() {
    let mut scene = scene();
    let mut surface = surface();
    let node = scene
        .add_shape(
            Shape::rect([(0.25, 0.5), (4.25, 3.5)]),
            None,
            None,
            ShapeDrawCommandOptions::new()
                .color(Color::WHITE)
                .transform(InstanceTransform::affine_2d(0.0, 2.0, -1.0, 0.0, 18.0, 4.0)),
        )
        .unwrap();
    let config = ShapeMaskEffectConfig::new().outsets(1.0, 2.0, 3.0, 4.0);
    scene
        .set_shape_mask_effect(node, 7, &[1, 2, 3, 4], config)
        .unwrap();
    let initial = scene.state.shape_mask_effect(node).unwrap().bounds;
    assert_eq!(initial.local_bounds, [(-2.0, -3.0), (9.0, 9.0)]);
    assert_eq!(
        initial.logical_screen_bounds,
        MathRect::new((9.0, 0.0).into(), (21.0, 22.0).into())
    );

    scene.change_scale_factor(2.0).unwrap();
    let scaled = scene.state.shape_mask_effect(node).unwrap().bounds;
    assert_eq!(scaled.local_bounds, [(-1.5, -2.0), (8.0, 8.0)]);
    scene.set_fringe_width(2.25).unwrap();
    let padded = scene.state.shape_mask_effect(node).unwrap().bounds;
    assert_eq!(padded.local_bounds, [(-2.5, -3.0), (9.0, 9.0)]);
    assert_eq!(
        padded.logical_screen_bounds,
        MathRect::new((9.0, -1.0).into(), (21.0, 22.0).into())
    );

    surface.resize((64, 64));
    scene.render(&mut surface).unwrap();
    assert_eq!(scene.state.shape_mask_effect(node).unwrap().bounds, padded);
    scene
        .set_shape_mask_effect(node, 7, &[1, 2, 3, 4], config.downsample(0.25))
        .unwrap();
    let downsampled = scene.state.shape_mask_effect(node).unwrap().bounds;
    assert_eq!(
        downsampled.logical_screen_bounds,
        padded.logical_screen_bounds
    );
    assert_eq!(downsampled.texture_size, [6, 6]);
    scene.render(&mut surface).unwrap();
    assert_eq!(
        scene.backend.root_scissor,
        Some(UnsignedPhysicalRect::new((18, 0).into(), (42, 44).into()))
    );
    let mask = scene
        .planner
        .plan(&scene.state, 4096)
        .find_map(|command| match command.operation {
            RenderOperation::DrawShapeMask(mask) => Some(mask),
            _ => None,
        })
        .unwrap();
    assert_eq!(mask.local_bounds, downsampled.local_bounds);
    assert_eq!(
        mask.local_physical_origin,
        downsampled.local_physical_origin
    );
    assert_eq!(mask.scale_factor, 2.0);
    assert_eq!(mask.fringe_width, 2.25);

    surface.resize((32, 32));
    scene.render(&mut surface).unwrap();
    assert_eq!(
        scene.state.shape_mask_effect(node).unwrap().bounds,
        downsampled
    );
    scene
        .set_shape_mask_effect(node, 7, &[1, 2, 3, 4], config.downsample(0.25))
        .unwrap();
    scene.render(&mut surface).unwrap();
    assert_eq!(
        scene.backend.root_scissor,
        Some(UnsignedPhysicalRect::new((18, 0).into(), (32, 32).into()))
    );
    assert_eq!(
        scene.state.shape_mask_effect(node).unwrap().bounds,
        downsampled
    );
    scene.clear();
    scene.render(&mut surface).unwrap();
    assert_eq!(
        scene.backend.root_scissor,
        Some(UnsignedPhysicalRect::new((18, 0).into(), (32, 32).into()))
    );
}

#[test]
fn shape_mask_effect_damage_uses_configured_bounds_with_other_effects() {
    for backdrop in [false, true] {
        let mut scene = scene();
        let mut surface = surface();
        let node = scene
            .add_shape(
                Shape::rect([(10.0, 10.0), (14.0, 14.0)]),
                None,
                None,
                ShapeDrawCommandOptions::new().color(Color::WHITE),
            )
            .unwrap();
        if backdrop {
            scene
                .set_backdrop_effect(node, 7, &[1, 2, 3, 4], BackdropEffectConfig::default())
                .unwrap();
        } else {
            scene
                .set_content_effect(node, 7, &[1, 2, 3, 4], ContentEffectConfig::default())
                .unwrap();
        }
        scene.render(&mut surface).unwrap();
        let bounds = UnsignedPhysicalRect::new((6, 6).into(), (18, 18).into());
        scene
            .set_shape_mask_effect(
                node,
                7,
                &[1, 2, 3, 4],
                ShapeMaskEffectConfig::new().outset(3.0),
            )
            .unwrap();
        scene.render(&mut surface).unwrap();
        assert_eq!(scene.backend.root_scissor, Some(bounds));
        scene
            .set_shape_mask_effect(
                node,
                7,
                &[1, 2, 3, 4],
                ShapeMaskEffectConfig::new().outset(1.0),
            )
            .unwrap();
        scene.render(&mut surface).unwrap();
        assert_eq!(scene.backend.root_scissor, Some(bounds));
        let smaller_bounds = UnsignedPhysicalRect::new((8, 8).into(), (16, 16).into());
        scene
            .set_shape_mask_effect(
                node,
                7,
                &[1, 2, 3, 4],
                ShapeMaskEffectConfig::new().outset(1.0),
            )
            .unwrap();
        scene.render(&mut surface).unwrap();
        assert_eq!(scene.backend.root_scissor, Some(smaller_bounds));
        scene.load_effect(7, &["replacement shader"]).unwrap();
        scene.render(&mut surface).unwrap();
        assert_eq!(scene.backend.root_scissor, Some(smaller_bounds));
        scene
            .set_content_effect(node, 8, &[1, 2, 3, 4], ContentEffectConfig::default())
            .unwrap();
        scene
            .set_shape_mask_effect(
                node,
                7,
                &[1, 2, 3, 4],
                ShapeMaskEffectConfig::new().outset(3.0),
            )
            .unwrap();
        scene.render(&mut surface).unwrap();
        scene.clear();
        scene.render(&mut surface).unwrap();
        assert_eq!(scene.backend.root_scissor, Some(bounds));
    }
}

#[test]
fn removing_effect_subtrees_clears_shadow_pixels_and_allows_id_reuse() {
    let mut scene = scene();
    let mut surface = surface();
    let parent = scene
        .add_clipping_rect(
            [(0.0, 0.0), (32.0, 32.0)],
            None,
            None::<InstanceTransform>,
            false,
        )
        .unwrap();
    let node = scene
        .add_shape(
            Shape::rect([(10.0, 10.0), (14.0, 14.0)]),
            Some(parent),
            None,
            ShapeDrawCommandOptions::new().color(Color::WHITE),
        )
        .unwrap();
    scene
        .set_shape_mask_effect(
            node,
            7,
            &[1, 2, 3, 4],
            ShapeMaskEffectConfig::new().outset(3.0),
        )
        .unwrap();
    scene.render(&mut surface).unwrap();
    scene.remove_subtrees([node, parent, node], |_| {});
    scene.render(&mut surface).unwrap();
    assert_eq!(
        scene.backend.root_scissor,
        Some(UnsignedPhysicalRect::new((6, 6).into(), (18, 18).into()))
    );
    scene
        .add_shape(
            Shape::rect([(22.0, 22.0), (24.0, 24.0)]),
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::WHITE),
        )
        .unwrap();
    scene.render(&mut surface).unwrap();
    assert_eq!(
        scene.backend.root_scissor,
        Some(UnsignedPhysicalRect::new((21, 21).into(), (25, 25).into()))
    );
}
