use super::{renderer, surface, TestBackend};
use crate::commands::RenderOperation;
use crate::core::effect::ShapeEffectBounds;
use crate::core::vertex::InstanceTransform;
use crate::core::{
    BackdropEffectConfig, Color, MathRect, Shape, ShapeDrawCommandOptions, ShapeEffectConfig,
    UnsignedPhysicalRect,
};
use crate::renderer::Renderer;
use crate::scene::effects::ShapeEffectInstance;
use crate::scene::SceneError;

fn add_effects_with_mixed_coordinate_ranges(renderer: &mut Renderer<TestBackend>) -> [usize; 2] {
    let nodes = [0, 1].map(|_| {
        renderer
            .add_shape(
                Shape::rect([(10.0, 10.0), (14.0, 14.0)]),
                None,
                None,
                ShapeDrawCommandOptions::new().color(Color::WHITE),
            )
            .unwrap()
    });
    for node in nodes {
        renderer
            .set_shape_effect(node, 7, &[1, 2, 3, 4], ShapeEffectConfig::new().outset(3.0))
            .unwrap();
    }
    // Put the failing attachment last so rejection follows a successful bounds update.
    let translated = *renderer.scene.shape_effects.keys().last().unwrap();
    renderer
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

fn assert_shape_effects_preserved(
    renderer: &Renderer<TestBackend>,
    previous: [(usize, ShapeEffectInstance); 2],
) {
    assert_eq!(renderer.scene.shape_effects.len(), previous.len());
    for (node, previous) in previous {
        let current = renderer.scene.shape_effect(node).unwrap();
        assert_eq!(current.effect_id, previous.effect_id);
        assert_eq!(current.parameters.range, previous.parameters.range);
        assert_eq!(current.parameters.hash, previous.parameters.hash);
        assert_eq!(current.config, previous.config);
        assert_eq!(current.bounds, previous.bounds);
    }
}

fn assert_shape_effect_bounds_match_settings(renderer: &Renderer<TestBackend>, nodes: [usize; 2]) {
    for node in nodes {
        let shape = renderer.scene.shape(node).unwrap();
        let effect = renderer.scene.shape_effect(node).unwrap();
        assert_eq!(
            Some(effect.bounds),
            ShapeEffectBounds::new(
                shape.cached_shape.tessellation.local_bounds,
                effect.config,
                shape.transform,
                renderer.scale_factor(),
                renderer.fringe_width(),
            )
        );
    }
}

#[test]
fn rejected_scale_change_preserves_all_effects_and_allows_a_later_valid_change() {
    let mut renderer = renderer();
    let mut surface = surface();
    let nodes = add_effects_with_mixed_coordinate_ranges(&mut renderer);
    renderer.render(&mut surface).unwrap();
    assert_eq!(surface.resource().shape_masks, 2);
    let previous = nodes.map(|node| (node, *renderer.scene.shape_effect(node).unwrap()));

    assert!(matches!(
        renderer.change_scale_factor(2.0),
        Err(SceneError::InvalidShapeEffectBounds(node)) if node == nodes[1]
    ));
    assert_eq!(renderer.scale_factor(), 1.0);
    assert_eq!(renderer.fringe_width(), 0.75);
    assert_eq!(renderer.size(), (32, 32));
    assert_eq!(renderer.backend.size, None);
    assert_eq!(renderer.dirty_bounds, None);
    assert_shape_effects_preserved(&renderer, previous);
    assert_shape_effect_bounds_match_settings(&renderer, nodes);

    renderer.change_scale_factor(0.5).unwrap();
    assert_eq!(renderer.scale_factor(), 0.5);
    assert_eq!(renderer.fringe_width(), 0.75);
    assert_eq!(renderer.backend.size, Some((32, 32)));
    assert_shape_effect_bounds_match_settings(&renderer, nodes);
    for (node, previous) in previous {
        let current = renderer.scene.shape_effect(node).unwrap();
        assert_ne!(current.bounds, previous.bounds);
        assert_eq!(current.config, previous.config);
        assert_eq!(current.parameters.range, previous.parameters.range);
        assert_eq!(current.parameters.hash, previous.parameters.hash);
    }
    renderer.render(&mut surface).unwrap();
    assert_eq!(surface.resource().shape_masks, 2);
}

#[test]
fn rejected_fringe_change_preserves_all_effects_and_allows_a_later_valid_change() {
    let mut renderer = renderer();
    let mut surface = surface();
    let nodes = add_effects_with_mixed_coordinate_ranges(&mut renderer);
    renderer.render(&mut surface).unwrap();
    assert_eq!(surface.resource().shape_masks, 2);
    let previous = nodes.map(|node| (node, *renderer.scene.shape_effect(node).unwrap()));

    assert!(matches!(
        renderer.set_fringe_width(700_000_000.0),
        Err(SceneError::InvalidShapeEffectBounds(node)) if node == nodes[1]
    ));
    assert_eq!(renderer.scale_factor(), 1.0);
    assert_eq!(renderer.fringe_width(), 0.75);
    assert_eq!(renderer.size(), (32, 32));
    assert_eq!(renderer.backend.size, None);
    assert_eq!(renderer.dirty_bounds, None);
    assert_shape_effects_preserved(&renderer, previous);
    assert_shape_effect_bounds_match_settings(&renderer, nodes);

    renderer.set_fringe_width(2.25).unwrap();
    assert_eq!(renderer.scale_factor(), 1.0);
    assert_eq!(renderer.fringe_width(), 2.25);
    assert_eq!(renderer.backend.size, Some((32, 32)));
    assert_shape_effect_bounds_match_settings(&renderer, nodes);
    for (node, previous) in previous {
        let current = renderer.scene.shape_effect(node).unwrap();
        assert_ne!(current.bounds, previous.bounds);
        assert_eq!(current.config, previous.config);
        assert_eq!(current.parameters.range, previous.parameters.range);
        assert_eq!(current.parameters.hash, previous.parameters.hash);
    }
    renderer.render(&mut surface).unwrap();
    assert_eq!(surface.resource().shape_masks, 2);
}

#[test]
fn retained_shape_effect_changes_clear_old_footprints() {
    let mut renderer = renderer();
    let mut surface = surface();
    let node = renderer
        .add_shape(
            Shape::rect([(10.0, 10.0), (14.0, 14.0)]),
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::WHITE),
        )
        .unwrap();
    renderer.load_effect(7, &["shape effect"]).unwrap();
    renderer.render(&mut surface).unwrap();
    let unrepresentable = ShapeEffectConfig::new().outset(f32::MAX);
    assert!(renderer
        .set_shape_effect(node, 7, &[1, 2, 3, 4], unrepresentable)
        .is_err());
    assert!(!renderer.scene.shape_effects.contains_key(&node));
    assert_eq!(renderer.dirty_bounds, None);
    renderer
        .set_shape_effect(node, 7, &[1, 2, 3, 4], ShapeEffectConfig::new().outset(3.0))
        .unwrap();
    let large = UnsignedPhysicalRect::new((6, 6).into(), (18, 18).into());
    renderer.render(&mut surface).unwrap();
    assert_eq!(renderer.backend.root_scissor, Some(large));

    renderer
        .set_shape_effect(node, 7, &[1, 2, 3, 4], ShapeEffectConfig::new().outset(1.0))
        .unwrap();
    renderer.render(&mut surface).unwrap();
    assert_eq!(renderer.backend.root_scissor, Some(large));
    let small = UnsignedPhysicalRect::new((8, 8).into(), (16, 16).into());
    renderer
        .set_shape_effect(node, 7, &[1, 2, 3, 4], ShapeEffectConfig::new().outset(1.0))
        .unwrap();
    renderer.render(&mut surface).unwrap();
    assert_eq!(renderer.backend.root_scissor, Some(small));

    let old_effect = *renderer.scene.shape_effect(node).unwrap();
    assert!(renderer
        .set_shape_effect(node, 7, &[3; 4], unrepresentable)
        .is_err());
    let effect = renderer.scene.shape_effect(node).unwrap();
    assert_eq!(effect.bounds, old_effect.bounds);
    assert_eq!(effect.config, old_effect.config);
    assert_eq!(effect.parameters.range, old_effect.parameters.range);
    assert_eq!(effect.parameters.hash, old_effect.parameters.hash);
    assert!(renderer
        .set_shape_effect(node, 7, &[3; 3], old_effect.config)
        .is_err());
    assert!(renderer
        .set_shape_effect(node, 7, &[3; 4], ShapeEffectConfig::new().outset(-1.0))
        .is_err());
    renderer.render(&mut surface).unwrap();
    assert_eq!(renderer.backend.root_scissor, None);

    renderer
        .set_shape_effect(
            node,
            7,
            &[1, 2, 3, 4],
            ShapeEffectConfig::new().outsets(5.0, 0.0, 0.0, 4.0),
        )
        .unwrap();
    renderer.backend.should_fail = true;
    assert!(renderer.render(&mut surface).is_err());
    renderer.remove_shape_effect(node);
    renderer.backend.should_fail = false;
    renderer.render(&mut surface).unwrap();
    assert_eq!(
        renderer.backend.root_scissor,
        Some(UnsignedPhysicalRect::new((4, 8).into(), (16, 19).into()))
    );
    renderer.remove_shape_effect(node);
    renderer.render(&mut surface).unwrap();
    assert_eq!(renderer.backend.root_scissor, None);

    for unload in [false, true] {
        renderer
            .set_shape_effect(node, 7, &[1, 2, 3, 4], ShapeEffectConfig::new().outset(3.0))
            .unwrap();
        renderer.render(&mut surface).unwrap();
        if unload {
            renderer.unload_effect(7);
        } else {
            renderer.load_effect(7, &["replacement shader"]).unwrap();
        }
        renderer.render(&mut surface).unwrap();
        assert_eq!(renderer.backend.root_scissor, Some(large));
        assert!(!renderer.scene.shape_effects.contains_key(&node));
    }
}

#[test]
fn cached_effect_bounds_follow_rasterization_and_keep_offscreen_coverage() {
    let mut renderer = renderer();
    let mut surface = surface();
    let node = renderer
        .add_shape(
            Shape::rect([(0.25, 0.5), (4.25, 3.5)]),
            None,
            None,
            ShapeDrawCommandOptions::new()
                .color(Color::WHITE)
                .transform(InstanceTransform::affine_2d(0.0, 2.0, -1.0, 0.0, 18.0, 4.0)),
        )
        .unwrap();
    let config = ShapeEffectConfig::new().outsets(1.0, 2.0, 3.0, 4.0);
    renderer
        .set_shape_effect(node, 7, &[1, 2, 3, 4], config)
        .unwrap();
    let initial = renderer.scene.shape_effect(node).unwrap().bounds;
    assert_eq!(initial.local_bounds, [(-2.0, -3.0), (9.0, 9.0)]);
    assert_eq!(
        initial.logical_screen_bounds,
        MathRect::new((9.0, 0.0).into(), (21.0, 22.0).into())
    );

    renderer.change_scale_factor(2.0).unwrap();
    let scaled = renderer.scene.shape_effect(node).unwrap().bounds;
    assert_eq!(scaled.local_bounds, [(-1.5, -2.0), (8.0, 8.0)]);
    renderer.set_fringe_width(2.25).unwrap();
    let padded = renderer.scene.shape_effect(node).unwrap().bounds;
    assert_eq!(padded.local_bounds, [(-2.5, -3.0), (9.0, 9.0)]);
    assert_eq!(
        padded.logical_screen_bounds,
        MathRect::new((9.0, -1.0).into(), (21.0, 22.0).into())
    );

    surface.resize((64, 64));
    renderer.render(&mut surface).unwrap();
    assert_eq!(renderer.scene.shape_effect(node).unwrap().bounds, padded);
    renderer
        .set_shape_effect(node, 7, &[1, 2, 3, 4], config.downsample(0.25))
        .unwrap();
    let downsampled = renderer.scene.shape_effect(node).unwrap().bounds;
    assert_eq!(
        downsampled.logical_screen_bounds,
        padded.logical_screen_bounds
    );
    assert_eq!(downsampled.texture_size, [6, 6]);
    renderer.render(&mut surface).unwrap();
    assert_eq!(
        renderer.backend.root_scissor,
        Some(UnsignedPhysicalRect::new((18, 0).into(), (42, 44).into()))
    );
    let plan = renderer.planner.plan(&renderer.scene, 4096);
    let mask = plan
        .instructions
        .iter()
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
    renderer.render(&mut surface).unwrap();
    assert_eq!(
        renderer.scene.shape_effect(node).unwrap().bounds,
        downsampled
    );
    renderer
        .set_shape_effect(node, 7, &[1, 2, 3, 4], config.downsample(0.25))
        .unwrap();
    renderer.render(&mut surface).unwrap();
    assert_eq!(
        renderer.backend.root_scissor,
        Some(UnsignedPhysicalRect::new((18, 0).into(), (32, 32).into()))
    );
    assert_eq!(
        renderer.scene.shape_effect(node).unwrap().bounds,
        downsampled
    );
    renderer.clear_draw_queue();
    renderer.render(&mut surface).unwrap();
    assert_eq!(
        renderer.backend.root_scissor,
        Some(UnsignedPhysicalRect::new((18, 0).into(), (32, 32).into()))
    );
}

#[test]
fn shape_effect_damage_uses_configured_bounds_with_other_effects() {
    for backdrop in [false, true] {
        let mut renderer = renderer();
        let mut surface = surface();
        let node = renderer
            .add_shape(
                Shape::rect([(10.0, 10.0), (14.0, 14.0)]),
                None,
                None,
                ShapeDrawCommandOptions::new().color(Color::WHITE),
            )
            .unwrap();
        if backdrop {
            renderer
                .set_shape_backdrop_effect(node, 7, &[1, 2, 3, 4], BackdropEffectConfig::default())
                .unwrap();
        } else {
            renderer.set_group_effect(node, 7, &[1, 2, 3, 4]).unwrap();
        }
        renderer.render(&mut surface).unwrap();
        let bounds = UnsignedPhysicalRect::new((6, 6).into(), (18, 18).into());
        renderer
            .set_shape_effect(node, 7, &[1, 2, 3, 4], ShapeEffectConfig::new().outset(3.0))
            .unwrap();
        renderer.render(&mut surface).unwrap();
        assert_eq!(renderer.backend.root_scissor, Some(bounds));
        renderer
            .set_shape_effect(node, 7, &[1, 2, 3, 4], ShapeEffectConfig::new().outset(1.0))
            .unwrap();
        renderer.render(&mut surface).unwrap();
        assert_eq!(renderer.backend.root_scissor, Some(bounds));
        let smaller_bounds = UnsignedPhysicalRect::new((8, 8).into(), (16, 16).into());
        renderer
            .set_shape_effect(node, 7, &[1, 2, 3, 4], ShapeEffectConfig::new().outset(1.0))
            .unwrap();
        renderer.render(&mut surface).unwrap();
        assert_eq!(renderer.backend.root_scissor, Some(smaller_bounds));
        renderer.load_effect(7, &["replacement shader"]).unwrap();
        renderer.render(&mut surface).unwrap();
        assert_eq!(renderer.backend.root_scissor, Some(smaller_bounds));
        renderer.set_group_effect(node, 8, &[1, 2, 3, 4]).unwrap();
        renderer
            .set_shape_effect(node, 7, &[1, 2, 3, 4], ShapeEffectConfig::new().outset(3.0))
            .unwrap();
        renderer.render(&mut surface).unwrap();
        renderer.clear_draw_queue();
        renderer.render(&mut surface).unwrap();
        assert_eq!(renderer.backend.root_scissor, Some(bounds));
    }
}

#[test]
fn removing_effect_subtrees_clears_shadow_pixels_and_allows_id_reuse() {
    let mut renderer = renderer();
    let mut surface = surface();
    let parent = renderer
        .add_clipping_rect(
            [(0.0, 0.0), (32.0, 32.0)],
            None,
            None::<InstanceTransform>,
            false,
        )
        .unwrap();
    let node = renderer
        .add_shape(
            Shape::rect([(10.0, 10.0), (14.0, 14.0)]),
            Some(parent),
            None,
            ShapeDrawCommandOptions::new().color(Color::WHITE),
        )
        .unwrap();
    renderer
        .set_shape_effect(node, 7, &[1, 2, 3, 4], ShapeEffectConfig::new().outset(3.0))
        .unwrap();
    renderer.render(&mut surface).unwrap();
    renderer.remove_subtrees([node, parent, node], |_| {});
    renderer.render(&mut surface).unwrap();
    assert_eq!(
        renderer.backend.root_scissor,
        Some(UnsignedPhysicalRect::new((6, 6).into(), (18, 18).into()))
    );
    renderer
        .add_shape(
            Shape::rect([(22.0, 22.0), (24.0, 24.0)]),
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::WHITE),
        )
        .unwrap();
    renderer.render(&mut surface).unwrap();
    assert_eq!(
        renderer.backend.root_scissor,
        Some(UnsignedPhysicalRect::new((21, 21).into(), (25, 25).into()))
    );
}
