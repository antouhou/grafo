use super::{queue_shape, scene, surface, DrawCommandError, Scene, TestBackend, TestBackendError};
use crate::core::vertex::InstanceTransform;
use crate::core::{
    BackdropCaptureArea, BackdropEffectConfig, ContentEffectConfig, Shape, ShapeDrawCommandOptions,
    UnsignedPhysicalRect,
};
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
