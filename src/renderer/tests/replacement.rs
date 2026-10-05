use super::{
    queue_shape, renderer, surface, DrawCommandError, Renderer, TestBackend, TestBackendError,
};
use crate::core::vertex::InstanceTransform;
use crate::core::{
    BackdropCaptureArea, BackdropEffectConfig, Shape, ShapeDrawCommandOptions, UnsignedPhysicalRect,
};
use crate::renderer::DrawCommandReplacement;
use crate::scene::SceneError;

fn cached(cache_key: u64) -> DrawCommandReplacement<'static> {
    DrawCommandReplacement::CachedShape {
        cache_key,
        options: ShapeDrawCommandOptions::new(),
    }
}

fn shape(renderer: &mut Renderer<TestBackend>, bounds: [(f32, f32); 2], parent: usize) -> usize {
    renderer
        .add_shape(
            Shape::rect(bounds),
            Some(parent),
            None,
            ShapeDrawCommandOptions::new(),
        )
        .unwrap()
}

fn clip(renderer: &mut Renderer<TestBackend>, node_id: usize, clips_children: bool) {
    renderer
        .replace_clipping_rect(
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
    let mut renderer = renderer();
    let root = queue_shape(&mut renderer, false);
    let parent = queue_shape(&mut renderer, true);
    let first = shape(&mut renderer, [(0.0, 0.0), (4.0, 4.0)], parent);
    let second = shape(&mut renderer, [(4.0, 4.0), (8.0, 8.0)], parent);
    let sibling = queue_shape(&mut renderer, false);
    renderer.set_group_effect(first, 8, &[1, 2, 3, 4]).unwrap();
    renderer
        .set_shape_backdrop_effect(parent, 7, &[1, 2, 3, 4], BackdropEffectConfig::default())
        .unwrap();
    let old_effect = *renderer.scene.shape_effect(parent).unwrap();
    let old_capture = renderer.scene.backdrop_effects[&parent].capture_region;
    let replacement = Shape::rect([(4.0, 5.0), (12.0, 13.0)]);
    renderer
        .replace_draw_commands([
            (
                parent,
                DrawCommandReplacement::Shape {
                    shape: &replacement,
                    geometry_id: None,
                    options: ShapeDrawCommandOptions::new(),
                },
            ),
            (first, cached(1)),
            (first, cached(1)),
        ])
        .unwrap();
    assert_eq!(renderer.scene.draw_tree.children(root), &[parent, sibling]);
    assert_eq!(renderer.scene.draw_tree.children(parent), &[first, second]);
    assert_eq!(
        renderer.scene.draw_tree.parent_index_unchecked(first),
        Some(parent)
    );
    let effect = renderer.scene.shape_effect(parent).unwrap();
    assert_ne!(effect.bounds, old_effect.bounds);
    assert_eq!(effect.parameters.hash, old_effect.parameters.hash);
    assert_ne!(
        renderer.scene.backdrop_effects[&parent].capture_region,
        old_capture
    );
    let mut surface = surface();
    renderer.render(&mut surface).unwrap();
    assert_eq!(surface.resource().shape_masks, 1);
    assert!(surface.resource().effects.contains(&8));

    clip(&mut renderer, parent, true);
    assert!(!renderer.backend.registered_shapes.contains(&parent));
    assert!(!renderer.scene.shape_effects.contains_key(&parent));
    assert!(!renderer.scene.backdrop_effects.contains_key(&parent));
    assert!(!renderer.scene.group_effects.contains_key(&parent));
    assert!(renderer.scene.group_effects.contains_key(&first));
    let plan = renderer.planner.plan(
        &renderer.scene,
        renderer.viewport,
        renderer.fringe_width,
        4096,
        None,
    );
    assert_eq!(plan.effect_parameters.len(), 16);
    renderer
        .replace_cached_shape(parent, 1, ShapeDrawCommandOptions::new())
        .unwrap();
    assert_eq!(renderer.scene.draw_tree.children(parent), &[first, second]);
    renderer.render(&mut surface).unwrap();
    assert_eq!(surface.resource().shape_masks, 0);
    assert!(surface.resource().effects.contains(&8));
    assert_eq!(
        renderer
            .planner
            .plan(
                &renderer.scene,
                renderer.viewport,
                renderer.fringe_width,
                4096,
                None,
            )
            .effect_parameters
            .len(),
        4
    );
}

#[test]
fn replacement_errors_preserve_the_failing_node_and_keep_the_applied_prefix() {
    let mut renderer = renderer();
    queue_shape(&mut renderer, false);
    let first = queue_shape(&mut renderer, false);
    let second = queue_shape(&mut renderer, true);
    let last = queue_shape(&mut renderer, false);
    renderer.load_shape(Shape::rect([(2.0, 3.0), (6.0, 7.0)]), 2, Some(2));
    let old_effect = *renderer.scene.shape_effect(second).unwrap();
    renderer.backend.registration_failure = Some(second);
    assert!(matches!(
        renderer.replace_draw_commands([
            (first, cached(2)),
            (second, cached(2)),
            (last, cached(2))
        ]),
        Err(DrawCommandError::Backend(TestBackendError))
    ));
    for (node_id, expected) in [(first, Some(2)), (second, Some(1)), (last, Some(1))] {
        assert_eq!(
            renderer
                .scene
                .shape(node_id)
                .unwrap()
                .cached_shape
                .geometry_id,
            expected
        );
    }
    let dirty = renderer.dirty_bounds;
    for (node_id, command) in [(usize::MAX, cached(1)), (first, cached(999))] {
        assert!(renderer
            .replace_draw_commands([(node_id, command)])
            .is_err());
        assert_eq!(renderer.dirty_bounds, dirty);
    }
    assert!(matches!(
        renderer.replace_shape(second, Shape::rect([(0.0, 0.0), (f32::MAX, f32::MAX)]), None, ShapeDrawCommandOptions::new()),
        Err(DrawCommandError::Scene(SceneError::InvalidShapeEffectBounds(id))) if id == second
    ));
    assert_eq!(
        renderer.scene.shape_effect(second).unwrap().bounds,
        old_effect.bounds
    );
    assert_eq!(renderer.dirty_bounds, dirty);
    renderer.backend.registration_failure = None;
    let mut surface = surface();
    renderer.render(&mut surface).unwrap();
    assert_eq!(surface.resource().shape_masks, 1);
    assert_eq!(renderer.backend.registered_shapes.len(), 4);
}

#[test]
fn clip_damage_is_bounded_and_deferred_before_backdrop_expansion_and_planning() {
    let mut renderer = renderer();
    let mut surface = surface();
    renderer
        .add_clipping_rect(
            [(0.0, 0.0), (32.0, 32.0)],
            None,
            None::<InstanceTransform>,
            false,
        )
        .unwrap();
    let parent = renderer
        .add_clipping_rect(
            [(8.0, 8.0), (16.0, 16.0)],
            Some(0),
            None::<InstanceTransform>,
            true,
        )
        .unwrap();
    shape(&mut renderer, [(20.0, 20.0), (24.0, 24.0)], parent);
    shape(&mut renderer, [(28.0, 28.0), (30.0, 30.0)], 0);
    renderer.render(&mut surface).unwrap();
    clip(&mut renderer, parent, true);
    renderer.render(&mut surface).unwrap();
    assert_eq!(renderer.backend.root_scissor, rect((7, 7), (19, 19)));

    renderer.load_effect(7, &["backdrop"]).unwrap();
    let backdrop = shape(&mut renderer, [(2.0, 20.0), (4.0, 24.0)], 0);
    renderer
        .set_shape_backdrop_effect(
            backdrop,
            7,
            &[1, 2, 3, 4],
            BackdropEffectConfig::new().capture_area(BackdropCaptureArea::ScreenRect([
                (20.0, 20.0),
                (24.0, 24.0),
            ])),
        )
        .unwrap();
    renderer.render(&mut surface).unwrap();
    for clips_children in [false, true, false] {
        clip(&mut renderer, parent, clips_children);
    }
    assert_eq!(renderer.dirty_bounds, rect((9, 9), (19, 19)));
    renderer.backend.should_fail = true;
    assert!(renderer.render(&mut surface).is_err());
    renderer.backend.should_fail = false;
    renderer.render(&mut surface).unwrap();
    assert_eq!(renderer.backend.root_scissor, rect((1, 9), (25, 25)));
    renderer.render(&mut surface).unwrap();
    assert_eq!(renderer.backend.root_scissor, None);
}
