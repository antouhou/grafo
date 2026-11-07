use super::{path_with_nan_positions, scene, surface, TestBackend, TestBackendError};
use crate::core::{
    BackdropEffectConfig, CachedTessellation, Color, ContentEffectConfig, LineCap, LineJoin,
    PathShape, Shape, ShapeDrawCommandOptions, ShapeMaskEffectConfig, StrokeOptions,
    TransformInstance, UnsignedPhysicalRect,
};
use crate::scene::{DrawCommandError, Scene};
use crate::scene_state::{SceneError, SceneStateContext};
use crate::StrokeOptionsError;
use lyon::tessellation::{GeometryBuilderError, TessellationError};
use std::sync::Arc;

fn buffer_addresses(geometry: &CachedTessellation) -> [usize; 3] {
    [
        geometry.vertex_buffers.vertices.as_ptr() as usize,
        geometry.vertex_buffers.indices.as_ptr() as usize,
        geometry.boundary_edges.as_ptr() as usize,
    ]
}

fn miter_stroke() -> Shape {
    Shape::builder()
        .begin((10.25, 18.25))
        .line_to((16.25, 10.25))
        .line_to((22.25, 18.25))
        .end()
        .build()
        .stroke(StrokeOptions::new(6.0).line_join(LineJoin::Miter))
}

#[test]
fn stroke_replacement_and_removal_redraw_transformed_caps_and_miters() {
    for cap in [LineCap::Round, LineCap::Square] {
        let mut scene = scene();
        let mut surface = surface();
        surface.resize((64, 64));
        scene.change_scale_factor(2.0).unwrap();
        scene.render(&mut surface).unwrap();
        let stroke = Shape::builder()
            .begin((8.0, 10.0))
            .line_to((20.0, 10.0))
            .end()
            .build()
            .stroke(StrokeOptions::new(4.0).line_cap(cap));
        let options = ShapeDrawCommandOptions::new()
            .color(Color::WHITE)
            .transform(TransformInstance::affine_2d(0.0, 1.0, -1.0, 0.0, 26.0, 1.0));
        scene.load_shape(stroke, 1, None).unwrap();
        let node = scene.add_cached_shape(1, None, options.clone()).unwrap();
        scene.render(&mut surface).unwrap();
        assert_eq!(surface.resource().draws, [node]);
        assert_eq!(
            scene.backend.root_scissor,
            Some(UnsignedPhysicalRect::new((28, 14).into(), (36, 46).into())),
            "{cap:?} cap"
        );

        // The 3-4-5 corner extends five units above the centerline, to y = 5.25.
        scene
            .replace_with_shape(node, miter_stroke(), None, options)
            .unwrap();
        scene.render(&mut surface).unwrap();
        assert_eq!(surface.resource().draws, [node]);
        assert_eq!(
            scene.backend.root_scissor,
            Some(UnsignedPhysicalRect::new((11, 14).into(), (42, 52).into())),
        );
        scene.render(&mut surface).unwrap();
        assert_eq!(scene.backend.root_scissor, None);

        scene.remove_subtrees([node], |_| {});
        scene.render(&mut surface).unwrap();
        assert!(surface.resource().draws.is_empty());
        assert_eq!(
            scene.backend.root_scissor,
            Some(UnsignedPhysicalRect::new((11, 17).into(), (42, 52).into())),
        );
    }
}

#[test]
fn stroke_effect_captures_follow_caps_and_miters_after_replacement() {
    let mut scene = scene();
    let mut surface = surface();
    let stroke = Shape::builder()
        .begin((8.0, 10.0))
        .line_to((20.0, 10.0))
        .end()
        .build()
        .stroke(StrokeOptions::new(4.0).line_cap(LineCap::Square));
    let options = ShapeDrawCommandOptions::new()
        .color(Color::WHITE)
        .transform(TransformInstance::translation(1.0, 2.0));
    let node = scene
        .add_shape(stroke, None, None, options.clone())
        .unwrap();
    scene
        .set_shape_mask_effect(
            node,
            7,
            &[1, 2, 3, 4],
            ShapeMaskEffectConfig::new().outset(1.0),
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
    scene
        .set_backdrop_effect(
            node,
            9,
            &[1, 2, 3, 4],
            BackdropEffectConfig::new().padding(3.0),
        )
        .unwrap();
    scene.render(&mut surface).unwrap();
    assert_eq!(surface.resource().shape_masks, 1);
    assert!(surface.resource().effects.contains(&7));
    assert!(surface.resource().effects.contains(&8));
    assert!(surface.resource().effects.contains(&9));
    assert_eq!(
        scene
            .state
            .shape_mask_effect(node)
            .unwrap()
            .bounds
            .local_bounds,
        [(4.0, 6.0), (24.0, 14.0)],
    );
    let content = scene.state.content_effects[&node]
        .bounds
        .unwrap()
        .capture_region;
    assert_eq!(content.bounds.min.to_tuple(), (3, 6));
    assert_eq!(content.bounds.max.to_tuple(), (27, 18));
    assert_eq!(content.source_rect.unwrap().min.to_tuple(), (5, 8));
    assert_eq!(content.source_rect.unwrap().max.to_tuple(), (25, 16));
    let backdrop = scene.state.backdrop_effects[&node].capture_region.unwrap();
    assert_eq!(backdrop.bounds.min.to_tuple(), (4, 7));
    assert_eq!(backdrop.bounds.max.to_tuple(), (26, 17));

    scene
        .replace_with_shape(node, miter_stroke(), None, options)
        .unwrap();
    scene.render(&mut surface).unwrap();
    assert_eq!(surface.resource().shape_masks, 1);
    assert!(surface.resource().effects.contains(&7));
    assert!(surface.resource().effects.contains(&8));
    assert!(surface.resource().effects.contains(&9));
    assert_eq!(
        scene
            .state
            .shape_mask_effect(node)
            .unwrap()
            .bounds
            .local_bounds,
        [(5.0, 3.0), (27.0, 23.0)],
    );
    let content = scene.state.content_effects[&node]
        .bounds
        .unwrap()
        .capture_region;
    assert_eq!(content.bounds.min.to_tuple(), (4, 3));
    assert_eq!(content.bounds.max.to_tuple(), (30, 27));
    assert_eq!(content.source_rect.unwrap().min.to_tuple(), (6, 5));
    assert_eq!(content.source_rect.unwrap().max.to_tuple(), (28, 25));
    let backdrop = scene.state.backdrop_effects[&node].capture_region.unwrap();
    assert_eq!(backdrop.bounds.min.to_tuple(), (5, 4));
    assert_eq!(backdrop.bounds.max.to_tuple(), (29, 26));
    assert_eq!(
        scene.backend.root_scissor,
        Some(UnsignedPhysicalRect::new((3, 3).into(), (30, 27).into())),
    );
}

#[test]
fn invalid_stroke_options_preserve_loaded_shapes_and_queued_draws() {
    let mut scene = scene();
    let stroke = Shape::rect([(4.0, 4.0), (20.0, 20.0)]).stroke(StrokeOptions::new(2.0));
    scene.load_shape(&stroke, 1, Some(10)).unwrap();
    let node = scene
        .add_cached_shape(1, None, ShapeDrawCommandOptions::new())
        .unwrap();
    let original = scene.state.loaded_shape(1).unwrap().tessellation;
    scene.render(&mut surface()).unwrap();

    let invalid_options = [-1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY]
        .into_iter()
        .map(|width| (StrokeOptions::new(width), StrokeOptionsError::InvalidWidth))
        .chain(
            [0.0, 0.5, f32::NAN, f32::INFINITY, f32::NEG_INFINITY]
                .into_iter()
                .map(|limit| {
                    (
                        StrokeOptions::new(2.0).miter_limit(limit),
                        StrokeOptionsError::InvalidMiterLimit,
                    )
                }),
        );
    for (options, expected_error) in invalid_options {
        let invalid = stroke.clone().stroke(options);
        assert!(matches!(
            scene.load_shape(&invalid, 1, Some(10)),
            Err(SceneError::InvalidStrokeOptions(error)) if error == expected_error
        ));
        assert!(matches!(
            scene.add_shape(&invalid, None, Some(10), ShapeDrawCommandOptions::new()),
            Err(DrawCommandError::Scene(SceneError::InvalidStrokeOptions(error)))
                if error == expected_error
        ));
        assert!(matches!(
            scene.replace_with_shape(node, &invalid, Some(10), ShapeDrawCommandOptions::new()),
            Err(DrawCommandError::Scene(SceneError::InvalidStrokeOptions(error)))
                if error == expected_error
        ));
        assert!(Arc::ptr_eq(
            &original,
            &scene.state.loaded_shape(1).unwrap().tessellation,
        ));
        assert!(Arc::ptr_eq(
            &original,
            &scene.state.shape(node).unwrap().cached_shape.tessellation,
        ));
        assert_eq!(scene.backend.registered_shapes, [node]);
        assert!(scene.dirty_bounds.is_none());
    }
}

#[test]
fn invalid_stroke_points_preserve_existing_geometry_and_allow_later_strokes() {
    let mut scene = scene();
    let stroke = Shape::rect([(4.0, 4.0), (20.0, 20.0)]).stroke(StrokeOptions::new(2.0));
    scene.load_shape(&stroke, 1, None).unwrap();
    let node = scene
        .add_cached_shape(1, None, ShapeDrawCommandOptions::new())
        .unwrap();
    let original = scene.state.loaded_shape(1).unwrap().tessellation;
    for invalid in [
        Shape::from(PathShape::new(path_with_nan_positions())),
        Shape::rect([(0.0, 0.0), (f32::INFINITY, 10.0)]),
    ] {
        let invalid = invalid.stroke(StrokeOptions::new(2.0));
        assert!(matches!(
            scene.load_shape(&invalid, 1, None),
            Err(SceneError::Tessellation(
                TessellationError::GeometryBuilder(GeometryBuilderError::InvalidVertex,)
            ))
        ));
        assert!(matches!(
            scene.replace_with_shape(node, &invalid, None, ShapeDrawCommandOptions::new()),
            Err(DrawCommandError::Scene(SceneError::Tessellation(
                TessellationError::GeometryBuilder(GeometryBuilderError::InvalidVertex),
            )))
        ));
        assert!(Arc::ptr_eq(
            &original,
            &scene.state.loaded_shape(1).unwrap().tessellation,
        ));
        assert!(Arc::ptr_eq(
            &original,
            &scene.state.shape(node).unwrap().cached_shape.tessellation,
        ));
    }

    let wider = stroke.stroke(StrokeOptions::new(4.0));
    scene.load_shape(&wider, 1, None).unwrap();
    scene
        .replace_with_cached_shape(node, 1, ShapeDrawCommandOptions::new())
        .unwrap();
    assert_eq!(
        scene.state.shape(node).unwrap().cached_shape.local_bounds(),
        [(2.0, 2.0), (22.0, 22.0)],
    );
}

#[test]
fn shared_strokes_reuse_buffers_after_scene_removal_and_failed_replacement() {
    let context = SceneStateContext::default();
    let mut first = Scene::from_parts(TestBackend::default(), context.clone());
    let mut second = Scene::from_parts(TestBackend::default(), context);
    let stroke = Shape::rect([(4.0, 4.0), (20.0, 20.0)]).stroke(StrokeOptions::new(2.0));
    first.load_shape(&stroke, 1, None).unwrap();
    first
        .add_cached_shape(1, None, ShapeDrawCommandOptions::new())
        .unwrap();
    let node = second
        .add_cached_shape(1, None, ShapeDrawCommandOptions::new())
        .unwrap();
    let original = Arc::clone(&second.state.shape(node).unwrap().cached_shape.tessellation);
    let original_addresses = buffer_addresses(&original);
    let original_vertex = original.vertex_buffers.vertices[0].position;
    let released = Arc::downgrade(&original);
    first.remove_shape(1);
    drop(first);

    let wider = stroke.stroke(StrokeOptions::new(4.0));
    let separate = second.state.tessellate(&wider, None).unwrap();
    let separate_addresses = buffer_addresses(&separate.tessellation);
    for (original_address, separate_address) in
        original_addresses.into_iter().zip(separate_addresses)
    {
        assert_ne!(original_address, separate_address);
    }
    assert_eq!(
        original.vertex_buffers.vertices[0].position,
        original_vertex
    );
    assert!(Arc::ptr_eq(
        &original,
        &second.state.shape(node).unwrap().cached_shape.tessellation,
    ));
    second.state.release_tessellation(separate.tessellation);
    second.clear();
    drop(original);
    assert!(released.upgrade().is_none());

    let node = second
        .add_shape(&wider, None, None, ShapeDrawCommandOptions::new())
        .unwrap();
    let current = Arc::clone(&second.state.shape(node).unwrap().cached_shape.tessellation);
    assert_eq!(buffer_addresses(&current), separate_addresses);
    let spare = second.state.tessellate(&wider, None).unwrap();
    let spare_addresses = buffer_addresses(&spare.tessellation);
    second.state.release_tessellation(spare.tessellation);
    second.backend.should_fail = true;
    assert!(matches!(
        second.replace_with_shape(node, &wider, None, ShapeDrawCommandOptions::new()),
        Err(DrawCommandError::Backend(TestBackendError))
    ));
    assert!(Arc::ptr_eq(
        &current,
        &second.state.shape(node).unwrap().cached_shape.tessellation,
    ));
    second.backend.should_fail = false;
    second
        .replace_with_shape(node, &wider, None, ShapeDrawCommandOptions::new())
        .unwrap();
    let surviving = Arc::clone(&second.state.shape(node).unwrap().cached_shape.tessellation);
    assert_eq!(buffer_addresses(&surviving), spare_addresses);
    drop(current);
    drop(second);
    assert_eq!(surviving.local_bounds, [(2.0, 2.0), (22.0, 22.0)]);
    assert!(!surviving.vertex_buffers.indices.is_empty());
    drop(surviving);
}

#[test]
fn uncached_strokes_reuse_buffers_after_replacement_removal_and_clear() {
    let mut scene = scene();
    let stroke = Shape::rect([(4.0, 4.0), (20.0, 20.0)]).stroke(StrokeOptions::new(2.0));
    let root = scene
        .add_shape(&stroke, None, None, ShapeDrawCommandOptions::new())
        .unwrap();
    let root_addresses =
        buffer_addresses(&scene.state.shape(root).unwrap().cached_shape.tessellation);
    let child = scene
        .add_shape(&stroke, Some(root), None, ShapeDrawCommandOptions::new())
        .unwrap();
    let first_addresses =
        buffer_addresses(&scene.state.shape(child).unwrap().cached_shape.tessellation);
    let wider = stroke.clone().stroke(StrokeOptions::new(4.0));
    scene
        .replace_with_shape(child, &wider, None, ShapeDrawCommandOptions::new())
        .unwrap();
    let second_addresses =
        buffer_addresses(&scene.state.shape(child).unwrap().cached_shape.tessellation);
    assert_ne!(first_addresses, second_addresses);
    for _ in 0..4 {
        for (shape, expected) in [(&stroke, first_addresses), (&wider, second_addresses)] {
            scene
                .replace_with_shape(child, shape, None, ShapeDrawCommandOptions::new())
                .unwrap();
            assert_eq!(
                buffer_addresses(&scene.state.shape(child).unwrap().cached_shape.tessellation),
                expected
            );
        }
    }
    scene.remove_subtrees([child], |_| {});
    assert_eq!(scene.backend.registered_shapes, [root]);
    let replacement = scene
        .add_shape(&stroke, Some(root), None, ShapeDrawCommandOptions::new())
        .unwrap();
    assert_eq!(
        buffer_addresses(
            &scene
                .state
                .shape(replacement)
                .unwrap()
                .cached_shape
                .tessellation
        ),
        second_addresses
    );
    scene.clear();
    assert!(scene.backend.registered_geometry.is_empty());

    let mut reused_addresses = [[0; 3]; 3];
    for addresses in &mut reused_addresses {
        let node = scene
            .add_shape(&stroke, None, None, ShapeDrawCommandOptions::new())
            .unwrap();
        *addresses = buffer_addresses(&scene.state.shape(node).unwrap().cached_shape.tessellation);
    }
    let mut expected_addresses = [root_addresses, first_addresses, second_addresses];
    reused_addresses.sort_unstable();
    expected_addresses.sort_unstable();
    assert_eq!(reused_addresses, expected_addresses);
}
