use super::{path_with_nan_positions, scene, surface, TestBackend, TestBackendError};
use crate::core::{CachedTessellation, PathShape, Shape, ShapeDrawCommandOptions, StrokeOptions};
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
fn stroke_complexity_errors_preserve_geometry_and_allow_later_strokes() {
    let mut scene = scene();
    let stroke = Shape::rect([(4.0, 4.0), (20.0, 20.0)]).stroke(StrokeOptions::new(2.0));
    scene.load_shape(&stroke, 1, None).unwrap();
    let node = scene
        .add_cached_shape(1, None, ShapeDrawCommandOptions::new())
        .unwrap();
    let original = scene.state.loaded_shape(1).unwrap().tessellation;
    scene.render(&mut surface()).unwrap();

    let mut builder = Shape::builder();
    for _ in 0..16_384 {
        builder = builder.begin((0.0, 0.0)).line_to((10.0, 0.0)).end();
    }
    let complex = builder.build().stroke(StrokeOptions::new(2.0));
    assert!(matches!(
        scene.load_shape(&complex, 1, None),
        Err(SceneError::StrokeTooComplex {
            triangle_count: 32_768,
            maximum_triangles: 32_767,
        })
    ));
    assert!(matches!(
        scene.add_shape(&complex, None, None, ShapeDrawCommandOptions::new()),
        Err(DrawCommandError::Scene(SceneError::StrokeTooComplex {
            triangle_count: 32_768,
            maximum_triangles: 32_767,
        }))
    ));
    assert!(matches!(
        scene.replace_with_shape(node, &complex, None, ShapeDrawCommandOptions::new()),
        Err(DrawCommandError::Scene(SceneError::StrokeTooComplex {
            triangle_count: 32_768,
            maximum_triangles: 32_767,
        }))
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

    let wider = stroke.stroke(StrokeOptions::new(4.0));
    scene.load_shape(&wider, 1, None).unwrap();
    scene
        .replace_with_shape(node, &wider, None, ShapeDrawCommandOptions::new())
        .unwrap();
    let added = scene
        .add_shape(&wider, None, None, ShapeDrawCommandOptions::new())
        .unwrap();
    for node in [node, added] {
        assert_eq!(
            scene.state.shape(node).unwrap().cached_shape.local_bounds(),
            [(2.0, 2.0), (22.0, 22.0)],
        );
    }
    assert_eq!(
        scene.state.loaded_shape(1).unwrap().local_bounds(),
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
