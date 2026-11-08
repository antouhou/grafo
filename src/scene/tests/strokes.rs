use super::{path_with_nan_positions, scene, surface, TestBackendError};
use crate::commands::ShapeDrawId;
use crate::core::{
    BackdropEffectConfig, Color, ContentEffectConfig, LineCap, LineJoin, PathShape, Shape,
    ShapeDrawCommandOptions, ShapeMaskEffectConfig, StrokeOptions, TransformInstance,
    UnsignedPhysicalRect,
};
use crate::scene::DrawCommandError;
use crate::scene_state::SceneError;
use crate::StrokeOptionsError;
use lyon::tessellation::{GeometryBuilderError, TessellationError};
use std::sync::Arc;

#[test]
fn stroke_replacement_and_removal_redraw_transformed_caps_and_miters() {
    for cap in [LineCap::Round, LineCap::Square] {
        let mut scene = scene();
        let mut surface = surface();
        surface.resize((64, 64));
        scene.change_scale_factor(2.0).unwrap();
        scene.render(&mut surface).unwrap();
        let line = Shape::builder()
            .begin((8.0, 10.0))
            .line_to((20.0, 10.0))
            .end()
            .build();
        let options = ShapeDrawCommandOptions::new()
            .transform(TransformInstance::affine_2d(0.0, 1.0, -1.0, 0.0, 26.0, 1.0));
        scene.load_shape(line, 1, None).unwrap();
        let node = scene
            .add_cached_shape(
                1,
                None,
                options
                    .clone()
                    .stroke(StrokeOptions::new(4.0).line_cap(cap), Color::WHITE),
            )
            .unwrap();
        scene.render(&mut surface).unwrap();
        assert_eq!(surface.resource().draws, [ShapeDrawId(node).stroke().0]);
        assert_eq!(
            scene.backend.root_scissor,
            Some(UnsignedPhysicalRect::new((28, 14).into(), (36, 46).into())),
            "{cap:?} cap"
        );

        let corner = Shape::builder()
            .begin((10.25, 18.25))
            .line_to((16.25, 10.25))
            .line_to((22.25, 18.25))
            .end()
            .build();
        scene
            .replace_with_shape(
                node,
                corner,
                None,
                options.stroke(
                    StrokeOptions::new(6.0).line_join(LineJoin::Miter),
                    Color::WHITE,
                ),
            )
            .unwrap();
        scene.render(&mut surface).unwrap();
        assert_eq!(
            surface.resource().draws,
            [node, ShapeDrawId(node).stroke().0]
        );
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
fn stroke_expands_content_capture_and_preserves_shape_effect_bounds() {
    let mut scene = scene();
    let mut surface = surface();
    let shape = Shape::rect([(8.0, 10.0), (20.0, 18.0)]);
    let node = scene
        .add_shape(
            shape,
            None,
            None,
            ShapeDrawCommandOptions::new()
                .stroke(StrokeOptions::new(4.0), Color::WHITE)
                .transform(TransformInstance::translation(1.0, 2.0)),
        )
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
    for effect in [7, 8, 9] {
        assert!(surface.resource().effects.contains(&effect));
    }
    assert_eq!(
        scene
            .state
            .shape_mask_effect(node)
            .unwrap()
            .bounds
            .local_bounds,
        [(6.0, 8.0), (22.0, 20.0)],
    );
    let content = scene.state.content_effects[&node]
        .bounds
        .unwrap()
        .capture_region;
    assert_eq!(content.bounds.min.to_tuple(), (5, 8));
    assert_eq!(content.bounds.max.to_tuple(), (25, 24));
    assert_eq!(content.source_rect.unwrap().min.to_tuple(), (7, 10));
    assert_eq!(content.source_rect.unwrap().max.to_tuple(), (23, 22));
    let backdrop = scene.state.backdrop_effects[&node].capture_region.unwrap();
    assert_eq!(backdrop.bounds.min.to_tuple(), (6, 9));
    assert_eq!(backdrop.bounds.max.to_tuple(), (24, 23));
}

#[test]
fn invalid_stroke_options_and_positions_preserve_queued_draws() {
    let mut scene = scene();
    let shape = Shape::rect([(4.0, 4.0), (20.0, 20.0)]);
    scene.load_shape(&shape, 1, Some(10)).unwrap();
    let node = scene
        .add_cached_shape(
            1,
            None,
            ShapeDrawCommandOptions::new().stroke(StrokeOptions::new(2.0), Color::WHITE),
        )
        .unwrap();
    let original = scene.state.loaded_shape(1).unwrap().tessellation;
    let original_stroke = Arc::clone(
        &scene
            .state
            .shape(node)
            .unwrap()
            .stroke
            .as_ref()
            .unwrap()
            .tessellation,
    );
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
        let options = ShapeDrawCommandOptions::new().stroke(options, Color::WHITE);
        assert!(matches!(
            scene.replace_with_cached_shape(node, 1, options),
            Err(DrawCommandError::Scene(SceneError::InvalidStrokeOptions(error)))
                if error == expected_error
        ));
    }
    for invalid in [
        Shape::from(PathShape::new(path_with_nan_positions())),
        Shape::rect([(0.0, 0.0), (f32::INFINITY, 10.0)]),
    ] {
        let options = ShapeDrawCommandOptions::new().stroke(StrokeOptions::new(2.0), Color::WHITE);
        assert!(matches!(
            scene.replace_with_shape(node, &invalid, None, options),
            Err(DrawCommandError::Scene(SceneError::Tessellation(
                TessellationError::GeometryBuilder(GeometryBuilderError::InvalidVertex),
            )))
        ));
    }
    assert!(Arc::ptr_eq(
        &original,
        &scene.state.loaded_shape(1).unwrap().tessellation
    ));
    let queued = scene.state.shape(node).unwrap();
    assert!(Arc::ptr_eq(&original, &queued.cached_shape.tessellation));
    assert!(Arc::ptr_eq(
        &original_stroke,
        &queued.stroke.as_ref().unwrap().tessellation
    ));
    assert_eq!(
        scene.backend.registered_shapes,
        [node, ShapeDrawId(node).stroke().0]
    );
    assert!(scene.dirty_bounds.is_none());
}

#[test]
fn cached_shapes_share_geometry_by_stroke_width_and_replace_atomically() {
    let mut scene = scene();
    let shape = Shape::rect([(4.0, 4.0), (20.0, 20.0)]);
    scene.load_shape(&shape, 1, None).unwrap();
    let options =
        |width| ShapeDrawCommandOptions::new().stroke(StrokeOptions::new(width), Color::WHITE);
    let thin = scene.add_cached_shape(1, None, options(2.0)).unwrap();
    let wide = scene.add_cached_shape(1, None, options(4.0)).unwrap();
    let repeated = scene.add_cached_shape(1, None, options(2.0)).unwrap();
    let stroke = |node| {
        Arc::clone(
            &scene
                .state
                .shape(node)
                .unwrap()
                .stroke
                .as_ref()
                .unwrap()
                .tessellation,
        )
    };
    let thin_geometry = stroke(thin);
    let wide_geometry = stroke(wide);
    assert!(Arc::ptr_eq(&thin_geometry, &stroke(repeated)));
    assert!(!Arc::ptr_eq(&thin_geometry, &wide_geometry));
    assert_eq!(thin_geometry.local_bounds, [(3.0, 3.0), (21.0, 21.0)]);
    assert_eq!(wide_geometry.local_bounds, [(2.0, 2.0), (22.0, 22.0)]);

    scene.backend.should_fail = true;
    assert!(matches!(
        scene.replace_with_cached_shape(wide, 1, options(2.0)),
        Err(DrawCommandError::Backend(TestBackendError))
    ));
    let wide_id = ShapeDrawId(wide).stroke().0;
    assert!(Arc::ptr_eq(
        &wide_geometry,
        &scene.backend.registered_geometry[&wide_id]
    ));

    scene.backend.should_fail = false;
    scene
        .replace_with_cached_shape(wide, 1, options(0.0))
        .unwrap();
    assert!(scene.state.shape(wide).unwrap().stroke.is_none());
    assert!(!scene.backend.registered_geometry.contains_key(&wide_id));
}
