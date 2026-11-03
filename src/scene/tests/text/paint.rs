use super::{glyph_ids, scene, surface, Glyphs};
use crate::core::{
    Color, Fill, Gradient, GradientStop, GradientUnits, GroupDrawCommandOptions,
    LinearGradientDesc, LinearGradientLine, Shape, ShapeDrawCommandOptions, ShapeInstance,
    TransformInstance,
};
use crate::scene::tests::TestBackend;
use crate::scene::{DrawCommandError, Scene};
use crate::scene_state::types::DrawTreeNode;
use crate::scene_state::SceneError;
use lyon::geom::euclid::default::Transform3D;
use std::{ptr, sync::Arc};

fn gradient(units: GradientUnits) -> Fill {
    Gradient::linear(
        LinearGradientDesc::new(
            LinearGradientLine {
                start: [0.0, 0.0],
                end: [24.0, 0.0],
            },
            [
                GradientStop::auto(Color::BLACK),
                GradientStop::auto(Color::WHITE),
            ],
        )
        .with_units(units),
    )
    .unwrap()
    .into()
}

fn instance(scene: &Scene<TestBackend>, id: usize) -> &ShapeInstance {
    let DrawTreeNode::CachedShape(shape) = scene.state.draw_tree.get(id).unwrap() else {
        panic!("expected a shape");
    };
    &shape.instance
}

fn assert_fill(instance: &ShapeInstance, expected: &Fill) {
    match (instance.fill.as_ref().unwrap(), expected) {
        (Fill::Solid(actual), Fill::Solid(expected)) => assert_eq!(actual, expected),
        (Fill::Gradient(actual), Fill::Gradient(expected)) => {
            assert!(ptr::eq(actual.data(), expected.data()))
        }
        _ => panic!("wrong fill kind"),
    }
}

fn assert_paint_position(instance: &ShapeInstance, position: (f32, f32), expected: (f32, f32)) {
    let actual = instance
        .paint_transform
        .transform_point2d(position.into())
        .unwrap();
    assert!(
        (actual.x - expected.0).abs() < 0.001 && (actual.y - expected.1).abs() < 0.001,
        "{actual:?} != {expected:?}"
    );
}

#[test]
fn group_fill_ownership_survives_overrides_and_container_replacement() {
    let mut scene = scene();
    let mut target = surface();
    let outer_fill = gradient(GradientUnits::Local);
    let inner_fill = gradient(GradientUnits::Local);
    let group = scene
        .add_group(
            None,
            GroupDrawCommandOptions::new()
                .fill(outer_fill.clone())
                .paint_transform(TransformInstance::scale(2.0, 2.0)),
        )
        .unwrap();
    let container = scene
        .add_group(
            Some(group),
            GroupDrawCommandOptions::new()
                .paint_transform(TransformInstance::translation(100.0, 100.0)),
        )
        .unwrap();
    let text = scene
        .add_text(
            Glyphs::new(&[10, 10]),
            Some(container),
            ShapeDrawCommandOptions::new().transform(TransformInstance::translation(4.0, 6.0)),
        )
        .unwrap();
    let glyphs = glyph_ids(&scene, text);
    let options =
        ShapeDrawCommandOptions::new().transform(TransformInstance::translation(8.0, 8.0));
    let explicit = scene
        .add_shape(
            Shape::rect([(0.0, 0.0), (8.0, 8.0)]),
            Some(container),
            None,
            options
                .clone()
                .fill(inner_fill.clone())
                .clips_children(false),
        )
        .unwrap();
    let transparent = scene
        .add_shape(
            Shape::rect([(0.0, 0.0), (8.0, 8.0)]),
            Some(explicit),
            None,
            options.clone().color(Color::TRANSPARENT),
        )
        .unwrap();
    let descendant = scene
        .add_shape(
            Shape::rect([(0.0, 0.0), (4.0, 4.0)]),
            Some(explicit),
            None,
            options.clone(),
        )
        .unwrap();
    let nested = scene
        .add_group(
            Some(container),
            GroupDrawCommandOptions::new()
                .fill(inner_fill.clone())
                .paint_transform(TransformInstance::translation(2.0, 2.0)),
        )
        .unwrap();
    let nested_text = scene
        .add_text(Glyphs::new(&[10]), Some(nested), options.clone())
        .unwrap();
    let nested_glyph = glyph_ids(&scene, nested_text)[0];

    scene.render(&mut target).unwrap();
    for &id in glyphs.iter().chain([descendant].iter()) {
        assert_fill(instance(&scene, id), &outer_fill);
    }
    assert_paint_position(instance(&scene, glyphs[1]), (1.0, 1.0), (5.5, 4.5));
    assert_fill(instance(&scene, explicit), &inner_fill);
    assert_eq!(
        instance(&scene, explicit).paint_transform,
        Transform3D::identity()
    );
    assert_fill(
        instance(&scene, transparent),
        &Fill::Solid(Color::TRANSPARENT),
    );
    assert_fill(instance(&scene, nested_glyph), &inner_fill);
    assert_paint_position(instance(&scene, nested_glyph), (1.0, 1.0), (7.0, 9.0));

    scene.backend.refreshed_paints.clear();
    scene.render(&mut target).unwrap();
    assert!(scene.backend.refreshed_paints.is_empty());
    scene
        .add_shape(
            Shape::rect([(0.0, 0.0), (2.0, 2.0)]),
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::BLACK),
        )
        .unwrap();
    scene.render(&mut target).unwrap();
    assert!(
        scene.backend.refreshed_paints.is_empty(),
        "unchanged fills must keep their backend resources"
    );

    scene
        .replace_with_clipping_rect(
            nested,
            [(0.0, 0.0), (32.0, 32.0)],
            None::<TransformInstance>,
            false,
        )
        .unwrap();
    scene.render(&mut target).unwrap();
    assert_eq!(scene.backend.refreshed_paints, [nested_glyph]);
    assert_fill(instance(&scene, nested_glyph), &outer_fill);
    assert_paint_position(instance(&scene, nested_glyph), (1.0, 1.0), (4.5, 5.5));

    scene
        .replace_with_shape(
            group,
            Shape::rect([(0.0, 0.0), (32.0, 32.0)]),
            None,
            ShapeDrawCommandOptions::new()
                .color(Color::WHITE)
                .clips_children(false),
        )
        .unwrap();
    scene.render(&mut target).unwrap();
    for &id in glyphs.iter().chain([descendant, nested_glyph].iter()) {
        assert!(instance(&scene, id).fill.is_none());
        assert_eq!(
            instance(&scene, id).paint_transform,
            Transform3D::identity()
        );
    }
    assert_fill(instance(&scene, explicit), &inner_fill);
    assert_fill(
        instance(&scene, transparent),
        &Fill::Solid(Color::TRANSPARENT),
    );
}

#[test]
fn text_gradients_share_coordinates_and_cached_geometry_across_replacements() {
    let mut scene = scene();
    let mut target = surface();
    target.resize((256, 256));
    scene.resize((256, 256));
    let parent_fill = Fill::Solid(Color::WHITE);
    let parent = scene
        .add_group(
            None,
            GroupDrawCommandOptions::new().fill(parent_fill.clone()),
        )
        .unwrap();
    let fill = gradient(GradientUnits::Local);
    let mut transform = TransformInstance::scale(1.25, 0.75)
        .then(&TransformInstance::rotation_z_deg(15.0))
        .then(&TransformInstance::translation(12.0, 18.0));
    transform.col0[3] = 0.002;
    transform.col1[3] = -0.001;
    // A singular unused z axis must not invalidate the projected paint plane.
    transform.col2 = [0.0; 4];
    let options = ShapeDrawCommandOptions::new()
        .fill(fill.clone())
        .transform(transform);
    let text = scene
        .add_text(
            Glyphs::new(&[10, 10, 10, 10]),
            Some(parent),
            options.clone(),
        )
        .unwrap();
    let original = glyph_ids(&scene, text);
    scene.render(&mut target).unwrap();
    let geometry = Arc::clone(instance(&scene, original[0]).cached_shape.vertex_buffers());
    for (index, &id) in original.iter().enumerate() {
        let shape = instance(&scene, id);
        assert_fill(shape, &fill);
        assert_paint_position(shape, (1.0, 1.0), (index as f32 * 6.0 + 1.0, 3.0));
        assert!(Arc::ptr_eq(&geometry, shape.cached_shape.vertex_buffers()));
    }

    let mut culled = Glyphs::new(&[10, 10, 10, 10]);
    culled.cursor = 2;
    scene.replace_with_text(text, culled, options).unwrap();
    scene.render(&mut target).unwrap();
    for (index, &id) in glyph_ids(&scene, text).iter().enumerate() {
        let shape = instance(&scene, id);
        assert_paint_position(shape, (1.0, 1.0), ((index + 2) as f32 * 6.0 + 1.0, 3.0));
        assert!(Arc::ptr_eq(&geometry, shape.cached_shape.vertex_buffers()));
    }

    scene
        .replace_with_text(text, Glyphs::new(&[10]), ShapeDrawCommandOptions::new())
        .unwrap();
    scene.render(&mut target).unwrap();
    let shape = instance(&scene, glyph_ids(&scene, text)[0]);
    assert_fill(shape, &parent_fill);
    assert_eq!(shape.color_override, Some(Color::WHITE.normalize()));
    assert_eq!(shape.paint_transform, Transform3D::identity());
}

#[test]
fn invalid_text_paint_transform_preserves_existing_scene() {
    let mut scene = scene();
    let mut target = surface();
    let fill = gradient(GradientUnits::Local);
    let text = scene
        .add_text(
            Glyphs::new(&[10, 10]),
            None,
            ShapeDrawCommandOptions::new().fill(fill.clone()),
        )
        .unwrap();
    scene.render(&mut target).unwrap();
    let original = glyph_ids(&scene, text);
    let registrations = scene.backend.registered_shapes.clone();
    for transform in [
        TransformInstance::scale(0.0, 1.0),
        TransformInstance::translation(f32::NAN, 0.0),
    ] {
        let result = scene.replace_with_text(
            text,
            Glyphs::new(&[11]),
            ShapeDrawCommandOptions::new()
                .fill(fill.clone())
                .transform(transform),
        );
        assert!(matches!(
            result,
            Err(DrawCommandError::Scene(SceneError::InvalidPaintTransform))
        ));
        assert_eq!(glyph_ids(&scene, text), original);
        assert_eq!(scene.backend.registered_shapes, registrations);
        assert!(scene.dirty_bounds.is_none());
    }
    scene.backend.registration_failure = Some(scene.state.next_node_id() + 1);
    assert!(scene
        .replace_with_text(
            text,
            Glyphs::new(&[11, 11]),
            ShapeDrawCommandOptions::new().color(Color::WHITE)
        )
        .is_err());
    scene.backend.registration_failure = None;
    scene.backend.refreshed_paints.clear();
    scene.render(&mut target).unwrap();
    assert_eq!(glyph_ids(&scene, text), original);
    assert!(scene.backend.refreshed_paints.is_empty());
    for id in original {
        assert_fill(instance(&scene, id), &fill);
    }

    let canvas = gradient(GradientUnits::Canvas);
    let group = scene
        .add_group(
            None,
            GroupDrawCommandOptions::new()
                .fill(canvas.clone())
                .paint_transform(TransformInstance::scale(0.0, 1.0)),
        )
        .unwrap();
    let canvas_text = scene
        .add_text(
            Glyphs::new(&[10]),
            Some(group),
            ShapeDrawCommandOptions::new(),
        )
        .unwrap();
    scene.render(&mut target).unwrap();
    let shape = instance(&scene, glyph_ids(&scene, canvas_text)[0]);
    assert_fill(shape, &canvas);
    assert_eq!(shape.paint_transform, Transform3D::identity());
}
