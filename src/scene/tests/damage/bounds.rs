use super::super::{scene, surface};
use crate::commands::RenderOperation;
use crate::core::vertex::InstanceTransform;
use crate::core::{
    BackdropEffectConfig, Color, MathRect, PhysicalRect, Shape, ShapeDrawCommandOptions,
    UnsignedPhysicalRect,
};

#[test]
fn negative_w_shape_addition_dirties_shader_footprint_after_empty_render() {
    let mut scene = scene();
    let mut surface = surface();
    scene.render(&mut surface).unwrap();
    let mut transform = InstanceTransform::identity();
    transform.col3[3] = -1.0;
    scene
        .add_shape(
            Shape::rect([(8.0, 8.0), (16.0, 16.0)]),
            None,
            None,
            ShapeDrawCommandOptions::new()
                .color(Color::WHITE)
                .transform(transform),
        )
        .unwrap();
    scene.render(&mut surface).unwrap();
    assert_eq!(
        scene.backend.root_scissor,
        Some(UnsignedPhysicalRect::new((8, 8).into(), (16, 16).into()))
    );
}

#[test]
fn negative_w_shape_removal_dirties_shader_footprint_after_render() {
    let mut scene = scene();
    let mut surface = surface();
    let mut transform = InstanceTransform::identity();
    transform.col3[3] = -1.0;
    let shape = scene
        .add_shape(
            Shape::rect([(8.0, 8.0), (16.0, 16.0)]),
            None,
            None,
            ShapeDrawCommandOptions::new()
                .color(Color::WHITE)
                .transform(transform),
        )
        .unwrap();
    scene.render(&mut surface).unwrap();
    scene.remove_subtrees([shape], |_| {});
    scene.render(&mut surface).unwrap();
    assert_eq!(
        scene.backend.root_scissor,
        Some(UnsignedPhysicalRect::new((8, 8).into(), (16, 16).into()))
    );
}

#[test]
fn uncertain_projection_redraws_viewport_on_removal_and_addition() {
    for (horizontal_w, origin_w) in [
        (0.125e-6, -0.5e-6),
        (-0.125e-6, 0.5e-6),
        (0.25, -3.0),
        (f32::MAX, 1.0),
    ] {
        let mut scene = scene();
        let mut surface = surface();
        let mut transform = InstanceTransform::identity();
        transform.col0[3] = horizontal_w;
        transform.col3[3] = origin_w;
        let shape = scene
            .add_shape(
                Shape::rect([(8.0, 8.0), (16.0, 16.0)]),
                None,
                None,
                ShapeDrawCommandOptions::new().transform(transform),
            )
            .unwrap();
        scene.render(&mut surface).unwrap();
        scene.remove_subtrees([shape], |_| {});
        scene.render(&mut surface).unwrap();
        assert_eq!(
            scene.backend.root_scissor,
            Some(UnsignedPhysicalRect::from_size((32, 32).into())),
            "transform: {transform:?}"
        );
        scene
            .add_shape(
                Shape::rect([(8.0, 8.0), (16.0, 16.0)]),
                None,
                None,
                ShapeDrawCommandOptions::new().transform(transform),
            )
            .unwrap();
        scene.render(&mut surface).unwrap();
        assert_eq!(
            scene.backend.root_scissor,
            Some(UnsignedPhysicalRect::from_size((32, 32).into())),
            "transform: {transform:?}"
        );
    }
}

#[test]
fn subtree_replacement_accumulates_bounds_until_a_successful_render() {
    let mut scene = scene();
    let mut surface = surface();
    scene.change_scale_factor(2.0).unwrap();
    scene.render(&mut surface).unwrap();
    assert!(!scene.is_dirty_region_overlay_enabled());
    scene.set_dirty_region_overlay_enabled(true);
    scene.render(&mut surface).unwrap();
    assert_eq!(scene.backend.root_scissor, None);
    let branch = scene
        .add_clipping_rect(
            [(0.0, 0.0), (16.0, 16.0)],
            None,
            None::<InstanceTransform>,
            false,
        )
        .unwrap();
    scene.load_shape(Shape::rect([(0.0, 0.0), (2.0, 2.0)]), 17, None);
    let first = scene
        .add_cached_shape(
            17,
            Some(branch),
            ShapeDrawCommandOptions::new()
                .color(Color::WHITE)
                .transform(InstanceTransform::translation(2.0, 3.0)),
        )
        .unwrap();
    scene
        .add_shape(
            Shape::rect([(10.0, 11.0), (12.0, 13.0)]),
            Some(branch),
            None,
            ShapeDrawCommandOptions::new().color(Color::WHITE),
        )
        .unwrap();
    scene.render(&mut surface).unwrap();
    let bounds = UnsignedPhysicalRect::new((4, 6).into(), (24, 26).into());
    assert_eq!(scene.backend.root_scissor, Some(bounds));
    scene.render(&mut surface).unwrap();
    assert_eq!(scene.backend.root_scissor, None);

    scene.remove_subtrees([branch, first, branch], |_| {});
    scene
        .add_shape(
            Shape::rect([(14.0, 1.0), (18.0, 3.0)]),
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::WHITE),
        )
        .unwrap();
    scene.backend.should_fail = true;
    assert!(scene.render(&mut surface).is_err());
    scene.backend.should_fail = false;
    scene.render(&mut surface).unwrap();
    assert!(scene.is_dirty_region_overlay_enabled());
    assert_eq!(
        scene.backend.root_scissor,
        Some(UnsignedPhysicalRect::new((4, 2).into(), (32, 26).into()))
    );
    scene.render(&mut surface).unwrap();
    assert_eq!(scene.backend.root_scissor, None);

    scene.clear();
    scene.render(&mut surface).unwrap();
    assert_eq!(
        scene.backend.root_scissor,
        Some(UnsignedPhysicalRect::new((28, 2).into(), (32, 6).into()))
    );
    scene.remove_subtrees([usize::MAX], |_| panic!("missing node"));
    scene.set_dirty_region_overlay_enabled(false);
    scene.render(&mut surface).unwrap();
    assert_eq!(scene.backend.root_scissor, None);
    assert!(!scene.is_dirty_region_overlay_enabled());
}

#[test]
fn retained_node_bounds_follow_viewport_changes_and_removal() {
    let mut scene = scene();
    let mut surface = surface();
    scene.render(&mut surface).unwrap();
    let branch = scene
        .add_clipping_rect(
            [(0.0, 0.0), (20.0, 20.0)],
            None,
            Some(InstanceTransform::translation(5.0, 4.0)),
            true,
        )
        .unwrap();
    scene.load_shape(Shape::rect([(0.0, 0.0), (4.0, 3.0)]), 17, None);
    let cached = scene
        .add_cached_shape(
            17,
            Some(branch),
            ShapeDrawCommandOptions::new()
                .color(Color::WHITE)
                .transform(InstanceTransform::affine_2d(-1.0, 0.0, 0.0, 1.0, 13.0, 7.0)),
        )
        .unwrap();
    scene
        .add_cached_shape(
            17,
            Some(cached),
            ShapeDrawCommandOptions::new()
                .color(Color::WHITE)
                .transform(InstanceTransform::translation(9.0, 7.0)),
        )
        .unwrap();
    let path = scene
        .add_shape(
            Shape::builder()
                .begin((0.0, 0.0))
                .line_to((4.0, 0.0))
                .line_to((4.0, 3.0))
                .line_to((0.0, 3.0))
                .close()
                .build(),
            Some(branch),
            None,
            ShapeDrawCommandOptions::new()
                .color(Color::WHITE)
                .transform(InstanceTransform::affine_2d(
                    1.0, 0.5, -0.25, 1.0, 17.0, 11.0,
                )),
        )
        .unwrap();
    scene.load_effect(7, &["backdrop effect"]).unwrap();
    scene
        .set_backdrop_effect(
            path,
            7,
            &[1, 2, 3, 4],
            BackdropEffectConfig::new().padding(0.5),
        )
        .unwrap();

    for (scale, size, expected_clip, expected_cached_clip, expected_capture) in [
        (
            1.0,
            (32, 32),
            UnsignedPhysicalRect::new((5, 4).into(), (25, 24).into()),
            UnsignedPhysicalRect::new((9, 7).into(), (13, 10).into()),
            PhysicalRect::new((15, 10).into(), (22, 17).into()),
        ),
        (
            2.0,
            (64, 48),
            UnsignedPhysicalRect::new((10, 8).into(), (50, 48).into()),
            UnsignedPhysicalRect::new((18, 14).into(), (26, 20).into()),
            PhysicalRect::new((31, 21).into(), (43, 33).into()),
        ),
    ] {
        if scale != 1.0 {
            surface.resize(size);
            scene.change_scale_factor(scale).unwrap();
        }
        scene.render(&mut surface).unwrap();
        if scale == 1.0 {
            assert_eq!(
                scene.backend.root_scissor,
                Some(UnsignedPhysicalRect::new((9, 7).into(), (22, 17).into()))
            );
        }
        assert_eq!(
            scene
                .state
                .draw_tree
                .get(path)
                .unwrap()
                .logical_screen_bounds(),
            MathRect::new((16.25, 11.0).into(), (21.0, 16.0).into())
        );
        let commands: Vec<_> = scene.planner.plan(&scene.state, 4096).collect();
        let capture = commands.iter().find_map(|command| match command.operation {
            RenderOperation::Capture(capture) => Some(capture.region.bounds),
            _ => None,
        });
        assert_eq!(capture, Some(expected_capture));
        for (node_id, expected_clip) in [(path, expected_clip), (cached, expected_cached_clip)] {
            let clip = commands.iter().find_map(|command| match command.operation {
                RenderOperation::DrawShape(draw)
                | RenderOperation::DrawShapeAndIncrementStencil(draw)
                    if draw.id.0 == node_id =>
                {
                    Some(command.clip.scissor)
                }
                _ => None,
            });
            assert_eq!(clip, Some(expected_clip));
        }
    }

    scene.remove_subtrees([cached], |_| {});
    scene.render(&mut surface).unwrap();
    assert_eq!(
        scene.backend.root_scissor,
        Some(UnsignedPhysicalRect::new((18, 14).into(), (26, 20).into()))
    );
    scene.remove_subtrees([branch], |_| {});
    scene.render(&mut surface).unwrap();
    assert_eq!(
        scene.backend.root_scissor,
        Some(UnsignedPhysicalRect::new((32, 22).into(), (42, 32).into()))
    );
}
