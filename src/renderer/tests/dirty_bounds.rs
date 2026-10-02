use super::{renderer, surface};
use crate::commands::RenderOperation;
use crate::core::vertex::InstanceTransform;
use crate::core::{
    BackdropEffectConfig, Color, MathRect, PhysicalRect, Shape, ShapeDrawCommandOptions,
    UnsignedPhysicalRect,
};

#[test]
fn subtree_replacement_accumulates_bounds_until_a_successful_render() {
    let mut renderer = renderer();
    let mut surface = surface();
    renderer.change_scale_factor(2.0);
    renderer.render(&mut surface).unwrap();
    assert!(!renderer.is_dirty_region_overlay_enabled());
    renderer.set_dirty_region_overlay_enabled(true);
    renderer.render(&mut surface).unwrap();
    assert_eq!(renderer.backend.root_scissor, None);
    let branch = renderer
        .add_clipping_rect(
            [(0.0, 0.0), (16.0, 16.0)],
            None,
            None::<InstanceTransform>,
            false,
        )
        .unwrap();
    renderer.load_shape(Shape::rect([(0.0, 0.0), (2.0, 2.0)]), 17, None);
    let first = renderer
        .add_cached_shape(
            17,
            Some(branch),
            ShapeDrawCommandOptions::new()
                .color(Color::WHITE)
                .transform(InstanceTransform::translation(2.0, 3.0)),
        )
        .unwrap();
    renderer
        .add_shape(
            Shape::rect([(10.0, 11.0), (12.0, 13.0)]),
            Some(branch),
            None,
            ShapeDrawCommandOptions::new().color(Color::WHITE),
        )
        .unwrap();
    renderer.render(&mut surface).unwrap();
    let bounds = UnsignedPhysicalRect::new((3, 5).into(), (25, 27).into());
    assert_eq!(renderer.backend.root_scissor, Some(bounds));
    renderer.render(&mut surface).unwrap();
    assert_eq!(renderer.backend.root_scissor, None);

    renderer.remove_subtrees([branch, first, branch], |_| {});
    renderer
        .add_shape(
            Shape::rect([(14.0, 1.0), (18.0, 3.0)]),
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::WHITE),
        )
        .unwrap();
    renderer.backend.should_fail = true;
    assert!(renderer.render(&mut surface).is_err());
    renderer.backend.should_fail = false;
    renderer.render(&mut surface).unwrap();
    assert!(renderer.is_dirty_region_overlay_enabled());
    assert_eq!(
        renderer.backend.root_scissor,
        Some(UnsignedPhysicalRect::new((3, 1).into(), (32, 27).into()))
    );
    renderer.render(&mut surface).unwrap();
    assert_eq!(renderer.backend.root_scissor, None);

    renderer.clear_draw_queue();
    renderer.render(&mut surface).unwrap();
    assert_eq!(
        renderer.backend.root_scissor,
        Some(UnsignedPhysicalRect::new((27, 1).into(), (32, 7).into()))
    );
    renderer.remove_subtrees([usize::MAX], |_| panic!("missing node"));
    renderer.set_dirty_region_overlay_enabled(false);
    renderer.render(&mut surface).unwrap();
    assert_eq!(renderer.backend.root_scissor, None);
    assert!(!renderer.is_dirty_region_overlay_enabled());
}

#[test]
fn retained_node_bounds_follow_viewport_changes_and_removal() {
    let mut renderer = renderer();
    let mut surface = surface();
    renderer.render(&mut surface).unwrap();
    let branch = renderer
        .add_clipping_rect(
            [(0.0, 0.0), (20.0, 20.0)],
            None,
            Some(InstanceTransform::translation(5.0, 4.0)),
            true,
        )
        .unwrap();
    renderer.load_shape(Shape::rect([(0.0, 0.0), (4.0, 3.0)]), 17, None);
    let cached = renderer
        .add_cached_shape(
            17,
            Some(branch),
            ShapeDrawCommandOptions::new()
                .color(Color::WHITE)
                .transform(InstanceTransform::affine_2d(-1.0, 0.0, 0.0, 1.0, 13.0, 7.0)),
        )
        .unwrap();
    renderer
        .add_cached_shape(
            17,
            Some(cached),
            ShapeDrawCommandOptions::new()
                .color(Color::WHITE)
                .transform(InstanceTransform::translation(9.0, 7.0)),
        )
        .unwrap();
    let path = renderer
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
    renderer.load_effect(7, &["backdrop effect"]).unwrap();
    renderer
        .set_shape_backdrop_effect(
            path,
            7,
            &[1, 2, 3, 4],
            BackdropEffectConfig::new().padding(0.5),
        )
        .unwrap();

    for (scale, size, fringe, expected_clip, expected_cached_clip, expected_capture) in [
        (
            1.0,
            (32, 32),
            0.75,
            UnsignedPhysicalRect::new((5, 4).into(), (25, 24).into()),
            UnsignedPhysicalRect::new((9, 7).into(), (13, 10).into()),
            PhysicalRect::new((15, 10).into(), (22, 17).into()),
        ),
        (
            2.0,
            (64, 48),
            2.0,
            UnsignedPhysicalRect::new((10, 8).into(), (50, 48).into()),
            UnsignedPhysicalRect::new((18, 14).into(), (26, 20).into()),
            PhysicalRect::new((31, 21).into(), (43, 33).into()),
        ),
    ] {
        if scale != 1.0 {
            surface.resize(size);
            renderer.change_scale_factor(scale);
            renderer.set_fringe_width(fringe);
        }
        renderer.render(&mut surface).unwrap();
        if scale == 1.0 {
            assert_eq!(
                renderer.backend.root_scissor,
                Some(UnsignedPhysicalRect::new((8, 6).into(), (22, 17).into()))
            );
        }
        assert_eq!(
            renderer
                .scene
                .draw_tree
                .get(path)
                .unwrap()
                .logical_screen_bounds(),
            MathRect::new((16.25, 11.0).into(), (21.0, 16.0).into())
        );
        let commands = renderer.planner.plan(
            &renderer.scene,
            renderer.viewport,
            renderer.fringe_width,
            4096,
            None,
        );
        let capture = commands
            .instructions
            .iter()
            .find_map(|command| match command.operation {
                RenderOperation::CaptureBackdrop(capture) => Some(capture.region.bounds),
                _ => None,
            });
        assert_eq!(capture, Some(expected_capture));
        for (node_id, expected_clip) in [(path, expected_clip), (cached, expected_cached_clip)] {
            let clip = commands
                .instructions
                .iter()
                .find_map(|command| match command.operation {
                    RenderOperation::DrawShape(draw) if draw.id.0 == node_id => {
                        Some(command.clip.scissor)
                    }
                    _ => None,
                });
            assert_eq!(clip, Some(expected_clip));
        }
    }

    renderer.remove_subtrees([cached], |_| {});
    renderer.render(&mut surface).unwrap();
    assert_eq!(
        renderer.backend.root_scissor,
        Some(UnsignedPhysicalRect::new((16, 12).into(), (44, 34).into()))
    );
    renderer.remove_subtrees([branch], |_| {});
    renderer.render(&mut surface).unwrap();
    assert_eq!(
        renderer.backend.root_scissor,
        Some(UnsignedPhysicalRect::new((30, 20).into(), (44, 34).into()))
    );
}
