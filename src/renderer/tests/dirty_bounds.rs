use super::{renderer, surface};
use crate::core::vertex::InstanceTransform;
use crate::core::{Color, Shape, ShapeDrawCommandOptions, UnsignedPhysicalRect};

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
