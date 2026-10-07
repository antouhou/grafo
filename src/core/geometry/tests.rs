use super::{
    logical_bounds_to_viewport_rect, logical_rect_to_physical_rect, logical_rect_to_scissor_rect,
    transformed_bounds_to_logical_screen_rect, unit_quad_transform,
};
use crate::core::{
    MathRect, PhysicalRect, Size, TransformInstance, UnsignedPhysicalRect, Viewport,
};
use lyon::geom::Point;

#[test]
fn transformed_screen_bounds_convert_to_scissor() {
    let transform = TransformInstance::affine_2d(2.0, 0.0, 0.0, -3.0, 10.0, 20.0);

    let scissor = logical_rect_to_scissor_rect(
        transformed_bounds_to_logical_screen_rect(
            MathRect::new(Point::new(0.0, 0.0), Point::new(10.0, 5.0)),
            Some(transform),
        ),
        1.0,
        Size::new(100, 100),
    );

    assert_eq!(
        scissor,
        Some(UnsignedPhysicalRect::new(
            Point::new(10, 5),
            Point::new(30, 20)
        ))
    );
}

#[test]
fn physical_capture_rect_preserves_requested_size_outside_viewport() {
    let requested_rect = logical_rect_to_physical_rect(
        MathRect::new(Point::new(-10.0, 5.0), Point::new(30.0, 25.0)),
        1.0,
    )
    .expect("capture rect should be non-empty");

    assert_eq!(
        requested_rect,
        PhysicalRect::new(Point::new(-10, 5), Point::new(30, 25))
    );
}

#[test]
fn transformed_screen_bounds_match_negative_w_shader_projection() {
    let transform = TransformInstance {
        col0: [2.0, 0.0, 0.0, 0.0],
        col1: [0.0, 3.0, 0.0, 0.0],
        col2: [0.0, 0.0, 1.0, 0.0],
        col3: [0.0, 0.0, 0.0, -2.0],
    };

    assert_eq!(
        transformed_bounds_to_logical_screen_rect(
            MathRect::new((1.0, 1.0).into(), (2.0, 2.0).into()),
            Some(transform),
        ),
        MathRect::new((1.0, 1.5).into(), (2.0, 3.0).into())
    );
}

#[test]
fn projection_bounds_fall_back_when_the_divisor_crosses_clamp_regions() {
    let viewport = Viewport {
        physical_size: (32, 24),
        scale_factor: 1.0,
    };
    for (horizontal_w, origin_w) in [(0.75e-6, 0.5e-6), (-0.75e-6, -0.5e-6), (1.0, -1.0)] {
        let mut transform = TransformInstance::identity();
        transform.col0[3] = horizontal_w;
        transform.col3[3] = origin_w;
        let bounds = transformed_bounds_to_logical_screen_rect(
            MathRect::new((0.0, 0.0).into(), (2.0, 2.0).into()),
            Some(transform),
        );
        assert!(!bounds.is_finite(), "transform: {transform:?}");
        assert_eq!(
            logical_bounds_to_viewport_rect(bounds, viewport, 0.75),
            Some(UnsignedPhysicalRect::from_size((32, 24).into()))
        );
    }
}

#[test]
fn projection_bounds_keep_constant_clamp_and_negative_divisor_regions() {
    for horizontal_w in [0.0, 0.5e-6] {
        let mut transform = TransformInstance::scale(1e-6, 1e-6);
        transform.col0[3] = horizontal_w;
        transform.col3[3] = -0.5e-6;
        assert_eq!(
            transformed_bounds_to_logical_screen_rect(
                MathRect::new((0.0, 0.0).into(), (2.0, 2.0).into()),
                Some(transform),
            ),
            MathRect::new((0.0, 0.0).into(), (2.0, 2.0).into())
        );
    }

    let mut transform = TransformInstance::identity();
    transform.col0[3] = -0.5;
    transform.col3[3] = -1.0;
    assert_eq!(
        transformed_bounds_to_logical_screen_rect(
            MathRect::new((0.0, 0.0).into(), (2.0, 2.0).into()),
            Some(transform),
        ),
        MathRect::new((0.0, 0.0).into(), (1.0, 2.0).into())
    );
}

#[test]
fn projection_bounds_fall_back_for_non_finite_homogeneous_coordinates() {
    for transform in [
        TransformInstance {
            col0: [1.0, 0.0, 0.0, f32::MAX],
            ..TransformInstance::identity()
        },
        TransformInstance {
            col0: [f32::MAX, 0.0, 0.0, 0.0],
            ..TransformInstance::identity()
        },
        TransformInstance {
            col3: [0.0, 0.0, 0.0, f32::NAN],
            ..TransformInstance::identity()
        },
    ] {
        let bounds = transformed_bounds_to_logical_screen_rect(
            MathRect::new((0.0, 0.0).into(), (2.0, 2.0).into()),
            Some(transform),
        );
        assert!(!bounds.is_finite(), "transform: {transform:?}");
    }
}

#[test]
fn physical_capture_rect_rejects_non_finite_coordinates() {
    let requested_rect = logical_rect_to_physical_rect(
        MathRect::new(Point::new(0.0, 0.0), Point::new(f32::INFINITY, 25.0)),
        1.0,
    );

    assert!(requested_rect.is_none());
}

#[test]
fn physical_capture_rect_rejects_unrepresentable_bounds_and_extents() {
    for bounds in [
        MathRect::new(Point::new(0.0, 0.0), Point::new(f32::MAX, 25.0)),
        MathRect::new(Point::new(i32::MIN as f32, 0.0), Point::new(1.0, 25.0)),
        MathRect::new(Point::new(i32::MAX as f32, 0.0), Point::new(f32::MAX, 25.0)),
    ] {
        assert!(logical_rect_to_physical_rect(bounds, 1.0).is_none());
    }
}

#[test]
fn scissor_conversion_clips_scaled_bounds_and_preserves_empty_regions() {
    let viewport_size = Size::new(100, 80);
    let transform = Some(TransformInstance::translation(-5.0, -10.0));
    let partially_visible = MathRect::new(Point::new(0.0, 0.0), Point::new(30.0, 25.0));
    assert_eq!(
        logical_rect_to_scissor_rect(
            transformed_bounds_to_logical_screen_rect(partially_visible, transform),
            2.0,
            viewport_size,
        ),
        Some(UnsignedPhysicalRect::new(
            Point::new(0, 0),
            Point::new(50, 30)
        ))
    );

    for bounds in [
        MathRect::new(Point::new(-30.0, -25.0), Point::new(0.0, 0.0)),
        MathRect::new(Point::new(60.0, 50.0), Point::new(80.0, 70.0)),
        MathRect::new(Point::new(20.0, 20.0), Point::new(20.0, 30.0)),
    ] {
        let scissor = logical_rect_to_scissor_rect(
            transformed_bounds_to_logical_screen_rect(bounds, transform),
            2.0,
            viewport_size,
        )
        .unwrap();
        assert!(scissor.is_empty());
        assert!(scissor.max.x <= viewport_size.width);
        assert!(scissor.max.y <= viewport_size.height);
    }
}

#[test]
fn transformed_capture_bounds_include_all_four_corners() {
    let transform = TransformInstance::affine_2d(1.0, 0.5, -0.25, 1.0, 12.0, -4.0);

    assert_eq!(
        transformed_bounds_to_logical_screen_rect(
            MathRect::new(Point::new(0.0, 0.0), Point::new(20.0, 10.0)),
            Some(transform)
        ),
        MathRect::new(Point::new(9.5, -4.0), Point::new(32.0, 16.0))
    );
}

#[test]
fn perspective_capture_bounds_include_projected_corners() {
    let mut transform = TransformInstance::identity();
    transform.col0[3] = 0.5;
    transform.col1[3] = 0.5;

    assert_eq!(
        transformed_bounds_to_logical_screen_rect(
            MathRect::new(Point::new(0.0, 0.0), Point::new(2.0, 2.0)),
            Some(transform),
        ),
        MathRect::new(Point::new(0.0, 0.0), Point::new(1.0, 1.0))
    );
}

#[test]
fn unit_quad_transform_maps_unit_quad_before_source_transform() {
    let transform = unit_quad_transform(
        [(-3.0, -4.0), (11.0, 15.0)],
        Some(TransformInstance::translation(5.0, 7.0)),
    );

    assert_eq!(transform.col0, [14.0, 0.0, 0.0, 0.0]);
    assert_eq!(transform.col1, [0.0, 19.0, 0.0, 0.0]);
    assert_eq!(transform.col3, [2.0, 3.0, 0.0, 1.0]);
}
