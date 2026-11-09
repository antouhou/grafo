use super::{CachedShapeHandle, LineCap, Shape, StrokeGeometryOptions};
use crate::core::util::ShapeResources;
use std::sync::Arc;

#[test]
fn tessellation_reuse_survives_queue_rebuilds_in_independent_caches() {
    let mut first = ShapeResources::new();
    let mut second = ShapeResources::new();
    let rectangle =
        Shape::rect([(0.0, 0.0), (20.0, 20.0)]).with_stroke(StrokeGeometryOptions::new(2.0));
    let triangle = Shape::builder()
        .begin((0.0, 0.0))
        .line_to((20.0, 0.0))
        .line_to((0.0, 20.0))
        .close()
        .build();
    let cached_rect = CachedShapeHandle::new(&rectangle, &mut first, Some(12)).unwrap();
    let cached_triangle = CachedShapeHandle::new(&triangle, &mut second, Some(12)).unwrap();
    assert!(!Arc::ptr_eq(
        &cached_rect.tessellation,
        &cached_triangle.tessellation
    ));
    let cached_stroke = cached_rect.stroke_tessellation.as_ref().unwrap();
    assert!(!Arc::ptr_eq(&cached_rect.tessellation, cached_stroke));
    assert!(cached_triangle.stroke_tessellation.is_none());
    for _ in 0..4 {
        first.tessellation_cache.end_frame();
        second.tessellation_cache.end_frame();
        let rebuilt_rect = CachedShapeHandle::new(&rectangle, &mut first, Some(12)).unwrap();
        let rebuilt_triangle = CachedShapeHandle::new(&triangle, &mut second, Some(12)).unwrap();
        assert!(Arc::ptr_eq(
            &cached_rect.tessellation,
            &rebuilt_rect.tessellation
        ));
        assert!(Arc::ptr_eq(
            cached_stroke,
            rebuilt_rect.stroke_tessellation.as_ref().unwrap()
        ));
        assert!(Arc::ptr_eq(
            &cached_triangle.tessellation,
            &rebuilt_triangle.tessellation
        ));
    }
}

#[test]
fn stroke_tessellation_preserves_bounds_for_wide_strokes() {
    let mut resources = ShapeResources::new();
    let rectangle =
        Shape::rect([(0.0, 0.0), (4.0, 4.0)]).with_stroke(StrokeGeometryOptions::new(10.0));
    let line = Shape::builder()
        .begin((0.0, 0.0))
        .line_to((20.0, 0.0))
        .end()
        .build()
        .with_stroke(StrokeGeometryOptions::new(1_000_000.0).line_cap(LineCap::Round));

    for (shape, expected_bounds) in [
        (rectangle, [(-5.0, -5.0), (9.0, 9.0)]),
        (line, [(-500_000.0, -500_000.0), (500_020.0, 500_000.0)]),
    ] {
        let cached = CachedShapeHandle::new(&shape, &mut resources, None).unwrap();
        let stroke = cached.stroke_tessellation.unwrap();
        assert_eq!(stroke.local_bounds, expected_bounds);
    }
}

#[test]
fn thin_strokes_keep_round_caps() {
    let shape = Shape::builder()
        .begin((0.0, 0.0))
        .line_to((10.0, 0.0))
        .end()
        .build()
        .with_stroke(StrokeGeometryOptions::new(0.1).line_cap(LineCap::Round));
    let cached_shape = CachedShapeHandle::new(&shape, &mut ShapeResources::new(), None).unwrap();
    let stroke = cached_shape.stroke_tessellation.unwrap();

    assert_eq!(stroke.local_bounds, [(-0.05, -0.05), (10.05, 0.05)]);
}

#[test]
fn short_segments_keep_square_end_caps() {
    let shape = Shape::builder()
        .begin((0.0, 0.0))
        .line_to((0.04, 0.0))
        .end()
        .build()
        .with_stroke(StrokeGeometryOptions::new(1.0).end_cap(LineCap::Square));
    let cached_shape = CachedShapeHandle::new(&shape, &mut ShapeResources::new(), None).unwrap();
    let stroke = cached_shape.stroke_tessellation.unwrap();

    assert_eq!(stroke.local_bounds, [(0.0, -0.5), (0.54, 0.5)]);
}
