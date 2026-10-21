use super::Shape;
use crate::core::util::ShapeResources;
use lyon::tessellation::FillTessellator;
use std::sync::Arc;

#[test]
fn tessellation_reuse_survives_queue_rebuilds_in_independent_caches() {
    let mut first = ShapeResources::new();
    let mut second = ShapeResources::new();
    let mut tessellator = FillTessellator::new();
    let rectangle = Shape::rect([(0.0, 0.0), (20.0, 20.0)]);
    let triangle = Shape::builder()
        .begin((0.0, 0.0))
        .line_to((20.0, 0.0))
        .line_to((0.0, 20.0))
        .close()
        .build();
    let cached_rect = rectangle.tessellate(&mut tessellator, &mut first, Some(12));
    let cached_triangle = triangle.tessellate(&mut tessellator, &mut second, Some(12));
    assert!(!Arc::ptr_eq(&cached_rect, &cached_triangle));
    for _ in 0..4 {
        first.tessellation_cache.end_frame();
        second.tessellation_cache.end_frame();
        let rebuilt_rect = rectangle.tessellate(&mut tessellator, &mut first, Some(12));
        let rebuilt_triangle = triangle.tessellate(&mut tessellator, &mut second, Some(12));
        assert!(Arc::ptr_eq(&cached_rect, &rebuilt_rect));
        assert!(Arc::ptr_eq(
            &cached_rect.vertex_buffers,
            &rebuilt_rect.vertex_buffers
        ));
        assert!(Arc::ptr_eq(&cached_triangle, &rebuilt_triangle));
        assert!(Arc::ptr_eq(
            &cached_triangle.vertex_buffers,
            &rebuilt_triangle.vertex_buffers
        ));
    }
}
