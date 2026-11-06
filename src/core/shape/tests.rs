use super::Shape;
use crate::core::util::ShapeResources;
use lyon::math::Point;
use lyon::path::PathEvent;
use lyon::tessellation::FillTessellator;
use std::sync::Arc;

#[test]
fn open_and_closed_subpaths_keep_their_ends_and_fill_separately() {
    let shape = Shape::builder()
        .begin((0.0, 0.0))
        .line_to((4.0, 0.0))
        .line_to((0.0, 3.0))
        .end()
        .begin((10.0, 0.0))
        .line_to((12.0, 0.0))
        .line_to((12.0, 2.0))
        .line_to((10.0, 2.0))
        .close()
        .begin((20.0, 0.0))
        .line_to((24.0, 0.0))
        .line_to((20.0, 3.0))
        .end()
        .build();
    let Shape::Path(path) = &shape else {
        panic!("expected a path shape");
    };
    let subpath_closed_flags: Vec<_> = path
        .path
        .iter()
        .filter_map(|event| match event {
            PathEvent::End { close, .. } => Some(close),
            _ => None,
        })
        .collect();
    assert_eq!(subpath_closed_flags, [false, true, false]);

    let mut tessellator = FillTessellator::new();
    let mut resources = ShapeResources::new();
    let tessellation = shape
        .tessellate(&mut tessellator, &mut resources, None)
        .unwrap();
    let buffers = &tessellation.vertex_buffers;
    let filled_area: f32 = buffers
        .indices
        .as_chunks::<3>()
        .0
        .iter()
        .map(|triangle| {
            let [origin, second_vertex, third_vertex] =
                triangle.map(|index| Point::from(buffers.vertices[index as usize].position));
            (second_vertex - origin).cross(third_vertex - origin).abs() * 0.5
        })
        .sum();
    assert!((filled_area - 16.0).abs() < 1e-6);
}

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
    let cached_rect = rectangle
        .tessellate(&mut tessellator, &mut first, Some(12))
        .unwrap();
    let cached_triangle = triangle
        .tessellate(&mut tessellator, &mut second, Some(12))
        .unwrap();
    assert!(!Arc::ptr_eq(&cached_rect, &cached_triangle));
    for _ in 0..4 {
        first.tessellation_cache.end_frame();
        second.tessellation_cache.end_frame();
        let rebuilt_rect = rectangle
            .tessellate(&mut tessellator, &mut first, Some(12))
            .unwrap();
        let rebuilt_triangle = triangle
            .tessellate(&mut tessellator, &mut second, Some(12))
            .unwrap();
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
