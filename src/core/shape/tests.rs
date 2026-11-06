use super::{
    BorderRadii, FillRule, LineCap, LineJoin, Shape, ShapeTessellationError, StrokeOptions,
};
use crate::core::cache::CachedTessellation;
use crate::core::util::ShapeResources;
use lyon::math::Point;
use lyon::path::PathEvent;
use std::{f32::consts::PI, sync::Arc};

fn tessellate_shape(shape: &Shape) -> Arc<CachedTessellation> {
    shape.tessellate(&mut ShapeResources::new(), None).unwrap()
}

fn mesh_area(tessellation: &CachedTessellation) -> f32 {
    let buffers = &tessellation.vertex_buffers;
    buffers
        .indices
        .as_chunks::<3>()
        .0
        .iter()
        .map(|triangle| {
            let [origin, second_vertex, third_vertex] =
                triangle.map(|index| Point::from(buffers.vertices[index as usize].position));
            (second_vertex - origin).cross(third_vertex - origin).abs() * 0.5
        })
        .sum()
}

fn mesh_contains(tessellation: &CachedTessellation, position: (f32, f32)) -> bool {
    let position = Point::from(position);
    let buffers = &tessellation.vertex_buffers;
    buffers.indices.as_chunks::<3>().0.iter().any(|triangle| {
        let [first_vertex, second_vertex, third_vertex] =
            triangle.map(|index| Point::from(buffers.vertices[index as usize].position));
        let doubled_signed_area = (second_vertex - first_vertex).cross(third_vertex - first_vertex);
        if doubled_signed_area == 0.0 {
            return false;
        }
        let tolerance = doubled_signed_area.abs() * f32::EPSILON * 8.0;
        let edge_sides = [
            (second_vertex - first_vertex).cross(position - first_vertex),
            (third_vertex - second_vertex).cross(position - second_vertex),
            (first_vertex - third_vertex).cross(position - third_vertex),
        ];
        edge_sides.iter().all(|side| *side >= -tolerance)
            || edge_sides.iter().all(|side| *side <= tolerance)
    })
}

fn assert_bounds(tessellation: &CachedTessellation, expected: [(f32, f32); 2], tolerance: f32) {
    let actual = tessellation.local_bounds;
    for (actual, expected) in actual.into_iter().zip(expected) {
        assert!(
            (actual.0 - expected.0).abs() <= tolerance
                && (actual.1 - expected.1).abs() <= tolerance,
            "expected {expected:?}, got {actual:?}"
        );
    }
}

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

    let mut resources = ShapeResources::new();
    let tessellation = shape.tessellate(&mut resources, None).unwrap();
    assert!((mesh_area(&tessellation) - 16.0).abs() < 1e-6);
}

#[test]
fn stroke_caps_cover_the_line_and_extend_its_ends() {
    let line = Shape::builder()
        .begin((2.0, 3.0))
        .line_to((12.0, 3.0))
        .end()
        .build();
    for (cap, expected_bounds, expected_area, extends_ends, covers_corner) in [
        (LineCap::Butt, [(2.0, 1.0), (12.0, 5.0)], 40.0, false, false),
        (LineCap::Square, [(0.0, 1.0), (14.0, 5.0)], 56.0, true, true),
        (
            LineCap::Round,
            [(0.0, 1.0), (14.0, 5.0)],
            40.0 + 4.0 * PI,
            true,
            false,
        ),
    ] {
        let stroke = line.clone().stroke(StrokeOptions::new(4.0).line_cap(cap));
        let tessellation = tessellate_shape(&stroke);
        assert_bounds(&tessellation, expected_bounds, 0.1);
        assert!((mesh_area(&tessellation) - expected_area).abs() < 0.6);
        assert!(mesh_contains(&tessellation, (7.0, 3.0)));
        assert!(!mesh_contains(&tessellation, (7.0, 5.1)));
        assert_eq!(mesh_contains(&tessellation, (1.0, 3.0)), extends_ends);
        assert_eq!(mesh_contains(&tessellation, (13.0, 3.0)), extends_ends);
        assert_eq!(mesh_contains(&tessellation, (0.25, 1.25)), covers_corner);
        assert!(!tessellation.boundary_edges.is_empty());
    }
}

#[test]
fn overlapping_strokes_tessellate_the_union_and_preserve_holes() {
    let crossing = Shape::builder()
        .begin((-5.0, 0.0))
        .line_to((5.0, 0.0))
        .end()
        .begin((0.0, -5.0))
        .line_to((0.0, 5.0))
        .end()
        .build();
    let retraced = Shape::builder()
        .fill_rule(FillRule::EvenOdd)
        .begin((-5.0, 0.0))
        .line_to((5.0, 0.0))
        .end()
        .begin((5.0, 0.0))
        .line_to((-5.0, 0.0))
        .end()
        .build();
    let self_crossing = Shape::builder()
        .begin((-5.0, 0.0))
        .line_to((5.0, 0.0))
        .line_to((5.0, 5.0))
        .line_to((0.0, 5.0))
        .line_to((0.0, -5.0))
        .end()
        .build();
    let opposite_rings = Shape::builder()
        .fill_rule(FillRule::EvenOdd)
        .begin((0.0, 0.0))
        .line_to((10.0, 0.0))
        .line_to((10.0, 10.0))
        .line_to((0.0, 10.0))
        .close()
        .begin((0.0, 0.0))
        .line_to((0.0, 10.0))
        .line_to((10.0, 10.0))
        .line_to((10.0, 0.0))
        .close()
        .build();
    for (name, shape, expected_area, inside, outside) in [
        (
            "crossing",
            crossing,
            36.0,
            [(0.0, 0.0), (4.0, 0.0), (0.0, 4.0)],
            (2.0, 2.0),
        ),
        (
            "retraced",
            retraced,
            20.0,
            [(0.0, 0.0), (4.0, 0.0), (-4.0, 0.0)],
            (0.0, 2.0),
        ),
        (
            "self crossing",
            self_crossing,
            56.0,
            [(0.0, 0.0), (5.0, 3.0), (3.0, 5.0)],
            (2.5, 2.5),
        ),
        (
            "opposite rings",
            opposite_rings,
            80.0,
            [(0.0, 5.0), (10.0, 5.0), (5.0, 0.0)],
            (5.0, 5.0),
        ),
    ] {
        let tessellation = tessellate_shape(&shape.stroke(StrokeOptions::new(2.0)));
        let area = mesh_area(&tessellation);
        assert!(
            (area - expected_area).abs() < 1e-4,
            "{name}: expected area {expected_area}, got {area}"
        );
        for position in inside {
            assert!(
                mesh_contains(&tessellation, position),
                "{name}: {position:?}"
            );
        }
        assert!(
            !mesh_contains(&tessellation, outside),
            "{name}: {outside:?}"
        );
    }
}

#[test]
fn thin_diagonal_crossings_preserve_union_area_at_small_scales() {
    for (scale, translation) in [
        (1.0, (0.0, 0.0)),
        (0.001, (0.0, 0.0)),
        (0.001, (0.25, -0.5)),
    ] {
        let extent = 5.0 * scale;
        let width = 0.02 * scale;
        let translate = |(x, y)| (x + translation.0, y + translation.1);
        let crossing = Shape::builder()
            .begin(translate((-extent, -extent)))
            .line_to(translate((extent, extent)))
            .end()
            .begin(translate((-extent, extent)))
            .line_to(translate((extent, -extent)))
            .end()
            .build()
            .stroke(StrokeOptions::new(width));
        let tessellation = tessellate_shape(&crossing);
        let expected_area = 4.0 * extent * 2.0_f32.sqrt() * width - width * width;
        let area = mesh_area(&tessellation);
        let relative_tolerance = if translation == (0.0, 0.0) {
            0.001
        } else {
            // Translation leaves fewer significant bits for the tiny stroke width.
            0.005
        };
        assert!(
            (area - expected_area).abs() / expected_area < relative_tolerance,
            "width {width}, translation {translation:?}: expected area {expected_area}, got {area}"
        );
        for position in [
            (0.0, 0.0),
            (3.0 * scale, 3.0 * scale),
            (3.0 * scale, -3.0 * scale),
        ] {
            assert!(
                mesh_contains(&tessellation, translate(position)),
                "width {width}, translation {translation:?}: {position:?}"
            );
        }
        assert!(!mesh_contains(&tessellation, translate((0.0, 3.0 * scale))));
    }
}

#[test]
fn long_thin_butt_strokes_preserve_area_and_both_ends() {
    for (length, width, is_vertical) in [
        (1_000.0, 0.02, false),
        (1_000.0, 0.02, true),
        (100_000.0, 2.0, false),
        (100_000.0, 2.0, true),
    ] {
        let position = |distance| {
            if is_vertical {
                (0.0, distance)
            } else {
                (distance, 0.0)
            }
        };
        let stroke = Shape::builder()
            .begin(position(0.0))
            .line_to(position(length))
            .end()
            .build()
            .stroke(StrokeOptions::new(width));
        let tessellation = tessellate_shape(&stroke);
        let expected_area = length * width;
        let area = mesh_area(&tessellation);
        assert!(
            (area - expected_area).abs() / expected_area < 0.0001,
            "length {length}, width {width}, vertical {is_vertical}: expected area {expected_area}, got {area}"
        );
        for distance in [width, length * 0.5, length - width] {
            assert!(
                mesh_contains(&tessellation, position(distance)),
                "length {length}, width {width}, vertical {is_vertical}: distance {distance}"
            );
        }
        for distance in [-width, length + width] {
            assert!(!mesh_contains(&tessellation, position(distance)));
        }
    }
}

#[test]
fn distant_thin_stroke_subpaths_preserve_nearby_geometry() {
    let stroke = Shape::builder()
        .begin((0.0, 0.0))
        .line_to((0.0, 10.0))
        .end()
        .begin((10_000_000.0, 0.0))
        .line_to((10_000_016.0, 0.0))
        .end()
        .build()
        .stroke(StrokeOptions::new(0.02));
    let tessellation = tessellate_shape(&stroke);
    let area = mesh_area(&tessellation);
    assert!(
        (area - 0.52).abs() < 0.0001,
        "expected area 0.52, got {area}"
    );
    for position in [(0.0, 0.01), (0.0, 5.0), (0.0, 9.99), (10_000_008.0, 0.0)] {
        assert!(mesh_contains(&tessellation, position), "{position:?}");
    }
}

#[test]
fn many_stroke_subpaths_preserve_a_long_thin_segment() {
    let mut builder = Shape::builder();
    for index in 0..8_193 {
        let (start, end) = if index == 4_096 {
            ((0.0, 0.0), (0.0, 1_000.0))
        } else {
            ((10.0, -20.0), (20.0, -20.0))
        };
        builder = builder.begin(start).line_to(end).end();
    }
    let tessellation = tessellate_shape(&builder.build().stroke(StrokeOptions::new(0.02)));
    let area = mesh_area(&tessellation);
    assert!(
        (area - 20.2).abs() < 0.0001,
        "expected area 20.2, got {area}"
    );
    for position in [(0.0, 0.02), (0.0, 500.0), (0.0, 999.98), (15.0, -20.0)] {
        assert!(mesh_contains(&tessellation, position), "{position:?}");
    }
    assert!(!mesh_contains(&tessellation, (0.02, 500.0)));
}

#[test]
fn stroke_complexity_limit_preserves_valid_subsequent_tessellation() {
    let mut resources = ShapeResources::new();
    for subpath_count in [16_383, 32_769, 1] {
        let mut builder = Shape::builder();
        for _ in 0..subpath_count {
            builder = builder.begin((0.0, 0.0)).line_to((10.0, 0.0)).end();
        }
        let result = builder
            .build()
            .stroke(StrokeOptions::new(2.0))
            .tessellate(&mut resources, None);
        if subpath_count == 32_769 {
            assert!(matches!(
                result,
                Err(ShapeTessellationError::StrokeTooComplex {
                    triangle_count: 65_538,
                    maximum_triangles: 32_767,
                })
            ));
            continue;
        }
        let tessellation = result.unwrap();
        assert!((mesh_area(&tessellation) - 20.0).abs() < 0.0001);
        for position in [(0.01, 0.0), (5.0, 0.0), (9.99, 0.0)] {
            assert!(mesh_contains(&tessellation, position));
        }
        assert!(!mesh_contains(&tessellation, (5.0, 2.0)));
    }
}

#[test]
fn wide_butt_strokes_preserve_short_segments() {
    let stroke = Shape::builder()
        .begin((0.0, 0.0))
        .line_to((2.0, 0.0))
        .end()
        .build()
        .stroke(StrokeOptions::new(100_000_000.0));
    let tessellation = tessellate_shape(&stroke);
    assert!(mesh_contains(&tessellation, (1.0, 0.0)));
    assert_bounds(
        &tessellation,
        [(0.0, -50_000_000.0), (2.0, 50_000_000.0)],
        0.1,
    );
    assert!((mesh_area(&tessellation) - 200_000_000.0).abs() < 1.0);
}

#[test]
fn wide_round_caps_produce_finite_geometry_and_cover_both_ends() {
    let stroke = Shape::builder()
        .begin((0.0, 0.0))
        .line_to((40_000_000.0, 0.0))
        .end()
        .build()
        .stroke(StrokeOptions::new(20_000_000.0).line_cap(LineCap::Round));
    let tessellation = tessellate_shape(&stroke);
    assert!(!tessellation.vertex_buffers.indices.is_empty());
    assert!(tessellation
        .vertex_buffers
        .vertices
        .iter()
        .all(|vertex| vertex
            .position
            .into_iter()
            .chain(vertex.tex_coords)
            .all(f32::is_finite)));
    assert_bounds(
        &tessellation,
        [(-10_000_000.0, -10_000_000.0), (50_000_000.0, 10_000_000.0)],
        10.0,
    );
    assert!(mesh_contains(&tessellation, (-9_000_000.0, 0.0)));
    assert!(mesh_contains(&tessellation, (49_000_000.0, 0.0)));
}

#[test]
fn stroke_joins_use_corner_tip_distance_for_the_miter_limit() {
    let corner = Shape::builder()
        .begin((0.0, 0.0))
        .line_to((10.0, 0.0))
        .line_to((10.0, 10.0))
        .end()
        .build();
    for (join, miter_limit, covers_tip, covers_arc) in [
        (LineJoin::Miter, 1.0, false, false),
        (LineJoin::Miter, 1.4, false, false),
        (LineJoin::Miter, 1.5, true, true),
        (LineJoin::Miter, 4.0, true, true),
        (LineJoin::Bevel, 4.0, false, false),
        (LineJoin::Round, 4.0, false, true),
    ] {
        let stroke = corner.clone().stroke(
            StrokeOptions::new(2.0)
                .line_join(join)
                .miter_limit(miter_limit),
        );
        let tessellation = tessellate_shape(&stroke);
        assert!(mesh_contains(&tessellation, (5.0, 0.0)));
        assert!(mesh_contains(&tessellation, (10.0, 5.0)));
        assert_eq!(
            mesh_contains(&tessellation, (10.8, -0.8)),
            covers_tip,
            "join {join:?}, miter limit {miter_limit}"
        );
        assert_eq!(
            mesh_contains(&tessellation, (10.65, -0.65)),
            covers_arc,
            "join {join:?}, miter limit {miter_limit}"
        );
    }
}

#[test]
fn stroked_rectangles_keep_empty_interiors_without_fill_shortcuts() {
    for shape in [
        Shape::rect([(0.0, 0.0), (20.0, 10.0)]),
        Shape::rounded_rect([(0.0, 0.0), (20.0, 10.0)], BorderRadii::new(3.0)),
    ] {
        let butt = tessellate_shape(&shape.clone().stroke(StrokeOptions::new(2.0)));
        let square =
            tessellate_shape(&shape.stroke(StrokeOptions::new(2.0).line_cap(LineCap::Square)));
        for tessellation in [&butt, &square] {
            assert!(!tessellation.is_rect);
            assert!(tessellation.rounded_rect_boundary.is_none());
            assert_bounds(tessellation, [(-1.0, -1.0), (21.0, 11.0)], 0.1);
            assert!(mesh_contains(tessellation, (10.0, 0.0)));
            assert!(mesh_contains(tessellation, (0.0, 5.0)));
            assert!(!mesh_contains(tessellation, (10.0, 5.0)));
        }
        assert!((mesh_area(&butt) - mesh_area(&square)).abs() < 1e-5);
    }
}

#[test]
fn strokes_follow_curves_and_preserve_open_and_closed_subpaths() {
    let shape = Shape::builder()
        .begin((0.0, 0.0))
        .quadratic_bezier_to((10.0, 20.0), (20.0, 0.0))
        .cubic_bezier_to((25.0, -10.0), (35.0, -10.0), (40.0, 0.0))
        .end()
        .begin((50.0, 0.0))
        .line_to((60.0, 0.0))
        .line_to((60.0, 10.0))
        .line_to((50.0, 10.0))
        .close()
        .build()
        .stroke(StrokeOptions::new(2.0).line_cap(LineCap::Round));
    let tessellation = tessellate_shape(&shape);
    for position in [(10.0, 10.0), (30.0, -7.5), (50.0, 5.0), (55.0, 0.0)] {
        assert!(mesh_contains(&tessellation, position), "{position:?}");
    }
    for position in [(10.0, 3.0), (30.0, -3.0), (55.0, 5.0), (45.0, 0.0)] {
        assert!(!mesh_contains(&tessellation, position), "{position:?}");
    }
    assert!(tessellation
        .vertex_buffers
        .vertices
        .iter()
        .all(|vertex| vertex
            .position
            .into_iter()
            .chain(vertex.tex_coords)
            .all(f32::is_finite)));
}

#[test]
fn zero_width_and_degenerate_strokes_respect_cap_geometry() {
    let point = Shape::builder()
        .begin((5.0, 5.0))
        .line_to((5.0, 5.0))
        .line_to((5.0, 5.0))
        .end()
        .build();
    for cap in [LineCap::Butt, LineCap::Square, LineCap::Round] {
        for shape in [point.clone(), Shape::rect([(0.0, 0.0), (20.0, 10.0)])] {
            let tessellation =
                tessellate_shape(&shape.stroke(StrokeOptions::new(0.0).line_cap(cap)));
            assert!(tessellation.vertex_buffers.indices.is_empty());
            assert!(tessellation.boundary_edges.is_empty());
        }
        let tessellation =
            tessellate_shape(&point.clone().stroke(StrokeOptions::new(4.0).line_cap(cap)));
        if cap == LineCap::Butt {
            assert!(tessellation.vertex_buffers.indices.is_empty());
        } else {
            assert_bounds(&tessellation, [(3.0, 3.0), (7.0, 7.0)], 0.1);
            assert!(mesh_contains(&tessellation, (5.0, 5.0)));
            assert_eq!(
                mesh_contains(&tessellation, (3.25, 3.25)),
                cap == LineCap::Square
            );
        }
        let move_only = Shape::builder()
            .begin((5.0, 5.0))
            .end()
            .build()
            .stroke(StrokeOptions::new(4.0).line_cap(cap));
        assert!(tessellate_shape(&move_only)
            .vertex_buffers
            .indices
            .is_empty());
    }
}

#[test]
fn degenerate_closed_strokes_keep_their_cap_fallbacks() {
    let closed_point = Shape::builder().begin((5.0, 5.0)).close().build();
    let closed_segment = Shape::builder()
        .begin((5.0, 5.0))
        .line_to((15.0, 5.0))
        .close()
        .build();
    for cap in [LineCap::Butt, LineCap::Square, LineCap::Round] {
        let options = StrokeOptions::new(4.0).line_cap(cap);
        let cap_area = match cap {
            LineCap::Butt => 0.0,
            LineCap::Square => 16.0,
            LineCap::Round => 4.0 * PI,
        };
        for point_shape in [closed_point.clone(), Shape::rect([(5.0, 5.0), (5.0, 5.0)])] {
            let tessellation = tessellate_shape(&point_shape.stroke(options));
            assert!((mesh_area(&tessellation) - cap_area).abs() < 0.6);
            if cap == LineCap::Butt {
                assert!(tessellation.vertex_buffers.indices.is_empty());
                assert!(tessellation.boundary_edges.is_empty());
            } else {
                assert_bounds(&tessellation, [(3.0, 3.0), (7.0, 7.0)], 0.1);
                assert!(mesh_contains(&tessellation, (5.0, 5.0)));
                assert_eq!(
                    mesh_contains(&tessellation, (3.25, 3.25)),
                    cap == LineCap::Square
                );
            }
        }
        for (segment_shape, center, before_start, after_end, butt_bounds, capped_bounds) in [
            (
                closed_segment.clone(),
                (10.0, 5.0),
                (4.0, 5.0),
                (16.0, 5.0),
                [(5.0, 3.0), (15.0, 7.0)],
                [(3.0, 3.0), (17.0, 7.0)],
            ),
            (
                Shape::rect([(5.0, 5.0), (5.0, 15.0)]),
                (5.0, 10.0),
                (5.0, 4.0),
                (5.0, 16.0),
                [(3.0, 5.0), (7.0, 15.0)],
                [(3.0, 3.0), (7.0, 17.0)],
            ),
        ] {
            let tessellation = tessellate_shape(&segment_shape.stroke(options));
            let bounds = if cap == LineCap::Butt {
                butt_bounds
            } else {
                capped_bounds
            };
            assert_bounds(&tessellation, bounds, 0.1);
            assert!((mesh_area(&tessellation) - 40.0 - cap_area).abs() < 0.6);
            assert!(mesh_contains(&tessellation, center));
            assert_eq!(
                mesh_contains(&tessellation, before_start),
                cap != LineCap::Butt
            );
            assert_eq!(
                mesh_contains(&tessellation, after_end),
                cap != LineCap::Butt
            );
        }
    }
}

#[test]
fn stroke_geometry_reuses_its_cache_entry_and_keeps_distinct_widths() {
    let path = Shape::builder()
        .begin((0.0, 0.0))
        .line_to((10.0, 0.0))
        .end()
        .build();
    let thin = path.clone().stroke(StrokeOptions::new(2.0));
    let thick = path.stroke(StrokeOptions::new(4.0));
    let mut resources = ShapeResources::new();
    let thin_geometry = thin.tessellate(&mut resources, Some(21)).unwrap();
    let thick_geometry = thick.tessellate(&mut resources, Some(22)).unwrap();
    assert!(!Arc::ptr_eq(&thin_geometry, &thick_geometry));
    assert!(!mesh_contains(&thin_geometry, (5.0, 1.5)));
    assert!(mesh_contains(&thick_geometry, (5.0, 1.5)));
    for _ in 0..3 {
        resources.tessellation_cache.end_frame();
        for (shape, cache_key, cached) in
            [(&thin, 21, &thin_geometry), (&thick, 22, &thick_geometry)]
        {
            let reused = shape.tessellate(&mut resources, Some(cache_key)).unwrap();
            assert!(Arc::ptr_eq(&reused, cached));
        }
    }
}

#[test]
fn tessellation_reuse_survives_queue_rebuilds_in_independent_caches() {
    let mut first = ShapeResources::new();
    let mut second = ShapeResources::new();
    let rectangle = Shape::rect([(0.0, 0.0), (20.0, 20.0)]);
    let triangle = Shape::builder()
        .begin((0.0, 0.0))
        .line_to((20.0, 0.0))
        .line_to((0.0, 20.0))
        .close()
        .build();
    let cached_rect = rectangle.tessellate(&mut first, Some(12)).unwrap();
    let cached_triangle = triangle.tessellate(&mut second, Some(12)).unwrap();
    assert!(!Arc::ptr_eq(&cached_rect, &cached_triangle));
    for _ in 0..4 {
        first.tessellation_cache.end_frame();
        second.tessellation_cache.end_frame();
        let rebuilt_rect = rectangle.tessellate(&mut first, Some(12)).unwrap();
        let rebuilt_triangle = triangle.tessellate(&mut second, Some(12)).unwrap();
        assert!(Arc::ptr_eq(&cached_rect, &rebuilt_rect));
        assert!(Arc::ptr_eq(&cached_triangle, &rebuilt_triangle));
    }
}
