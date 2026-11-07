use super::{coverage_at, CoverageTestMesh};
use crate::core::cache::CachedTessellation;
use crate::core::shape::boundary::PathBoundaryEdge;
use crate::core::{FillRule, Shape, StrokeOptions, TransformInstance, Viewport};
use crate::scene_state::{SceneState, SceneStateContext};

fn signed_polygon_area(polygon: &[[f64; 2]]) -> f64 {
    let Some(origin) = polygon.first() else {
        return 0.0;
    };
    polygon[1..]
        .windows(2)
        .map(|pair| {
            (pair[0][0] - origin[0]) * (pair[1][1] - origin[1])
                - (pair[0][1] - origin[1]) * (pair[1][0] - origin[0])
        })
        .sum::<f64>()
        * 0.5
}

fn clip_to_edge(
    polygon: &[[f64; 2]],
    start: [f64; 2],
    end: [f64; 2],
    orientation: f64,
) -> Vec<[f64; 2]> {
    let side = |point: [f64; 2]| {
        ((end[0] - start[0]) * (point[1] - start[1]) - (end[1] - start[1]) * (point[0] - start[0]))
            * orientation
    };
    let mut clipped = Vec::new();
    for index in 0..polygon.len() {
        let first = polygon[index];
        let second = polygon[(index + 1) % polygon.len()];
        let first_side = side(first);
        let second_side = side(second);
        let first_inside = first_side >= 0.0;
        if first_inside {
            clipped.push(first);
        }
        if first_inside != (second_side >= 0.0) {
            let fraction = first_side / (first_side - second_side);
            clipped.push([
                first[0] + fraction * (second[0] - first[0]),
                first[1] + fraction * (second[1] - first[1]),
            ]);
        }
    }
    clipped
}

fn intersect_polygons(first: &[[f64; 2]], second: &[[f64; 2]]) -> Vec<[f64; 2]> {
    let area = signed_polygon_area(second);
    if area == 0.0 {
        return Vec::new();
    }
    let orientation = area.signum();
    let mut intersection = first.to_vec();
    for index in 0..second.len() {
        intersection = clip_to_edge(
            &intersection,
            second[index],
            second[(index + 1) % second.len()],
            orientation,
        );
    }
    intersection
}

fn polygon_union_pixel_area(first: &[[f64; 2]], second: &[[f64; 2]], x: u32, y: u32) -> f64 {
    let left = f64::from(x);
    let top = f64::from(y);
    let pixel = [
        [left, top],
        [left + 1.0, top],
        [left + 1.0, top + 1.0],
        [left, top + 1.0],
    ];
    let first = intersect_polygons(first, &pixel);
    let second = intersect_polygons(second, &pixel);
    let intersection = intersect_polygons(&first, &second);
    (signed_polygon_area(&first).abs() + signed_polygon_area(&second).abs()
        - signed_polygon_area(&intersection).abs())
    .clamp(0.0, 1.0)
}

fn project_polygon(
    polygon: &[[f64; 2]],
    transform: TransformInstance,
    scale: f64,
) -> Vec<[f64; 2]> {
    polygon
        .iter()
        .map(|&[x, y]| {
            let divisor = (x * f64::from(transform.col0[3])
                + y * f64::from(transform.col1[3])
                + f64::from(transform.col3[3]))
            .abs()
            .max(1e-6);
            [0, 1].map(|axis| {
                (x * f64::from(transform.col0[axis])
                    + y * f64::from(transform.col1[axis])
                    + f64::from(transform.col3[axis]))
                    * scale
                    / divisor
            })
        })
        .collect()
}

fn rectangle_pixel_area(bounds: [[f64; 2]; 2], x: u32, y: u32) -> f64 {
    [x, y]
        .into_iter()
        .enumerate()
        .map(|(axis, pixel)| {
            (bounds[1][axis].min(f64::from(pixel) + 1.0) - bounds[0][axis].max(f64::from(pixel)))
                .max(0.0)
        })
        .product()
}

fn butt_stroke_rectangle(start: [f32; 2], end: [f32; 2], width: f32) -> [[f64; 2]; 4] {
    let start = start.map(f64::from);
    let end = end.map(f64::from);
    let direction = [end[0] - start[0], end[1] - start[1]];
    let half_width = f64::from(width) * 0.5;
    let length = direction[0].hypot(direction[1]);
    let normal = [
        -direction[1] * half_width / length,
        direction[0] * half_width / length,
    ];
    [
        [start[0] + normal[0], start[1] + normal[1]],
        [end[0] + normal[0], end[1] + normal[1]],
        [end[0] - normal[0], end[1] - normal[1]],
        [start[0] - normal[0], start[1] - normal[1]],
    ]
}

#[test]
fn stroke_crossings_and_partial_retracing_cover_each_pixel_region_once() {
    let mut scene = SceneState::new(
        SceneStateContext::default(),
        Viewport {
            physical_size: (8, 8),
            scale_factor: 1.0,
        },
    );
    let crossing = Shape::builder()
        .begin((-2.0, 0.125))
        .line_to((2.0, 0.125))
        .end()
        .begin((0.125, -2.0))
        .line_to((0.125, 2.0))
        .end()
        .build()
        .stroke(StrokeOptions::new(0.25));
    let retraced = Shape::builder()
        .begin((-0.25, 0.5))
        .line_to((0.75, 0.5))
        .end()
        .begin((1.25, 0.5))
        .line_to((0.25, 0.5))
        .end()
        .begin((0.75, 0.5))
        .line_to((-0.25, 0.5))
        .end()
        .build()
        .stroke(StrokeOptions::new(0.5));
    let mut mesh = CoverageTestMesh::default();
    for (shape, expected) in [(crossing, 7.0 / 16.0), (retraced, 0.5)] {
        let cached = scene.tessellate(&shape, None).unwrap();
        for transform in [
            TransformInstance::identity(),
            TransformInstance::affine_2d(0.0, 1.0, -1.0, 0.0, 1.0, 0.0),
        ] {
            mesh.build(&cached.tessellation, Some(transform), 1.0, (8, 8));
            assert_eq!(coverage_at(&mesh, 0, 0), expected);
        }
    }
}

#[test]
fn crossing_a_very_long_stroke_preserves_fractional_coverage_after_rotation() {
    let mut scene = SceneState::new(
        SceneStateContext::default(),
        Viewport {
            physical_size: (4, 4),
            scale_factor: 1.0,
        },
    );
    let shape = Shape::builder()
        .begin((-1e17, 0.5))
        .line_to((1e17, 0.5))
        .end()
        .begin((0.5, -1.0))
        .line_to((0.5, 2.0))
        .end()
        .build()
        .stroke(StrokeOptions::new(0.5));
    let cached = scene.tessellate(&shape, None).unwrap();
    let mut mesh = CoverageTestMesh::default();
    for transform in [
        TransformInstance::identity(),
        TransformInstance::affine_2d(0.0, 1.0, -1.0, 0.0, 1.0, 0.0),
    ] {
        mesh.build(&cached.tessellation, Some(transform), 1.0, (4, 4));
        assert_eq!(coverage_at(&mesh, 0, 0), 0.75, "transform {transform:?}");
    }
}

#[test]
fn very_wide_slanted_contours_preserve_clipping_traversal_order() {
    let limit = 1e17;
    let corners = [[limit, 0.0], [limit, 1.0], [-limit, 2.0], [-limit, 1.0]];
    let mut tessellation = CachedTessellation::empty();
    tessellation.may_have_overlapping_triangles = true;
    tessellation.local_bounds = [(-limit, 0.0), (limit, 2.0)];
    for index in 0..corners.len() {
        tessellation.boundary_edges.push(PathBoundaryEdge {
            start: corners[index],
            end: corners[(index + 1) % corners.len()],
            winding: 1,
        });
    }
    let mut mesh = CoverageTestMesh::default();
    mesh.build(&tessellation, None, 1.0, (8, 2));
    // Each slanted side crosses this narrow viewport halfway through its row.
    for y in 0..2 {
        for x in 0..8 {
            assert_eq!(coverage_at(&mesh, x, y), 0.5, "pixel {x},{y}");
        }
    }
}

#[test]
fn overlapping_triangles_with_clipped_vertices_match_polygon_union() {
    let triangles = [
        [[1.375, 1.5], [-0.875, 0.25], [2.5, 1.75]],
        [[3.625, 0.25], [0.25, 0.75], [-1.0, -0.125]],
    ];
    let mut tessellation = CachedTessellation::empty();
    tessellation.may_have_overlapping_triangles = true;
    tessellation.local_bounds = [(-1.0, -0.25), (4.0, 2.25)];
    for triangle in triangles {
        for index in 0..3 {
            tessellation.boundary_edges.push(PathBoundaryEdge {
                start: triangle[index].map(|coordinate| coordinate as f32),
                end: triangle[(index + 1) % 3].map(|coordinate| coordinate as f32),
                winding: 1,
            });
        }
    }
    let mut mesh = CoverageTestMesh::default();
    for copies in 1..=2 {
        if copies == 2 {
            tessellation.boundary_edges.extend_from_within(..);
            // Contained contours add distinct endpoints without changing the union.
            for index in 0..10 {
                let y = 0.25 + index as f32 * 0.02;
                let contained = [[0.5, y], [0.625, y], [0.5, y + 0.01]];
                for index in 0..3 {
                    tessellation.boundary_edges.push(PathBoundaryEdge {
                        start: contained[index],
                        end: contained[(index + 1) % 3],
                        winding: 1,
                    });
                }
            }
        }
        mesh.build(&tessellation, None, 1.0, (4, 2));
        for y in 0..2 {
            for x in 0..4 {
                let expected = polygon_union_pixel_area(&triangles[0], &triangles[1], x, y);
                assert!(
                    (f64::from(coverage_at(&mesh, x, y)) - expected).abs() < 1e-6,
                    "pixel {x},{y}, copies {copies}: {} != {expected}",
                    coverage_at(&mesh, x, y)
                );
            }
        }
    }
}

#[test]
fn opposite_stroke_contours_preserve_the_ring_hole_and_fractional_edges() {
    let mut scene = SceneState::new(
        SceneStateContext::default(),
        Viewport {
            physical_size: (16, 16),
            scale_factor: 1.0,
        },
    );
    let shape = Shape::builder()
        .fill_rule(FillRule::EvenOdd)
        .begin((2.25, 2.5))
        .line_to((10.25, 2.5))
        .line_to((10.25, 10.5))
        .line_to((2.25, 10.5))
        .close()
        .begin((2.25, 2.5))
        .line_to((2.25, 10.5))
        .line_to((10.25, 10.5))
        .line_to((10.25, 2.5))
        .close()
        .build()
        .stroke(StrokeOptions::new(2.0));
    let cached = scene.tessellate(&shape, None).unwrap();
    let mut mesh = CoverageTestMesh::default();
    mesh.build(&cached.tessellation, None, 1.0, (16, 16));
    for y in 0..16 {
        for x in 0..16 {
            let expected = rectangle_pixel_area([[1.25, 1.5], [11.25, 11.5]], x, y)
                - rectangle_pixel_area([[3.25, 3.5], [9.25, 9.5]], x, y);
            assert_eq!(
                f64::from(coverage_at(&mesh, x, y)),
                expected,
                "pixel {x},{y}"
            );
        }
    }
}

#[test]
fn slanted_stroke_crossings_match_independent_polygon_union_under_transforms() {
    let size = (64, 48);
    let mut scene = SceneState::new(
        SceneStateContext::default(),
        Viewport {
            physical_size: size,
            scale_factor: 1.0,
        },
    );
    let shape = Shape::builder()
        .begin((3.25, 4.125))
        .line_to((19.25, 16.125))
        .end()
        .begin((4.625, 18.375))
        .line_to((20.625, 6.375))
        .end()
        .build()
        .stroke(StrokeOptions::new(2.5));
    let cached = scene.tessellate(&shape, None).unwrap();
    // Both centerlines have length 20, so the perpendicular offsets are exact binary fractions.
    let first = [[4.0, 3.125], [20.0, 15.125], [18.5, 17.125], [2.5, 5.125]];
    let second = [
        [3.875, 17.375],
        [19.875, 5.375],
        [21.375, 7.375],
        [5.375, 19.375],
    ];
    let mut perspective = TransformInstance::translation(3.25, 5.5);
    perspective.col0[3] = 0.0125;
    perspective.col1[3] = -0.003125;
    let transforms = [
        TransformInstance::identity(),
        TransformInstance::rotation_z_deg(23.0)
            .then(&TransformInstance::translation(12.375, 3.125)),
        TransformInstance::affine_2d(-1.0, 0.125, 0.25, 1.0, 31.25, 2.5),
        perspective,
    ];
    let mut mesh = CoverageTestMesh::default();
    for transform in transforms {
        for scale in [1.0, 1.5] {
            let first = project_polygon(&first, transform, scale);
            let second = project_polygon(&second, transform, scale);
            mesh.build(&cached.tessellation, Some(transform), scale, size);
            for y in 0..size.1 {
                for x in 0..size.0 {
                    let expected = polygon_union_pixel_area(&first, &second, x, y);
                    let actual = f64::from(coverage_at(&mesh, x, y));
                    assert!(
                        (actual - expected).abs() < 0.0001,
                        "pixel {x},{y}, scale {scale}, transform {transform:?}: {actual} != {expected}"
                    );
                }
            }
        }
    }
}

#[test]
fn thin_crossings_and_coincident_endpoints_match_polygon_union_within_one_row() {
    let size = (8, 2);
    let mut scene = SceneState::new(
        SceneStateContext::default(),
        Viewport {
            physical_size: size,
            scale_factor: 1.0,
        },
    );
    let mut mesh = CoverageTestMesh::default();
    for case in 0..32 {
        let mut first = [
            [-0.25, 0.125 + (case % 8) as f32 / 64.0],
            [8.25, 0.875 - ((case * 7) % 32) as f32 / 128.0],
        ];
        let mut second = [
            [-0.125, 0.875 - ((case * 5) % 16) as f32 / 64.0],
            [8.125, 0.125 + ((case * 3) % 32) as f32 / 128.0],
        ];
        match case % 4 {
            0 => second[1] = first[1],
            1 => second[0] = first[0],
            _ => {}
        }
        if case % 2 == 0 {
            first.reverse();
        }
        if case % 3 == 0 {
            second.reverse();
        }
        let width = [0.015625, 0.046875, 0.0625, 0.125][case % 4];
        let first_rectangle = butt_stroke_rectangle(first[0], first[1], width);
        let second_rectangle = butt_stroke_rectangle(second[0], second[1], width);
        for has_retracing in [false, true] {
            let mut path = Shape::builder()
                .begin((first[0][0], first[0][1]))
                .line_to((first[1][0], first[1][1]))
                .end()
                .begin((second[0][0], second[0][1]))
                .line_to((second[1][0], second[1][1]))
                .end();
            if has_retracing {
                path = path
                    .begin((first[1][0], first[1][1]))
                    .line_to((first[0][0], first[0][1]))
                    .end();
            }
            let shape = path.build().stroke(StrokeOptions::new(width));
            let cached = scene.tessellate(&shape, None).unwrap();
            mesh.build(&cached.tessellation, None, 1.0, size);
            for y in 0..size.1 {
                for x in 0..size.0 {
                    let expected =
                        polygon_union_pixel_area(&first_rectangle, &second_rectangle, x, y);
                    let actual = f64::from(coverage_at(&mesh, x, y));
                    assert!(
                        (actual - expected).abs() < 0.00001,
                        "case {case}, retracing {has_retracing}, pixel {x},{y}: {actual} != {expected}"
                    );
                }
            }
        }
    }
}

#[test]
fn repeated_and_tall_overlapping_strokes_preserve_coverage_after_viewport_changes() {
    let mut scene = SceneState::new(
        SceneStateContext::default(),
        Viewport {
            physical_size: (8, 70),
            scale_factor: 1.0,
        },
    );
    let mut mesh = CoverageTestMesh::default();
    for count in [32_769, 129] {
        let has_distinct_endpoints = count == 129;
        let mut builder = Shape::builder();
        for index in 0..count {
            let index = if has_distinct_endpoints { index } else { 0 };
            let center_x = 2.25 + index as f32 / 64.0;
            let inset = if index == 0 || index == 128 {
                0.0
            } else {
                index as f32 / 8.0
            };
            builder = builder
                .begin((center_x, -0.25 + inset))
                .line_to((center_x, 64.75 - inset))
                .end();
        }
        let shape = builder.build().stroke(StrokeOptions::new(4.0));
        let cached = scene.tessellate(&shape, None).unwrap();
        let right = if has_distinct_endpoints { 6.25 } else { 4.25 };
        for size in [(8, 70), (3, 17), (8, 70)] {
            mesh.build(&cached.tessellation, None, 1.0, size);
            for y in 0..size.1 {
                for x in 0..size.0 {
                    let expected = rectangle_pixel_area([[0.25, -0.25], [right, 64.75]], x, y);
                    assert_eq!(
                        f64::from(coverage_at(&mesh, x, y)),
                        expected,
                        "count {count}, size {size:?}, pixel {x},{y}"
                    );
                }
            }
            mesh.build(&CachedTessellation::empty(), None, 1.0, size);
            assert!(mesh.tiles.is_empty());
            mesh.build(
                &cached.tessellation,
                Some(TransformInstance::translation(100.0, 100.0)),
                1.0,
                size,
            );
            assert!(mesh.tiles.is_empty());
        }
    }
}
