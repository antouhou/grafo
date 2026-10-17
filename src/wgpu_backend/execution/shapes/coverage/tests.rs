use super::geometry::{CoverageBuilder, SOLID_COVERAGE};
use crate::core::{Shape, TransformInstance, Viewport};
use crate::scene::{Scene, SceneContext};
use crate::wgpu_backend::pipeline::SHAPE_SHADER;
use naga::valid::{Capabilities, ValidationFlags, Validator};

fn integral(value: f32) -> f32 {
    if value <= 0.0 {
        0.0
    } else if value >= 1.0 {
        value - 0.5
    } else {
        value * value * 0.5
    }
}

fn coverage_at(mesh: &CoverageBuilder, x: u32, y: u32) -> f32 {
    for vertices in mesh.vertices.as_chunks::<4>().0 {
        if (x as f32) < vertices[0].position[0]
            || (x as f32) >= vertices[2].position[0]
            || (y as f32) < vertices[0].position[1]
            || (y as f32) >= vertices[2].position[1]
        {
            continue;
        }
        if vertices[0].row_start == SOLID_COVERAGE {
            return 1.0;
        }
        let row = mesh.rows[(vertices[0].row_start + y - vertices[0].tile_y) as usize];
        let mut area = row.baseline;
        for edge in &mesh.edges[row.edge_start as usize..(row.edge_start + row.edge_count) as usize]
        {
            let a = edge.x0 - (x - row.tile_x) as f32;
            let b = edge.x1 - (x - row.tile_x) as f32;
            area += edge.height
                * if a.min(b) >= 1.0 {
                    1.0
                } else if a.min(b) >= 0.0 && a.max(b) <= 1.0 {
                    (a + b) * 0.5
                } else if a.max(b) > 0.0 {
                    (integral(b) - integral(a)) * edge.inverse_width
                } else {
                    0.0
                };
        }
        return area.clamp(0.0, 1.0);
    }
    0.0
}

fn clip_polygon(polygon: &[[f32; 2]], axis: usize, boundary: f32, direction: f32) -> Vec<[f32; 2]> {
    let mut result = Vec::new();
    for index in 0..polygon.len() {
        let a = polygon[index];
        let b = polygon[(index + 1) % polygon.len()];
        let a_inside = direction * (a[axis] - boundary) >= 0.0;
        let b_inside = direction * (b[axis] - boundary) >= 0.0;
        if a_inside {
            result.push(a);
        }
        if a_inside != b_inside {
            let fraction = (boundary - a[axis]) / (b[axis] - a[axis]);
            result.push([
                a[0] + fraction * (b[0] - a[0]),
                a[1] + fraction * (b[1] - a[1]),
            ]);
        }
    }
    result
}

fn triangle_pixel_area(triangle: &[[f32; 2]], x: u32, y: u32) -> f32 {
    let mut polygon = triangle.to_vec();
    for (axis, boundary, direction) in [
        (0, x as f32, 1.0),
        (0, x as f32 + 1.0, -1.0),
        (1, y as f32, 1.0),
        (1, y as f32 + 1.0, -1.0),
    ] {
        polygon = clip_polygon(&polygon, axis, boundary, direction);
    }
    let mut area = 0.0_f64;
    for index in 0..polygon.len() {
        let a = polygon[index].map(f64::from);
        let b = polygon[(index + 1) % polygon.len()].map(f64::from);
        area += a[0] * b[1] - b[0] * a[1];
    }
    (area.abs() * 0.5) as f32
}

#[test]
fn path_tiles_match_independently_clipped_triangles() {
    let shapes = [
        Shape::builder()
            .begin((8.375, 4.0))
            .line_to((8.625, 4.0))
            .line_to((8.625, 28.0))
            .line_to((8.375, 28.0))
            .close()
            .build(),
        Shape::builder()
            .begin((-8.25, 6.125))
            .line_to((51.625, 8.875))
            .line_to((17.25, 14.125))
            .line_to((9.375, 49.5))
            .close()
            .build(),
        Shape::builder()
            .begin((2.25, 2.5))
            .line_to((43.875, 43.125))
            .line_to((4.625, 45.25))
            .line_to((44.5, 2.75))
            .close()
            .build(),
        Shape::builder()
            .begin((2.25, 2.25))
            .line_to((51.75, 2.25))
            .line_to((51.75, 51.75))
            .line_to((2.25, 51.75))
            .close()
            .begin((8.375, 8.625))
            .line_to((44.25, 8.625))
            .line_to((44.25, 44.75))
            .line_to((8.375, 44.75))
            .close()
            .build(),
        Shape::rounded_rect(
            [(3.125, 5.625), (50.25, 47.875)],
            crate::BorderRadii::new(13.0),
        ),
    ];
    let mut scene = Scene::new(
        SceneContext::default(),
        Viewport {
            physical_size: (64, 64),
            scale_factor: 1.0,
        },
        0.75,
    );
    let mut mesh = CoverageBuilder::default();
    for shape in shapes {
        let cached = scene.tessellate(&shape, None);
        let geometry = cached.vertex_buffers();
        assert!(geometry
            .vertices
            .iter()
            .all(|vertex| vertex.coverage == 1.0));
        mesh.build(
            &geometry.vertices,
            &geometry.indices,
            None,
            1.0,
            (64, 64),
            cached.local_bounds(),
        );
        for y in 0..64 {
            for x in 0..64 {
                let expected: f32 = geometry
                    .indices
                    .as_chunks::<3>()
                    .0
                    .iter()
                    .map(|triangle| {
                        let points = [triangle[0], triangle[1], triangle[2]]
                            .map(|index| geometry.vertices[index as usize].position);
                        triangle_pixel_area(&points, x, y)
                    })
                    .sum();
                let actual = coverage_at(&mesh, x, y);
                assert!(
                    (actual - expected).abs() < 0.0003,
                    "pixel {x},{y}: {actual} != {expected}"
                );
            }
        }
    }
}

#[test]
fn transformed_tiles_preserve_thin_features_and_rebuild_for_viewport_changes() {
    let mut scene = Scene::new(
        SceneContext::default(),
        Viewport {
            physical_size: (32, 32),
            scale_factor: 1.0,
        },
        0.75,
    );
    let cached = scene.tessellate(
        &Shape::builder()
            .begin((0.0, 0.0))
            .line_to((0.25, 0.0))
            .line_to((0.25, 24.0))
            .line_to((0.0, 24.0))
            .close()
            .build(),
        None,
    );
    let geometry = cached.vertex_buffers();
    let mut mesh = CoverageBuilder::default();
    mesh.build(
        &geometry.vertices,
        &geometry.indices,
        Some(TransformInstance::translation(8.375, 4.0)),
        1.0,
        (32, 32),
        cached.local_bounds(),
    );
    assert!((coverage_at(&mesh, 8, 16) - 0.25).abs() < 1e-6);
    mesh.build(
        &geometry.vertices,
        &geometry.indices,
        Some(TransformInstance::translation(8.375, 4.0)),
        2.0,
        (64, 64),
        cached.local_bounds(),
    );
    assert!((coverage_at(&mesh, 16, 32) - 0.25).abs() < 1e-6);
    assert!((coverage_at(&mesh, 17, 32) - 0.25).abs() < 1e-6);
    mesh.build(
        &geometry.vertices,
        &geometry.indices,
        Some(TransformInstance::translation(-100.0, 4.0)),
        1.0,
        (32, 32),
        cached.local_bounds(),
    );
    assert!(mesh.vertices.is_empty());
}

#[test]
fn projected_path_tiles_match_the_projected_triangle_areas() {
    let mut scene = Scene::new(
        SceneContext::default(),
        Viewport {
            physical_size: (64, 64),
            scale_factor: 1.0,
        },
        0.75,
    );
    let cached = scene.tessellate(
        &Shape::builder()
            .begin((0.25, 0.375))
            .line_to((25.75, 3.125))
            .line_to((7.625, 27.25))
            .close()
            .build(),
        None,
    );
    let geometry = cached.vertex_buffers();
    let mut perspective = TransformInstance::translation(3.25, 5.5);
    perspective.col0[3] = 0.0125;
    perspective.col1[3] = 0.0075;
    let mut negative_w = TransformInstance::translation(8.125, 4.75);
    negative_w.col3[3] = -1.0;
    let mut clamped_w = TransformInstance::affine_2d(1e-6, 0.0, 0.0, 1e-6, 0.0, 0.0);
    clamped_w.col3[3] = 0.0;
    let transforms = [
        perspective,
        negative_w,
        clamped_w,
        TransformInstance::affine_2d(-1.25, 0.375, 0.625, 1.75, 35.25, 4.125),
    ];
    let mut mesh = CoverageBuilder::default();
    for transform in transforms {
        mesh.build(
            &geometry.vertices,
            &geometry.indices,
            Some(transform),
            1.5,
            (64, 64),
            cached.local_bounds(),
        );
        let project = |position: [f32; 2]| {
            let [x, y] = position;
            let inverse_w = 1.5
                / (x * transform.col0[3] + y * transform.col1[3] + transform.col3[3])
                    .abs()
                    .max(1e-6);
            [
                (x * transform.col0[0] + y * transform.col1[0] + transform.col3[0]) * inverse_w,
                (x * transform.col0[1] + y * transform.col1[1] + transform.col3[1]) * inverse_w,
            ]
        };
        let triangle = [
            geometry.indices[0],
            geometry.indices[1],
            geometry.indices[2],
        ]
        .map(|index| project(geometry.vertices[index as usize].position));
        for y in 0..64 {
            for x in 0..64 {
                let expected = triangle_pixel_area(&triangle, x, y);
                let actual = coverage_at(&mesh, x, y);
                assert!(
                    (actual - expected).abs() < 0.0003,
                    "pixel {x},{y}: {actual} != {expected}"
                );
            }
        }
    }
}

#[test]
fn coverage_shader_entry_points_validate_without_a_gpu() {
    let source = SHAPE_SHADER;
    let module = naga::front::wgsl::parse_str(source)
        .unwrap_or_else(|error| panic!("{}", error.emit_to_string(source)));
    Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .unwrap();
}
