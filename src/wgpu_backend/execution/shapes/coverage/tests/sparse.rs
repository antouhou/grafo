use super::{coverage_at, triangle_pixel_area, CoverageTestMesh};
use crate::core::{BorderRadii, Shape, Viewport};
use crate::scene_state::{SceneState, SceneStateContext};

#[test]
fn finished_coverage_combines_many_edges_inside_a_single_pixel() {
    let mut scene = SceneState::new(
        SceneStateContext::default(),
        Viewport {
            physical_size: (40, 40),
            scale_factor: 1.0,
        },
    );
    let mut mesh = CoverageTestMesh::default();
    for is_horizontal in [false, true] {
        let mut builder = Shape::builder();
        for stripe in 0..32 {
            let left = 8.0 + stripe as f32 / 32.0;
            let mut points = [
                (left, 4.25),
                (left + 1.0 / 64.0, 4.25),
                (left + 1.0 / 64.0, 24.75),
                (left, 24.75),
            ];
            if is_horizontal {
                for point in &mut points {
                    *point = (point.1, point.0);
                }
            }
            builder = builder
                .begin(points[0])
                .line_to(points[1])
                .line_to(points[2])
                .line_to(points[3])
                .close();
        }
        let cached = scene.tessellate(&builder.build(), None);
        mesh.build(&cached.tessellation, None, 1.0, (40, 40));
        for y in 0..40 {
            for x in 0..40 {
                let (across, along) = if is_horizontal { (y, x) } else { (x, y) };
                let expected = if across != 8 {
                    0.0
                } else if along == 4 || along == 24 {
                    0.375
                } else if (5..24).contains(&along) {
                    0.5
                } else {
                    0.0
                };
                assert_eq!(coverage_at(&mesh, x, y), expected, "pixel {x},{y}");
            }
        }
    }
}

#[test]
fn sparse_sweeps_preserve_empty_regions_and_reused_storage() {
    let mut scene = SceneState::new(
        SceneStateContext::default(),
        Viewport {
            physical_size: (128, 128),
            scale_factor: 1.0,
        },
    );
    let mut mesh = CoverageTestMesh::default();
    for size in [(128, 128), (19, 35), (96, 81), (128, 128)] {
        for shape in [
            Shape::rounded_rect([(3.1, 5.7), (120.8, 115.3)], BorderRadii::new(46.25)),
            Shape::builder()
                .begin((-8.125, -12.375))
                .line_to((121.2, 6.6))
                .line_to((12.7, 118.3))
                .close()
                .build(),
        ] {
            let cached = scene.tessellate(&shape, None);
            let geometry = cached.vertex_buffers();
            mesh.build(&cached.tessellation, None, 1.0, size);
            for y in 0..size.1 {
                for x in 0..size.0 {
                    let expected: f32 = geometry
                        .indices
                        .as_chunks::<3>()
                        .0
                        .iter()
                        .map(|triangle| {
                            triangle_pixel_area(
                                &triangle.map(|index| geometry.vertices[index as usize].position),
                                x,
                                y,
                            )
                        })
                        .sum();
                    let actual = coverage_at(&mesh, x, y);
                    assert!((actual - expected).abs() < 0.0003, "pixel {x},{y}");
                    if expected == 0.0 {
                        assert_eq!(actual, 0.0, "empty pixel {x},{y}");
                    }
                }
            }
        }
    }
}
