use super::{coverage_at, triangle_pixel_area, CoverageTestMesh};
use crate::core::{BorderRadii, Shape, TransformInstance, Viewport};
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
        let cached = scene.tessellate(&builder.build(), None).unwrap();
        mesh.build(&cached.fill_tessellation, None, 1.0, (40, 40));
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
            let cached = scene.tessellate(&shape, None).unwrap();
            let geometry = cached.fill_vertex_buffers();
            mesh.build(&cached.fill_tessellation, None, 1.0, size);
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

#[test]
fn repeated_vertical_rows_share_samples_across_bands() {
    let size = (8, 160);
    let mut scene = SceneState::new(
        SceneStateContext::default(),
        Viewport {
            physical_size: size,
            scale_factor: 1.0,
        },
    );
    let shape = Shape::builder()
        .begin((-1.0, -3.5))
        .line_to((0.25, -3.5))
        .line_to((0.25, 71.25))
        .line_to((-1.0, 71.25))
        .close()
        .begin((0.5, 9.75))
        .line_to((1.25, 9.75))
        .line_to((1.25, 59.5))
        .line_to((0.5, 59.5))
        .close()
        .begin((0.5, 97.25))
        .line_to((1.25, 97.25))
        .line_to((1.25, 99.75))
        .line_to((0.5, 99.75))
        .close()
        .begin((0.125, 128.5))
        .line_to((0.375, 128.5))
        .line_to((0.375, 176.0))
        .line_to((0.125, 176.0))
        .close()
        .build();
    let cached = scene.tessellate(&shape, None).unwrap();
    let mut mesh = CoverageTestMesh::default();
    for height in [160, 31, 160] {
        mesh.build(&cached.fill_tessellation, None, 1.0, (size.0, height));
        for y in 0..height {
            let first = match y {
                0..=70 => 0.25,
                71 => 0.0625,
                _ => 0.0,
            };
            let second = match y {
                9 => 0.25,
                10..=58 => 1.0,
                59 => 0.5,
                _ => 0.0,
            };
            let third = match y {
                128 => 0.125,
                129.. => 0.25,
                _ => 0.0,
            };
            let short = match y {
                97 | 99 => 0.75,
                98 => 1.0,
                _ => 0.0,
            };
            assert_eq!(
                coverage_at(&mesh, 0, y),
                first + (second + short) * 0.5 + third
            );
            assert_eq!(coverage_at(&mesh, 1, y), (second + short) * 0.25);
        }
        if height == size.1 {
            assert_eq!(mesh.tiles.len(), 14);
            assert!(mesh.coverage.len() < 8 * 80);
        }
    }
}

#[test]
fn shallow_contours_preserve_coverage_at_clipped_range_ends() {
    let mut builder = Shape::builder();
    for index in 0..128 {
        let y = 4.0 * index as f32;
        builder = builder
            .begin((-0.25, y))
            .line_to((153.75, y + 1.0))
            .line_to((153.75, y + 3.0))
            .line_to((-0.25, y + 2.0))
            .close();
    }
    let shape = builder
        .begin((177.125, 0.0))
        .line_to((185.75, 0.0))
        .line_to((185.75, 1024.0))
        .line_to((177.125, 1024.0))
        .close()
        .build();
    let size = (192, 8);
    let mut scene = SceneState::new(
        SceneStateContext::default(),
        Viewport {
            physical_size: size,
            scale_factor: 1.0,
        },
    );
    let cached = scene.tessellate(&shape, None).unwrap();
    let mut mesh = CoverageTestMesh::default();
    for width in [192, 64, 0, 192] {
        mesh.build(
            &cached.fill_tessellation,
            Some(TransformInstance::scale(1.0, 1.0 / 1024.0)),
            1.0,
            (width, size.1),
        );
        if width == 0 {
            assert!(mesh.tiles.is_empty());
        }
        for x in 0..width {
            let expected = match x {
                0..=152 => 0.25,
                153 => 0.1875,
                177 => 0.875,
                178..=184 => 1.0,
                185 => 0.75,
                _ => 0.0,
            };
            assert_eq!(coverage_at(&mesh, x, 0), expected, "column {x}");
            assert_eq!(coverage_at(&mesh, x, 1), 0.0, "column {x}");
        }
    }
}
