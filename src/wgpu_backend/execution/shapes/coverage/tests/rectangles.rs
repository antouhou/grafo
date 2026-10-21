use super::{coverage_at, triangle_pixel_area, CoverageTestMesh};
use crate::core::cache::CachedTessellation;
use crate::core::{geometry, MathRect, Shape, TransformInstance, Viewport};
use crate::scene_state::{SceneState, SceneStateContext};
use crate::wgpu_backend::execution::shapes::coverage::geometry::{
    CoverageBuilder, SOLID_COVERAGE, TILE_SIZE,
};
use crate::wgpu_backend::execution::shapes::coverage::projection::Projection;
use crate::wgpu_backend::execution::shapes::coverage::tiles::{
    CoverageDrawMetadata, CoverageGeometry,
};
use std::sync::Arc;

fn rasterize(
    geometry: &CoverageGeometry,
    draw: &CoverageDrawMetadata,
    size: (u32, u32),
) -> Vec<f32> {
    let mut pixels = vec![0.0; (size.0 * size.1) as usize];
    let mut occupied = vec![false; pixels.len()];
    let start = draw.range.tile_start as usize;
    for (index, tile) in geometry.tiles[start..start + draw.range.tile_count as usize]
        .iter()
        .enumerate()
    {
        assert!(tile.min[0] < tile.max[0] && tile.min[1] < tile.max[1]);
        assert!(tile.max[0] <= size.0 && tile.max[1] <= size.1);
        assert_eq!(
            index < draw.range.solid_tile_count as usize,
            tile.coverage_start == SOLID_COVERAGE
        );
        assert!((tile.projection_index as usize) < geometry.projections.len());
        if tile.coverage_start != SOLID_COVERAGE {
            assert!(tile.max[0] - tile.min[0] <= TILE_SIZE);
            assert!(tile.max[1] - tile.min[1] <= TILE_SIZE);
        }
        for y in tile.min[1]..tile.max[1] {
            for x in tile.min[0]..tile.max[0] {
                let index = (y * size.0 + x) as usize;
                assert!(!occupied[index], "overlapping tile at {x},{y}");
                occupied[index] = true;
                let value = if tile.coverage_start == SOLID_COVERAGE {
                    1.0
                } else {
                    geometry.coverage[(tile.coverage_start + (y - tile.min[1]) * TILE_SIZE + x
                        - tile.min[0]) as usize]
                };
                assert!(value > 0.0 && value <= 1.0, "invalid value {value}");
                pixels[index] = value;
            }
        }
    }
    pixels
}

fn random_fraction(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 40) as f32 / (1u32 << 24) as f32
}

#[test]
fn rectangle_tiles_match_pixel_overlap_and_general_coverage() {
    let size = (64, 48);
    let mut scene = SceneState::new(
        SceneStateContext::default(),
        Viewport {
            physical_size: size,
            scale_factor: 1.0,
        },
    );
    let mut builder = CoverageBuilder::default();
    let mut geometry = CoverageGeometry::default();
    let mut general = CoverageTestMesh::default();
    let mut random_state = 827_315_u64;
    for index in 0..2500 {
        let origin = [
            random_fraction(&mut random_state) * 100.0 - 30.0,
            random_fraction(&mut random_state) * 80.0 - 25.0,
        ];
        let widths = [0.0, 0.001, 0.125, 0.75, 1.0, 1.25, 8.0, 32.0, 80.0];
        let bounds = [
            (origin[0], origin[1]),
            (
                origin[0] + widths[index % widths.len()],
                origin[1] + widths[(index / widths.len()) % widths.len()],
            ),
        ];
        let transform = TransformInstance::affine_2d(
            if index % 3 == 0 { -1.25 } else { 1.0 },
            0.0,
            0.0,
            if index % 5 == 0 { -0.75 } else { 1.0 },
            if index % 3 == 0 { 48.375 } else { 0.125 },
            if index % 5 == 0 { 40.625 } else { 0.375 },
        );
        let scale = [0.5, 1.0, 1.25, 2.0][index % 4];
        let cached = scene.tessellate(&Shape::rect(bounds), None);
        let generic = CachedTessellation {
            is_rect: false,
            boundary_edges: cached.tessellation.boundary_edges.clone(),
            rounded_rect_boundary: None,
            vertex_buffers: Arc::clone(cached.vertex_buffers()),
            local_bounds: bounds,
            texture_mapping_size: cached.texture_mapping_size(),
        };
        // Append several draws before clearing to exercise shared sample offsets.
        if index % 7 == 0 {
            geometry.clear();
        }
        let draw = builder.build(
            &cached.tessellation,
            Some(transform),
            scale,
            size,
            &mut geometry,
        );
        let pixels = rasterize(&geometry, &draw, size);
        general.build(&generic, Some(transform), scale, size);
        let corners = bounds.map(|(x, y)| {
            [0, 1].map(|axis| {
                ((f64::from(transform.col0[axis]) * f64::from(x)
                    + f64::from(transform.col1[axis]) * f64::from(y)
                    + f64::from(transform.col3[axis]))
                    * scale) as f32
            })
        });
        for y in 0..size.1 {
            for x in 0..size.0 {
                let area = [x, y]
                    .into_iter()
                    .enumerate()
                    .map(|(axis, pixel)| {
                        let minimum = f64::from(corners[0][axis].min(corners[1][axis]));
                        let maximum = f64::from(corners[0][axis].max(corners[1][axis]));
                        (maximum.min(f64::from(pixel) + 1.0) - minimum.max(f64::from(pixel)))
                            .max(0.0)
                    })
                    .product::<f64>() as f32;
                let expected = if area <= 1e-12 { 0.0 } else { area };
                let actual = pixels[(y * size.0 + x) as usize];
                assert_eq!(actual, expected, "case {index}, pixel {x},{y}");
                assert_eq!(
                    actual,
                    coverage_at(&general, x, y),
                    "case {index}, pixel {x},{y}"
                );
            }
        }
    }
}

#[test]
fn transformed_rectangles_match_independently_clipped_triangles() {
    let size = (64, 48);
    let mut scene = SceneState::new(
        SceneStateContext::default(),
        Viewport {
            physical_size: size,
            scale_factor: 1.0,
        },
    );
    let cached = scene.tessellate(&Shape::rect([(2.25, 3.5), (28.75, 24.25)]), None);
    let mut perspective = TransformInstance::translation(10.0, 4.0);
    perspective.col0[3] = 0.0125;
    let mut negative_w = TransformInstance::translation(4.25, 3.5);
    negative_w.col3[3] = -2.0;
    let mut mesh = CoverageTestMesh::default();
    for transform in [
        TransformInstance::rotation_z_deg(23.0),
        TransformInstance::affine_2d(-1.0, 0.125, 0.25, 1.0, 40.0, 4.0),
        TransformInstance::affine_2d(0.0, 1.0, -1.0, 0.0, 40.0, 4.0),
        perspective,
        negative_w,
    ] {
        for scale in [1.0, 1.5, 2.0] {
            let projection =
                Projection::new(Some(transform), scale, cached.local_bounds()).unwrap();
            mesh.build(&cached.tessellation, Some(transform), scale, size);
            for y in 0..size.1 {
                for x in 0..size.0 {
                    let expected: f32 = cached
                        .vertex_buffers()
                        .indices
                        .as_chunks::<3>()
                        .0
                        .iter()
                        .map(|indices| {
                            let triangle = indices.map(|index| {
                                projection.project(
                                    cached.vertex_buffers().vertices[index as usize].position,
                                )
                            });
                            triangle_pixel_area(&triangle, x, y)
                        })
                        .sum();
                    assert!(
                        (coverage_at(&mesh, x, y) - expected).abs() < 0.0001,
                        "pixel {x},{y}, scale {scale}, transform {transform:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn rectangle_interior_and_repeated_edge_samples_stay_compact() {
    let mut scene = SceneState::new(
        SceneStateContext::default(),
        Viewport {
            physical_size: (4096, 4096),
            scale_factor: 1.0,
        },
    );
    let mut mesh = CoverageTestMesh::default();
    for extent in [80.0, 1080.0, 4000.0] {
        for phase in [0.0, 0.25] {
            let cached = scene.tessellate(
                &Shape::rect([(phase, phase), (extent + phase, extent + phase)]),
                None,
            );
            mesh.build(&cached.tessellation, None, 1.0, (4096, 4096));
            assert_eq!(mesh.range.solid_tile_count, 1);
            if phase == 0.0 {
                assert_eq!(mesh.tiles.len(), 1);
                assert!(mesh.coverage.is_empty());
            } else {
                assert_eq!(mesh.coverage.len(), 134);
            }
        }
    }
}

#[test]
fn rectangle_damage_bounds_contain_coverage_after_large_translations() {
    let mut scene = SceneState::new(
        SceneStateContext::default(),
        Viewport {
            physical_size: (64, 64),
            scale_factor: 1.0,
        },
    );
    let mut builder = CoverageBuilder::default();
    let mut geometry = CoverageGeometry::default();
    for (bounds, transform) in [
        (
            [(999_980.0, 0.0), (1_000_000.0, 20.0)],
            TransformInstance::affine_2d(1.1, 0.0, 0.0, 1.0, -1_099_970.0, 8.0),
        ),
        (
            [(999_980.0, 0.0), (1_000_000.0, 20.0)],
            TransformInstance::affine_2d(-1.1, 0.0, 0.0, 1.0, 1_100_010.0, 8.0),
        ),
    ] {
        let cached = scene.tessellate(&Shape::rect(bounds), None);
        let logical_bounds = geometry::transformed_bounds_to_logical_screen_rect(
            MathRect::new(bounds[0].into(), bounds[1].into()),
            Some(transform),
        );
        for scale_factor in [0.7, 1.0, 1.3, 2.0] {
            let viewport = Viewport {
                physical_size: (64, 64),
                scale_factor,
            };
            geometry.clear();
            let draw = builder.build(
                &cached.tessellation,
                Some(transform),
                scale_factor,
                viewport.physical_size,
                &mut geometry,
            );
            let damage =
                geometry::logical_bounds_to_viewport_rect(logical_bounds, viewport).unwrap();
            assert!(
                damage.contains_box(&draw.bounds),
                "damage {damage:?} excludes coverage {:?}, scale {scale_factor}",
                draw.bounds
            );
        }
    }
}
