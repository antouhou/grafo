use super::{CoverageBuilder, CoverageGeometry, Projection};
use crate::core::{Shape, StrokeOptions, TransformInstance, Viewport};
use crate::scene_state::{SceneState, SceneStateContext};
use bytemuck::cast_slice;

#[test]
fn hidden_stroke_crossings_are_culled_without_changing_visible_coverage() {
    let size = (10, 2);
    let mut scene = SceneState::new(
        SceneStateContext::default(),
        Viewport {
            physical_size: size,
            scale_factor: 1.0,
        },
    );
    let options = StrokeOptions::new(1.0);
    let mut builder = CoverageBuilder::default();
    let mut actual_geometry = CoverageGeometry::default();
    let mut expected_geometry = CoverageGeometry::default();
    for has_protruding_stroke in [false, true] {
        let mut visible = Shape::builder().begin((0.0, 0.5)).line_to((8.0, 0.5)).end();
        let mut all_strokes = Shape::builder();
        for index in 0..512 {
            if index == 256 {
                all_strokes = all_strokes.begin((0.0, 0.5)).line_to((8.0, 0.5)).end();
            }
            let fraction = index as f32 / 512.0;
            let end_offset = 0.05 * fraction * fraction + 0.01 * fraction * fraction * fraction;
            all_strokes = all_strokes
                .begin((4.0 + 0.1 * fraction, 0.32))
                .line_to((4.1 - end_offset, 0.68))
                .end();
        }
        if has_protruding_stroke {
            visible = visible.begin((7.75, 0.625)).line_to((8.5, 0.625)).end();
            all_strokes = all_strokes.begin((7.75, 0.625)).line_to((8.5, 0.625)).end();
        }
        let expected = scene
            .tessellate(&visible.build().stroke(options), None)
            .unwrap();
        let actual = scene
            .tessellate(&all_strokes.build().stroke(options), None)
            .unwrap();
        for transform in [
            None,
            Some(TransformInstance::affine_2d(
                -1.0, 0.0, 0.0, 0.75, 8.75, 0.875,
            )),
            Some(TransformInstance::affine_2d(
                1.0, 0.02, 0.14, 0.6, 0.125, 0.1,
            )),
        ] {
            actual_geometry.clear();
            expected_geometry.clear();
            builder.build(
                &expected.tessellation,
                transform,
                1.0,
                size,
                &mut expected_geometry,
            );
            let expected_edge_count = builder.boundary_edges.len();
            builder.build(
                &actual.tessellation,
                transform,
                1.0,
                size,
                &mut actual_geometry,
            );
            assert_eq!(
                builder.boundary_edges.len(),
                expected_edge_count,
                "protruding stroke {has_protruding_stroke}, transform {transform:?}",
            );
            assert_eq!(
                cast_slice::<_, u8>(&actual_geometry.tiles),
                cast_slice::<_, u8>(&expected_geometry.tiles),
            );
            assert_eq!(
                actual_geometry.coverage.len(),
                expected_geometry.coverage.len()
            );
            for (&actual, &expected) in actual_geometry
                .coverage
                .iter()
                .zip(&expected_geometry.coverage)
            {
                assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
            }
        }
    }
}

#[test]
fn clipped_projected_strokes_keep_the_original_winding_boundary() {
    let size = (8, 1);
    let mut scene = SceneState::new(
        SceneStateContext::default(),
        Viewport {
            physical_size: size,
            scale_factor: 1.0,
        },
    );
    let mut strokes = Shape::builder()
        .begin((2.0, 1.0))
        .line_to((0.0, 1.0))
        .end()
        .begin((1.2, 1.0))
        .line_to((16.0, 1.0))
        .end();
    for index in 0..64 {
        let fraction = index as f32 / 64.0;
        strokes = strokes
            .begin((2.4 + 0.04 * fraction, 0.9))
            .line_to((2.44 - 0.04 * fraction * fraction, 1.06))
            .end();
    }
    let shape = strokes.build().stroke(StrokeOptions::new(2.0));
    let cached = scene.tessellate(&shape, None).unwrap();
    let tessellation = &cached.tessellation;
    let factor = (1_u64 << 40) as f32;
    let transform = TransformInstance {
        col0: [1.5 * factor, factor, 0.0, 0.5 * factor],
        col1: [-0.625 * factor, -0.5 * factor, 0.0, -0.25 * factor],
        col2: [0.0, 0.0, 1.0, 0.0],
        col3: [-2.0 * factor, -factor, 0.0, 0.0],
    };
    let projection = Projection::new(Some(transform), 1.0, tessellation.local_bounds).unwrap();
    let mut builder = CoverageBuilder::default();
    builder.collect_boundary(tessellation, &projection);
    let original_edge_count = builder.boundary_edges.len();
    let mesh = &tessellation.vertex_buffers;
    let culled_edge_count = builder
        .boundary_builder
        .cull_covered_triangles(
            &mesh.vertices,
            &mesh.indices,
            |position| projection.project(position),
            projection.boundary_orientation,
        )
        .unwrap()
        .count();
    assert!(culled_edge_count < original_edge_count);

    // The perspective transform sends a retained edge far above the viewport.
    // Interpolating that edge at the first row can change its winding there.
    let mut geometry = CoverageGeometry::default();
    builder.build(tessellation, Some(transform), 1.0, size, &mut geometry);
    assert_eq!(builder.boundary_edges.len(), original_edge_count);
}
