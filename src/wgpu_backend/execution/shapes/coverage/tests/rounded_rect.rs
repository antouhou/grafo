use super::{coverage_at, CoverageTestMesh};
use crate::core::cache::CachedTessellation;
use crate::core::{BorderRadii, Shape, TransformInstance, Viewport};
use crate::scene::{Scene, SceneContext};
use std::sync::Arc;

#[test]
fn rounded_rect_sweep_matches_general_coverage() {
    let size = (96, 80);
    let mut scene = Scene::new(
        SceneContext::default(),
        Viewport {
            physical_size: size,
            scale_factor: 1.0,
        },
        0.75,
    );
    let mut specialized = CoverageTestMesh::default();
    let mut general = CoverageTestMesh::default();
    let mut perspective = TransformInstance::translation(4.25, 3.5);
    perspective.col0[3] = 0.0125;
    let mut negative_w = TransformInstance::translation(4.25, 3.5);
    negative_w.col3[3] = -2.0;
    let transforms = [
        TransformInstance::identity(),
        TransformInstance::translation(8.125, 7.375),
        TransformInstance::translation(-21.75, -13.25),
        TransformInstance::translation(73.5, 58.25),
        TransformInstance::translation(-120.0, 4.0),
        TransformInstance::scale(0.0075, 0.0125),
        TransformInstance::affine_2d(1.375, 0.0, 0.0, 0.625, 0.125, 2.75),
        TransformInstance::affine_2d(-1.25, 0.0, 0.0, 0.875, 75.25, 4.125),
        TransformInstance::affine_2d(1.125, 0.0, 0.0, -0.75, 5.5, 67.25),
        TransformInstance::affine_2d(-1.125, 0.0, 0.0, -0.75, 75.5, 67.25),
        TransformInstance::affine_2d(1.0, 0.125, 0.25, 1.0, 3.25, 4.5),
        TransformInstance::rotation_z_deg(23.0),
        perspective,
        negative_w,
    ];
    for bounds in [
        [(3.125, 5.625), (62.25, 58.875)],
        [(8.375, 6.125), (8.625, 57.875)],
        [(4.125, 8.375), (64.25, 8.625)],
        [(-32.625, -18.375), (62.25, 58.875)],
    ] {
        for radii in [
            BorderRadii::new(0.0),
            BorderRadii::new(0.125),
            BorderRadii::new(13.0),
            BorderRadii::new(1000.0),
            BorderRadii {
                top_left: 0.0,
                top_right: 8.25,
                bottom_left: 40.0,
                bottom_right: 200.0,
            },
        ] {
            let cached = scene.tessellate(&Shape::rounded_rect(bounds, radii), None);
            let tessellation = &cached.tessellation;
            assert!(
                tessellation.rounded_rect_boundary.is_some(),
                "{bounds:?}, {radii:?}"
            );
            let generic = CachedTessellation {
                boundary_edges: tessellation.boundary_edges.clone(),
                rounded_rect_boundary: None,
                vertex_buffers: Arc::clone(&tessellation.vertex_buffers),
                local_bounds: tessellation.local_bounds,
                texture_mapping_size: tessellation.texture_mapping_size,
            };
            for scale in [1.0, 1.5, 2.0] {
                for transform in transforms {
                    specialized.build(tessellation, Some(transform), scale, size);
                    general.build(&generic, Some(transform), scale, size);
                    for y in 0..size.1 {
                        for x in 0..size.0 {
                            let actual = coverage_at(&specialized, x, y);
                            let expected = coverage_at(&general, x, y);
                            assert!(
                                (actual - expected).abs() < 0.000001,
                                "pixel {x},{y}: {actual} != {expected}, bounds {bounds:?}, \
                                 radii {radii:?}, scale {scale}, transform {transform:?}"
                            );
                            if expected == 0.0 || expected == 1.0 {
                                assert_eq!(actual, expected, "solid or empty pixel {x},{y}");
                            }
                        }
                    }
                    specialized.build(&generic, Some(transform), scale, size);
                }
            }
        }
    }
}
