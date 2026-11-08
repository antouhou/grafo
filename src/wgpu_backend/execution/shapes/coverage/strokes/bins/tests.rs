use super::{requires_precise_projection, TriangleBins, ATLAS_SIDE};
use crate::core::vertex::{CustomVertex, InstanceTransform};
use crate::core::UnsignedPhysicalRect;

#[test]
fn large_coordinates_that_cancel_in_a_small_viewport_use_precise_projection() {
    let bounds = [
        (100_000_016.0, 99_999_992.0),
        (100_000_032.0, 100_000_008.0),
    ];
    let transform = InstanceTransform::affine_2d(1.0, 0.0, 1.0, 1.0, -200_000_000.0, -99_999_984.0);
    assert!(requires_precise_projection(Some(transform), bounds, 1.0));
    assert!(!requires_precise_projection(
        Some(InstanceTransform::translation(120.0, 80.0)),
        [(0.0, 0.0), (160.0, 48.0)],
        2.0,
    ));
}

#[test]
fn oversized_diagonal_only_populates_intersecting_tiles() {
    let size = ATLAS_SIDE * 4;
    let vertices = [
        [10.0, 10.0],
        [size as f32 - 10.0, size as f32 - 10.0],
        [10.0, 12.0],
    ]
    .map(|position| CustomVertex {
        position,
        tex_coords: [0.0; 2],
    });
    let mut bins = TriangleBins::default();
    bins.prepare(
        &vertices,
        &[0, 1, 2],
        InstanceTransform::identity(),
        1.0,
        UnsignedPhysicalRect::new((0, 0).into(), (size, size).into()),
    );
    let occupied: Vec<_> = bins.occupied().collect();
    assert_eq!(occupied.len(), 7);
    for &(bounds, indices) in &occupied {
        assert_eq!(indices, [0, 1, 2]);
        assert!(bounds.width() <= ATLAS_SIDE && bounds.height() <= ATLAS_SIDE);
        if bounds.min.x / ATLAS_SIDE != bounds.min.y / ATLAS_SIDE {
            assert!(bounds.width() <= 2 && bounds.height() <= 2);
        }
    }

    // Reusing bins for a translated mesh drops the old regions.
    bins.prepare(
        &vertices,
        &[0, 1, 2],
        InstanceTransform::translation(size as f32, 0.0),
        1.0,
        UnsignedPhysicalRect::new((0, 0).into(), (size, size).into()),
    );
    assert_eq!(bins.occupied().count(), 0);
}
