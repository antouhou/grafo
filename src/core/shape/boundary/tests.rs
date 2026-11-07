use super::PathBoundaryBuilder;
use crate::core::vertex::CustomVertex;

#[test]
fn a_triangle_omitted_from_the_boundary_cannot_prove_coverage() {
    let vertices = [
        [-1e18, 1e19],
        [-1e11, 1e19 / 16_777_216.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [1.25, 1.0],
        [1.5, 2.0],
        [1.75, 1.0],
    ]
    .map(|position| CustomVertex {
        position,
        tex_coords: [0.0; 2],
    });
    let indices = [0, 1, 2, 0, 2, 3, 4, 5, 6];
    let mut builder = PathBoundaryBuilder::default();
    // At vertex 0 the second triangle's area rounds to zero. It contributes
    // no winding to the boundary built from this mesh.
    assert!(builder
        .cull_covered_triangles(&vertices, &indices, |position| position, Some(1))
        .is_none());
}

#[test]
fn culling_keeps_triangles_near_an_imprecisely_clipped_cover_edge() {
    let vertices = [
        [1.0, 1.0],
        [-2e18, -1e18],
        [8.0, -1e18],
        [8.0, 1.0],
        [0.55, 0.75],
        [0.65, 0.8],
        [0.7, 0.8],
    ]
    .map(|position| CustomVertex {
        position,
        tex_coords: [0.0; 2],
    });
    let indices = [0, 1, 3, 1, 2, 3, 4, 5, 6];
    let mut builder = PathBoundaryBuilder::default();
    // Row clipping can round the long left edge past the small triangle.
    assert!(builder
        .cull_covered_triangles(&vertices, &indices, |position| position, Some(1))
        .is_none());
}
