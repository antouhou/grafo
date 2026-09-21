use super::{
    build_boundary_data, generate_aa_fringe, AaFringeScratch, BoundaryEdge, BoundaryVertexKey,
    RectShape, Shape,
};
use crate::util::ShapeResources;
use crate::vertex::CustomVertex;
use crate::Stroke;
use lyon::tessellation::FillTessellator;
use std::collections::BTreeSet;

fn find_boundary_edges<'a>(
    vertices: &[CustomVertex],
    indices: &[u16],
    scratch: &'a mut AaFringeScratch,
) -> &'a [BoundaryEdge] {
    build_boundary_data(vertices, indices, scratch);
    &scratch.boundary_edges
}

fn filled_vertex(position: [f32; 2]) -> CustomVertex {
    CustomVertex {
        position,
        tex_coords: [0.0, 0.0],
        normal: [0.0, 0.0],
        coverage: 1.0,
    }
}

#[test]
fn aa_fringe_ignores_internal_seams_with_duplicate_vertices() {
    let mut vertices = vec![
        filled_vertex([0.0, 0.0]),
        filled_vertex([1.0, 0.0]),
        filled_vertex([1.0, 1.0]),
        filled_vertex([0.0, 0.0]),
        filled_vertex([1.0, 1.0]),
        filled_vertex([0.0, 1.0]),
    ];
    let mut indices = vec![0, 1, 2, 3, 4, 5];
    let mut aa_fringe_scratch = AaFringeScratch::new();

    let boundary_edges = find_boundary_edges(&vertices, &indices, &mut aa_fringe_scratch);
    assert_eq!(boundary_edges.len(), 4);

    generate_aa_fringe(&mut vertices, &mut indices, &mut aa_fringe_scratch);

    assert_eq!(vertices.len(), 10);
    assert_eq!(indices.len(), 30);

    let outer_vertex_count = vertices
        .iter()
        .filter(|vertex| vertex.coverage == 0.0)
        .count();
    assert_eq!(outer_vertex_count, 4);

    let unique_outer_vertex_positions = vertices
        .iter()
        .filter(|vertex| vertex.coverage == 0.0)
        .map(|vertex| BoundaryVertexKey::from_position(vertex.position))
        .collect::<BTreeSet<_>>();
    assert_eq!(unique_outer_vertex_positions.len(), 4);
}

#[test]
fn rect_tessellation_uses_shared_quad_corners() {
    let rect_shape = RectShape::new([(10.0, 20.0), (30.0, 50.0)], Stroke::default());
    let mut tessellator = FillTessellator::new();
    let mut shape_resources = ShapeResources::new();

    let tessellated_geometry =
        Shape::Rect(rect_shape).tessellate(&mut tessellator, &mut shape_resources, None);

    assert_eq!(tessellated_geometry.vertex_buffers.vertices.len(), 8);
    assert_eq!(tessellated_geometry.vertex_buffers.indices.len(), 30);

    let fill_vertex_count = tessellated_geometry
        .vertex_buffers
        .vertices
        .iter()
        .filter(|vertex| vertex.coverage == 1.0)
        .count();
    assert_eq!(fill_vertex_count, 4);
}

#[test]
fn aa_fringe_keeps_distinct_corners_for_point_touching_triangles() {
    let mut vertices = vec![
        filled_vertex([0.0, 0.0]),
        filled_vertex([1.0, 0.0]),
        filled_vertex([0.0, 1.0]),
        filled_vertex([0.0, 0.0]),
        filled_vertex([-1.0, 0.0]),
        filled_vertex([0.0, -1.0]),
    ];
    let mut indices = vec![0, 1, 2, 3, 4, 5];
    let mut aa_fringe_scratch = AaFringeScratch::new();

    generate_aa_fringe(&mut vertices, &mut indices, &mut aa_fringe_scratch);

    let outer_vertices = vertices
        .iter()
        .filter(|vertex| vertex.coverage == 0.0)
        .collect::<Vec<_>>();
    assert_eq!(outer_vertices.len(), 6);

    let shared_point_outer_vertices = outer_vertices
        .iter()
        .filter(|vertex| {
            BoundaryVertexKey::from_position(vertex.position)
                == BoundaryVertexKey::from_position([0.0, 0.0])
        })
        .count();
    assert_eq!(shared_point_outer_vertices, 2);
}

#[test]
fn aa_fringe_skips_zero_length_boundary_edges() {
    let mut vertices = vec![
        filled_vertex([5.0, 5.0]),
        filled_vertex([5.0, 5.0]),
        filled_vertex([6.0, 5.0]),
    ];
    let mut indices = vec![0, 1, 2];
    let mut aa_fringe_scratch = AaFringeScratch::new();

    generate_aa_fringe(&mut vertices, &mut indices, &mut aa_fringe_scratch);

    assert_eq!(vertices.len(), 3);
    assert_eq!(indices.len(), 3);
}
