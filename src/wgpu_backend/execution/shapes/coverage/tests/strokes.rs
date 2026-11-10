use super::{coverage_at, CoverageTestMesh};
use crate::core::{LineCap, Shape, StrokeGeometryOptions, Viewport};
use crate::scene_state::{SceneState, SceneStateContext};

#[test]
fn cached_stroke_meshes_use_path_coverage() {
    let size = (32, 32);
    let mut scene = SceneState::new(
        SceneStateContext::default(),
        Viewport {
            physical_size: size,
            scale_factor: 1.0,
        },
    );
    let line = Shape::builder()
        .begin((8.5, 8.25))
        .line_to((20.25, 8.25))
        .end()
        .build()
        .with_stroke(StrokeGeometryOptions::new(2.0).start_cap(LineCap::Square));
    let rectangle =
        Shape::rect([(8.25, 8.25), (20.25, 20.25)]).with_stroke(StrokeGeometryOptions::new(2.0));
    scene.load_shape(line, 1, Some(1)).unwrap();
    scene.load_shape(rectangle, 2, Some(2)).unwrap();

    let mut mesh = CoverageTestMesh::default();
    let line = scene.loaded_shape(1).unwrap();
    assert!(line.fill_vertex_buffers().indices.is_empty());
    mesh.build(line.stroke_tessellation.as_ref().unwrap(), None, 1.0, size);
    assert_eq!(coverage_at(&mesh, 7, 7), 0.375);
    assert_eq!(coverage_at(&mesh, 20, 8), 0.25);

    let rectangle = scene.loaded_shape(2).unwrap();
    mesh.build(
        rectangle.stroke_tessellation.as_ref().unwrap(),
        None,
        1.0,
        size,
    );
    assert_eq!(coverage_at(&mesh, 12, 7), 0.75);
    assert_eq!(coverage_at(&mesh, 12, 8), 1.0);
    assert_eq!(coverage_at(&mesh, 12, 12), 0.0);
}
