use super::preparation::{self, InstanceTextureData};
use super::{ShapeDrawLocation, ShapeDrawResources, ShapeExecutionResources};
use crate::commands::ShapeDrawId;
use crate::core::vertex::TextureUvTransform;
use crate::core::{Shape, ShapeDrawCommandOptions, ShapeInstance};
use crate::scene::Scene;

fn insert_draw(resources: &mut ShapeExecutionResources, id: usize, shape: &ShapeInstance) {
    let geometry_range = preparation::append_aggregated_geometry_for_shape(
        shape,
        &mut resources.vertices,
        &mut resources.indices,
        &mut resources.geometry_ranges,
    )
    .unwrap()
    .unwrap();
    let instance_index = preparation::append_instance_data(
        &mut resources.instance_transforms,
        &mut resources.instance_colors,
        &mut resources.instance_metadata,
        shape.transform,
        Some([id as f32; 4]),
        InstanceTextureData {
            texture_presence: [false; 2],
            texture_uv_transforms: [TextureUvTransform::IDENTITY; 2],
        },
    );
    resources.draws.insert(
        id,
        ShapeDrawResources {
            location: Some(ShapeDrawLocation {
                geometry_range,
                instance_index,
            }),
            ..Default::default()
        },
    );
}

fn assert_draw_data(resources: &ShapeExecutionResources, id: usize, shape: &ShapeInstance) {
    let location = resources.draws[&id].location.unwrap();
    let geometry = location.geometry_range;
    let expected = shape.cached_shape.vertex_buffers();
    let vertices = &resources.vertices
        [geometry.vertex_start as usize..geometry.vertex_start as usize + geometry.vertex_count];
    assert_eq!(
        bytemuck::cast_slice::<_, u8>(vertices),
        bytemuck::cast_slice::<_, u8>(&expected.vertices)
    );
    assert_eq!(
        &resources.indices[geometry.index_start as usize..geometry.indices().end as usize],
        expected.indices.as_slice()
    );
    assert_eq!(
        resources.instance_colors[location.instance_index].color,
        [id as f32; 4]
    );
    if let Some(geometry_id) = shape.cached_shape.geometry_id {
        assert_eq!(resources.geometry_ranges[&geometry_id], geometry);
    }
}

#[test]
fn removal_compacts_buffers_and_keeps_shared_geometry_until_its_last_draw() {
    let mut scene = Scene::default();
    let shared = ShapeInstance::new(
        scene.tessellate(&Shape::rect([(1.0, 2.0), (10.0, 12.0)]), Some(1)),
        ShapeDrawCommandOptions::new(),
    );
    let unique = ShapeInstance::new(
        scene.tessellate(&Shape::rect([(20.0, 3.0), (27.0, 9.0)]), Some(2)),
        ShapeDrawCommandOptions::new(),
    );
    let uncached = ShapeInstance::new(
        scene.tessellate(&Shape::rect([(3.0, 5.0), (18.0, 21.0)]), None),
        ShapeDrawCommandOptions::new(),
    );
    let mut resources = ShapeExecutionResources::new();
    insert_draw(&mut resources, 1, &shared);
    insert_draw(&mut resources, 2, &unique);
    insert_draw(&mut resources, 3, &shared);
    insert_draw(&mut resources, 4, &uncached);
    let initial_vertex_count = resources.vertices.len();
    resources.remove_draw(ShapeDrawId(1));
    assert_eq!(resources.vertices.len(), initial_vertex_count);
    for (id, shape) in [(2, &unique), (3, &shared), (4, &uncached)] {
        assert_draw_data(&resources, id, shape);
    }
    resources.remove_draw(ShapeDrawId(3));
    assert!(!resources.geometry_ranges.contains_key(&1));
    assert_draw_data(&resources, 2, &unique);
    assert_draw_data(&resources, 4, &uncached);
    resources.remove_draw(ShapeDrawId(2));
    assert_draw_data(&resources, 4, &uncached);
    let retained_vertex_count = resources.vertices.len();
    for _ in 0..20 {
        insert_draw(&mut resources, 1, &shared);
        insert_draw(&mut resources, 2, &unique);
        assert_draw_data(&resources, 1, &shared);
        assert_draw_data(&resources, 2, &unique);
        resources.remove_draw(ShapeDrawId(1));
        assert_draw_data(&resources, 2, &unique);
        resources.remove_draw(ShapeDrawId(2));
        assert_draw_data(&resources, 4, &uncached);
        assert_eq!(resources.vertices.len(), retained_vertex_count);
        assert_eq!(resources.instance_transforms.len(), 1);
        assert_eq!(resources.instance_colors.len(), 1);
        assert_eq!(resources.instance_metadata.len(), 1);
    }
    resources.remove_draw(ShapeDrawId(4));
    resources.remove_draw(ShapeDrawId(4));
    assert!(resources.vertices.is_empty());
    assert!(resources.indices.is_empty());
    assert!(resources.geometry_ranges.is_empty());
    assert!(resources.instance_transforms.is_empty());
    assert!(resources.instance_colors.is_empty());
    assert!(resources.instance_metadata.is_empty());
}
