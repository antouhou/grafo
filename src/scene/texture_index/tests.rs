use super::TextureIndex;
use crate::core::{Shape, ShapeDrawCommandOptions, UnsignedPhysicalRect, Viewport};
use crate::render_backend::texture_manager::TextureUpdatePublisher;
use crate::scene::types::{CachedShapeDrawData, DrawTreeNode};
use crate::scene::{Scene, SceneContext};

impl TextureIndex {
    fn textures_used_by(&self, shape_id: usize) -> &[u64] {
        self.textures_by_shape
            .get(&shape_id)
            .map_or(&[], |textures| textures.as_slice())
    }
}

fn scene() -> Scene {
    Scene::new(
        SceneContext::default(),
        Viewport {
            physical_size: (32, 32),
            scale_factor: 1.0,
        },
        0.75,
    )
}

fn shape(scene: &mut Scene, options: ShapeDrawCommandOptions) -> DrawTreeNode {
    let cached_shape = scene.tessellate(&Shape::rect([(0.0, 0.0), (16.0, 16.0)]), Some(1));
    DrawTreeNode::CachedShape(CachedShapeDrawData::new(cached_shape, options))
}

fn textured_scene(context: SceneContext, bounds: [(f32, f32); 2]) -> Scene {
    let mut scene = Scene::new(
        context,
        Viewport {
            physical_size: (32, 32),
            scale_factor: 1.0,
        },
        0.75,
    );
    let shape = scene.tessellate(&Shape::rect(bounds), None);
    scene.insert_node(
        DrawTreeNode::CachedShape(CachedShapeDrawData::new(
            shape,
            ShapeDrawCommandOptions::new()
                .background_texture_id(11)
                .foreground_texture_id(11),
        )),
        None,
    );
    scene
}

fn assert_texture_shapes(scene: &Scene, texture_id: u64, expected: &[usize]) {
    let mut shapes: Vec<_> = scene.texture_index.shapes_using(texture_id).collect();
    shapes.sort_unstable();
    assert_eq!(shapes, expected);
}

#[test]
fn insertion_indexes_shared_textures_and_deduplicates_layers_per_instance() {
    let mut scene = scene();
    let node = shape(
        &mut scene,
        ShapeDrawCommandOptions::new()
            .background_texture_id(0)
            .foreground_texture_id(2),
    );
    let first = scene.insert_node(node, None);
    let node = shape(
        &mut scene,
        ShapeDrawCommandOptions::new()
            .background_texture_id(2)
            .foreground_texture_id(2),
    );
    let second = scene.insert_node(node, Some(first));
    let node = shape(
        &mut scene,
        ShapeDrawCommandOptions::new().foreground_texture_id(u64::MAX),
    );
    let foreground_only = scene.insert_node(node, Some(first));
    let node = shape(&mut scene, ShapeDrawCommandOptions::new());
    let untextured = scene.insert_node(node, Some(first));
    let clip = Scene::prepare_clipping_rect([(0.0, 0.0), (16.0, 16.0)], None, true).unwrap();
    let clip = scene.insert_node(clip, Some(first));

    assert_eq!(scene.texture_index.textures_used_by(first), &[0, 2]);
    assert_eq!(scene.texture_index.textures_used_by(second), &[2]);
    assert_eq!(
        scene.texture_index.textures_used_by(foreground_only),
        &[u64::MAX]
    );
    assert!(scene.texture_index.textures_used_by(untextured).is_empty());
    assert!(scene.texture_index.textures_used_by(clip).is_empty());
    assert!(scene.texture_index.textures_used_by(usize::MAX).is_empty());
    assert_texture_shapes(&scene, 0, &[first]);
    assert_texture_shapes(&scene, 2, &[first, second]);
    assert_texture_shapes(&scene, u64::MAX, &[foreground_only]);
    assert_texture_shapes(&scene, 99, &[]);
}

#[test]
fn replacements_update_only_the_shapes_texture_associations() {
    let mut scene = scene();
    let node = shape(
        &mut scene,
        ShapeDrawCommandOptions::new()
            .background_texture_id(1)
            .foreground_texture_id(2),
    );
    let parent = scene.insert_node(node, None);
    let node = shape(
        &mut scene,
        ShapeDrawCommandOptions::new().background_texture_id(1),
    );
    let child = scene.insert_node(node, Some(parent));
    for options in [
        ShapeDrawCommandOptions::new()
            .background_texture_id(2)
            .foreground_texture_id(3),
        ShapeDrawCommandOptions::new()
            .background_texture_id(3)
            .foreground_texture_id(2),
        ShapeDrawCommandOptions::new()
            .background_texture_id(3)
            .foreground_texture_id(2),
    ] {
        let node = shape(&mut scene, options);
        scene.replace_node(parent, node, None, 4096);
        assert_texture_shapes(&scene, 1, &[child]);
        assert_texture_shapes(&scene, 2, &[parent]);
        assert_texture_shapes(&scene, 3, &[parent]);
        assert_eq!(scene.texture_index.textures_used_by(child), &[1]);
    }

    let clip = Scene::prepare_clipping_rect([(0.0, 0.0), (16.0, 16.0)], None, true).unwrap();
    scene.replace_node(parent, clip, None, 4096);
    assert!(scene.texture_index.textures_used_by(parent).is_empty());
    assert_texture_shapes(&scene, 1, &[child]);
    assert_texture_shapes(&scene, 2, &[]);
    assert_texture_shapes(&scene, 3, &[]);
    assert_eq!(scene.draw_tree.children(parent), &[child]);

    let node = shape(
        &mut scene,
        ShapeDrawCommandOptions::new().background_texture_id(4),
    );
    scene.replace_node(parent, node, None, 4096);
    assert_eq!(scene.texture_index.textures_used_by(parent), &[4]);
    assert_texture_shapes(&scene, 4, &[parent]);
    let node = shape(&mut scene, ShapeDrawCommandOptions::new());
    scene.replace_node(parent, node, None, 4096);
    assert!(scene.texture_index.textures_used_by(parent).is_empty());
    assert_texture_shapes(&scene, 4, &[]);
    assert!(scene.texture_index.shapes_by_texture.contains_key(&1));
    assert_eq!(scene.texture_index.shapes_by_texture.len(), 1);
}

#[test]
fn subtree_removal_clears_descendants_and_keeps_reused_ids_free_of_old_textures() {
    let mut scene = scene();
    let root = Scene::prepare_clipping_rect([(0.0, 0.0), (32.0, 32.0)], None, true).unwrap();
    let root = scene.insert_node(root, None);
    let branch = Scene::prepare_clipping_rect([(0.0, 0.0), (16.0, 16.0)], None, true).unwrap();
    let branch = scene.insert_node(branch, Some(root));
    let node = shape(
        &mut scene,
        ShapeDrawCommandOptions::new()
            .background_texture_id(1)
            .foreground_texture_id(2),
    );
    let child = scene.insert_node(node, Some(branch));
    let node = shape(
        &mut scene,
        ShapeDrawCommandOptions::new().background_texture_id(2),
    );
    let descendant = scene.insert_node(node, Some(child));
    let node = shape(
        &mut scene,
        ShapeDrawCommandOptions::new().background_texture_id(1),
    );
    let sibling = scene.insert_node(node, Some(root));
    scene.remove_subtrees_with([child, branch, child, usize::MAX], |_, _, _| {});

    for removed in [branch, child, descendant] {
        assert!(scene.texture_index.textures_used_by(removed).is_empty());
    }
    assert_texture_shapes(&scene, 1, &[sibling]);
    assert_texture_shapes(&scene, 2, &[]);

    let node = shape(
        &mut scene,
        ShapeDrawCommandOptions::new().foreground_texture_id(3),
    );
    let reused = scene.insert_node(node, Some(root));
    assert!([branch, child, descendant].contains(&reused));
    assert_eq!(scene.texture_index.textures_used_by(reused), &[3]);
    assert_texture_shapes(&scene, 1, &[sibling]);
    assert_texture_shapes(&scene, 3, &[reused]);
    scene.remove_subtrees_with([root, reused, root], |_, _, _| {});
    assert!(scene.texture_index.shapes_by_texture.is_empty());
    assert!(scene.texture_index.textures_by_shape.is_empty());

    let node = shape(&mut scene, ShapeDrawCommandOptions::new());
    let rebuilt = scene.insert_node(node, None);
    assert_eq!(rebuilt, 0);
    assert!(scene.texture_index.textures_used_by(rebuilt).is_empty());
    assert!(scene.texture_index.shapes_by_texture.is_empty());
}

#[test]
fn shared_context_scenes_consume_texture_damage_independently() {
    let updates = TextureUpdatePublisher::default();
    let context = SceneContext::new(updates.clone());
    let mut first = textured_scene(context.clone(), [(8.0, 8.0), (16.0, 16.0)]);
    let mut second = textured_scene(context.clone(), [(20.0, 20.0), (28.0, 28.0)]);
    let first_bounds = Some(UnsignedPhysicalRect::new((7, 7).into(), (17, 17).into()));
    let second_bounds = Some(UnsignedPhysicalRect::new((19, 19).into(), (29, 29).into()));

    updates.publish(11);
    updates.publish(11);
    assert_eq!(first.texture_damage(), first_bounds);
    assert_eq!(first.texture_damage(), None);

    updates.publish(11);
    drop(first);
    let mut replacement = textured_scene(context, [(8.0, 8.0), (16.0, 16.0)]);
    assert_eq!(replacement.texture_damage(), None);
    assert_eq!(second.texture_damage(), second_bounds);
    assert_eq!(second.texture_damage(), None);

    updates.clone().publish(11);
    assert_eq!(replacement.texture_damage(), first_bounds);
    assert_eq!(second.texture_damage(), second_bounds);
}
