use super::{scene, surface, Scene, TestBackend};
use crate::core::{
    BlendMode, Color, ContentEffectConfig, PhysicalRect, Shape, ShapeDrawOptions,
    ShapeMaskEffectConfig, TransformInstance, UnsignedPhysicalRect,
};
use crate::scene_state::types::DrawTreeNode;

fn add_shape(
    scene: &mut Scene<TestBackend>,
    parent: usize,
    bounds: [(f32, f32); 2],
    blend_mode: BlendMode,
) -> usize {
    scene
        .add_shape(
            Shape::rect(bounds),
            Some(parent),
            None,
            ShapeDrawOptions::new()
                .color(Color::WHITE)
                .blend_mode(blend_mode)
                .clips_children(false),
        )
        .unwrap()
}

fn rect(min: (u32, u32), max: (u32, u32)) -> Option<UnsignedPhysicalRect> {
    Some(UnsignedPhysicalRect::new(min.into(), max.into()))
}

fn output(min: (i32, i32), max: (i32, i32)) -> Option<PhysicalRect> {
    Some(PhysicalRect::new(min.into(), max.into()))
}

fn assert_effect_free_group(
    scene: &Scene<TestBackend>,
    node_id: usize,
    input_bounds: Option<UnsignedPhysicalRect>,
    output_bounds: Option<PhysicalRect>,
) {
    let group = scene.state.content_groups[&node_id];
    assert!(group.effect.is_none());
    assert_eq!(group.input_bounds, input_bounds);
    assert_eq!(group.output_bounds, output_bounds);
}

#[test]
fn blend_only_shape_commands_create_and_replace_content_groups() {
    let mut scene = scene();
    let mut surface = surface();
    let root = scene.add_group(None, Default::default()).unwrap();
    let bounds = [(4.0, 6.0), (12.0, 14.0)];
    scene.load_shape(Shape::rect(bounds), 1, None).unwrap();
    let direct = add_shape(&mut scene, root, bounds, BlendMode::Multiply);
    assert_effect_free_group(
        &scene,
        direct,
        rect((4, 6), (12, 14)),
        output((4, 6), (12, 14)),
    );
    let cached = scene
        .add_cached_shape(
            1,
            Some(root),
            ShapeDrawOptions::new()
                .color(Color::WHITE)
                .blend_mode(BlendMode::Screen)
                .transform(TransformInstance::translation(10.0, 0.0)),
        )
        .unwrap();
    assert_effect_free_group(
        &scene,
        cached,
        rect((14, 6), (22, 14)),
        output((14, 6), (22, 14)),
    );
    assert!(scene.state.effect_parameters.bytes.is_empty());

    let replacement = add_shape(&mut scene, root, bounds, BlendMode::SourceOver);
    assert!(!scene.state.content_groups.contains_key(&replacement));
    scene
        .replace_with_cached_shape(
            replacement,
            1,
            ShapeDrawOptions::new().blend_mode(BlendMode::Multiply),
        )
        .unwrap();
    assert_effect_free_group(
        &scene,
        replacement,
        rect((4, 6), (12, 14)),
        output((4, 6), (12, 14)),
    );
    scene
        .replace_with_shape(
            replacement,
            Shape::rect(bounds),
            None,
            ShapeDrawOptions::new(),
        )
        .unwrap();
    assert!(!scene.state.content_groups.contains_key(&replacement));

    scene
        .set_content_effect(
            direct,
            8,
            &[1, 2, 3, 4],
            ContentEffectConfig::new().outset(2.0),
        )
        .unwrap();
    scene
        .replace_with_cached_shape(direct, 1, ShapeDrawOptions::new().color(Color::WHITE))
        .unwrap();
    assert_eq!(
        scene.state.content_groups[&direct]
            .effect
            .unwrap()
            .effect_id,
        8
    );
    assert_eq!(
        scene.state.content_groups[&direct].output_bounds,
        output((2, 4), (14, 16))
    );
    scene.render(&mut surface).unwrap();
    assert_eq!(surface.resource().effects, [8]);
}

#[test]
fn blend_only_bounds_follow_masks_viewports_and_failed_replacements() {
    let mut scene = scene();
    let mut surface = surface();
    let root = scene.add_group(None, Default::default()).unwrap();
    let node = add_shape(
        &mut scene,
        root,
        [(20.0, 20.0), (40.0, 40.0)],
        BlendMode::Multiply,
    );
    assert_effect_free_group(
        &scene,
        node,
        rect((20, 20), (32, 32)),
        output((20, 20), (40, 40)),
    );
    scene
        .set_shape_mask_effect(
            node,
            7,
            &[1, 2, 3, 4],
            ShapeMaskEffectConfig::new().outset(2.0),
        )
        .unwrap();
    assert_effect_free_group(
        &scene,
        node,
        rect((17, 17), (32, 32)),
        output((17, 17), (43, 43)),
    );
    scene.remove_shape_mask_effect(node);
    assert_effect_free_group(
        &scene,
        node,
        rect((20, 20), (32, 32)),
        output((20, 20), (40, 40)),
    );
    scene
        .set_shape_mask_effect(
            node,
            7,
            &[1, 2, 3, 4],
            ShapeMaskEffectConfig::new().outset(2.0),
        )
        .unwrap();
    scene.unload_effect(7);
    assert_effect_free_group(
        &scene,
        node,
        rect((20, 20), (32, 32)),
        output((20, 20), (40, 40)),
    );

    scene.change_scale_factor(2.0).unwrap();
    assert_effect_free_group(&scene, node, None, output((40, 40), (80, 80)));
    scene.resize((16, 16));
    assert_effect_free_group(&scene, node, None, None);
    scene.resize((64, 64));
    assert_effect_free_group(
        &scene,
        node,
        rect((40, 40), (64, 64)),
        output((40, 40), (80, 80)),
    );
    scene.change_scale_factor(1.0).unwrap();
    assert_effect_free_group(
        &scene,
        node,
        rect((20, 20), (40, 40)),
        output((20, 20), (40, 40)),
    );
    scene.render(&mut surface).unwrap();
    assert!(scene.state.effect_parameters.bytes.is_empty());
    let before = scene.state.content_groups[&node];
    assert!(scene
        .set_shape_mask_effect(
            node,
            7,
            &[1, 2, 3, 4],
            ShapeMaskEffectConfig::new().outset(-1.0)
        )
        .is_err());
    scene.backend.registration_failure = Some(node);
    assert!(scene
        .replace_with_shape(
            node,
            Shape::rect([(1.0, 1.0), (3.0, 3.0)]),
            None,
            ShapeDrawOptions::new(),
        )
        .is_err());
    assert_effect_free_group(&scene, node, before.input_bounds, before.output_bounds);
    let DrawTreeNode::CachedShape(shape) = scene.state.draw_tree.get(node).unwrap() else {
        panic!("expected a shape node");
    };
    assert_eq!(shape.blend_mode, BlendMode::Multiply);
    let next_node = scene.state.next_node_id();
    let group_count = scene.state.content_groups.len();
    scene.backend.registration_failure = Some(next_node);
    assert!(scene
        .add_shape(
            Shape::rect([(1.0, 1.0), (3.0, 3.0)]),
            Some(root),
            None,
            ShapeDrawOptions::new().blend_mode(BlendMode::Screen),
        )
        .is_err());
    assert_eq!(scene.state.content_groups.len(), group_count);
    assert!(!scene.state.content_groups.contains_key(&next_node));
    assert!(scene.dirty_bounds.is_none());
}

#[test]
fn subtree_removal_clear_and_id_reuse_remove_blend_group_state() {
    let mut scene = scene();
    let mut surface = surface();
    let root = scene.add_group(None, Default::default()).unwrap();
    let parent = add_shape(
        &mut scene,
        root,
        [(4.0, 4.0), (12.0, 12.0)],
        BlendMode::Multiply,
    );
    let child = add_shape(
        &mut scene,
        parent,
        [(5.0, 5.0), (9.0, 9.0)],
        BlendMode::Screen,
    );
    add_shape(
        &mut scene,
        root,
        [(20.0, 20.0), (28.0, 28.0)],
        BlendMode::SourceOver,
    );
    assert!(scene.state.content_groups.contains_key(&parent));
    assert!(scene.state.content_groups.contains_key(&child));
    scene
        .set_content_effect(
            child,
            8,
            &[1, 2, 3, 4],
            ContentEffectConfig::new().outset(1.0),
        )
        .unwrap();
    scene.render(&mut surface).unwrap();

    let mut removed = Vec::new();
    scene.remove_subtrees([parent], |node_id| removed.push(node_id));
    assert_eq!(removed.len(), 2);
    assert!(removed.contains(&parent) && removed.contains(&child));
    assert!(scene.state.content_groups.is_empty());
    scene.render(&mut surface).unwrap();
    assert_eq!(scene.backend.root_scissor, rect((4, 4), (12, 12)));
    assert!(scene.state.effect_parameters.bytes.is_empty());

    let reused = add_shape(
        &mut scene,
        root,
        [(2.0, 2.0), (3.0, 3.0)],
        BlendMode::SourceOver,
    );
    assert!(removed.contains(&reused));
    assert!(!scene.state.content_groups.contains_key(&reused));
    scene.render(&mut surface).unwrap();
    assert_eq!(scene.backend.root_scissor, rect((2, 2), (3, 3)));
    scene
        .replace_with_shape(
            reused,
            Shape::rect([(16.0, 16.0), (20.0, 20.0)]),
            None,
            ShapeDrawOptions::new().blend_mode(BlendMode::Multiply),
        )
        .unwrap();
    assert!(scene.state.content_groups.contains_key(&reused));
    scene.clear();
    assert!(scene.state.draw_tree.is_empty());
    assert!(scene.state.content_groups.is_empty());
    assert!(scene.state.effect_parameters.bytes.is_empty());
    assert!(scene.backend.registered_shapes.is_empty());
}
