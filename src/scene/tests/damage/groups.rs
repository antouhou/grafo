use super::super::{Scene, TestBackend, TestSurface};
use crate::core::vertex::InstanceTransform;
use crate::core::{
    BackdropCaptureArea, BackdropEffectConfig, ContentEffectConfig, Shape, ShapeDrawCommandOptions,
    ShapeMaskEffectConfig, UnsignedPhysicalRect,
};
use crate::render_backend::render_target::Surface;
use crate::scene_state::SceneStateContext;

fn scene() -> (Scene<TestBackend>, Surface<TestSurface>) {
    let size = (128, 128);
    let mut scene = Scene::from_parts(
        TestBackend {
            size: Some(size),
            ..Default::default()
        },
        SceneStateContext::default(),
    );
    scene
        .add_clipping_rect(
            [(0.0, 0.0), (128.0, 128.0)],
            None,
            None::<InstanceTransform>,
            false,
        )
        .unwrap();
    let surface = Surface::from_resource(TestSurface::default(), size, true);
    (scene, surface)
}

fn shape(scene: &mut Scene<TestBackend>, parent: usize, bounds: [(f32, f32); 2]) -> usize {
    scene
        .add_shape(
            Shape::rect(bounds),
            Some(parent),
            None,
            ShapeDrawCommandOptions::new().clips_children(false),
        )
        .unwrap()
}

fn group(scene: &mut Scene<TestBackend>, node: usize, outset: f32) {
    scene
        .set_content_effect(
            node,
            8,
            &[1, 2, 3, 4],
            ContentEffectConfig::new().outset(outset),
        )
        .unwrap();
}

fn rect(min: (u32, u32), max: (u32, u32)) -> Option<UnsignedPhysicalRect> {
    Some(UnsignedPhysicalRect::new(min.into(), max.into()))
}

#[test]
fn child_damage_reaches_group_output_backdrop_and_another_group_on_the_same_node() {
    let (mut scene, mut surface) = scene();
    let first = shape(&mut scene, 0, [(20.0, 20.0), (40.0, 40.0)]);
    let child = shape(&mut scene, first, [(26.0, 26.0), (28.0, 28.0)]);
    group(&mut scene, first, 10.0);
    let second = shape(&mut scene, 0, [(90.0, 20.0), (100.0, 30.0)]);
    scene
        .set_backdrop_effect(
            second,
            7,
            &[1, 2, 3, 4],
            BackdropEffectConfig::new().capture_area(BackdropCaptureArea::ScreenRect([
                (45.0, 20.0),
                (50.0, 30.0),
            ])),
        )
        .unwrap();
    group(&mut scene, second, 5.0);
    scene.render(&mut surface).unwrap();

    scene.remove_subtrees([child], |_| {});
    scene.backend.should_fail = true;
    assert!(scene.render(&mut surface).is_err());
    scene.backend.should_fail = false;
    scene.render(&mut surface).unwrap();
    assert_eq!(scene.backend.root_scissor, rect((10, 10), (105, 50)));
    scene.render(&mut surface).unwrap();
    assert_eq!(scene.backend.root_scissor, None);
}

#[test]
fn group_updates_clear_old_outputs_and_rejected_updates_preserve_damage() {
    let (mut scene, mut surface) = scene();
    let node = shape(&mut scene, 0, [(20.0, 20.0), (40.0, 40.0)]);
    scene.render(&mut surface).unwrap();
    group(&mut scene, node, 6.0);
    scene.render(&mut surface).unwrap();
    assert_eq!(scene.backend.root_scissor, rect((14, 14), (46, 46)));
    group(&mut scene, node, 2.0);
    scene.render(&mut surface).unwrap();
    assert_eq!(scene.backend.root_scissor, rect((14, 14), (46, 46)));
    group(&mut scene, node, 2.0);
    scene.render(&mut surface).unwrap();
    assert_eq!(scene.backend.root_scissor, rect((18, 18), (42, 42)));

    for (params, config) in [
        (&[1, 2, 3][..], ContentEffectConfig::default()),
        (&[1, 2, 3, 4][..], ContentEffectConfig::new().outset(-1.0)),
    ] {
        assert!(scene.set_content_effect(node, 8, params, config).is_err());
        assert_eq!(scene.state.content_effects[&node].config.left_outset, 2.0);
        scene.render(&mut surface).unwrap();
        assert_eq!(scene.backend.root_scissor, None);
    }
    scene.remove_content_effect(node);
    scene.render(&mut surface).unwrap();
    assert_eq!(scene.backend.root_scissor, rect((18, 18), (42, 42)));
    scene.remove_content_effect(node);
    scene.render(&mut surface).unwrap();
    assert_eq!(scene.backend.root_scissor, None);
}

#[test]
fn shape_mask_effect_changes_clear_the_old_padded_group_output() {
    for unload in [false, true] {
        let (mut scene, mut surface) = scene();
        let node = shape(&mut scene, 0, [(20.0, 20.0), (40.0, 40.0)]);
        group(&mut scene, node, 3.0);
        scene.render(&mut surface).unwrap();
        scene
            .set_shape_mask_effect(
                node,
                7,
                &[1, 2, 3, 4],
                ShapeMaskEffectConfig::new().outset(4.0),
            )
            .unwrap();
        scene.render(&mut surface).unwrap();
        assert_eq!(scene.backend.root_scissor, rect((12, 12), (48, 48)));
        scene
            .set_shape_mask_effect(
                node,
                7,
                &[1, 2, 3, 4],
                ShapeMaskEffectConfig::new().outset(1.0),
            )
            .unwrap();
        scene.render(&mut surface).unwrap();
        assert_eq!(scene.backend.root_scissor, rect((12, 12), (48, 48)));
        if unload {
            scene.unload_effect(7);
        } else {
            scene.remove_shape_mask_effect(node);
        }
        scene.render(&mut surface).unwrap();
        assert_eq!(scene.backend.root_scissor, rect((15, 15), (45, 45)));
    }
}

#[test]
fn removing_group_nodes_clears_outputs_and_removes_dependencies_before_id_reuse() {
    let (mut scene, mut surface) = scene();
    let node = shape(&mut scene, 0, [(20.0, 20.0), (40.0, 40.0)]);
    group(&mut scene, node, 10.0);
    scene.render(&mut surface).unwrap();
    scene.remove_subtrees([node], |_| {});
    scene.render(&mut surface).unwrap();
    assert_eq!(scene.backend.root_scissor, rect((10, 10), (50, 50)));
    assert_eq!(shape(&mut scene, 0, [(80.0, 80.0), (90.0, 90.0)]), node);
    scene.render(&mut surface).unwrap();
    shape(&mut scene, 0, [(26.0, 26.0), (28.0, 28.0)]);
    scene.render(&mut surface).unwrap();
    assert_eq!(scene.backend.root_scissor, rect((26, 26), (28, 28)));
}

#[test]
fn group_attachment_changes_damage_only_the_resolved_output() {
    let (mut scene, mut surface) = scene();
    let parent = shape(&mut scene, 0, [(20.0, 20.0), (40.0, 40.0)]);
    let child = shape(&mut scene, parent, [(90.0, 90.0), (100.0, 100.0)]);
    group(&mut scene, child, 10.0);
    scene.render(&mut surface).unwrap();
    group(&mut scene, parent, 10.0);
    scene.render(&mut surface).unwrap();
    assert_eq!(scene.backend.root_scissor, rect((10, 10), (50, 50)));
    group(&mut scene, parent, 5_000.0);
    assert!(scene.state.content_effects[&parent].bounds.is_none());
    scene.render(&mut surface).unwrap();
    assert_eq!(scene.backend.root_scissor, rect((10, 10), (50, 50)));
    group(&mut scene, parent, 10.0);
    scene.render(&mut surface).unwrap();
    assert_eq!(scene.backend.root_scissor, rect((10, 10), (50, 50)));
    scene.remove_content_effect(parent);
    scene.render(&mut surface).unwrap();
    assert_eq!(scene.backend.root_scissor, rect((10, 10), (50, 50)));
}

#[test]
fn replacing_a_group_marks_old_new_and_removed_group_outputs() {
    let (mut scene, mut surface) = scene();
    let node = shape(&mut scene, 0, [(20.0, 20.0), (40.0, 40.0)]);
    let child = shape(&mut scene, node, [(110.0, 110.0), (115.0, 115.0)]);
    group(&mut scene, child, 3.0);
    group(&mut scene, node, 10.0);
    scene.render(&mut surface).unwrap();
    scene
        .replace_with_shape(
            node,
            Shape::rect([(60.0, 60.0), (80.0, 80.0)]),
            None,
            ShapeDrawCommandOptions::new().clips_children(false),
        )
        .unwrap();
    scene.render(&mut surface).unwrap();
    assert_eq!(scene.backend.root_scissor, rect((10, 10), (90, 90)));
    scene
        .replace_with_clipping_rect(
            node,
            [(60.0, 60.0), (80.0, 80.0)],
            None::<InstanceTransform>,
            false,
        )
        .unwrap();
    scene.render(&mut surface).unwrap();
    assert_eq!(scene.backend.root_scissor, rect((50, 50), (90, 90)));
}

#[test]
fn group_shader_removal_damages_the_previous_output() {
    for unload in [false, true] {
        let (mut scene, mut surface) = scene();
        let node = shape(&mut scene, 0, [(20.0, 20.0), (40.0, 40.0)]);
        group(&mut scene, node, 10.0);
        scene.render(&mut surface).unwrap();
        if unload {
            scene.unload_effect(8);
        } else {
            scene.load_effect(8, &["changed group shader"]).unwrap();
        }
        scene.render(&mut surface).unwrap();
        assert_eq!(scene.backend.root_scissor, rect((10, 10), (50, 50)));
    }
}

#[test]
fn damage_in_group_padding_does_not_damage_the_whole_group() {
    let (mut scene, mut surface) = scene();
    let node = shape(&mut scene, 0, [(20.0, 20.0), (40.0, 40.0)]);
    group(&mut scene, node, 10.0);
    scene.render(&mut surface).unwrap();
    shape(&mut scene, 0, [(12.0, 25.0), (13.0, 26.0)]);
    scene.render(&mut surface).unwrap();
    assert_eq!(scene.backend.root_scissor, rect((12, 25), (13, 26)));
}

#[test]
fn viewport_changes_refresh_group_input_and_full_resolution_output() {
    let (mut scene, mut surface) = scene();
    let node = shape(&mut scene, 0, [(20.0, 20.0), (40.0, 40.0)]);
    scene
        .set_content_effect(
            node,
            8,
            &[1, 2, 3, 4],
            ContentEffectConfig::new().outset(5.0).downsample(0.25),
        )
        .unwrap();
    scene.render(&mut surface).unwrap();
    scene.change_scale_factor(1.5).unwrap();
    surface.resize((192, 192));
    scene.render(&mut surface).unwrap();
    let child = shape(&mut scene, node, [(30.0, 30.0), (32.0, 32.0)]);
    scene.render(&mut surface).unwrap();
    assert_eq!(scene.backend.root_scissor, rect((22, 22), (68, 68)));
    scene.remove_subtrees([child], |_| {});
    scene.render(&mut surface).unwrap();
    assert_eq!(scene.backend.root_scissor, rect((22, 22), (68, 68)));
}

#[test]
fn offscreen_input_keeps_visible_group_output_without_a_damage_dependency() {
    let (mut scene, mut surface) = scene();
    let node = shape(&mut scene, 0, [(-8.0, 20.0), (-2.0, 30.0)]);
    scene.render(&mut surface).unwrap();
    group(&mut scene, node, 6.0);
    scene.render(&mut surface).unwrap();
    assert_eq!(scene.backend.root_scissor, rect((0, 14), (4, 36)));
    shape(&mut scene, 0, [(1.0, 24.0), (2.0, 25.0)]);
    scene.render(&mut surface).unwrap();
    assert_eq!(scene.backend.root_scissor, rect((1, 24), (2, 25)));
    scene.remove_subtrees([node], |_| {});
    scene.render(&mut surface).unwrap();
    assert_eq!(scene.backend.root_scissor, rect((0, 14), (4, 36)));
}

#[test]
fn shape_mask_effect_changes_refresh_group_damage_when_allocation_is_skipped_or_restored() {
    let (mut scene, mut surface) = scene();
    let node = shape(&mut scene, 0, [(20.0, 20.0), (40.0, 40.0)]);
    scene
        .set_content_effect(
            node,
            8,
            &[1, 2, 3, 4],
            ContentEffectConfig::new().outsets(0.0, 0.0, 240.0, 0.0),
        )
        .unwrap();
    scene.render(&mut surface).unwrap();
    scene
        .set_shape_mask_effect(
            node,
            7,
            &[1, 2, 3, 4],
            ShapeMaskEffectConfig::new().outset(71.0),
        )
        .unwrap();
    assert!(scene.state.content_effects[&node].bounds.is_none());
    scene.render(&mut surface).unwrap();
    assert_eq!(scene.backend.root_scissor, rect((0, 0), (128, 112)));
    scene.unload_effect(7);
    assert!(scene.state.content_effects[&node].bounds.is_some());
    scene.render(&mut surface).unwrap();
    assert_eq!(scene.backend.root_scissor, rect((0, 0), (128, 112)));
}

#[test]
fn changing_ancestor_clipping_damages_descendant_group_outputs() {
    let (mut scene, mut surface) = scene();
    let parent = shape(&mut scene, 0, [(20.0, 20.0), (40.0, 40.0)]);
    let child = shape(&mut scene, parent, [(90.0, 90.0), (100.0, 100.0)]);
    group(&mut scene, child, 10.0);
    scene.render(&mut surface).unwrap();
    scene
        .replace_with_shape(
            parent,
            Shape::rect([(20.0, 20.0), (40.0, 40.0)]),
            None,
            ShapeDrawCommandOptions::new(),
        )
        .unwrap();
    scene.render(&mut surface).unwrap();
    assert_eq!(scene.backend.root_scissor, rect((20, 20), (110, 110)));
}
