use super::super::{Scene, TestBackend, TestSurface};
use crate::core::vertex::InstanceTransform;
use crate::core::{
    BackdropCaptureArea, BackdropEffectConfig, Shape, ShapeDrawCommandOptions, UnsignedPhysicalRect,
};
use crate::render_backend::render_target::Surface;
use crate::scene_state::SceneStateContext;

fn scene() -> (Scene<TestBackend>, Surface<TestSurface>) {
    let size = (1024, 512);
    let mut scene = Scene::from_parts(
        TestBackend {
            size: Some(size),
            ..Default::default()
        },
        SceneStateContext::default(),
    );
    scene
        .add_clipping_rect(
            [(0.0, 0.0), (1024.0, 512.0)],
            None,
            None::<InstanceTransform>,
            false,
        )
        .unwrap();
    let surface = Surface::from_resource(TestSurface::default(), size, true);
    (scene, surface)
}

fn shape(scene: &mut Scene<TestBackend>, bounds: [(f32, f32); 2]) -> usize {
    scene
        .add_shape(
            Shape::rect(bounds),
            Some(0),
            None,
            ShapeDrawCommandOptions::new(),
        )
        .unwrap()
}

fn capture(bounds: [(f32, f32); 2]) -> BackdropEffectConfig {
    BackdropEffectConfig::new().capture_area(BackdropCaptureArea::ScreenRect(bounds))
}

fn backdrop(
    scene: &mut Scene<TestBackend>,
    output: [(f32, f32); 2],
    input: [(f32, f32); 2],
) -> usize {
    let node = shape(scene, output);
    scene
        .set_backdrop_effect(node, 7, &[1, 2, 3, 4], capture(input))
        .unwrap();
    node
}

fn rect(min: (u32, u32), max: (u32, u32)) -> Option<UnsignedPhysicalRect> {
    Some(UnsignedPhysicalRect::new(min.into(), max.into()))
}

#[test]
fn removing_input_propagates_through_separate_capture_and_output_regions() {
    let (mut scene, mut surface) = scene();
    let input = shape(&mut scene, [(8.0, 8.0), (16.0, 16.0)]);
    backdrop(
        &mut scene,
        [(256.0, 8.0), (280.0, 32.0)],
        [(0.0, 0.0), (64.0, 64.0)],
    );
    backdrop(
        &mut scene,
        [(512.0, 8.0), (536.0, 32.0)],
        [(256.0, 0.0), (320.0, 64.0)],
    );
    scene.render(&mut surface).unwrap();

    scene.remove_subtrees([input], |_| {});
    scene.backend.should_fail = true;
    assert!(scene.render(&mut surface).is_err());
    scene.backend.should_fail = false;
    scene.render(&mut surface).unwrap();
    assert_eq!(scene.backend.root_scissor, rect((0, 0), (536, 64)));
    scene.render(&mut surface).unwrap();
    assert_eq!(scene.backend.root_scissor, None);

    let unrelated = shape(&mut scene, [(900.0, 400.0), (916.0, 416.0)]);
    scene.render(&mut surface).unwrap();
    assert_eq!(scene.backend.root_scissor, rect((900, 400), (916, 416)));
    scene.remove_subtrees([unrelated], |_| {});
    scene.render(&mut surface).unwrap();
    assert_eq!(scene.backend.root_scissor, rect((900, 400), (916, 416)));

    scene.change_scale_factor(2.0).unwrap();
    surface.resize((2048, 1024));
    scene.render(&mut surface).unwrap();
    shape(&mut scene, [(8.0, 8.0), (16.0, 16.0)]);
    scene.render(&mut surface).unwrap();
    assert_eq!(scene.backend.root_scissor, rect((0, 0), (1072, 128)));
}

#[test]
fn backdrop_mutations_damage_shape_and_reconstruct_current_capture() {
    let (mut scene, mut surface) = scene();
    let node = shape(&mut scene, [(16.0, 128.0), (48.0, 160.0)]);
    scene.render(&mut surface).unwrap();
    let config = capture([(512.0, 128.0), (544.0, 160.0)]);
    scene
        .set_backdrop_effect(node, 7, &[1, 2, 3, 4], config)
        .unwrap();
    scene.render(&mut surface).unwrap();
    assert_eq!(scene.backend.root_scissor, rect((16, 128), (544, 160)));

    scene
        .set_backdrop_effect(node, 7, &[1, 2, 3, 4], config)
        .unwrap();
    scene.render(&mut surface).unwrap();
    assert_eq!(scene.backend.root_scissor, rect((16, 128), (544, 160)));

    let moved = capture([(768.0, 128.0), (800.0, 160.0)]);
    scene
        .set_backdrop_effect(node, 7, &[1, 2, 3, 4], moved)
        .unwrap();
    scene.render(&mut surface).unwrap();
    assert_eq!(scene.backend.root_scissor, rect((16, 128), (800, 160)));

    let previous_capture = scene.state.backdrop_effects[&node].capture_region;
    assert!(scene
        .set_backdrop_effect(node, 7, &[1, 2, 3, 4], config.padding(-1.0))
        .is_err());
    let attachment = scene.state.backdrop_effects[&node];
    assert_eq!(attachment.config, moved);
    assert_eq!(attachment.capture_region, previous_capture);
    scene.render(&mut surface).unwrap();
    assert_eq!(scene.backend.root_scissor, None);

    let rejected_input = shape(&mut scene, [(524.0, 140.0), (528.0, 144.0)]);
    scene.render(&mut surface).unwrap();
    assert_eq!(scene.backend.root_scissor, rect((524, 140), (528, 144)));
    let retained_input = shape(&mut scene, [(780.0, 140.0), (784.0, 144.0)]);
    scene.render(&mut surface).unwrap();
    assert_eq!(scene.backend.root_scissor, rect((16, 128), (800, 160)));
    scene.remove_subtrees([rejected_input, retained_input], |_| {});
    scene.render(&mut surface).unwrap();

    let oversized = capture([(0.0, 0.0), (1_000_000.0, 512.0)]);
    scene
        .set_backdrop_effect(node, 7, &[1, 2, 3, 4], oversized)
        .unwrap();
    scene.render(&mut surface).unwrap();
    assert_eq!(scene.backend.root_scissor, rect((16, 128), (48, 160)));

    scene
        .set_backdrop_effect(node, 7, &[4, 3, 2, 1], oversized)
        .unwrap();
    scene.render(&mut surface).unwrap();
    assert_eq!(scene.backend.root_scissor, rect((16, 128), (48, 160)));

    scene
        .set_backdrop_effect(node, 7, &[1, 2, 3, 4], oversized)
        .unwrap();
    scene.render(&mut surface).unwrap();
    assert_eq!(scene.backend.root_scissor, rect((16, 128), (48, 160)));

    scene.remove_backdrop_effect(node);
    scene.render(&mut surface).unwrap();
    assert_eq!(scene.backend.root_scissor, rect((16, 128), (48, 160)));
    let input = shape(&mut scene, [(780.0, 140.0), (784.0, 144.0)]);
    scene.render(&mut surface).unwrap();
    assert_eq!(scene.backend.root_scissor, rect((780, 140), (784, 144)));
    scene.remove_subtrees([input], |_| {});
}

#[test]
fn removed_backdrops_do_not_leave_dependencies_when_node_ids_are_reused() {
    let (mut scene, mut surface) = scene();
    let output = [(512.0, 8.0), (536.0, 32.0)];
    let input = [(0.0, 0.0), (64.0, 64.0)];
    let node = backdrop(&mut scene, output, input);
    scene.render(&mut surface).unwrap();
    let oversized = capture([(0.0, 0.0), (1_000_000.0, 512.0)]);
    scene
        .set_backdrop_effect(node, 7, &[1, 2, 3, 4], oversized)
        .unwrap();
    scene.render(&mut surface).unwrap();
    scene.unload_effect(7);
    scene.render(&mut surface).unwrap();
    assert_eq!(scene.backend.root_scissor, rect((512, 8), (536, 32)));

    scene
        .set_backdrop_effect(node, 7, &[1, 2, 3, 4], oversized)
        .unwrap();
    scene.render(&mut surface).unwrap();
    scene.load_effect(7, &["changed shader"]).unwrap();
    scene.render(&mut surface).unwrap();
    assert_eq!(scene.backend.root_scissor, rect((512, 8), (536, 32)));

    scene
        .set_backdrop_effect(node, 7, &[1, 2, 3, 4], capture(input))
        .unwrap();
    scene.render(&mut surface).unwrap();
    scene.remove_subtrees([node], |_| {});
    assert_eq!(shape(&mut scene, output), node);
    scene.render(&mut surface).unwrap();
    shape(&mut scene, [(8.0, 8.0), (16.0, 16.0)]);
    scene.render(&mut surface).unwrap();
    assert_eq!(scene.backend.root_scissor, rect((8, 8), (16, 16)));
}
