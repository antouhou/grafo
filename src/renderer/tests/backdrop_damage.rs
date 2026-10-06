use super::{Renderer, TestBackend, TestSurface};
use crate::core::vertex::InstanceTransform;
use crate::core::{
    BackdropCaptureArea, BackdropEffectConfig, Shape, ShapeDrawCommandOptions, UnsignedPhysicalRect,
};
use crate::render_backend::render_target::Surface;
use crate::scene::SceneContext;

fn scene() -> (Renderer<TestBackend>, Surface<TestSurface>) {
    let size = (1024, 512);
    let mut renderer = Renderer::from_backend(
        TestBackend {
            size: Some(size),
            ..Default::default()
        },
        SceneContext::default(),
    );
    renderer
        .add_clipping_rect(
            [(0.0, 0.0), (1024.0, 512.0)],
            None,
            None::<InstanceTransform>,
            false,
        )
        .unwrap();
    let surface = Surface::from_resource(TestSurface::default(), size, true);
    (renderer, surface)
}

fn shape(renderer: &mut Renderer<TestBackend>, bounds: [(f32, f32); 2]) -> usize {
    renderer
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
    renderer: &mut Renderer<TestBackend>,
    output: [(f32, f32); 2],
    input: [(f32, f32); 2],
) -> usize {
    let node = shape(renderer, output);
    renderer
        .set_shape_backdrop_effect(node, 7, &[1, 2, 3, 4], capture(input))
        .unwrap();
    node
}

fn rect(min: (u32, u32), max: (u32, u32)) -> Option<UnsignedPhysicalRect> {
    Some(UnsignedPhysicalRect::new(min.into(), max.into()))
}

#[test]
fn removing_input_propagates_through_separate_capture_and_output_regions() {
    let (mut renderer, mut surface) = scene();
    let input = shape(&mut renderer, [(8.0, 8.0), (16.0, 16.0)]);
    backdrop(
        &mut renderer,
        [(256.0, 8.0), (280.0, 32.0)],
        [(0.0, 0.0), (64.0, 64.0)],
    );
    backdrop(
        &mut renderer,
        [(512.0, 8.0), (536.0, 32.0)],
        [(256.0, 0.0), (320.0, 64.0)],
    );
    renderer.render(&mut surface).unwrap();

    renderer.remove_subtrees([input], |_| {});
    renderer.backend.should_fail = true;
    assert!(renderer.render(&mut surface).is_err());
    renderer.backend.should_fail = false;
    renderer.render(&mut surface).unwrap();
    assert_eq!(renderer.backend.root_scissor, rect((0, 0), (537, 64)));
    renderer.render(&mut surface).unwrap();
    assert_eq!(renderer.backend.root_scissor, None);

    let unrelated = shape(&mut renderer, [(900.0, 400.0), (916.0, 416.0)]);
    renderer.render(&mut surface).unwrap();
    assert_eq!(renderer.backend.root_scissor, rect((899, 399), (917, 417)));
    renderer.remove_subtrees([unrelated], |_| {});
    renderer.render(&mut surface).unwrap();
    assert_eq!(renderer.backend.root_scissor, rect((899, 399), (917, 417)));

    renderer.change_scale_factor(2.0).unwrap();
    renderer.set_fringe_width(2.0).unwrap();
    surface.resize((2048, 1024));
    renderer.render(&mut surface).unwrap();
    shape(&mut renderer, [(8.0, 8.0), (16.0, 16.0)]);
    renderer.render(&mut surface).unwrap();
    assert_eq!(renderer.backend.root_scissor, rect((0, 0), (1074, 128)));
}

#[test]
fn backdrop_mutations_damage_shape_and_reconstruct_current_capture() {
    let (mut renderer, mut surface) = scene();
    let node = shape(&mut renderer, [(16.0, 128.0), (48.0, 160.0)]);
    renderer.render(&mut surface).unwrap();
    let config = capture([(512.0, 128.0), (544.0, 160.0)]);
    renderer
        .set_shape_backdrop_effect(node, 7, &[1, 2, 3, 4], config)
        .unwrap();
    renderer.render(&mut surface).unwrap();
    assert_eq!(renderer.backend.root_scissor, rect((15, 127), (544, 161)));

    renderer
        .update_backdrop_effect_params(node, &[1, 2, 3, 4])
        .unwrap();
    renderer.render(&mut surface).unwrap();
    assert_eq!(renderer.backend.root_scissor, rect((15, 127), (544, 161)));

    let moved = capture([(768.0, 128.0), (800.0, 160.0)]);
    renderer.update_backdrop_effect_config(node, moved).unwrap();
    renderer.render(&mut surface).unwrap();
    assert_eq!(renderer.backend.root_scissor, rect((15, 127), (800, 161)));

    assert!(renderer
        .update_backdrop_effect_config(node, moved.padding(-1.0))
        .is_err());
    renderer.render(&mut surface).unwrap();
    assert_eq!(renderer.backend.root_scissor, None);

    let oversized = capture([(0.0, 0.0), (1_000_000.0, 512.0)]);
    renderer
        .update_backdrop_effect_config(node, oversized)
        .unwrap();
    renderer.render(&mut surface).unwrap();
    assert_eq!(renderer.backend.root_scissor, rect((15, 127), (49, 161)));

    renderer
        .update_backdrop_effect_params(node, &[4, 3, 2, 1])
        .unwrap();
    renderer.render(&mut surface).unwrap();
    assert_eq!(renderer.backend.root_scissor, rect((15, 127), (49, 161)));

    renderer
        .set_shape_backdrop_effect(node, 7, &[1, 2, 3, 4], oversized)
        .unwrap();
    renderer.render(&mut surface).unwrap();
    assert_eq!(renderer.backend.root_scissor, rect((15, 127), (49, 161)));

    renderer.remove_backdrop_effect(node);
    renderer.render(&mut surface).unwrap();
    assert_eq!(renderer.backend.root_scissor, rect((15, 127), (49, 161)));
    let input = shape(&mut renderer, [(780.0, 140.0), (784.0, 144.0)]);
    renderer.render(&mut surface).unwrap();
    assert_eq!(renderer.backend.root_scissor, rect((779, 139), (785, 145)));
    renderer.remove_subtrees([input], |_| {});
}

#[test]
fn removed_backdrops_do_not_leave_dependencies_when_node_ids_are_reused() {
    let (mut renderer, mut surface) = scene();
    let output = [(512.0, 8.0), (536.0, 32.0)];
    let input = [(0.0, 0.0), (64.0, 64.0)];
    let node = backdrop(&mut renderer, output, input);
    renderer.render(&mut surface).unwrap();
    let oversized = capture([(0.0, 0.0), (1_000_000.0, 512.0)]);
    renderer
        .update_backdrop_effect_config(node, oversized)
        .unwrap();
    renderer.render(&mut surface).unwrap();
    renderer.unload_effect(7);
    renderer.render(&mut surface).unwrap();
    assert_eq!(renderer.backend.root_scissor, rect((511, 7), (537, 33)));

    renderer
        .set_shape_backdrop_effect(node, 7, &[1, 2, 3, 4], oversized)
        .unwrap();
    renderer.render(&mut surface).unwrap();
    renderer.load_effect(7, &["changed shader"]).unwrap();
    renderer.render(&mut surface).unwrap();
    assert_eq!(renderer.backend.root_scissor, rect((511, 7), (537, 33)));

    renderer
        .set_shape_backdrop_effect(node, 7, &[1, 2, 3, 4], capture(input))
        .unwrap();
    renderer.render(&mut surface).unwrap();
    renderer.remove_subtrees([node], |_| {});
    assert_eq!(shape(&mut renderer, output), node);
    renderer.render(&mut surface).unwrap();
    shape(&mut renderer, [(8.0, 8.0), (16.0, 16.0)]);
    renderer.render(&mut surface).unwrap();
    assert_eq!(renderer.backend.root_scissor, rect((7, 7), (17, 17)));
}
