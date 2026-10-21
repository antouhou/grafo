use super::super::{Scene, TestBackend, TestSurface, TestTextureManager};
use crate::core::vertex::InstanceTransform;
use crate::core::{
    BackdropCaptureArea, BackdropEffectConfig, Shape, ShapeDrawCommandOptions, UnsignedPhysicalRect,
};
use crate::render_backend::render_target::Surface;
use crate::render_backend::texture_manager::TextureManager;
use crate::scene_state::SceneStateContext;

fn scene(texture_manager: TestTextureManager) -> (Scene<TestBackend>, Surface<TestSurface>) {
    let size = (128, 96);
    let context = SceneStateContext::new(texture_manager.updates.clone());
    let mut scene = Scene::from_parts(
        TestBackend {
            texture_manager,
            size: Some(size),
            ..TestBackend::default()
        },
        context,
    );
    scene
        .add_clipping_rect(
            [(0.0, 0.0), (128.0, 96.0)],
            None,
            None::<InstanceTransform>,
            false,
        )
        .unwrap();
    for (bounds, options) in [
        (
            [(8.0, 8.0), (16.0, 16.0)],
            ShapeDrawCommandOptions::new().background_texture_id(11),
        ),
        (
            [(40.0, 16.0), (48.0, 24.0)],
            ShapeDrawCommandOptions::new().foreground_texture_id(11),
        ),
        (
            [(100.0, 64.0), (108.0, 72.0)],
            ShapeDrawCommandOptions::new().background_texture_id(12),
        ),
    ] {
        scene
            .add_shape(Shape::rect(bounds), Some(0), None, options)
            .unwrap();
    }
    let surface = Surface::from_resource(TestSurface::default(), size, true);
    (scene, surface)
}

fn rect(min: (u32, u32), max: (u32, u32)) -> Option<UnsignedPhysicalRect> {
    Some(UnsignedPhysicalRect::new(min.into(), max.into()))
}

#[test]
fn texture_mutations_damage_current_users_and_survive_failed_rendering() {
    let manager = TestTextureManager::default();
    let (mut scene, mut surface) = scene(manager.clone());
    manager.allocate_texture_with_data(11, (1, 1), &[255; 4]);
    manager.allocate_texture(12, (1, 1));
    scene.render(&mut surface).unwrap();

    manager.allocate_texture_with_data(11, (1, 1), &[0; 4]);
    manager.allocate_texture_with_data(11, (1, 1), &[255; 4]);
    scene.render(&mut surface).unwrap();
    assert_eq!(scene.backend.root_scissor, rect((8, 8), (48, 24)));
    scene.render(&mut surface).unwrap();
    assert_eq!(scene.backend.root_scissor, None);

    manager.load_data_into_texture(11, (1, 1), &[0; 4]).unwrap();
    manager.load_data_into_texture(12, (1, 1), &[0; 4]).unwrap();
    scene.backend.should_fail = true;
    assert!(scene.render(&mut surface).is_err());
    assert_eq!(scene.dirty_bounds, rect((8, 8), (108, 72)));
    scene.backend.should_fail = false;
    scene.render(&mut surface).unwrap();
    assert_eq!(scene.backend.root_scissor, rect((8, 8), (108, 72)));

    manager.remove_texture(11);
    scene.render(&mut surface).unwrap();
    assert_eq!(scene.backend.root_scissor, rect((8, 8), (48, 24)));
    assert!(manager.load_data_into_texture(11, (1, 1), &[0; 4]).is_err());
    manager.remove_texture(11);
    manager.allocate_texture(99, (1, 1));
    scene.render(&mut surface).unwrap();
    assert_eq!(scene.backend.root_scissor, None);

    manager.allocate_texture(11, (1, 1));
    scene.render(&mut surface).unwrap();
    assert_eq!(scene.backend.root_scissor, rect((8, 8), (48, 24)));
    manager.clear();
    scene.render(&mut surface).unwrap();
    assert_eq!(scene.backend.root_scissor, rect((8, 8), (108, 72)));
    manager.clear();
    scene.render(&mut surface).unwrap();
    assert_eq!(scene.backend.root_scissor, None);
}

#[test]
fn shared_texture_updates_are_consumed_independently_by_scenes() {
    let manager = TestTextureManager::default();
    manager.allocate_texture(11, (1, 1));
    let (mut first, mut first_surface) = scene(manager.clone());
    let (mut second, mut second_surface) = scene(manager.clone());
    first.render(&mut first_surface).unwrap();
    second.render(&mut second_surface).unwrap();

    manager
        .load_data_into_texture(11, (1, 1), &[255; 4])
        .unwrap();
    first.render(&mut first_surface).unwrap();
    assert_eq!(first.backend.root_scissor, rect((8, 8), (48, 24)));
    first.render(&mut first_surface).unwrap();
    assert_eq!(first.backend.root_scissor, None);
    second.render(&mut second_surface).unwrap();
    assert_eq!(second.backend.root_scissor, rect((8, 8), (48, 24)));
}

#[test]
fn texture_damage_is_expanded_through_chained_backdrop_captures() {
    let manager = TestTextureManager::default();
    let (mut scene, mut surface) = scene(manager.clone());
    for (output, input) in [
        ([(48.0, 8.0), (56.0, 16.0)], [(0.0, 0.0), (24.0, 24.0)]),
        ([(80.0, 8.0), (88.0, 16.0)], [(48.0, 0.0), (64.0, 32.0)]),
    ] {
        let node = scene
            .add_shape(
                Shape::rect(output),
                Some(0),
                None,
                ShapeDrawCommandOptions::new(),
            )
            .unwrap();
        scene
            .set_backdrop_effect(
                node,
                7,
                &[1, 2, 3, 4],
                BackdropEffectConfig::new().capture_area(BackdropCaptureArea::ScreenRect(input)),
            )
            .unwrap();
    }
    scene.render(&mut surface).unwrap();
    manager.allocate_texture_with_data(11, (1, 1), &[255; 4]);
    scene.render(&mut surface).unwrap();
    assert_eq!(scene.backend.root_scissor, rect((0, 0), (88, 32)));
    scene.render(&mut surface).unwrap();
    assert_eq!(scene.backend.root_scissor, None);
}

#[test]
fn texture_damage_uses_current_associations_after_node_slot_reuse() {
    let manager = TestTextureManager::default();
    let (mut scene, mut surface) = scene(manager.clone());
    scene.render(&mut surface).unwrap();
    scene.remove_subtrees([1, 2], |_| {});
    let reused = scene
        .add_shape(
            Shape::rect([(80.0, 80.0), (88.0, 88.0)]),
            Some(0),
            None,
            ShapeDrawCommandOptions::new().background_texture_id(12),
        )
        .unwrap();
    assert!([1, 2].contains(&reused));
    scene.render(&mut surface).unwrap();

    manager.allocate_texture(11, (1, 1));
    scene.render(&mut surface).unwrap();
    assert_eq!(scene.backend.root_scissor, None);
    manager.allocate_texture(12, (1, 1));
    scene.render(&mut surface).unwrap();
    assert_eq!(scene.backend.root_scissor, rect((80, 64), (108, 88)));
}

#[test]
fn updates_during_rendering_remain_pending_for_the_next_render() {
    let manager = TestTextureManager::default();
    let (mut scene, mut surface) = scene(manager.clone());
    manager.allocate_texture(11, (1, 1));
    scene.render(&mut surface).unwrap();
    scene.backend.texture_upload_during_render = Some(11);
    scene.render(&mut surface).unwrap();
    assert_eq!(scene.backend.root_scissor, None);
    scene.render(&mut surface).unwrap();
    assert_eq!(scene.backend.root_scissor, rect((8, 8), (48, 24)));
    scene.render(&mut surface).unwrap();
    assert_eq!(scene.backend.root_scissor, None);
}

#[test]
fn texture_damage_respects_transforms_scale_and_viewport_clipping() {
    let manager = TestTextureManager::default();
    let size = (64, 48);
    let mut scene = Scene::from_parts(
        TestBackend {
            texture_manager: manager.clone(),
            size: Some(size),
            ..TestBackend::default()
        },
        SceneStateContext::new(manager.updates.clone()),
    );
    scene.change_scale_factor(2.0).unwrap();
    scene
        .add_shape(
            Shape::rect([(0.0, 0.0), (16.0, 8.0)]),
            None,
            None,
            ShapeDrawCommandOptions::new()
                .background_texture_id(11)
                .transform(InstanceTransform::translation(24.0, 8.0)),
        )
        .unwrap();
    let mut surface = Surface::from_resource(TestSurface::default(), size, true);
    scene.render(&mut surface).unwrap();
    manager.allocate_texture(11, (1, 1));
    scene.render(&mut surface).unwrap();
    assert_eq!(scene.backend.root_scissor, rect((48, 16), (64, 32)));
}
