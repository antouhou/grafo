use super::{create_headless_scene_with_size_and_scale, render_bgra};
use grafo::{
    BackdropEffectConfig, BorderRadii, Color, ContentEffectConfig, Scene, Shape,
    ShapeDrawCommandOptions,
};
use grafo_test_scenes::shaders::PASSTHROUGH_WGSL;

#[derive(Clone, Copy, Debug)]
enum Effects {
    None,
    Content,
    ContentAndBackdrop,
}

fn populate_motion_scene(scene: &mut Scene, effects: Effects) -> [usize; 4] {
    let root = scene
        .add_shape(
            Shape::rect([(0.0, 0.0), (160.0, 120.0)]),
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::WHITE),
        )
        .unwrap();
    let background = scene
        .add_shape(
            Shape::rect([(10.0, 10.0), (70.0, 90.0)]),
            Some(root),
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(255, 0, 0)),
        )
        .unwrap();
    let group = scene
        .add_shape(
            Shape::rounded_rect([(20.0, 20.0), (100.0, 100.0)], BorderRadii::new(8.0)),
            Some(root),
            None,
            ShapeDrawCommandOptions::new().color(Color::rgba(0, 255, 0, 128)),
        )
        .unwrap();
    let child = scene
        .add_shape(
            Shape::rounded_rect([(40.0, 30.0), (110.0, 70.0)], BorderRadii::new(5.0)),
            Some(group),
            None,
            ShapeDrawCommandOptions::new().color(Color::rgba(0, 0, 255, 128)),
        )
        .unwrap();
    if !matches!(effects, Effects::None) {
        scene
            .set_content_effect(group, 9_401, &[], ContentEffectConfig::new().outset(4.0))
            .unwrap();
        scene
            .set_content_effect(
                child,
                9_401,
                &[],
                ContentEffectConfig::new().outset(2.0).downsample(0.5),
            )
            .unwrap();
    }
    if matches!(effects, Effects::ContentAndBackdrop) {
        scene
            .set_backdrop_effect(
                child,
                9_401,
                &[],
                BackdropEffectConfig::default().padding(6.0),
            )
            .unwrap();
    }
    let foreground = scene
        .add_shape(
            Shape::rect([(60.0, 10.0), (80.0, 80.0)]),
            Some(root),
            None,
            ShapeDrawCommandOptions::new().color(Color::rgba(255, 0, 255, 128)),
        )
        .unwrap();
    [background, foreground, child, group]
}

fn check_motion(scene: &mut Scene, pixels: &mut Vec<u8>, configuration: (u32, f64, Effects)) {
    let (samples, scale, effects) = configuration;
    scene.clear();
    scene.set_msaa_samples(samples);
    scene.change_scale_factor(scale).unwrap();
    let [background, foreground, child, group] = populate_motion_scene(scene, effects);
    render_bgra(scene, pixels).unwrap();

    // Movement must restore both the old position and the newly covered pixels.
    // A static tile cannot exercise retained-output damage or target reuse.
    let movements = [
        background, foreground, child, group, background, child, foreground, group,
    ];
    for (step, node) in movements.into_iter().enumerate() {
        let left = 5.0 + step as f32 * 7.0;
        scene
            .replace_with_shape(
                node,
                Shape::rounded_rect([(left, 15.0), (left + 45.0, 65.0)], BorderRadii::new(6.0)),
                None,
                ShapeDrawCommandOptions::new().color(Color::rgba(20 * step as u8, 180, 220, 160)),
            )
            .unwrap();
        render_bgra(scene, pixels).unwrap();
        let incremental = pixels.clone();
        scene.resize(scene.size());
        render_bgra(scene, pixels).unwrap();
        let differences = incremental
            .as_chunks::<4>()
            .0
            .iter()
            .zip(pixels.as_chunks::<4>().0)
            .filter(|(partial, full)| partial != full)
            .count();
        assert_eq!(
            differences, 0,
            "configuration {configuration:?}, movement {step}"
        );
    }
}

#[test]
fn moving_layers_preserve_damage_and_compositing_order() {
    let Some(mut scene) = create_headless_scene_with_size_and_scale((160, 120), 1.0) else {
        return;
    };
    scene.load_effect(9_401, &[PASSTHROUGH_WGSL]).unwrap();
    let mut pixels = Vec::new();
    for samples in [1, 4] {
        for scale in [1.0, 1.25, 2.0] {
            for effects in [Effects::None, Effects::Content, Effects::ContentAndBackdrop] {
                check_motion(&mut scene, &mut pixels, (samples, scale, effects));
            }
        }
    }
}
