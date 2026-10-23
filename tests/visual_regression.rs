//! Checks rendered pixels against expected colors.
//! Run with `cargo test --test visual_regression`.

use futures::executor::block_on;
use grafo::{
    BackdropCaptureArea, BackdropEffectConfig, BorderRadii, Color, ColorInterpolation,
    ContentEffectConfig, DrawCommandError, EffectError, EffectResourceError, Fill, Gradient,
    GradientStop, GradientStopOffset, LinearGradientDesc, LinearGradientLine, PixelFormat,
    PixelLayout, Pixmap, PixmapMut, Scene, SceneCreationError, SceneError, Shape,
    ShapeDrawCommandOptions, ShapeMaskEffectConfig, ShapeTextureFitMode, ShapeTextureOptions,
    TextureManager, TransformInstance, WgpuBackendError,
};
use grafo_test_scenes::shaders::{PASSTHROUGH_WGSL, SHAPE_DROP_WGSL};
use grafo_test_scenes::{
    build_bounded_targets_scene, build_main_scene, build_nested_targets_scene, check_pixels,
    PixelExpectation, CANVAS_HEIGHT, CANVAS_WIDTH,
};

#[path = "visual_regression/bounded_targets.rs"]
mod bounded_targets;

const CONSTANT_BLUE_SHAPE_MASK_EFFECT: &str = r#"
@fragment
fn effect_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 1.0, 1.0);
}
"#;

/// Creates a headless scene. If no suitable GPU adapter is available,
/// prints a skip message and returns `None`.
fn create_headless_scene() -> Option<Scene> {
    create_headless_scene_with_size_and_scale((CANVAS_WIDTH, CANVAS_HEIGHT), 1.0)
}

fn create_headless_scene_with_size_and_scale(
    physical_size: (u32, u32),
    scale_factor: f64,
) -> Option<Scene> {
    match block_on(Scene::try_new(physical_size, scale_factor, 1)) {
        Ok(r) => Some(r),
        Err(SceneCreationError::AdapterNotAvailable(_)) => {
            println!("Skipping test: no suitable GPU adapter available.");
            None
        }
        Err(e) => panic!("Failed to create headless scene: {e}"),
    }
}

fn assert_pixels_match(pixel_buffer: &[u8], expectations: &[PixelExpectation]) {
    let failures = check_pixels(pixel_buffer, CANVAS_WIDTH, CANVAS_HEIGHT, expectations);
    if !failures.is_empty() {
        let message = format!(
            "{} pixel expectation(s) failed:\n{}",
            failures.len(),
            failures.join("\n"),
        );
        panic!("{message}");
    }
}

fn render_bgra(scene: &mut Scene, pixels: &mut Vec<u8>) -> Result<(), WgpuBackendError> {
    let size = scene.size();
    pixels.resize(size.0 as usize * size.1 as usize * 4, 0);
    scene.render(PixmapMut::bgra8(pixels, size)?)
}

#[test]
fn root_replacement_and_partial_updates_preserve_retained_pixels() {
    let Some(mut scene) = create_headless_scene_with_size_and_scale((64, 64), 1.0) else {
        return;
    };
    let mut pixels = Vec::new();
    for samples in [1, 4] {
        scene.set_msaa_samples(samples);
        let root = scene
            .add_shape(
                Shape::rounded_rect([(0.0, 0.0), (64.0, 64.0)], BorderRadii::new(8.0)),
                None,
                None,
                ShapeDrawCommandOptions::new().color(Color::WHITE),
            )
            .unwrap();
        render_bgra(&mut scene, &mut pixels).unwrap();
        let child = scene
            .add_shape(
                Shape::rect([(4.0, 4.0), (12.0, 12.0)]),
                Some(root),
                None,
                ShapeDrawCommandOptions::new().color(Color::rgb(255, 0, 0)),
            )
            .unwrap();
        render_bgra(&mut scene, &mut pixels).unwrap();
        assert_eq!(read_pixel_rgba(&pixels, 64, 8, 8), [255, 0, 0, 255]);
        assert_eq!(read_pixel_rgba(&pixels, 64, 56, 56), [255; 4]);

        scene
            .replace_with_shape(
                root,
                Shape::builder()
                    .begin((0.0, 0.0))
                    .line_to((64.0, 0.0))
                    .line_to((0.0, 64.0))
                    .close()
                    .build(),
                None,
                ShapeDrawCommandOptions::new().color(Color::rgb(0, 255, 0)),
            )
            .unwrap();
        scene
            .replace_with_shape(
                child,
                Shape::rect([(4.0, 4.0), (12.0, 12.0)]),
                None,
                ShapeDrawCommandOptions::new().color(Color::rgb(0, 0, 255)),
            )
            .unwrap();
        render_bgra(&mut scene, &mut pixels).unwrap();
        assert_eq!(read_pixel_rgba(&pixels, 64, 8, 8), [0, 0, 255, 255]);
        assert_eq!(read_pixel_rgba(&pixels, 64, 20, 8), [0, 255, 0, 255]);
        assert_eq!(read_pixel_rgba(&pixels, 64, 56, 56), [0; 4]);

        scene
            .replace_with_shape(
                child,
                Shape::rect([(16.0, 16.0), (24.0, 24.0)]),
                None,
                ShapeDrawCommandOptions::new().color(Color::rgb(255, 0, 0)),
            )
            .unwrap();
        render_bgra(&mut scene, &mut pixels).unwrap();
        assert_eq!(read_pixel_rgba(&pixels, 64, 8, 8), [0, 255, 0, 255]);
        assert_eq!(read_pixel_rgba(&pixels, 64, 20, 20), [255, 0, 0, 255]);
        assert_eq!(read_pixel_rgba(&pixels, 64, 56, 56), [0; 4]);
        scene.clear();
        render_bgra(&mut scene, &mut pixels).unwrap();
        assert!(pixels.iter().all(|&byte| byte == 0));
    }
}

#[test]
fn projected_rect_addition_and_removal_update_retained_pixels() {
    let size = (64, 64);
    let Some(mut scene) = create_headless_scene_with_size_and_scale(size, 1.0) else {
        return;
    };
    scene
        .load_effect(9_301, &[CACHED_SHAPE_MASK_EFFECT_RED_MASK])
        .unwrap();
    let mut transform = TransformInstance::identity();
    transform.col3[3] = -1.0;
    let cases = [
        ([(8.0, 8.0), (16.0, 16.0)], transform),
        (
            [(999_980.0, 8.0), (1_000_000.0, 16.0)],
            TransformInstance::affine_2d(1.1, 0.0, 0.0, 1.0, -1_099_984.0, 0.0),
        ),
        (
            [(999_981.5, 8.0), (1_000_000.0, 28.0)],
            TransformInstance::translation(-999_970.0, 0.0),
        ),
        (
            [(8.0, 999_981.5), (28.0, 1_000_000.0)],
            TransformInstance::translation(0.0, -999_970.0),
        ),
    ];
    let mut pixels = Vec::new();

    // A populated first render cannot exercise retained-output damage.
    for (samples, scale_factor) in [1, 4].into_iter().flat_map(|samples| {
        [1.0, 1.25, 1.5, 0.7, 1.3, 2.0].map(|scale_factor| (samples, scale_factor))
    }) {
        scene.set_msaa_samples(samples);
        scene.change_scale_factor(scale_factor).unwrap();
        for (bounds, transform, has_shape_mask_effect) in
            cases.into_iter().flat_map(|(bounds, transform)| {
                [false, true].map(|has_effect| (bounds, transform, has_effect))
            })
        {
            render_bgra(&mut scene, &mut pixels).unwrap();
            assert!(pixels.iter().all(|&byte| byte == 0));
            let node = scene
                .add_shape(
                    Shape::rect(bounds),
                    None,
                    None,
                    ShapeDrawCommandOptions::new()
                        .color(Color::rgb(255, 0, 0))
                        .transform(transform),
                )
                .unwrap();
            if has_shape_mask_effect {
                scene
                    .set_shape_mask_effect(
                        node,
                        9_301,
                        &[],
                        ShapeMaskEffectConfig::new().outset(3.0),
                    )
                    .unwrap();
            }
            render_bgra(&mut scene, &mut pixels).unwrap();
            let center = (14.0 * scale_factor) as u32;
            assert_eq!(
                read_pixel_rgba(&pixels, size.0, center, center),
                [255, 0, 0, 255]
            );
            let incremental = pixels.clone();
            scene.resize(size);
            render_bgra(&mut scene, &mut pixels).unwrap();
            let redraw_difference = pixels
                .as_chunks::<4>()
                .0
                .iter()
                .zip(incremental.as_chunks::<4>().0)
                .filter(|(full, incremental)| full != incremental)
                .count();
            scene.remove_subtrees([node], |_| {});
            render_bgra(&mut scene, &mut pixels).unwrap();
            let residual_pixels = pixels
                .as_chunks::<4>()
                .0
                .iter()
                .filter(|pixel| pixel.iter().any(|&byte| byte != 0))
                .count();
            assert_eq!(
                (redraw_difference, residual_pixels),
                (0, 0),
                "bounds {bounds:?}, transform {transform:?}, scale {scale_factor}, \
                 samples {samples}, shape mask effect {has_shape_mask_effect}"
            );
        }
    }
}

#[test]
fn rejected_rasterization_changes_preserve_shape_mask_effect_pixels() {
    let Some(mut scene) = create_headless_scene_with_size_and_scale((512, 256), 1.0) else {
        return;
    };
    scene
        .load_effect(9_302, &[CONSTANT_BLUE_SHAPE_MASK_EFFECT])
        .unwrap();
    let node = scene
        .add_shape(
            Shape::rect([(1_500_000_000.0, 0.0), (1_500_000_128.0, 64.0)]),
            None,
            None,
            ShapeDrawCommandOptions::new()
                .transform(TransformInstance::translation(-1_500_000_000.0, 0.0)),
        )
        .unwrap();
    scene
        .set_shape_mask_effect(node, 9_302, &[], ShapeMaskEffectConfig::new().outset(128.0))
        .unwrap();
    let mut pixels = Vec::new();
    render_bgra(&mut scene, &mut pixels).unwrap();
    assert_eq!(read_pixel_rgba(&pixels, 512, 192, 32), [0, 0, 255, 255]);
    assert_eq!(read_pixel_rgba(&pixels, 512, 384, 32), [0, 0, 0, 0]);
    let initial = pixels.clone();

    let _ = scene.change_scale_factor(2.0);
    render_bgra(&mut scene, &mut pixels).unwrap();
    assert_eq!(pixels, initial);
    scene.change_scale_factor(0.5).unwrap();
    render_bgra(&mut scene, &mut pixels).unwrap();
    assert_eq!(read_pixel_rgba(&pixels, 512, 64, 16), [0, 0, 255, 255]);
    assert_eq!(read_pixel_rgba(&pixels, 512, 192, 32), [0, 0, 0, 0]);
    let recovered = pixels.clone();

    render_bgra(&mut scene, &mut pixels).unwrap();
    assert_eq!(pixels, recovered);
    scene.change_scale_factor(1.0).unwrap();
    render_bgra(&mut scene, &mut pixels).unwrap();
    assert_eq!(pixels, initial);
    scene.remove_shape_mask_effect(node);
    render_bgra(&mut scene, &mut pixels).unwrap();
    assert!(pixels.iter().all(|&byte| byte == 0));
}

#[test]
fn shape_mask_effect_is_resolved_before_backdrop_capture_with_msaa() {
    // This test passes when no GPU present on purpose: I don't have a GPU on my CI just yet, so I'm
    //  running this test and the visual regression example manually before commiting changes
    let Some(mut scene) = create_headless_scene_with_size_and_scale((64, 64), 1.0) else {
        return;
    };
    scene.set_msaa_samples(4);
    scene
        .load_effect(9_101, &[SHAPE_DROP_WGSL])
        .expect("to load the MSAA shape mask effect");
    scene
        .load_effect(9_102, &[PASSTHROUGH_WGSL])
        .expect("to load the MSAA backdrop effect");

    scene
        .add_shape(
            Shape::rect([(0.0, 0.0), (64.0, 64.0)]),
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(220, 200, 50)),
        )
        .unwrap();
    let panel_id = scene
        .add_shape(
            Shape::rect([(16.0, 16.0), (48.0, 48.0)]),
            None,
            Some(9_103),
            ShapeDrawCommandOptions::new(),
        )
        .unwrap();
    scene
        .set_shape_mask_effect(
            panel_id,
            9_101,
            &[],
            ShapeMaskEffectConfig::new().outset(12.0),
        )
        .expect("to attach the MSAA shape mask effect");
    scene
        .set_backdrop_effect(panel_id, 9_102, &[], BackdropEffectConfig::default())
        .expect("to attach the MSAA backdrop effect");

    let mut pixel_buffer = Vec::new();
    render_bgra(&mut scene, &mut pixel_buffer).unwrap();

    assert_eq!(read_pixel_rgba(&pixel_buffer, 64, 32, 32), [0, 0, 255, 255]);
    assert_eq!(read_pixel_rgba(&pixel_buffer, 64, 52, 32), [0, 0, 255, 255]);
}

fn read_pixel_rgba(pixel_buffer: &[u8], width: u32, x: u32, y: u32) -> [u8; 4] {
    let stride = (width as usize) * 4;
    let offset = (y as usize) * stride + (x as usize) * 4;

    [
        pixel_buffer[offset + 2],
        pixel_buffer[offset + 1],
        pixel_buffer[offset],
        pixel_buffer[offset + 3],
    ]
}

const CACHED_SHAPE_MASK_EFFECT_RED_MASK: &str = r#"
@fragment
fn effect_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let coverage = textureSample(t_input, s_input, uv).a;
    return vec4<f32>(coverage, 0.0, 0.0, coverage);
}
"#;

#[test]
fn invalid_effect_can_be_replaced_with_a_valid_shader() {
    let Some(mut scene) = create_headless_scene_with_size_and_scale((32, 32), 1.0) else {
        return;
    };

    // Parsing and semantic failures must not poison the scene's shared validator.
    for source in [
        "@fragment fn effect_main(",
        "",
        "@fragment fn effect_main() -> @location(0) vec4<f32> { return vec3<f32>(1.0); }",
    ] {
        assert!(matches!(
            scene.load_effect(9_201, &[source]),
            Err(EffectError::Backend(WgpuBackendError::Effect(
                EffectResourceError::InvalidShader { pass_index: 0, .. }
            )))
        ));
    }
    scene.load_effect(9_201, &[PASSTHROUGH_WGSL]).unwrap();
    let shape_id = scene
        .add_shape(
            Shape::rect([(8.0, 8.0), (24.0, 24.0)]),
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(255, 0, 0)),
        )
        .unwrap();
    scene
        .set_content_effect(shape_id, 9_201, &[], ContentEffectConfig::default())
        .unwrap();

    let mut pixel_buffer = Vec::new();
    render_bgra(&mut scene, &mut pixel_buffer).unwrap();
    assert_eq!(read_pixel_rgba(&pixel_buffer, 32, 16, 16), [255, 0, 0, 255]);

    for _ in 0..2 {
        assert!(matches!(
            scene.load_effect(9_201, &[PASSTHROUGH_WGSL, ""]),
            Err(EffectError::Backend(WgpuBackendError::Effect(
                EffectResourceError::InvalidShader { pass_index: 1, .. }
            )))
        ));
        // Force a redraw so the preserved shader runs after each failed reload.
        scene.resize(scene.size());
        render_bgra(&mut scene, &mut pixel_buffer).unwrap();
        assert_eq!(read_pixel_rgba(&pixel_buffer, 32, 16, 16), [255, 0, 0, 255]);
    }
}

#[test]
fn shape_mask_effects_follow_geometry_and_placement_across_queue_rebuilds() {
    let Some(mut scene) = create_headless_scene_with_size_and_scale((96, 48), 1.0) else {
        return;
    };
    scene
        .load_effect(8_101, &[CACHED_SHAPE_MASK_EFFECT_RED_MASK])
        .unwrap();
    scene.load_shape(Shape::rect([(0.0, 0.0), (24.0, 24.0)]), 8_102, Some(8_103));
    scene.load_shape(
        Shape::builder()
            .begin((0.0, 0.0))
            .line_to((24.0, 0.0))
            .line_to((0.0, 24.0))
            .close()
            .build(),
        8_104,
        Some(8_105),
    );

    let mut pixels = Vec::new();
    // Reorder and move distinct shapes with identical mask bounds after each queue clear.
    for placements in [
        [(8_102, 4), (8_102, 36), (8_104, 68)],
        [(8_104, 36), (8_102, 68), (8_102, 4)],
        [(8_102, 36), (8_104, 4), (8_102, 68)],
    ] {
        scene.clear();
        let root_id = scene
            .add_shape(
                Shape::rect([(0.0, 0.0), (96.0, 48.0)]),
                None,
                None,
                ShapeDrawCommandOptions::new().color(Color::WHITE),
            )
            .unwrap();
        for (shape_key, left) in placements {
            let node_id = scene
                .add_cached_shape(
                    shape_key,
                    Some(root_id),
                    ShapeDrawCommandOptions::new()
                        .transform(TransformInstance::translation(left as f32, 12.0)),
                )
                .unwrap();
            scene
                .set_shape_mask_effect(
                    node_id,
                    8_101,
                    &[],
                    ShapeMaskEffectConfig::new().outset(3.0),
                )
                .unwrap();
        }
        render_bgra(&mut scene, &mut pixels).unwrap();

        for (shape_key, left) in placements {
            assert_eq!(read_pixel_rgba(&pixels, 96, left + 6, 18), [255, 0, 0, 255]);
            let corner = if shape_key == 8_102 {
                [255, 0, 0, 255]
            } else {
                [255; 4]
            };
            assert_eq!(read_pixel_rgba(&pixels, 96, left + 18, 30), corner);
        }
    }
}

#[cfg(feature = "render_metrics")]
#[test]
fn cached_shape_mask_effects_composite_with_solid_shapes() {
    let Some(mut scene) = create_headless_scene_with_size_and_scale((96, 48), 1.0) else {
        return;
    };
    scene.load_effect(8_151, &[PASSTHROUGH_WGSL]).unwrap();
    scene.load_shape(Shape::rect([(0.0, 0.0), (24.0, 24.0)]), 8_152, Some(8_153));
    let root_id = scene
        .add_shape(
            Shape::rect([(0.0, 0.0), (96.0, 48.0)]),
            None,
            None,
            ShapeDrawCommandOptions::new().clips_children(false),
        )
        .unwrap();

    for (translation_x, color) in [
        (8.0, Color::rgb(220, 50, 50)),
        (48.0, Color::rgb(50, 90, 220)),
    ] {
        let node_id = scene
            .add_cached_shape(
                8_152,
                Some(root_id),
                ShapeDrawCommandOptions::new()
                    .color(color)
                    .transform(TransformInstance::translation(translation_x, 12.0)),
            )
            .unwrap();
        scene
            .set_shape_mask_effect(
                node_id,
                8_151,
                &[],
                ShapeMaskEffectConfig::new().outset(3.0),
            )
            .unwrap();
    }

    let mut pixels = Vec::new();
    render_bgra(&mut scene, &mut pixels).unwrap();
    render_bgra(&mut scene, &mut pixels).unwrap();

    // Queue a transparent shape to redraw and exercise the retained effect cache.
    scene
        .add_shape(
            Shape::rect([(0.0, 0.0), (96.0, 48.0)]),
            Some(root_id),
            None,
            ShapeDrawCommandOptions::new(),
        )
        .unwrap();
    render_bgra(&mut scene, &mut pixels).unwrap();

    let cache_metrics = scene.backend().last_shape_mask_effect_cache_metrics();
    assert_eq!(cache_metrics.hits, 2);
    assert_eq!(cache_metrics.misses, 0);

    assert_eq!(read_pixel_rgba(&pixels, 96, 20, 24), [220, 50, 50, 255]);
    assert_eq!(read_pixel_rgba(&pixels, 96, 60, 24), [50, 90, 220, 255]);
}

#[cfg(feature = "render_metrics")]
#[test]
fn cached_shape_mask_effect_is_invalidated_by_normal_pipeline_recreation() {
    let Some(mut scene) = create_headless_scene_with_size_and_scale((64, 64), 1.0) else {
        return;
    };
    scene.load_effect(8_161, &[SHAPE_DROP_WGSL]).unwrap();
    let shape_id = scene
        .add_shape(
            Shape::rect([(16.0, 16.0), (48.0, 48.0)]),
            None,
            Some(8_162),
            ShapeDrawCommandOptions::new().color(Color::rgb(220, 200, 50)),
        )
        .unwrap();
    scene
        .set_shape_mask_effect(
            shape_id,
            8_161,
            &[],
            ShapeMaskEffectConfig::new().outset(12.0),
        )
        .unwrap();

    let mut pixels = Vec::new();
    render_bgra(&mut scene, &mut pixels).unwrap();
    scene.set_msaa_samples(4);
    render_bgra(&mut scene, &mut pixels).unwrap();

    let cache_metrics = scene.backend().last_shape_mask_effect_cache_metrics();
    assert_eq!(cache_metrics.hits, 0);
    assert_eq!(cache_metrics.misses, 1);
    assert_eq!(read_pixel_rgba(&pixels, 64, 52, 32), [0, 0, 255, 255]);
}

fn assert_scene_restored_after_overlay_removal(
    scene: &mut Scene,
    pixel_buffer: &mut Vec<u8>,
    expectations: &[PixelExpectation],
) {
    let overlay = scene
        .add_clipping_rect(
            [(0.0, 0.0), (CANVAS_WIDTH as f32, CANVAS_HEIGHT as f32)],
            None,
            None::<TransformInstance>,
            true,
        )
        .unwrap();
    let overlay_shape = scene
        .add_shape(
            Shape::rect([(0.0, 0.0), (CANVAS_WIDTH as f32, CANVAS_HEIGHT as f32)]),
            Some(overlay),
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(255, 0, 255)),
        )
        .unwrap();
    render_bgra(scene, pixel_buffer).unwrap();
    assert_eq!(
        read_pixel_rgba(pixel_buffer, CANVAS_WIDTH, 40, 40),
        [255, 0, 255, 255]
    );
    scene.remove_subtrees([overlay_shape, overlay, overlay_shape], |_| {});
    render_bgra(scene, pixel_buffer).unwrap();
    assert_pixels_match(pixel_buffer, expectations);
}

#[test]
fn main_scene_pixel_expectations() {
    let Some(mut scene) = create_headless_scene() else {
        return;
    };

    let mut pixel_buffer: Vec<u8> = Vec::new();
    // Start with MSAA so a rejected submission cannot reuse a prior valid output.
    for sample_count in [4, 1, 4, 1] {
        scene.set_msaa_samples(sample_count);
        // Exercise both full clears and root removal after a completed render.
        for should_remove_root in [false, true] {
            if should_remove_root {
                scene.remove_subtrees([0], |_| {});
            } else {
                scene.clear();
            }
            let expectations = build_main_scene(&mut scene);
            render_bgra(&mut scene, &mut pixel_buffer).unwrap();
            assert_pixels_match(&pixel_buffer, &expectations);
            if should_remove_root {
                assert_scene_restored_after_overlay_removal(
                    &mut scene,
                    &mut pixel_buffer,
                    &expectations,
                );
            }
        }
        for _ in 0..2 {
            scene.clear();
            let expectations = build_nested_targets_scene(&mut scene);
            render_bgra(&mut scene, &mut pixel_buffer).unwrap();
            assert_pixels_match(&pixel_buffer, &expectations);
        }
        for scale_factor in [1.25, 2.0] {
            scene.clear();
            let width = (f64::from(CANVAS_WIDTH) * scale_factor) as u32;
            let height = (f64::from(CANVAS_HEIGHT) * scale_factor) as u32;
            scene.resize((width, height));
            scene.change_scale_factor(scale_factor).unwrap();
            let mut expectations = build_bounded_targets_scene(&mut scene);
            for expectation in &mut expectations {
                expectation.x = ((f64::from(expectation.x) + 0.5) * scale_factor) as u32;
                expectation.y = ((f64::from(expectation.y) + 0.5) * scale_factor) as u32;
            }
            render_bgra(&mut scene, &mut pixel_buffer).unwrap();
            let failures = check_pixels(&pixel_buffer, width, height, &expectations);
            assert!(
                failures.is_empty(),
                "scale {scale_factor}: {}",
                failures.join("\n")
            );
        }
        scene.resize((CANVAS_WIDTH, CANVAS_HEIGHT));
        scene.change_scale_factor(1.0).unwrap();
    }
}

fn rebuild_texture_material_scene(
    scene: &mut Scene,
    gradient_color: Color,
    offset: f32,
    swaps_fills: bool,
) {
    scene.clear();
    scene
        .add_shape(
            Shape::rect([(0.0, 0.0), (64.0, 32.0)]),
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::WHITE),
        )
        .unwrap();
    for left in [14.0, 46.0] {
        scene
            .add_shape(
                Shape::rect([(left, 0.0), (left + 4.0, 32.0)]),
                None,
                None,
                ShapeDrawCommandOptions::new().color(Color::BLACK),
            )
            .unwrap();
    }
    for panel_index in 0..2 {
        let left = panel_index as f32 * 32.0 + offset;
        let fill = if (panel_index == 1) != swaps_fills {
            Fill::Gradient(
                Gradient::linear(LinearGradientDesc::new(
                    LinearGradientLine {
                        start: [left, 0.0],
                        end: [left + 24.0, 0.0],
                    },
                    [
                        GradientStop::at_position(
                            GradientStopOffset::linear_radial(0.0),
                            gradient_color,
                        ),
                        GradientStop::at_position(
                            GradientStopOffset::linear_radial(1.0),
                            Color::rgba(0, 0, 0, 0),
                        ),
                    ],
                ))
                .unwrap(),
            )
        } else {
            Fill::Solid(Color::rgba(0, 0, 255, 128))
        };
        let panel = scene
            .add_shape(
                Shape::rect([(left, 4.0), (left + 24.0, 28.0)]),
                None,
                None,
                ShapeDrawCommandOptions::new().fill(fill),
            )
            .unwrap();
        scene
            .set_backdrop_effect(panel, 9_301, &[], BackdropEffectConfig::default())
            .unwrap();
    }
}

#[test]
fn texture_materials_follow_scene_changes_across_queue_rebuilds() {
    let Some(mut scene) = create_headless_scene_with_size_and_scale((64, 32), 1.0) else {
        return;
    };
    scene.load_effect(9_301, &[PASSTHROUGH_WGSL]).unwrap();
    let mut pixels = Vec::new();
    for samples in [1, 4, 1] {
        scene.set_msaa_samples(samples);
        for _ in 0..2 {
            rebuild_texture_material_scene(&mut scene, Color::rgba(255, 0, 0, 128), 2.0, false);
            render_bgra(&mut scene, &mut pixels).unwrap();
            assert_eq!(read_pixel_rgba(&pixels, 64, 63, 16), [255; 4]);
            let solid = read_pixel_rgba(&pixels, 64, 10, 16);
            assert!(solid[0] < 200 && solid[2] == 255);
            let gradient = read_pixel_rgba(&pixels, 64, 38, 16);
            assert!(gradient[0] > gradient[1] + 20);
        }
    }

    for (color, offset, swaps_fills) in [
        (Color::rgba(255, 0, 0, 128), 6.0, false),
        (Color::rgba(0, 255, 0, 128), 6.0, false),
        (Color::rgba(0, 255, 0, 128), 6.0, false),
        (Color::rgba(0, 255, 0, 128), 6.0, true),
        (Color::rgba(0, 255, 0, 128), 6.0, true),
    ] {
        rebuild_texture_material_scene(&mut scene, color, offset, swaps_fills);
        render_bgra(&mut scene, &mut pixels).unwrap();
        let solid_left = if swaps_fills { 32 } else { 0 };
        let solid_over_black = read_pixel_rgba(&pixels, 64, solid_left + 16, 16);
        assert_eq!(
            &solid_over_black[..2],
            &[0, 0],
            "sampling must follow the moved capture"
        );
        assert!((187..=189).contains(&solid_over_black[2]));
        let gradient_left = if swaps_fills { 0 } else { 32 };
        let gradient = read_pixel_rgba(&pixels, 64, gradient_left + 10, 16);
        if color == Color::rgba(255, 0, 0, 128) {
            assert!(gradient[0] > gradient[1] + 20);
        } else {
            assert!(gradient[1] > gradient[0] + 20);
        }
    }
}

#[test]
fn dirty_subtree_addition_and_removal_preserve_retained_pixels() {
    let Some(mut scene) = create_headless_scene_with_size_and_scale((64, 64), 1.0) else {
        return;
    };
    scene.set_dirty_region_overlay_enabled(true);
    let mut pixels = Vec::new();
    for samples in [1, 4] {
        scene.clear();
        scene.set_msaa_samples(samples);
        let root = scene
            .add_clipping_rect(
                [(0.0, 0.0), (64.0, 64.0)],
                None,
                None::<TransformInstance>,
                true,
            )
            .unwrap();
        scene
            .add_shape(
                Shape::rect([(2.0, 2.0), (48.0, 48.0)]),
                Some(root),
                None,
                ShapeDrawCommandOptions::new().color(Color::rgba(255, 0, 0, 128)),
            )
            .unwrap();
        render_bgra(&mut scene, &mut pixels).unwrap();
        let background = pixels.clone();
        let branch = scene
            .add_clipping_rect(
                [(8.0, 8.0), (60.0, 60.0)],
                Some(root),
                None::<TransformInstance>,
                true,
            )
            .unwrap();
        scene
            .add_shape(
                Shape::rect([(10.25, 10.25), (20.75, 20.75)]),
                Some(branch),
                None,
                ShapeDrawCommandOptions::new().color(Color::rgba(0, 0, 255, 128)),
            )
            .unwrap();
        scene
            .add_shape(
                Shape::rect([(50.25, 50.25), (56.75, 56.75)]),
                Some(branch),
                None,
                ShapeDrawCommandOptions::new().color(Color::WHITE),
            )
            .unwrap();
        render_bgra(&mut scene, &mut pixels).unwrap();
        assert_eq!(read_pixel_rgba(&pixels, 64, 54, 54), [255, 255, 255, 255]);
        assert_eq!(
            read_pixel_rgba(&pixels, 64, 4, 4),
            read_pixel_rgba(&background, 64, 4, 4)
        );
        let overlay = pixels.clone();
        pixels.fill(0x71);
        render_bgra(&mut scene, &mut pixels).unwrap();
        assert_eq!(pixels, overlay, "unchanged renders must preserve the image");
        scene.remove_subtrees([branch], |_| {});
        render_bgra(&mut scene, &mut pixels).unwrap();
        assert_eq!(
            pixels, background,
            "removal must restore transparent pixels and AA edges"
        );
        scene.clear();
        render_bgra(&mut scene, &mut pixels).unwrap();
        assert!(pixels.iter().all(|&byte| byte == 0));
    }
}

fn assert_incremental_pixels_match_full_redraw(scene: &mut Scene, pixels: &mut Vec<u8>) {
    let incremental = pixels.clone();
    scene.resize(scene.size());
    render_bgra(scene, pixels).unwrap();
    assert_eq!(
        pixels, &incremental,
        "partial updates must match a full redraw"
    );
}

enum TextureMutation {
    Allocate([u8; 4]),
    Upload([u8; 4]),
    Remove,
    Clear,
}

fn fitted_texture_data(fit_mode: ShapeTextureFitMode, size: (u32, u32)) -> Vec<u8> {
    let colors = [
        [255, 0, 0, 255],
        [0, 255, 0, 255],
        [0, 0, 255, 255],
        [255, 255, 0, 255],
    ];
    let mut pixels = Vec::new();
    for y in 0..size.1 {
        for x in 0..size.0 {
            let stripe = if fit_mode == ShapeTextureFitMode::OriginalSize {
                usize::from(size.0 > size.1)
            } else if size.0 < size.1 {
                (y * 4 / size.1) as usize
            } else {
                (x * 4 / size.0) as usize
            };
            pixels.extend_from_slice(&colors[stripe]);
        }
    }
    pixels
}

fn assert_fitted_texture_pixels(pixels: &[u8], fit_mode: ShapeTextureFitMode, size: (u32, u32)) {
    let red = [255, 0, 0, 255];
    let green = [0, 255, 0, 255];
    let blue = [0, 0, 255, 255];
    let yellow = [255, 255, 0, 255];
    let white = [255; 4];
    let expectations: &[((u32, u32), [u8; 4])] = match (fit_mode, size) {
        (ShapeTextureFitMode::OriginalSize, (8, 16)) => {
            &[((2, 2), red), ((12, 2), white), ((2, 12), red)]
        }
        (ShapeTextureFitMode::OriginalSize, (16, 8)) => {
            &[((2, 2), green), ((12, 2), green), ((2, 12), white)]
        }
        (ShapeTextureFitMode::Cover, (8, 16)) => &[((16, 6), green), ((16, 26), blue)],
        (ShapeTextureFitMode::Cover, (16, 8)) => &[((6, 16), green), ((26, 16), blue)],
        (ShapeTextureFitMode::Contain, (8, 16)) => &[
            ((4, 16), white),
            ((28, 16), white),
            ((16, 4), red),
            ((16, 28), yellow),
        ],
        (ShapeTextureFitMode::Contain, (16, 8)) => &[
            ((16, 4), white),
            ((16, 28), white),
            ((4, 16), red),
            ((28, 16), yellow),
        ],
        _ => unreachable!(),
    };
    for &((x, y), expected) in expectations {
        assert_eq!(
            read_pixel_rgba(pixels, 96, 8 + x, 8 + y),
            expected,
            "{fit_mode:?}, texture {size:?}, offset ({x}, {y})"
        );
    }
}

#[test]
fn retained_texture_fits_follow_allocation_dimensions() {
    let Some(mut scene) = create_headless_scene_with_size_and_scale((96, 96), 1.0) else {
        return;
    };
    let texture_id = 9_410;
    scene.load_effect(9_410, &[PASSTHROUGH_WGSL]).unwrap();
    let manager = scene.texture_manager().clone();
    let mut pixels = Vec::new();
    for samples in [1, 4] {
        scene.set_msaa_samples(samples);
        for fit_mode in [
            ShapeTextureFitMode::OriginalSize,
            ShapeTextureFitMode::Cover,
            ShapeTextureFitMode::Contain,
        ] {
            for is_foreground in [false, true] {
                scene.clear();
                manager.clear();
                scene.change_scale_factor(1.0).unwrap();
                let root = scene
                    .add_clipping_rect(
                        [(0.0, 0.0), (96.0, 96.0)],
                        None,
                        None::<TransformInstance>,
                        false,
                    )
                    .unwrap();
                let discarded = scene
                    .add_shape(
                        Shape::rect([(90.0, 90.0), (94.0, 94.0)]),
                        Some(root),
                        None,
                        ShapeDrawCommandOptions::new().color(Color::WHITE),
                    )
                    .unwrap();
                let texture = ShapeTextureOptions::new(texture_id).fit_mode(fit_mode);
                let options = ShapeDrawCommandOptions::new().color(Color::WHITE);
                let options = if is_foreground {
                    options.foreground_texture(texture)
                } else {
                    options.background_texture(texture)
                };
                let shape = scene
                    .add_shape(
                        Shape::rect([(8.0, 8.0), (40.0, 40.0)]),
                        Some(root),
                        None,
                        options,
                    )
                    .unwrap();
                scene
                    .set_shape_mask_effect(shape, 9_410, &[], ShapeMaskEffectConfig::new())
                    .unwrap();
                render_bgra(&mut scene, &mut pixels).unwrap();
                assert_eq!(read_pixel_rgba(&pixels, 96, 10, 10), [255; 4]);
                scene.remove_subtrees([discarded], |_| {});

                for size in [(8, 16), (16, 8)] {
                    manager.allocate_texture_with_data(
                        texture_id,
                        size,
                        &fitted_texture_data(fit_mode, size),
                    );
                    render_bgra(&mut scene, &mut pixels).unwrap();
                    assert_fitted_texture_pixels(&pixels, fit_mode, size);
                    assert_incremental_pixels_match_full_redraw(&mut scene, &mut pixels);
                }
                manager.remove_texture(texture_id);
                render_bgra(&mut scene, &mut pixels).unwrap();
                assert_eq!(read_pixel_rgba(&pixels, 96, 10, 10), [255; 4]);
                manager.allocate_texture_with_data(
                    texture_id,
                    (8, 16),
                    &fitted_texture_data(fit_mode, (8, 16)),
                );
                render_bgra(&mut scene, &mut pixels).unwrap();
                assert_fitted_texture_pixels(&pixels, fit_mode, (8, 16));
            }
        }
    }
}

#[test]
fn texture_mutations_redraw_retained_layers_and_backdrop_captures() {
    let Some(mut scene) = create_headless_scene_with_size_and_scale((96, 64), 1.0) else {
        return;
    };
    let texture_id = 9_402;
    scene.load_effect(9_402, &[PASSTHROUGH_WGSL]).unwrap();
    let manager = scene.texture_manager().clone();
    let mut pixels = Vec::new();
    for samples in [1, 4] {
        scene.set_msaa_samples(samples);
        let root = scene
            .add_shape(
                Shape::rect([(0.0, 0.0), (96.0, 64.0)]),
                None,
                None,
                ShapeDrawCommandOptions::new().color(Color::rgb(30, 40, 50)),
            )
            .unwrap();
        for (bounds, options) in [
            (
                [(8.0, 8.0), (24.0, 24.0)],
                ShapeDrawCommandOptions::new()
                    .color(Color::WHITE)
                    .background_texture_id(texture_id),
            ),
            (
                [(8.0, 36.0), (24.0, 52.0)],
                ShapeDrawCommandOptions::new()
                    .color(Color::WHITE)
                    .foreground_texture_id(texture_id),
            ),
            (
                [(80.0, 36.0), (88.0, 44.0)],
                ShapeDrawCommandOptions::new().color(Color::rgb(0, 0, 255)),
            ),
        ] {
            scene
                .add_shape(Shape::rect(bounds), Some(root), None, options)
                .unwrap();
        }
        let backdrop = scene
            .add_shape(
                Shape::rect([(48.0, 8.0), (64.0, 24.0)]),
                Some(root),
                None,
                ShapeDrawCommandOptions::new().color(Color::TRANSPARENT),
            )
            .unwrap();
        scene
            .set_backdrop_effect(
                backdrop,
                9_402,
                &[],
                BackdropEffectConfig::new()
                    .capture_area(BackdropCaptureArea::ScreenRect([(8.0, 8.0), (24.0, 24.0)])),
            )
            .unwrap();
        render_bgra(&mut scene, &mut pixels).unwrap();
        assert_eq!(read_pixel_rgba(&pixels, 96, 16, 16), [255; 4]);
        assert_eq!(read_pixel_rgba(&pixels, 96, 16, 44), [255; 4]);
        assert_eq!(read_pixel_rgba(&pixels, 96, 56, 16), [255; 4]);

        for (mutation, expected) in [
            (
                TextureMutation::Allocate([255, 0, 0, 255]),
                [255, 0, 0, 255],
            ),
            (TextureMutation::Upload([0, 255, 0, 255]), [0, 255, 0, 255]),
            (
                TextureMutation::Allocate([0, 0, 255, 255]),
                [0, 0, 255, 255],
            ),
            (TextureMutation::Remove, [255; 4]),
            (
                TextureMutation::Allocate([255, 0, 0, 255]),
                [255, 0, 0, 255],
            ),
            (TextureMutation::Clear, [255; 4]),
        ] {
            match mutation {
                TextureMutation::Allocate(data) => {
                    manager.allocate_texture_with_data(texture_id, (1, 1), &data);
                }
                TextureMutation::Upload(data) => {
                    manager
                        .load_data_into_texture(texture_id, (1, 1), &data)
                        .unwrap();
                }
                TextureMutation::Remove => manager.remove_texture(texture_id),
                TextureMutation::Clear => manager.clear(),
            }
            render_bgra(&mut scene, &mut pixels).unwrap();
            for (x, y) in [(16, 16), (16, 44), (56, 16)] {
                assert_eq!(read_pixel_rgba(&pixels, 96, x, y), expected);
            }
            assert_eq!(read_pixel_rgba(&pixels, 96, 84, 40), [0, 0, 255, 255]);
            assert_eq!(read_pixel_rgba(&pixels, 96, 40, 44), [30, 40, 50, 255]);
            assert_incremental_pixels_match_full_redraw(&mut scene, &mut pixels);
        }
        scene.clear();
    }
}

#[test]
fn command_replacements_update_retained_geometry_effects_and_child_clips() {
    let Some(mut scene) = create_headless_scene_with_size_and_scale((96, 96), 1.0) else {
        return;
    };
    scene.load_effect(9_401, &[SHAPE_DROP_WGSL]).unwrap();
    scene.load_shape(
        Shape::builder()
            .begin((20.25, 42.25))
            .line_to((40.75, 42.25))
            .line_to((20.25, 62.75))
            .close()
            .build(),
        9_402,
        None,
    );
    let background = [190, 50, 20, 255];
    let blue = [0, 0, 255, 255];
    let mut pixels = Vec::new();
    for samples in [1, 4] {
        scene.clear();
        scene.set_msaa_samples(samples);
        let root = scene
            .add_shape(
                Shape::rect([(0.0, 0.0), (96.0, 96.0)]),
                None,
                None,
                ShapeDrawCommandOptions::new().color(Color::rgb(190, 50, 20)),
            )
            .unwrap();
        let parent = scene
            .add_shape(
                Shape::rect([(10.25, 10.25), (30.75, 30.75)]),
                Some(root),
                None,
                ShapeDrawCommandOptions::new().clips_children(false),
            )
            .unwrap();
        scene
            .set_shape_mask_effect(
                parent,
                9_401,
                &[],
                ShapeMaskEffectConfig::new().outset(12.0),
            )
            .unwrap();
        scene
            .add_shape(
                Shape::rect([(78.25, 12.25), (88.75, 22.75)]),
                Some(parent),
                None,
                ShapeDrawCommandOptions::new().color(Color::WHITE),
            )
            .unwrap();
        render_bgra(&mut scene, &mut pixels).unwrap();
        assert_eq!(read_pixel_rgba(&pixels, 96, 25, 25), blue);
        assert_eq!(read_pixel_rgba(&pixels, 96, 36, 36), blue);
        assert_eq!(read_pixel_rgba(&pixels, 96, 87, 15), [255; 4]);

        scene
            .replace_with_shape(
                parent,
                Shape::rect([(40.25, 10.25), (60.75, 30.75)]),
                None,
                ShapeDrawCommandOptions::new().clips_children(false),
            )
            .unwrap();
        render_bgra(&mut scene, &mut pixels).unwrap();
        assert_eq!(read_pixel_rgba(&pixels, 96, 25, 25), background);
        assert_eq!(read_pixel_rgba(&pixels, 96, 36, 36), background);
        assert_eq!(read_pixel_rgba(&pixels, 96, 55, 25), blue);
        assert_eq!(read_pixel_rgba(&pixels, 96, 66, 36), blue);
        assert_eq!(read_pixel_rgba(&pixels, 96, 87, 15), [255; 4]);
        assert_incremental_pixels_match_full_redraw(&mut scene, &mut pixels);

        scene
            .replace_with_cached_shape(
                parent,
                9_402,
                ShapeDrawCommandOptions::new().clips_children(false),
            )
            .unwrap();
        render_bgra(&mut scene, &mut pixels).unwrap();
        assert_eq!(read_pixel_rgba(&pixels, 96, 55, 25), background);
        assert_eq!(read_pixel_rgba(&pixels, 96, 66, 36), background);
        assert_eq!(read_pixel_rgba(&pixels, 96, 32, 54), blue);
        assert_eq!(read_pixel_rgba(&pixels, 96, 30, 64), blue);
        assert_eq!(read_pixel_rgba(&pixels, 96, 45, 67), background);
        assert_eq!(read_pixel_rgba(&pixels, 96, 87, 15), [255; 4]);
        assert_incremental_pixels_match_full_redraw(&mut scene, &mut pixels);

        scene
            .replace_with_clipping_rect(
                parent,
                [(80.0, 10.0), (86.0, 18.0)],
                None::<TransformInstance>,
                true,
            )
            .unwrap();
        render_bgra(&mut scene, &mut pixels).unwrap();
        assert_eq!(read_pixel_rgba(&pixels, 96, 32, 54), background);
        assert_eq!(read_pixel_rgba(&pixels, 96, 30, 64), background);
        assert_eq!(read_pixel_rgba(&pixels, 96, 82, 15), [255; 4]);
        assert_eq!(read_pixel_rgba(&pixels, 96, 87, 15), background);
        assert_eq!(read_pixel_rgba(&pixels, 96, 82, 21), background);
        assert_incremental_pixels_match_full_redraw(&mut scene, &mut pixels);

        scene
            .replace_with_shape(
                parent,
                Shape::rect([(40.25, 42.25), (60.75, 62.75)]),
                None,
                ShapeDrawCommandOptions::new()
                    .color(Color::rgb(0, 255, 0))
                    .clips_children(false),
            )
            .unwrap();
        render_bgra(&mut scene, &mut pixels).unwrap();
        assert_eq!(read_pixel_rgba(&pixels, 96, 45, 50), [0, 255, 0, 255]);
        assert_eq!(read_pixel_rgba(&pixels, 96, 66, 66), background);
        assert_eq!(read_pixel_rgba(&pixels, 96, 87, 15), [255; 4]);
        assert_incremental_pixels_match_full_redraw(&mut scene, &mut pixels);
    }
}

#[test]
fn empty_draw_queue() {
    let Some(mut scene) = create_headless_scene() else {
        return;
    };

    let mut pixel_buffer = Vec::new();
    render_bgra(&mut scene, &mut pixel_buffer).unwrap();
    assert_eq!(
        pixel_buffer.len(),
        (CANVAS_WIDTH * CANVAS_HEIGHT * 4) as usize
    );
    assert!(pixel_buffer.iter().all(|&byte| byte == 0));
    for samples in [4, 1] {
        scene.set_msaa_samples(samples);
        scene.clear();
        scene
            .add_shape(
                Shape::rect([(0.0, 0.0), (CANVAS_WIDTH as f32, CANVAS_HEIGHT as f32)]),
                None,
                None,
                ShapeDrawCommandOptions::new().color(Color::WHITE),
            )
            .unwrap();
        render_bgra(&mut scene, &mut pixel_buffer).unwrap();
        assert!(pixel_buffer.iter().all(|&byte| byte == 255));
        scene.clear();
        render_bgra(&mut scene, &mut pixel_buffer).unwrap();
        assert_eq!(
            pixel_buffer.len(),
            (CANVAS_WIDTH * CANVAS_HEIGHT * 4) as usize
        );
        assert!(
            pixel_buffer.iter().all(|&byte| byte == 0),
            "empty scene must clear the previous output"
        );
    }
}

#[test]
fn single_root_no_children() {
    let Some(mut scene) = create_headless_scene() else {
        return;
    };

    let shape = Shape::rect([(10.0, 10.0), (100.0, 100.0)]);
    scene
        .add_shape(
            shape,
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(200, 50, 50)),
        )
        .unwrap();

    let mut pixel_buffer: Vec<u8> = Vec::new();
    render_bgra(&mut scene, &mut pixel_buffer).unwrap();

    let expectations = vec![
        PixelExpectation::opaque(55, 55, 200, 50, 50, "center_red"),
        PixelExpectation::transparent(5, 5, "outside_rect"),
    ];

    assert_pixels_match(&pixel_buffer, &expectations);
}

#[test]
fn original_size_texture_fit_uses_physical_pixels_on_hidpi() {
    let physical_size = (200, 200);
    let scale_factor = 2.0;
    let Some(mut scene) = create_headless_scene_with_size_and_scale(physical_size, scale_factor)
    else {
        return;
    };

    let green_texture_id = 9_001u64;
    let green_texture_with_transparent_border_20x20 = (0..20u32)
        .flat_map(|y| {
            (0..20u32).flat_map(move |x| {
                if x == 0 || x == 19 || y == 0 || y == 19 {
                    [0u8, 0u8, 0u8, 0u8]
                } else {
                    [0u8, 255u8, 0u8, 255u8]
                }
            })
        })
        .collect::<Vec<_>>();
    scene.texture_manager().allocate_texture_with_data(
        green_texture_id,
        (20, 20),
        &green_texture_with_transparent_border_20x20,
    );

    let shape = Shape::rect([(10.0, 10.0), (70.0, 70.0)]);
    scene
        .add_shape(
            shape,
            None,
            None,
            ShapeDrawCommandOptions::new()
                .background_texture(
                    ShapeTextureOptions::new(green_texture_id)
                        .fit_mode(ShapeTextureFitMode::OriginalSize),
                )
                .color(Color::WHITE),
        )
        .unwrap();

    let mut pixel_buffer: Vec<u8> = Vec::new();
    render_bgra(&mut scene, &mut pixel_buffer).unwrap();

    let expectations = vec![
        PixelExpectation::opaque(30, 30, 0, 255, 0, "inside_20px_physical_texture_region"),
        PixelExpectation::opaque(60, 30, 255, 255, 255, "outside_texture_region_inside_shape"),
        PixelExpectation::transparent(5, 5, "outside_shape"),
    ];

    let failures = check_pixels(
        &pixel_buffer,
        physical_size.0,
        physical_size.1,
        &expectations,
    );
    if !failures.is_empty() {
        let message = format!(
            "{} pixel expectation(s) failed:\n{}",
            failures.len(),
            failures.join("\n"),
        );
        panic!("{message}");
    }
}

#[test]
fn cover_and_contain_texture_fit_preserve_aspect_ratio() {
    let physical_size = (160, 64);
    let Some(mut scene) = create_headless_scene_with_size_and_scale(physical_size, 1.0) else {
        return;
    };

    let texture_id = 9_002u64;
    let texture_data = (0..10u32)
        .flat_map(|_| {
            (0..20u32).flat_map(|x| {
                if x < 5 {
                    [255u8, 0u8, 0u8, 255u8]
                } else if x < 15 {
                    [0u8, 255u8, 0u8, 255u8]
                } else {
                    [0u8, 0u8, 255u8, 255u8]
                }
            })
        })
        .collect::<Vec<_>>();
    scene
        .texture_manager()
        .allocate_texture_with_data(texture_id, (20, 10), &texture_data);

    scene
        .add_shape(
            Shape::rect([(8.0, 8.0), (56.0, 56.0)]),
            None,
            None,
            ShapeDrawCommandOptions::new()
                .clips_children(false)
                .background_texture(
                    ShapeTextureOptions::new(texture_id).fit_mode(ShapeTextureFitMode::Cover),
                )
                .color(Color::WHITE),
        )
        .unwrap();
    scene
        .add_shape(
            Shape::rect([(88.0, 8.0), (136.0, 56.0)]),
            None,
            None,
            ShapeDrawCommandOptions::new()
                .background_texture(
                    ShapeTextureOptions::new(texture_id).fit_mode(ShapeTextureFitMode::Contain),
                )
                .color(Color::WHITE),
        )
        .unwrap();

    let mut pixel_buffer = Vec::new();
    render_bgra(&mut scene, &mut pixel_buffer).unwrap();

    assert_eq!(
        read_pixel_rgba(&pixel_buffer, physical_size.0, 12, 32),
        [0, 255, 0, 255],
        "cover should crop the left edge of the centered texture",
    );
    assert_eq!(
        read_pixel_rgba(&pixel_buffer, physical_size.0, 52, 32),
        [0, 255, 0, 255],
        "cover should crop the right edge of the centered texture",
    );
    assert_eq!(
        read_pixel_rgba(&pixel_buffer, physical_size.0, 92, 32),
        [255, 0, 0, 255],
        "contain should preserve the left half of the texture",
    );
    assert_eq!(
        read_pixel_rgba(&pixel_buffer, physical_size.0, 132, 32),
        [0, 0, 255, 255],
        "contain should preserve the right half of the texture",
    );
    assert_eq!(
        read_pixel_rgba(&pixel_buffer, physical_size.0, 100, 12),
        [255, 255, 255, 255],
        "contain should reveal the fill above the centered texture",
    );
    assert_eq!(
        read_pixel_rgba(&pixel_buffer, physical_size.0, 100, 52),
        [255, 255, 255, 255],
        "contain should reveal the fill below the centered texture",
    );
}

#[test]
fn clipping_rect_clips_child_without_visible_surface() {
    let Some(mut scene) = create_headless_scene() else {
        return;
    };

    let clip_rect_id = scene
        .add_clipping_rect(
            [(20.0, 20.0), (80.0, 80.0)],
            None,
            None::<TransformInstance>,
            true,
        )
        .unwrap();
    let child = Shape::rect([(0.0, 0.0), (100.0, 100.0)]);
    scene
        .add_shape(
            child,
            Some(clip_rect_id),
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(200, 50, 50)),
        )
        .unwrap();

    let mut pixel_buffer: Vec<u8> = Vec::new();
    render_bgra(&mut scene, &mut pixel_buffer).unwrap();

    let expectations = vec![
        PixelExpectation::opaque(50, 50, 200, 50, 50, "inside_clip_rect"),
        PixelExpectation::transparent(10, 50, "left_of_clip_rect"),
        PixelExpectation::transparent(50, 10, "above_clip_rect"),
        PixelExpectation::transparent(90, 50, "right_of_clip_rect"),
        PixelExpectation::transparent(50, 90, "below_clip_rect"),
    ];

    assert_pixels_match(&pixel_buffer, &expectations);
}

#[test]
fn partially_offscreen_backdrop_capture_clears_reused_texture_space() {
    let physical_size = (100, 80);
    let Some(mut scene) = create_headless_scene_with_size_and_scale(physical_size, 1.0) else {
        return;
    };

    const AVERAGE_WITH_RIGHT_NEIGHBOR_EFFECT_ID: u64 = 9_101;
    const AVERAGE_WITH_RIGHT_NEIGHBOR_WGSL: &str = r#"
const LOOKAHEAD_UV: vec2<f32> = vec2<f32>(0.4, 0.0);

@fragment
fn effect_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let base = textureSample(t_input, s_input, uv);
    let lookahead = textureSample(t_input, s_input, uv + LOOKAHEAD_UV);
    // Keep alpha opaque so cleared capture pixels darken the output.
    return vec4<f32>(0.5 * (base.rgb + lookahead.rgb), 1.0);
}
"#;

    scene
        .load_effect(
            AVERAGE_WITH_RIGHT_NEIGHBOR_EFFECT_ID,
            &[AVERAGE_WITH_RIGHT_NEIGHBOR_WGSL],
        )
        .expect("Failed to compile deterministic backdrop test effect");

    let seeded_blue_panel = Shape::rect([(20.0, 20.0), (60.0, 60.0)]);
    scene
        .add_shape(
            seeded_blue_panel.clone(),
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(40, 40, 220)),
        )
        .unwrap();
    let seeded_blue_panel_id = scene
        .add_shape(
            seeded_blue_panel,
            None,
            None,
            ShapeDrawCommandOptions::new(),
        )
        .unwrap();
    scene
        .set_backdrop_effect(
            seeded_blue_panel_id,
            AVERAGE_WITH_RIGHT_NEIGHBOR_EFFECT_ID,
            &[],
            BackdropEffectConfig::new().capture_area(BackdropCaptureArea::ScreenRect([
                (20.0, 20.0),
                (60.0, 60.0),
            ])),
        )
        .unwrap();

    let mut seeded_frame: Vec<u8> = Vec::new();
    render_bgra(&mut scene, &mut seeded_frame).unwrap();

    scene.clear();

    let visible_red_source = Shape::rect([(70.0, 20.0), (100.0, 60.0)]);
    scene
        .add_shape(
            visible_red_source,
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(220, 40, 40)),
        )
        .unwrap();

    let partially_offscreen_panel = Shape::rect([(70.0, 20.0), (100.0, 60.0)]);
    let partially_offscreen_panel_id = scene
        .add_shape(
            partially_offscreen_panel,
            None,
            None,
            ShapeDrawCommandOptions::new(),
        )
        .unwrap();
    scene
        .set_backdrop_effect(
            partially_offscreen_panel_id,
            AVERAGE_WITH_RIGHT_NEIGHBOR_EFFECT_ID,
            &[],
            BackdropEffectConfig::new()
                .capture_area(BackdropCaptureArea::ScreenRect([
                    (80.0, 30.0),
                    (100.0, 50.0),
                ]))
                .padding(10.0),
        )
        .unwrap();

    let mut pixel_buffer: Vec<u8> = Vec::new();
    render_bgra(&mut scene, &mut pixel_buffer).unwrap();

    let failures = check_pixels(
        &pixel_buffer,
        physical_size.0,
        physical_size.1,
        &[
            PixelExpectation::opaque(75, 40, 220, 40, 40, "both_samples_inside_red_source"),
            // The second sample lands outside the viewport. Half the linear red
            // source color encodes to roughly [161, 27, 27] in sRGB.
            PixelExpectation::opaque_approx(86, 40, 161, 27, 27, 2, "lookahead_in_cleared_padding"),
        ],
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn standalone_clipping_rect_does_not_panic() {
    let Some(mut scene) = create_headless_scene() else {
        return;
    };

    scene
        .add_clipping_rect(
            [(20.0, 20.0), (80.0, 80.0)],
            None,
            None::<TransformInstance>,
            true,
        )
        .unwrap();

    let mut pixel_buffer: Vec<u8> = Vec::new();
    render_bgra(&mut scene, &mut pixel_buffer).unwrap();

    assert!(
        pixel_buffer.iter().all(|&byte| byte == 0),
        "Standalone clipping rect should not draw any pixels",
    );
}

#[test]
fn clipping_rect_rejects_non_axis_aligned_transform() {
    let Some(mut scene) = create_headless_scene() else {
        return;
    };

    let clip_rect_id = scene
        .add_clipping_rect(
            [(20.0, 20.0), (80.0, 80.0)],
            None,
            None::<TransformInstance>,
            true,
        )
        .unwrap();
    assert!(matches!(
        scene.add_clipping_rect(
            [(20.0, 20.0), (80.0, 80.0)],
            None,
            Some(TransformInstance::rotation_z_deg(45.0)),
            true,
        ),
        Err(DrawCommandError::Scene(
            SceneError::UnsupportedClipRectTransform
        ))
    ));

    let child = Shape::rect([(0.0, 0.0), (100.0, 100.0)]);
    scene
        .add_shape(
            child,
            Some(clip_rect_id),
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(200, 50, 50)),
        )
        .unwrap();

    let mut pixel_buffer: Vec<u8> = Vec::new();
    render_bgra(&mut scene, &mut pixel_buffer).unwrap();

    let expectations = vec![
        PixelExpectation::opaque(50, 50, 200, 50, 50, "inside_unrotated_clip_rect"),
        PixelExpectation::transparent(10, 50, "outside_unrotated_clip_rect"),
    ];

    assert_pixels_match(&pixel_buffer, &expectations);
}

#[test]
fn gradient_fill_basic() {
    let Some(mut scene) = create_headless_scene() else {
        return;
    };

    let root = Shape::rect([(0.0, 0.0), (100.0, 100.0)]);
    let root_id = scene
        .add_shape(
            root,
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::WHITE),
        )
        .unwrap();

    let gradient = Gradient::linear(
        LinearGradientDesc::new(
            LinearGradientLine {
                start: [10.0, 50.0],
                end: [90.0, 50.0],
            },
            [
                GradientStop::at_position(
                    GradientStopOffset::linear_radial(0.0),
                    Color::rgb(255, 0, 0),
                ),
                GradientStop::at_position(
                    GradientStopOffset::linear_radial(1.0),
                    Color::rgb(0, 0, 255),
                ),
            ],
        )
        .with_interpolation(ColorInterpolation::Srgb),
    )
    .expect("valid gradient");

    scene
        .add_shape(
            Shape::rect([(10.0, 10.0), (90.0, 90.0)]),
            Some(root_id),
            None,
            ShapeDrawCommandOptions::new().fill(Fill::from(gradient)),
        )
        .unwrap();

    let mut pixel_buffer: Vec<u8> = Vec::new();
    render_bgra(&mut scene, &mut pixel_buffer).unwrap();

    // At each pixel center, t = (x + 0.5 - 10) / 80 and sRGB = 255 * [1 - t, 0, t].
    let expectations = [
        PixelExpectation::opaque_approx(20, 50, 222, 0, 33, 3, "gradient_left"),
        PixelExpectation::opaque_approx(50, 50, 126, 0, 129, 3, "gradient_center"),
        PixelExpectation::opaque_approx(80, 50, 30, 0, 225, 3, "gradient_right"),
    ];
    assert_pixels_match(&pixel_buffer, &expectations);
}

#[test]
fn gradient_survives_pipeline_recreation() {
    let Some(mut scene) = create_headless_scene() else {
        return;
    };

    let gradient = Gradient::linear(
        LinearGradientDesc::new(
            LinearGradientLine {
                start: [10.0, 50.0],
                end: [90.0, 50.0],
            },
            [
                GradientStop::at_position(
                    GradientStopOffset::linear_radial(0.0),
                    Color::rgb(255, 0, 0),
                ),
                GradientStop::at_position(
                    GradientStopOffset::linear_radial(1.0),
                    Color::rgb(0, 0, 255),
                ),
            ],
        )
        .with_interpolation(ColorInterpolation::Srgb),
    )
    .expect("valid gradient");

    scene
        .add_shape(
            Shape::rect([(10.0, 10.0), (90.0, 90.0)]),
            None,
            None,
            ShapeDrawCommandOptions::new().fill(Fill::from(gradient)),
        )
        .unwrap();

    // At each pixel center, t = (x + 0.5 - 10) / 80 and sRGB = 255 * [1 - t, 0, t].
    let expectations = [
        PixelExpectation::opaque_approx(20, 50, 222, 0, 33, 3, "gradient_left"),
        PixelExpectation::opaque_approx(50, 50, 126, 0, 129, 3, "gradient_center"),
        PixelExpectation::opaque_approx(80, 50, 30, 0, 225, 3, "gradient_right"),
    ];
    let mut pixel_buffer = Vec::new();
    render_bgra(&mut scene, &mut pixel_buffer).unwrap();
    assert_pixels_match(&pixel_buffer, &expectations);

    // Changing MSAA recreates pipelines and their bind group layouts.
    scene.set_msaa_samples(4);

    pixel_buffer.clear();
    render_bgra(&mut scene, &mut pixel_buffer).unwrap();
    assert_pixels_match(&pixel_buffer, &expectations);
}

/// Rounded parents force stencil clipping. Sample exposed parent pixels so an
/// opaque child cannot hide an incorrectly inherited gradient.
#[test]
fn stencil_increment_gradient_does_not_leak_to_solid_parent() {
    let Some(mut scene) = create_headless_scene() else {
        return;
    };

    // Full-canvas rect root so all children are visible.
    let root = scene
        .add_shape(
            Shape::rect([(0.0, 0.0), (CANVAS_WIDTH as f32, CANVAS_HEIGHT as f32)]),
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::rgba(0, 0, 0, 0)),
        )
        .unwrap();

    let radii = BorderRadii::new(8.0);

    let gradient = Gradient::linear(
        LinearGradientDesc::new(
            LinearGradientLine {
                start: [10.0, 50.0],
                end: [140.0, 50.0],
            },
            [
                GradientStop::at_position(
                    GradientStopOffset::linear_radial(0.0),
                    Color::rgb(255, 0, 0),
                ),
                GradientStop::at_position(
                    GradientStopOffset::linear_radial(1.0),
                    Color::rgb(0, 0, 255),
                ),
            ],
        )
        .with_interpolation(ColorInterpolation::Srgb),
    )
    .expect("valid gradient");

    let gradient_parent = scene
        .add_shape(
            Shape::rounded_rect([(10.0, 10.0), (140.0, 90.0)], radii),
            Some(root),
            None,
            ShapeDrawCommandOptions::new().fill(Fill::from(gradient)),
        )
        .unwrap();

    scene
        .add_shape(
            Shape::rect([(20.0, 20.0), (130.0, 80.0)]),
            Some(gradient_parent),
            None,
            ShapeDrawCommandOptions::new().color(Color::WHITE),
        )
        .unwrap();

    let solid_parent = scene
        .add_shape(
            Shape::rounded_rect([(160.0, 10.0), (290.0, 90.0)], radii),
            Some(root),
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(0, 200, 0)),
        )
        .unwrap();

    scene
        .add_shape(
            Shape::rect([(170.0, 20.0), (280.0, 80.0)]),
            Some(solid_parent),
            None,
            ShapeDrawCommandOptions::new().color(Color::WHITE),
        )
        .unwrap();

    let mut pixel_buffer = Vec::new();
    render_bgra(&mut scene, &mut pixel_buffer).unwrap();
    assert_pixels_match(
        &pixel_buffer,
        &[
            PixelExpectation::opaque(165, 50, 0, 200, 0, "solid_parent_left"),
            PixelExpectation::opaque(285, 50, 0, 200, 0, "solid_parent_right"),
            PixelExpectation::opaque(225, 50, 255, 255, 255, "solid_child"),
        ],
    );
}

/// Touching triangle subpaths must not create an AA seam along their shared diagonal.
#[test]
fn multi_subpath_fill_has_no_internal_seam() {
    let Some(mut scene) = create_headless_scene() else {
        return;
    };

    let canvas_root = Shape::rect([(0.0, 0.0), (CANVAS_WIDTH as f32, CANVAS_HEIGHT as f32)]);
    let canvas_root_id = scene
        .add_shape(
            canvas_root,
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::WHITE),
        )
        .unwrap();

    let shape = Shape::builder()
        .begin((10.0, 10.0))
        .line_to((100.0, 10.0))
        .line_to((100.0, 100.0))
        .close()
        .begin((10.0, 10.0))
        .line_to((100.0, 100.0))
        .line_to((10.0, 100.0))
        .close()
        .build();
    scene
        .add_shape(
            shape,
            Some(canvas_root_id),
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(200, 50, 50)),
        )
        .unwrap();

    let rect = Shape::rect([(140.0, 10.0), (230.0, 100.0)]);
    scene
        .add_shape(
            rect,
            Some(canvas_root_id),
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(200, 50, 50)),
        )
        .unwrap();

    let mut pixel_buffer: Vec<u8> = Vec::new();
    render_bgra(&mut scene, &mut pixel_buffer).unwrap();

    let expectations = vec![
        PixelExpectation::opaque(30, 30, 200, 50, 50, "diag_top_left"),
        PixelExpectation::opaque(55, 55, 200, 50, 50, "diag_center"),
        PixelExpectation::opaque(80, 80, 200, 50, 50, "diag_bottom_right"),
        PixelExpectation::opaque(5, 5, 255, 255, 255, "outside_shape"),
        PixelExpectation::opaque(185, 55, 200, 50, 50, "rect_center"),
        PixelExpectation::opaque(145, 15, 200, 50, 50, "rect_near_corner"),
        PixelExpectation::opaque(235, 55, 255, 255, 255, "outside_rect"),
    ];

    assert_pixels_match(&pixel_buffer, &expectations);
}

fn assert_layered_backdrop_pixels(scene: &mut Scene, pixels: &mut Vec<u8>) {
    render_bgra(scene, pixels).unwrap();
    let width = scene.size().0;
    assert_eq!(read_pixel_rgba(pixels, width, 24, 32), [255, 0, 0, 255]);
    assert_eq!(read_pixel_rgba(pixels, width, 40, 32), [0, 255, 0, 255]);
}

#[test]
fn layered_backdrop_survives_capture_and_resource_changes() {
    let Some(mut scene) = create_headless_scene_with_size_and_scale((64, 64), 1.0) else {
        return;
    };
    let effect_id = 9_301;
    scene
        .load_effect(
            effect_id,
            &[PASSTHROUGH_WGSL, PASSTHROUGH_WGSL, PASSTHROUGH_WGSL],
        )
        .unwrap();
    scene
        .add_shape(
            Shape::rect([(0.0, 0.0), (64.0, 64.0)]),
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(255, 0, 0)),
        )
        .unwrap();
    let group = scene
        .add_shape(
            Shape::rect([(4.0, 4.0), (60.0, 60.0)]),
            None,
            None,
            ShapeDrawCommandOptions::new(),
        )
        .unwrap();
    scene
        .set_content_effect(group, effect_id, &[], ContentEffectConfig::default())
        .unwrap();
    scene
        .add_shape(
            Shape::rect([(32.0, 4.0), (60.0, 60.0)]),
            Some(group),
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(0, 255, 0)),
        )
        .unwrap();
    let panel = scene
        .add_shape(
            Shape::rect([(16.0, 16.0), (48.0, 48.0)]),
            Some(group),
            None,
            ShapeDrawCommandOptions::new(),
        )
        .unwrap();
    scene
        .set_backdrop_effect(
            panel,
            effect_id,
            &[],
            BackdropEffectConfig::new().padding(4.0).downsample(0.5),
        )
        .unwrap();

    let mut pixels = Vec::new();
    for _ in 0..4 {
        assert_layered_backdrop_pixels(&mut scene, &mut pixels);
    }

    let moved_capture = BackdropEffectConfig::new()
        .capture_area(BackdropCaptureArea::ScreenRect([
            (24.0, 16.0),
            (56.0, 48.0),
        ]))
        .padding(4.0)
        .downsample(0.5);
    scene
        .set_backdrop_effect(panel, effect_id, &[], moved_capture)
        .unwrap();
    assert_layered_backdrop_pixels(&mut scene, &mut pixels);

    for samples in [4, 1] {
        scene.set_msaa_samples(samples);
        for _ in 0..2 {
            assert_layered_backdrop_pixels(&mut scene, &mut pixels);
        }
    }

    scene.resize((80, 64));
    for _ in 0..2 {
        assert_layered_backdrop_pixels(&mut scene, &mut pixels);
    }

    scene
        .set_backdrop_effect(panel, effect_id, &[], moved_capture.downsample(1.0))
        .unwrap();
    assert_layered_backdrop_pixels(&mut scene, &mut pixels);

    scene.remove_backdrop_effect(panel);
    scene
        .set_backdrop_effect(panel, effect_id, &[], moved_capture)
        .unwrap();
    assert_layered_backdrop_pixels(&mut scene, &mut pixels);
}

fn assert_repeated_readback_pixels(
    scene: &mut Scene,
    pixels: &mut Vec<u8>,
    size: (u32, u32),
    format: PixelFormat,
) {
    let layout = PixelLayout::tightly_packed(size, format).unwrap();
    let expected_pixel = match format {
        PixelFormat::Bgra8 => [153, 102, 51, 255],
        PixelFormat::Rgba8 => [51, 102, 153, 255],
        PixelFormat::Argb32 => 0xff33_6699_u32.to_ne_bytes(),
    };
    pixels.resize(layout.byte_len(), 0);
    for _ in 0..3 {
        pixels.fill(0x71);
        scene
            .render(PixmapMut::new(pixels, layout).unwrap())
            .unwrap();
        for pixel in pixels.as_chunks::<4>().0 {
            assert_eq!(*pixel, expected_pixel, "format {format:?}");
        }
    }
}

#[test]
fn readback_targets_survive_alternating_formats_and_resize() {
    let Some(mut scene) = create_headless_scene_with_size_and_scale((65, 7), 1.0) else {
        return;
    };
    scene.set_dirty_region_overlay_enabled(true);
    scene
        .add_shape(
            Shape::rect([(0.0, 0.0), (130.0, 10.0)]),
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(51, 102, 153)),
        )
        .unwrap();
    let mut pixels = Vec::new();
    for samples in [1, 4] {
        scene.set_msaa_samples(samples);
        for size in [(65, 7), (129, 5), (65, 7)] {
            for format in [
                PixelFormat::Bgra8,
                PixelFormat::Rgba8,
                PixelFormat::Bgra8,
                PixelFormat::Argb32,
                PixelFormat::Rgba8,
                PixelFormat::Argb32,
            ] {
                assert_repeated_readback_pixels(&mut scene, &mut pixels, size, format);
            }
        }
    }
}

#[test]
fn main_scene_matches_owned_and_borrowed_surfaces_with_padding() {
    let Some(mut scene) = create_headless_scene() else {
        return;
    };
    let expectations = build_main_scene(&mut scene);
    let size = (CANVAS_WIDTH, CANVAS_HEIGHT);
    let mut reference = Pixmap::new(size, PixelFormat::Bgra8).unwrap();
    let row_bytes = CANVAS_WIDTH as usize * 4;
    for samples in [1, 4] {
        scene.set_msaa_samples(samples);
        scene.render(&mut reference).unwrap();
        assert_pixels_match(reference.pixels(), &expectations);
        for format in [PixelFormat::Bgra8, PixelFormat::Rgba8, PixelFormat::Argb32] {
            let mut owned = Pixmap::new(size, format).unwrap();
            scene.render(&mut owned).unwrap();
            let layout = PixelLayout::new(size, format, row_bytes + 12).unwrap();
            let mut storage = vec![99; layout.byte_len() + 16];
            let mut borrowed = PixmapMut::new(&mut storage, layout).unwrap();
            scene.render(&mut borrowed).unwrap();
            for row in 0..CANVAS_HEIGHT as usize {
                let borrowed_row =
                    &storage[row * layout.stride()..row * layout.stride() + row_bytes];
                let owned_row = &owned.pixels()[row * row_bytes..(row + 1) * row_bytes];
                assert_eq!(borrowed_row, owned_row, "row {row}, format {format:?}");
                for (pixel, reference_pixel) in owned_row.as_chunks::<4>().0.iter().zip(
                    reference.pixels()[row * row_bytes..(row + 1) * row_bytes]
                        .as_chunks::<4>()
                        .0
                        .iter(),
                ) {
                    let bgra = match format {
                        PixelFormat::Bgra8 => [pixel[0], pixel[1], pixel[2], pixel[3]],
                        PixelFormat::Rgba8 => [pixel[2], pixel[1], pixel[0], pixel[3]],
                        PixelFormat::Argb32 => u32::from_ne_bytes(*pixel).to_le_bytes(),
                    };
                    assert_eq!(
                        bgra, *reference_pixel,
                        "format {format:?}, samples {samples}"
                    );
                }
                let padding_end = ((row + 1) * layout.stride()).min(storage.len());
                assert!(storage[row * layout.stride() + row_bytes..padding_end]
                    .iter()
                    .all(|byte| *byte == 99));
            }
            assert!(storage[layout.byte_len()..].iter().all(|byte| *byte == 99));
        }
    }
}
