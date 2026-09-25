//! Checks rendered pixels against expected colors.
//! Run with `cargo test --test visual_regression`.

use futures::executor::block_on;
use grafo::{
    BackdropCaptureArea, BackdropEffectConfig, BorderRadii, Color, ColorInterpolation,
    DrawCommandError, EffectError, Fill, Gradient, GradientStop, GradientStopOffset,
    LinearGradientDesc, LinearGradientLine, Renderer, RendererContext, RendererCreationError,
    Shape, ShapeDrawCommandOptions, ShapeEffectConfig, ShapeTextureFitMode, ShapeTextureOptions,
    Stroke, TransformInstance,
};
use grafo::{EffectResourceError, SceneError, WgpuBackendError};
use grafo_test_scenes::shaders::{PASSTHROUGH_WGSL, SHAPE_DROP_WGSL};
use grafo_test_scenes::{
    build_main_scene, build_nested_targets_scene, check_pixels, PixelExpectation, CANVAS_HEIGHT,
    CANVAS_WIDTH,
};

/// Creates a headless renderer. If no suitable GPU adapter is available,
/// prints a skip message and returns `None`.
fn create_headless_renderer() -> Option<Renderer<'static>> {
    create_headless_renderer_with_size_and_scale((CANVAS_WIDTH, CANVAS_HEIGHT), 1.0)
}

fn create_headless_renderer_with_size_and_scale(
    physical_size: (u32, u32),
    scale_factor: f64,
) -> Option<Renderer<'static>> {
    match block_on(Renderer::try_new_headless(physical_size, scale_factor)) {
        Ok(r) => Some(r),
        Err(RendererCreationError::AdapterNotAvailable(_)) => {
            println!("Skipping test: no suitable GPU adapter available.");
            None
        }
        Err(e) => panic!("Failed to create headless renderer: {e}"),
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

#[test]
fn shape_effect_is_resolved_before_backdrop_capture_with_msaa() {
    // This test passes when no GPU present on purpose: I don't have a GPU on my CI just yet, so I'm
    //  running this test and the visual regression example manually before commiting changes
    let Some(mut renderer) = create_headless_renderer_with_size_and_scale((64, 64), 1.0) else {
        return;
    };
    renderer.set_msaa_samples(4);
    renderer
        .load_effect(9_101, &[SHAPE_DROP_WGSL])
        .expect("to load the MSAA shape effect");
    renderer
        .load_effect(9_102, &[PASSTHROUGH_WGSL])
        .expect("to load the MSAA backdrop effect");

    renderer
        .add_shape(
            Shape::rect([(0.0, 0.0), (64.0, 64.0)], Stroke::default()),
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(220, 200, 50)),
        )
        .unwrap();
    let panel_id = renderer
        .add_shape(
            Shape::rect([(16.0, 16.0), (48.0, 48.0)], Stroke::default()),
            None,
            Some(9_103),
            ShapeDrawCommandOptions::new(),
        )
        .unwrap();
    renderer
        .set_shape_effect(panel_id, 9_101, &[], ShapeEffectConfig::new().outset(12.0))
        .expect("to attach the MSAA shape effect");
    renderer
        .set_shape_backdrop_effect(panel_id, 9_102, &[], BackdropEffectConfig::default())
        .expect("to attach the MSAA backdrop effect");

    let mut pixel_buffer = Vec::new();
    renderer.render_to_buffer(&mut pixel_buffer).unwrap();

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

const CACHED_SHAPE_EFFECT_RED_MASK: &str = r#"
@fragment
fn effect_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let coverage = textureSample(t_input, s_input, uv).a;
    return vec4<f32>(coverage, 0.0, 0.0, coverage);
}
"#;

const PARAMETERIZED_COLOR_EFFECT: &str = r#"
struct Params {
    color: vec4<f32>,
}
@group(1) @binding(0) var<uniform> params: Params;

@fragment
fn effect_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    return params.color * textureSample(t_input, s_input, uv).a;
}
"#;

#[test]
fn effect_reload_removes_every_attachment_and_accepts_fresh_parameters() {
    let Some(mut renderer) = create_headless_renderer_with_size_and_scale((48, 32), 1.0) else {
        return;
    };
    let effect_id = 9_204;
    renderer
        .load_effect(effect_id, &[PARAMETERIZED_COLOR_EFFECT])
        .unwrap();
    let background = renderer
        .add_shape(
            Shape::rect([(0.0, 0.0), (48.0, 32.0)], Stroke::default()),
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::WHITE),
        )
        .unwrap();
    let group = renderer
        .add_shape(
            Shape::rect([(0.0, 0.0), (16.0, 32.0)], Stroke::default()),
            Some(background),
            None,
            ShapeDrawCommandOptions::new().color(Color::WHITE),
        )
        .unwrap();
    let backdrop = renderer
        .add_shape(
            Shape::rect([(16.0, 0.0), (32.0, 32.0)], Stroke::default()),
            Some(background),
            None,
            ShapeDrawCommandOptions::new(),
        )
        .unwrap();
    let shape = renderer
        .add_shape(
            Shape::rect([(32.0, 0.0), (48.0, 32.0)], Stroke::default()),
            Some(background),
            None,
            ShapeDrawCommandOptions::new(),
        )
        .unwrap();
    renderer
        .set_shape_effect(
            shape,
            effect_id,
            bytemuck::cast_slice(&[1.0_f32, 0.0, 1.0, 1.0]),
            ShapeEffectConfig::default(),
        )
        .unwrap();
    renderer
        .set_group_effect(
            group,
            effect_id,
            bytemuck::cast_slice(&[1.0_f32, 0.0, 0.0, 1.0]),
        )
        .unwrap();
    renderer
        .set_shape_backdrop_effect(
            backdrop,
            effect_id,
            bytemuck::cast_slice(&[0.0_f32, 0.0, 1.0, 1.0]),
            BackdropEffectConfig::default(),
        )
        .unwrap();

    for result in [
        renderer.set_shape_effect(shape, effect_id, &[], ShapeEffectConfig::default()),
        renderer.update_shape_effect_params(shape, &[]),
        renderer.set_group_effect(group, effect_id, &[]),
        renderer.set_shape_backdrop_effect(
            backdrop,
            effect_id,
            &[],
            BackdropEffectConfig::default(),
        ),
        renderer.update_group_effect_params(group, &[]),
        renderer.update_backdrop_effect_params(backdrop, &[]),
    ] {
        assert!(matches!(
            result,
            Err(EffectError::Backend(WgpuBackendError::Effect(
                EffectResourceError::InvalidParams(_)
            )))
        ));
    }

    let mut pixel_buffer = Vec::new();
    renderer.render_to_buffer(&mut pixel_buffer).unwrap();
    assert_eq!(read_pixel_rgba(&pixel_buffer, 48, 8, 16), [255, 0, 0, 255]);
    assert_eq!(read_pixel_rgba(&pixel_buffer, 48, 24, 16), [0, 0, 255, 255]);
    assert_eq!(
        read_pixel_rgba(&pixel_buffer, 48, 40, 16),
        [255, 0, 255, 255]
    );

    renderer
        .update_group_effect_params(group, bytemuck::cast_slice(&[0.0_f32, 1.0, 0.0, 1.0]))
        .unwrap();
    renderer
        .update_backdrop_effect_params(backdrop, bytemuck::cast_slice(&[1.0_f32, 1.0, 0.0, 1.0]))
        .unwrap();
    renderer
        .update_shape_effect_params(shape, bytemuck::cast_slice(&[0.0_f32, 1.0, 1.0, 1.0]))
        .unwrap();
    renderer.render_to_buffer(&mut pixel_buffer).unwrap();
    assert_eq!(read_pixel_rgba(&pixel_buffer, 48, 8, 16), [0, 255, 0, 255]);
    assert_eq!(
        read_pixel_rgba(&pixel_buffer, 48, 24, 16),
        [255, 255, 0, 255]
    );

    assert_eq!(
        read_pixel_rgba(&pixel_buffer, 48, 40, 16),
        [0, 255, 255, 255]
    );

    for params in [&[1.0_f32, 0.0, 1.0][..], &[1.0_f32, 0.0, 1.0, 1.0, 0.0][..]] {
        let params = bytemuck::cast_slice(params);
        for result in [
            renderer.update_group_effect_params(group, params),
            renderer.update_backdrop_effect_params(backdrop, params),
        ] {
            assert!(matches!(
                result,
                Err(EffectError::Scene(SceneError::ParameterSizeMismatch {
                    effect_id: rejected_effect_id,
                    expected_size: 16,
                    actual_size,
                })) if rejected_effect_id == effect_id && actual_size == params.len() as u64
            ));
        }
        renderer.render_to_buffer(&mut pixel_buffer).unwrap();
        assert_eq!(read_pixel_rgba(&pixel_buffer, 48, 8, 16), [0, 255, 0, 255]);
        assert_eq!(
            read_pixel_rgba(&pixel_buffer, 48, 24, 16),
            [255, 255, 0, 255]
        );
    }

    let reloaded_source =
        PARAMETERIZED_COLOR_EFFECT.replace("params.color *", "params.color.gbra *");
    let larger_uniform = PARAMETERIZED_COLOR_EFFECT
        .replace("color: vec4<f32>,", "color: vec4<f32>, extra: vec4<f32>,")
        .replace("params.color *", "params.extra *");
    for (pass_sources, params, expected) in [
        (
            &[PASSTHROUGH_WGSL, &reloaded_source, PASSTHROUGH_WGSL][..],
            &[0.0_f32, 1.0, 0.0, 1.0][..],
            [255, 0, 0, 255],
        ),
        (
            &[PASSTHROUGH_WGSL, &larger_uniform][..],
            &[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0][..],
            [0, 255, 255, 255],
        ),
        (
            &[PARAMETERIZED_COLOR_EFFECT][..],
            &[1.0, 1.0, 0.0, 1.0][..],
            [255, 255, 0, 255],
        ),
    ] {
        renderer.load_effect(effect_id, pass_sources).unwrap();
        let params = bytemuck::cast_slice(params);
        for result in [
            renderer.update_group_effect_params(group, params),
            renderer.update_backdrop_effect_params(backdrop, params),
            renderer.update_shape_effect_params(shape, params),
        ] {
            assert!(matches!(
                result,
                Err(EffectError::Scene(SceneError::NodeNotFound(_)))
            ));
        }
        renderer.render_to_buffer(&mut pixel_buffer).unwrap();
        for x in [8, 24, 40] {
            assert_eq!(read_pixel_rgba(&pixel_buffer, 48, x, 16), [255; 4]);
        }

        renderer.set_group_effect(group, effect_id, params).unwrap();
        renderer
            .set_shape_backdrop_effect(backdrop, effect_id, params, BackdropEffectConfig::default())
            .unwrap();
        renderer
            .set_shape_effect(shape, effect_id, params, ShapeEffectConfig::default())
            .unwrap();
        assert!(matches!(
            renderer.load_effect(effect_id, &[pass_sources[0], ""]),
            Err(EffectError::Backend(WgpuBackendError::Effect(
                EffectResourceError::InvalidShader { pass_index: 1, .. }
            )))
        ));
        for samples in [4, 1] {
            renderer.load_effect(effect_id, pass_sources).unwrap();
            renderer.set_msaa_samples(samples);
            renderer.render_to_buffer(&mut pixel_buffer).unwrap();
            for x in [8, 24, 40] {
                assert_eq!(read_pixel_rgba(&pixel_buffer, 48, x, 16), expected);
            }
        }
    }

    renderer
        .load_effect(effect_id, &[PASSTHROUGH_WGSL])
        .unwrap();
    renderer.render_to_buffer(&mut pixel_buffer).unwrap();
    assert_eq!(read_pixel_rgba(&pixel_buffer, 48, 8, 16), [255; 4]);
    assert_eq!(read_pixel_rgba(&pixel_buffer, 48, 24, 16), [255; 4]);
    assert_eq!(read_pixel_rgba(&pixel_buffer, 48, 40, 16), [255; 4]);

    renderer.set_group_effect(group, effect_id, &[]).unwrap();
    renderer
        .set_shape_backdrop_effect(backdrop, effect_id, &[], BackdropEffectConfig::default())
        .unwrap();
    renderer
        .set_shape_effect(shape, effect_id, &[], ShapeEffectConfig::default())
        .unwrap();
    let unexpected_params = bytemuck::cast_slice(&[1.0_f32, 0.0, 0.0, 1.0]);
    for result in [
        renderer.set_shape_effect(
            shape,
            effect_id,
            unexpected_params,
            ShapeEffectConfig::default(),
        ),
        renderer.update_shape_effect_params(shape, unexpected_params),
        renderer.set_group_effect(group, effect_id, unexpected_params),
        renderer.set_shape_backdrop_effect(
            backdrop,
            effect_id,
            unexpected_params,
            BackdropEffectConfig::default(),
        ),
        renderer.update_group_effect_params(group, unexpected_params),
        renderer.update_backdrop_effect_params(backdrop, unexpected_params),
    ] {
        assert!(matches!(
            result,
            Err(EffectError::Backend(WgpuBackendError::Effect(
                EffectResourceError::InvalidParams(_)
            )))
        ));
    }
    renderer.render_to_buffer(&mut pixel_buffer).unwrap();
    assert_eq!(read_pixel_rgba(&pixel_buffer, 48, 8, 16), [255; 4]);
    assert_eq!(read_pixel_rgba(&pixel_buffer, 48, 24, 16), [255; 4]);
    assert_eq!(read_pixel_rgba(&pixel_buffer, 48, 40, 16), [255; 4]);

    renderer
        .load_effect(effect_id, &[PARAMETERIZED_COLOR_EFFECT])
        .unwrap();
    let red = bytemuck::cast_slice(&[1.0_f32, 0.0, 0.0, 1.0]);
    let green = bytemuck::cast_slice(&[0.0_f32, 1.0, 0.0, 1.0]);
    let blue = bytemuck::cast_slice(&[0.0_f32, 0.0, 1.0, 1.0]);
    renderer.set_group_effect(backdrop, effect_id, red).unwrap();
    renderer
        .set_shape_backdrop_effect(backdrop, effect_id, green, BackdropEffectConfig::default())
        .unwrap();
    renderer
        .set_shape_effect(backdrop, effect_id, blue, ShapeEffectConfig::default())
        .unwrap();
    renderer.render_to_buffer(&mut pixel_buffer).unwrap();
    assert_eq!(read_pixel_rgba(&pixel_buffer, 48, 24, 16), [255, 0, 0, 255]);
    renderer.remove_group_effect(backdrop);
    renderer.render_to_buffer(&mut pixel_buffer).unwrap();
    assert_eq!(read_pixel_rgba(&pixel_buffer, 48, 24, 16), [0, 255, 0, 255]);
    renderer.remove_backdrop_effect(backdrop);
    renderer.render_to_buffer(&mut pixel_buffer).unwrap();
    assert_eq!(read_pixel_rgba(&pixel_buffer, 48, 24, 16), [0, 0, 255, 255]);
    renderer.set_group_effect(backdrop, effect_id, red).unwrap();
    renderer
        .set_shape_backdrop_effect(backdrop, effect_id, green, BackdropEffectConfig::default())
        .unwrap();
    renderer.unload_effect(effect_id);
    renderer.render_to_buffer(&mut pixel_buffer).unwrap();
    assert_eq!(read_pixel_rgba(&pixel_buffer, 48, 24, 16), [255; 4]);
    assert!(
        matches!(renderer.set_group_effect(backdrop, effect_id, red), Err(EffectError::Backend(WgpuBackendError::Effect(EffectResourceError::EffectNotLoaded(id)))) if id == effect_id)
    );
}

#[test]
fn invalid_effect_can_be_replaced_with_a_valid_shader() {
    let Some(mut renderer) = create_headless_renderer_with_size_and_scale((32, 32), 1.0) else {
        return;
    };

    // Parsing and semantic failures must not poison the renderer's shared validator.
    for source in [
        "@fragment fn effect_main(",
        "",
        "@fragment fn effect_main() -> @location(0) vec4<f32> { return vec3<f32>(1.0); }",
    ] {
        assert!(matches!(
            renderer.load_effect(9_201, &[source]),
            Err(EffectError::Backend(WgpuBackendError::Effect(
                EffectResourceError::InvalidShader { pass_index: 0, .. }
            )))
        ));
    }
    renderer.load_effect(9_201, &[PASSTHROUGH_WGSL]).unwrap();
    let shape_id = renderer
        .add_shape(
            Shape::rect([(8.0, 8.0), (24.0, 24.0)], Stroke::default()),
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(255, 0, 0)),
        )
        .unwrap();
    renderer.set_group_effect(shape_id, 9_201, &[]).unwrap();

    let mut pixel_buffer = Vec::new();
    renderer.render_to_buffer(&mut pixel_buffer).unwrap();
    assert_eq!(read_pixel_rgba(&pixel_buffer, 32, 16, 16), [255, 0, 0, 255]);

    for _ in 0..2 {
        assert!(matches!(
            renderer.load_effect(9_201, &[PASSTHROUGH_WGSL, ""]),
            Err(EffectError::Backend(WgpuBackendError::Effect(
                EffectResourceError::InvalidShader { pass_index: 1, .. }
            )))
        ));
        renderer.render_to_buffer(&mut pixel_buffer).unwrap();
        assert_eq!(read_pixel_rgba(&pixel_buffer, 32, 16, 16), [255, 0, 0, 255]);
    }
}

#[test]
fn shape_effects_follow_geometry_and_placement_across_queue_rebuilds() {
    let Some(mut renderer) = create_headless_renderer_with_size_and_scale((96, 48), 1.0) else {
        return;
    };
    renderer
        .load_effect(8_101, &[CACHED_SHAPE_EFFECT_RED_MASK])
        .unwrap();
    renderer.load_shape(
        Shape::rect([(0.0, 0.0), (24.0, 24.0)], Stroke::default()),
        8_102,
        Some(8_103),
    );
    renderer.load_shape(
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
        renderer.clear_draw_queue();
        let root_id = renderer
            .add_shape(
                Shape::rect([(0.0, 0.0), (96.0, 48.0)], Stroke::default()),
                None,
                None,
                ShapeDrawCommandOptions::new().color(Color::WHITE),
            )
            .unwrap();
        for (shape_key, left) in placements {
            let node_id = renderer
                .add_cached_shape(
                    shape_key,
                    Some(root_id),
                    ShapeDrawCommandOptions::new()
                        .transform(TransformInstance::translation(left as f32, 12.0)),
                )
                .unwrap();
            renderer
                .set_shape_effect(node_id, 8_101, &[], ShapeEffectConfig::new().outset(3.0))
                .unwrap();
        }
        renderer.render_to_buffer(&mut pixels).unwrap();

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
fn cached_shape_effects_share_the_normal_texture_pipeline() {
    let Some(mut renderer) = create_headless_renderer_with_size_and_scale((96, 48), 1.0) else {
        return;
    };
    renderer.load_effect(8_151, &[PASSTHROUGH_WGSL]).unwrap();
    renderer.load_shape(
        Shape::rect([(0.0, 0.0), (24.0, 24.0)], Stroke::default()),
        8_152,
        Some(8_153),
    );
    let root_id = renderer
        .add_shape(
            Shape::rect([(0.0, 0.0), (96.0, 48.0)], Stroke::default()),
            None,
            None,
            ShapeDrawCommandOptions::new().clips_children(false),
        )
        .unwrap();

    for (translation_x, color) in [
        (8.0, Color::rgb(220, 50, 50)),
        (48.0, Color::rgb(50, 90, 220)),
    ] {
        let node_id = renderer
            .add_cached_shape(
                8_152,
                Some(root_id),
                ShapeDrawCommandOptions::new()
                    .color(color)
                    .transform(TransformInstance::translation(translation_x, 12.0)),
            )
            .unwrap();
        renderer
            .set_shape_effect(node_id, 8_151, &[], ShapeEffectConfig::new().outset(3.0))
            .unwrap();
    }

    let mut pixels = Vec::new();
    renderer.render_to_buffer(&mut pixels).unwrap();
    renderer.render_to_buffer(&mut pixels).unwrap();

    let cache_metrics = renderer.last_shape_effect_cache_metrics();
    assert_eq!(cache_metrics.hits, 2);
    assert_eq!(cache_metrics.misses, 0);

    let pipeline_switches = renderer.last_pipeline_switch_counts();
    assert_eq!(pipeline_switches.to_leaf_draw, 1);
    assert_eq!(pipeline_switches.to_composite, 0);
    assert_eq!(pipeline_switches.total_switches, 1);
}

#[cfg(feature = "render_metrics")]
#[test]
fn cached_shape_effect_is_invalidated_by_normal_pipeline_recreation() {
    let Some(mut renderer) = create_headless_renderer_with_size_and_scale((64, 64), 1.0) else {
        return;
    };
    renderer.load_effect(8_161, &[SHAPE_DROP_WGSL]).unwrap();
    let shape_id = renderer
        .add_shape(
            Shape::rect([(16.0, 16.0), (48.0, 48.0)], Stroke::default()),
            None,
            Some(8_162),
            ShapeDrawCommandOptions::new().color(Color::rgb(220, 200, 50)),
        )
        .unwrap();
    renderer
        .set_shape_effect(shape_id, 8_161, &[], ShapeEffectConfig::new().outset(12.0))
        .unwrap();

    let mut pixels = Vec::new();
    renderer.render_to_buffer(&mut pixels).unwrap();
    renderer.set_msaa_samples(4);
    renderer.render_to_buffer(&mut pixels).unwrap();

    let cache_metrics = renderer.last_shape_effect_cache_metrics();
    assert_eq!(cache_metrics.hits, 0);
    assert_eq!(cache_metrics.misses, 1);
    assert_eq!(read_pixel_rgba(&pixels, 64, 52, 32), [0, 0, 255, 255]);
}

#[test]
fn shape_effect_parameter_updates_change_visible_color() {
    let Some(mut renderer) = create_headless_renderer_with_size_and_scale((48, 48), 1.0) else {
        return;
    };
    renderer
        .load_effect(8_201, &[PARAMETERIZED_COLOR_EFFECT])
        .unwrap();
    let shape_id = renderer
        .add_shape(
            Shape::rect([(12.0, 12.0), (36.0, 36.0)], Stroke::default()),
            None,
            Some(8_202),
            ShapeDrawCommandOptions::new().transform(TransformInstance::translation(6.0, 4.0)),
        )
        .unwrap();
    let blue = [0.0f32, 0.0, 1.0, 1.0];
    renderer
        .set_shape_effect(
            shape_id,
            8_201,
            bytemuck::bytes_of(&blue),
            ShapeEffectConfig::default(),
        )
        .unwrap();

    let mut pixels = Vec::new();
    renderer.render_to_buffer(&mut pixels).unwrap();
    assert_eq!(read_pixel_rgba(&pixels, 48, 14, 14), [0, 0, 0, 0]);
    assert_eq!(read_pixel_rgba(&pixels, 48, 30, 28), [0, 0, 255, 255]);

    let red = [1.0f32, 0.0, 0.0, 1.0];
    renderer
        .update_shape_effect_params(shape_id, bytemuck::bytes_of(&red))
        .unwrap();
    renderer.render_to_buffer(&mut pixels).unwrap();
    assert_eq!(read_pixel_rgba(&pixels, 48, 30, 28), [255, 0, 0, 255]);
}

#[test]
fn main_scene_pixel_expectations() {
    let Some(mut renderer) = create_headless_renderer() else {
        return;
    };

    let mut pixel_buffer: Vec<u8> = Vec::new();
    // Start with MSAA so a rejected submission cannot reuse a prior valid output.
    for sample_count in [4, 1, 4, 1] {
        renderer.set_msaa_samples(sample_count);
        // Rebuild the queue on every render, including when MSAA remains unchanged.
        for _ in 0..2 {
            renderer.clear_draw_queue();
            let expectations = build_main_scene(&mut renderer);
            renderer.render_to_buffer(&mut pixel_buffer).unwrap();
            assert_pixels_match(&pixel_buffer, &expectations);
        }
        for _ in 0..2 {
            renderer.clear_draw_queue();
            let expectations = build_nested_targets_scene(&mut renderer);
            renderer.render_to_buffer(&mut pixel_buffer).unwrap();
            assert_pixels_match(&pixel_buffer, &expectations);
        }
    }
}

fn rebuild_texture_material_scene(
    renderer: &mut Renderer,
    gradient_color: Color,
    offset: f32,
    swaps_fills: bool,
) {
    renderer.clear_draw_queue();
    renderer
        .add_shape(
            Shape::rect([(0.0, 0.0), (64.0, 32.0)], Stroke::default()),
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::WHITE),
        )
        .unwrap();
    for left in [14.0, 46.0] {
        renderer
            .add_shape(
                Shape::rect([(left, 0.0), (left + 4.0, 32.0)], Stroke::default()),
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
        let panel = renderer
            .add_shape(
                Shape::rect([(left, 4.0), (left + 24.0, 28.0)], Stroke::default()),
                None,
                None,
                ShapeDrawCommandOptions::new().fill(fill),
            )
            .unwrap();
        renderer
            .set_shape_backdrop_effect(panel, 9_301, &[], BackdropEffectConfig::default())
            .unwrap();
    }
}

#[test]
fn texture_materials_follow_scene_changes_across_queue_rebuilds() {
    let Some(mut renderer) = create_headless_renderer_with_size_and_scale((64, 32), 1.0) else {
        return;
    };
    renderer.load_effect(9_301, &[PASSTHROUGH_WGSL]).unwrap();
    let mut pixels = Vec::new();
    for samples in [1, 4, 1] {
        renderer.set_msaa_samples(samples);
        for _ in 0..2 {
            rebuild_texture_material_scene(&mut renderer, Color::rgba(255, 0, 0, 128), 2.0, false);
            renderer.render_to_buffer(&mut pixels).unwrap();
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
        rebuild_texture_material_scene(&mut renderer, color, offset, swaps_fills);
        renderer.render_to_buffer(&mut pixels).unwrap();
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
fn empty_draw_queue() {
    let Some(mut renderer) = create_headless_renderer() else {
        return;
    };

    let mut pixel_buffer = Vec::new();
    renderer.render_to_buffer(&mut pixel_buffer).unwrap();
    assert_eq!(
        pixel_buffer.len(),
        (CANVAS_WIDTH * CANVAS_HEIGHT * 4) as usize
    );
    assert!(pixel_buffer.iter().all(|&byte| byte == 0));
    for samples in [4, 1] {
        renderer.set_msaa_samples(samples);
        renderer.clear_draw_queue();
        renderer
            .add_shape(
                Shape::rect(
                    [(0.0, 0.0), (CANVAS_WIDTH as f32, CANVAS_HEIGHT as f32)],
                    Stroke::default(),
                ),
                None,
                None,
                ShapeDrawCommandOptions::new().color(Color::WHITE),
            )
            .unwrap();
        renderer.render_to_buffer(&mut pixel_buffer).unwrap();
        assert!(pixel_buffer.iter().all(|&byte| byte == 255));
        renderer.clear_draw_queue();
        renderer.render_to_buffer(&mut pixel_buffer).unwrap();
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

/// Renderers created from the same context must keep independent draw queues while sharing GPU
/// resources such as textures.
#[test]
fn renderers_from_one_context_share_resources_and_keep_draw_queues_independent() {
    let context = match block_on(RendererContext::try_new()) {
        Ok(context) => context,
        Err(RendererCreationError::AdapterNotAvailable(_)) => {
            println!("Skipping test: no suitable GPU adapter available.");
            return;
        }
        Err(error) => panic!("Failed to create renderer context: {error}"),
    };

    let mut first = Renderer::try_new_headless_with_context(context.clone(), (16, 16), 1.0)
        .expect("to create first headless renderer");
    let mut second = Renderer::try_new_headless_with_context(context, (16, 16), 1.0)
        .expect("to create second headless renderer");

    first
        .texture_manager()
        .allocate_texture_with_data(42, (1, 1), &[0, 0, 0, 0]);
    assert!(second.texture_manager().is_texture_loaded(42));

    first.load_shape(
        Shape::rect([(0.0, 0.0), (16.0, 16.0)], Stroke::default()),
        99,
        Some(99),
    );
    second
        .add_cached_shape(
            99,
            None,
            ShapeDrawCommandOptions::new()
                .color(Color::rgb(0, 255, 0))
                .foreground_texture_id(42),
        )
        .expect("to add shape loaded by first renderer");

    first
        .add_shape(
            Shape::rect([(0.0, 0.0), (16.0, 16.0)], Stroke::default()),
            None,
            None,
            ShapeDrawCommandOptions::new()
                .color(Color::rgb(255, 0, 0))
                .background_texture_id(42),
        )
        .expect("to add shape to first renderer");

    let mut first_pixels = Vec::new();
    let mut second_pixels = Vec::new();
    first.render_to_buffer(&mut first_pixels).unwrap();
    second.render_to_buffer(&mut second_pixels).unwrap();

    assert_eq!(read_pixel_rgba(&first_pixels, 16, 8, 8), [255, 0, 0, 255]);
    assert_eq!(read_pixel_rgba(&second_pixels, 16, 8, 8), [0, 255, 0, 255]);
    assert_eq!(first.texture_manager().size(), (1, 1));

    for samples in [4, 1, 4] {
        first.set_msaa_samples(samples);
        first.resize((20, 20));
        first.render_to_buffer(&mut first_pixels).unwrap();
        second.render_to_buffer(&mut second_pixels).unwrap();

        assert_eq!(read_pixel_rgba(&first_pixels, 20, 8, 8), [255, 0, 0, 255]);
        assert_eq!(read_pixel_rgba(&second_pixels, 16, 8, 8), [0, 255, 0, 255]);
        assert_eq!(
            first.texture_manager().size(),
            (1, 1),
            "MSAA changes must reuse the texture binding across renderers and layers",
        );
    }

    first
        .texture_manager()
        .allocate_texture_with_data(42, (1, 1), &[0, 0, 255, 255]);
    assert_eq!(second.texture_manager().size(), (1, 0));
    first.render_to_buffer(&mut first_pixels).unwrap();
    second.render_to_buffer(&mut second_pixels).unwrap();
    assert_eq!(read_pixel_rgba(&first_pixels, 20, 8, 8), [0, 0, 255, 255]);
    assert_eq!(read_pixel_rgba(&second_pixels, 16, 8, 8), [0, 0, 255, 255]);
    assert_eq!(first.texture_manager().size(), (1, 1));

    first.texture_manager().remove_texture(42);
    assert_eq!(second.texture_manager().size(), (0, 0));
    first.render_to_buffer(&mut first_pixels).unwrap();
    second.render_to_buffer(&mut second_pixels).unwrap();
    assert_eq!(read_pixel_rgba(&first_pixels, 20, 8, 8), [255, 0, 0, 255]);
    assert_eq!(read_pixel_rgba(&second_pixels, 16, 8, 8), [0, 255, 0, 255]);

    second
        .load_effect(99_101, &[CACHED_SHAPE_EFFECT_RED_MASK])
        .unwrap();
    second.clear_draw_queue();
    let shared_rect = second
        .add_cached_shape(99, None, ShapeDrawCommandOptions::new())
        .unwrap();
    second
        .set_shape_effect(shared_rect, 99_101, &[], ShapeEffectConfig::default())
        .unwrap();
    second.render_to_buffer(&mut second_pixels).unwrap();
    assert_eq!(
        read_pixel_rgba(&second_pixels, 16, 12, 12),
        [255, 0, 0, 255]
    );

    second.clear_draw_queue();
    let triangle = second
        .add_shape(
            Shape::builder()
                .begin((0.0, 0.0))
                .line_to((16.0, 0.0))
                .line_to((0.0, 16.0))
                .close()
                .build(),
            None,
            None,
            ShapeDrawCommandOptions::new(),
        )
        .unwrap();
    second
        .set_shape_effect(triangle, 99_101, &[], ShapeEffectConfig::default())
        .unwrap();
    second.render_to_buffer(&mut second_pixels).unwrap();
    assert_eq!(read_pixel_rgba(&second_pixels, 16, 3, 3), [255, 0, 0, 255]);
    assert_eq!(read_pixel_rgba(&second_pixels, 16, 12, 12), [0, 0, 0, 0]);
}

#[test]
fn single_root_no_children() {
    let Some(mut renderer) = create_headless_renderer() else {
        return;
    };

    let shape = Shape::rect([(10.0, 10.0), (100.0, 100.0)], Stroke::default());
    renderer
        .add_shape(
            shape,
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(200, 50, 50)),
        )
        .unwrap();

    let mut pixel_buffer: Vec<u8> = Vec::new();
    renderer.render_to_buffer(&mut pixel_buffer).unwrap();

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
    let Some(mut renderer) =
        create_headless_renderer_with_size_and_scale(physical_size, scale_factor)
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
    renderer.texture_manager().allocate_texture_with_data(
        green_texture_id,
        (20, 20),
        &green_texture_with_transparent_border_20x20,
    );

    let shape = Shape::rect([(10.0, 10.0), (70.0, 70.0)], Stroke::default());
    renderer
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
    renderer.render_to_buffer(&mut pixel_buffer).unwrap();

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
    let Some(mut renderer) = create_headless_renderer_with_size_and_scale(physical_size, 1.0)
    else {
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
    renderer
        .texture_manager()
        .allocate_texture_with_data(texture_id, (20, 10), &texture_data);

    renderer
        .add_shape(
            Shape::rect([(8.0, 8.0), (56.0, 56.0)], Stroke::default()),
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
    renderer
        .add_shape(
            Shape::rect([(88.0, 8.0), (136.0, 56.0)], Stroke::default()),
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
    renderer.render_to_buffer(&mut pixel_buffer).unwrap();

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
    let Some(mut renderer) = create_headless_renderer() else {
        return;
    };

    let clip_rect_id = renderer
        .add_clipping_rect(
            [(20.0, 20.0), (80.0, 80.0)],
            None,
            None::<TransformInstance>,
            true,
        )
        .unwrap();
    let child = Shape::rect([(0.0, 0.0), (100.0, 100.0)], Stroke::default());
    renderer
        .add_shape(
            child,
            Some(clip_rect_id),
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(200, 50, 50)),
        )
        .unwrap();

    let mut pixel_buffer: Vec<u8> = Vec::new();
    renderer.render_to_buffer(&mut pixel_buffer).unwrap();

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
    let Some(mut renderer) = create_headless_renderer_with_size_and_scale(physical_size, 1.0)
    else {
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

    renderer
        .load_effect(
            AVERAGE_WITH_RIGHT_NEIGHBOR_EFFECT_ID,
            &[AVERAGE_WITH_RIGHT_NEIGHBOR_WGSL],
        )
        .expect("Failed to compile deterministic backdrop test effect");

    let seeded_blue_panel = Shape::rect([(20.0, 20.0), (60.0, 60.0)], Stroke::default());
    renderer
        .add_shape(
            seeded_blue_panel.clone(),
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(40, 40, 220)),
        )
        .unwrap();
    let seeded_blue_panel_id = renderer
        .add_shape(
            seeded_blue_panel,
            None,
            None,
            ShapeDrawCommandOptions::new(),
        )
        .unwrap();
    renderer
        .set_shape_backdrop_effect(
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
    renderer.render_to_buffer(&mut seeded_frame).unwrap();

    renderer.clear_draw_queue();

    let visible_red_source = Shape::rect([(70.0, 20.0), (100.0, 60.0)], Stroke::default());
    renderer
        .add_shape(
            visible_red_source,
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(220, 40, 40)),
        )
        .unwrap();

    let partially_offscreen_panel = Shape::rect([(70.0, 20.0), (100.0, 60.0)], Stroke::default());
    let partially_offscreen_panel_id = renderer
        .add_shape(
            partially_offscreen_panel,
            None,
            None,
            ShapeDrawCommandOptions::new(),
        )
        .unwrap();
    renderer
        .set_shape_backdrop_effect(
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
    renderer.render_to_buffer(&mut pixel_buffer).unwrap();

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
    let Some(mut renderer) = create_headless_renderer() else {
        return;
    };

    renderer
        .add_clipping_rect(
            [(20.0, 20.0), (80.0, 80.0)],
            None,
            None::<TransformInstance>,
            true,
        )
        .unwrap();

    let mut pixel_buffer: Vec<u8> = Vec::new();
    renderer.render_to_buffer(&mut pixel_buffer).unwrap();

    assert!(
        pixel_buffer.iter().all(|&byte| byte == 0),
        "Standalone clipping rect should not draw any pixels",
    );
}

#[test]
fn clipping_rect_rejects_non_axis_aligned_transform() {
    let Some(mut renderer) = create_headless_renderer() else {
        return;
    };

    let clip_rect_id = renderer
        .add_clipping_rect(
            [(20.0, 20.0), (80.0, 80.0)],
            None,
            None::<TransformInstance>,
            true,
        )
        .unwrap();
    assert!(matches!(
        renderer.add_clipping_rect(
            [(20.0, 20.0), (80.0, 80.0)],
            None,
            Some(TransformInstance::rotation_z_deg(45.0)),
            true,
        ),
        Err(DrawCommandError::Scene(
            SceneError::UnsupportedClipRectTransform
        ))
    ));

    let child = Shape::rect([(0.0, 0.0), (100.0, 100.0)], Stroke::default());
    renderer
        .add_shape(
            child,
            Some(clip_rect_id),
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(200, 50, 50)),
        )
        .unwrap();

    let mut pixel_buffer: Vec<u8> = Vec::new();
    renderer.render_to_buffer(&mut pixel_buffer).unwrap();

    let expectations = vec![
        PixelExpectation::opaque(50, 50, 200, 50, 50, "inside_unrotated_clip_rect"),
        PixelExpectation::transparent(10, 50, "outside_unrotated_clip_rect"),
    ];

    assert_pixels_match(&pixel_buffer, &expectations);
}

#[test]
fn gradient_fill_basic() {
    let Some(mut renderer) = create_headless_renderer() else {
        return;
    };

    let root = Shape::rect([(0.0, 0.0), (100.0, 100.0)], Stroke::default());
    let root_id = renderer
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

    renderer
        .add_shape(
            Shape::rect([(10.0, 10.0), (90.0, 90.0)], Stroke::default()),
            Some(root_id),
            None,
            ShapeDrawCommandOptions::new().fill(Fill::from(gradient)),
        )
        .unwrap();

    let mut pixel_buffer: Vec<u8> = Vec::new();
    renderer.render_to_buffer(&mut pixel_buffer).unwrap();

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
    let Some(mut renderer) = create_headless_renderer() else {
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

    renderer
        .add_shape(
            Shape::rect([(10.0, 10.0), (90.0, 90.0)], Stroke::default()),
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
    renderer.render_to_buffer(&mut pixel_buffer).unwrap();
    assert_pixels_match(&pixel_buffer, &expectations);

    // Changing MSAA recreates pipelines and their bind group layouts.
    renderer.set_msaa_samples(4);

    pixel_buffer.clear();
    renderer.render_to_buffer(&mut pixel_buffer).unwrap();
    assert_pixels_match(&pixel_buffer, &expectations);
}

/// Rounded parents force stencil clipping. Sample exposed parent pixels so an
/// opaque child cannot hide an incorrectly inherited gradient.
#[test]
fn stencil_increment_gradient_does_not_leak_to_solid_parent() {
    let Some(mut renderer) = create_headless_renderer() else {
        return;
    };

    // Full-canvas rect root so all children are visible.
    let root = renderer
        .add_shape(
            Shape::rect(
                [(0.0, 0.0), (CANVAS_WIDTH as f32, CANVAS_HEIGHT as f32)],
                Stroke::default(),
            ),
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

    let gradient_parent = renderer
        .add_shape(
            Shape::rounded_rect([(10.0, 10.0), (140.0, 90.0)], radii, Stroke::default()),
            Some(root),
            None,
            ShapeDrawCommandOptions::new().fill(Fill::from(gradient)),
        )
        .unwrap();

    renderer
        .add_shape(
            Shape::rect([(20.0, 20.0), (130.0, 80.0)], Stroke::default()),
            Some(gradient_parent),
            None,
            ShapeDrawCommandOptions::new().color(Color::WHITE),
        )
        .unwrap();

    let solid_parent = renderer
        .add_shape(
            Shape::rounded_rect([(160.0, 10.0), (290.0, 90.0)], radii, Stroke::default()),
            Some(root),
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(0, 200, 0)),
        )
        .unwrap();

    renderer
        .add_shape(
            Shape::rect([(170.0, 20.0), (280.0, 80.0)], Stroke::default()),
            Some(solid_parent),
            None,
            ShapeDrawCommandOptions::new().color(Color::WHITE),
        )
        .unwrap();

    let mut pixel_buffer = Vec::new();
    renderer.render_to_buffer(&mut pixel_buffer).unwrap();
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
    let Some(mut renderer) = create_headless_renderer() else {
        return;
    };

    let canvas_root = Shape::rect(
        [(0.0, 0.0), (CANVAS_WIDTH as f32, CANVAS_HEIGHT as f32)],
        Stroke::default(),
    );
    let canvas_root_id = renderer
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
    renderer
        .add_shape(
            shape,
            Some(canvas_root_id),
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(200, 50, 50)),
        )
        .unwrap();

    let rect = Shape::rect([(140.0, 10.0), (230.0, 100.0)], Stroke::default());
    renderer
        .add_shape(
            rect,
            Some(canvas_root_id),
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(200, 50, 50)),
        )
        .unwrap();

    let mut pixel_buffer: Vec<u8> = Vec::new();
    renderer.render_to_buffer(&mut pixel_buffer).unwrap();

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

fn assert_layered_backdrop_pixels(renderer: &mut Renderer, pixels: &mut Vec<u8>) {
    renderer.render_to_buffer(pixels).unwrap();
    let width = renderer.size().0;
    assert_eq!(read_pixel_rgba(pixels, width, 24, 32), [255, 0, 0, 255]);
    assert_eq!(read_pixel_rgba(pixels, width, 40, 32), [0, 255, 0, 255]);
}

#[test]
fn layered_backdrop_survives_capture_and_resource_changes() {
    let Some(mut renderer) = create_headless_renderer_with_size_and_scale((64, 64), 1.0) else {
        return;
    };
    let effect_id = 9_301;
    renderer
        .load_effect(
            effect_id,
            &[PASSTHROUGH_WGSL, PASSTHROUGH_WGSL, PASSTHROUGH_WGSL],
        )
        .unwrap();
    renderer
        .add_shape(
            Shape::rect([(0.0, 0.0), (64.0, 64.0)], Stroke::default()),
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(255, 0, 0)),
        )
        .unwrap();
    let group = renderer
        .add_shape(
            Shape::rect([(4.0, 4.0), (60.0, 60.0)], Stroke::default()),
            None,
            None,
            ShapeDrawCommandOptions::new(),
        )
        .unwrap();
    renderer.set_group_effect(group, effect_id, &[]).unwrap();
    renderer
        .add_shape(
            Shape::rect([(32.0, 4.0), (60.0, 60.0)], Stroke::default()),
            Some(group),
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(0, 255, 0)),
        )
        .unwrap();
    let panel = renderer
        .add_shape(
            Shape::rect([(16.0, 16.0), (48.0, 48.0)], Stroke::default()),
            Some(group),
            None,
            ShapeDrawCommandOptions::new(),
        )
        .unwrap();
    renderer
        .set_shape_backdrop_effect(
            panel,
            effect_id,
            &[],
            BackdropEffectConfig::new().padding(4.0).downsample(0.5),
        )
        .unwrap();

    let mut pixels = Vec::new();
    for _ in 0..4 {
        assert_layered_backdrop_pixels(&mut renderer, &mut pixels);
    }

    let moved_capture = BackdropEffectConfig::new()
        .capture_area(BackdropCaptureArea::ScreenRect([
            (24.0, 16.0),
            (56.0, 48.0),
        ]))
        .padding(4.0)
        .downsample(0.5);
    renderer
        .update_backdrop_effect_config(panel, moved_capture)
        .unwrap();
    assert_layered_backdrop_pixels(&mut renderer, &mut pixels);

    for samples in [4, 1] {
        renderer.set_msaa_samples(samples);
        for _ in 0..2 {
            assert_layered_backdrop_pixels(&mut renderer, &mut pixels);
        }
    }

    renderer.resize((80, 64));
    for _ in 0..2 {
        assert_layered_backdrop_pixels(&mut renderer, &mut pixels);
    }

    renderer
        .update_backdrop_effect_config(panel, moved_capture.downsample(1.0))
        .unwrap();
    assert_layered_backdrop_pixels(&mut renderer, &mut pixels);

    renderer.remove_backdrop_effect(panel);
    renderer
        .set_shape_backdrop_effect(panel, effect_id, &[], moved_capture)
        .unwrap();
    assert_layered_backdrop_pixels(&mut renderer, &mut pixels);
}

#[test]
fn readback_targets_survive_alternating_formats_and_resize() {
    let Some(mut renderer) = create_headless_renderer_with_size_and_scale((65, 7), 1.0) else {
        return;
    };
    renderer
        .add_shape(
            Shape::rect([(0.0, 0.0), (130.0, 10.0)], Stroke::default()),
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(51, 102, 153)),
        )
        .unwrap();
    let mut bgra_pixels = Vec::new();
    let mut argb_pixels = Vec::new();
    for size in [(65, 7), (129, 5), (65, 7)] {
        renderer.resize(size);
        argb_pixels.resize((size.0 * size.1) as usize, 0);
        for _ in 0..3 {
            renderer.render_to_buffer(&mut bgra_pixels).unwrap();
            renderer.render_to_argb32(&mut argb_pixels).unwrap();
            assert_eq!(bgra_pixels.len(), argb_pixels.len() * 4);
            for (bytes, pixel) in bgra_pixels.as_chunks::<4>().0.iter().zip(&argb_pixels) {
                assert_eq!(*bytes, [153, 102, 51, 255]);
                assert_eq!(*pixel, 0xff33_6699);
            }
        }
    }
}
