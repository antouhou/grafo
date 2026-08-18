/// Visual regression tests for the Grafo renderer.
///
/// These tests use the headless renderer to render scenes into a pixel buffer,
/// then validate specific pixel locations against expected colors.
///
/// Run with:   cargo test --test visual_regression
use futures::executor::block_on;
use grafo_test_scenes::{build_main_scene, check_pixels, CANVAS_HEIGHT, CANVAS_WIDTH};

/// Creates a headless renderer, returning `None` (and printing a skip message)
/// when no suitable GPU adapter is available.
fn create_headless_renderer() -> Option<grafo::Renderer<'static>> {
    create_headless_renderer_with_size_and_scale((CANVAS_WIDTH, CANVAS_HEIGHT), 1.0)
}

fn create_headless_renderer_with_size_and_scale(
    physical_size: (u32, u32),
    scale_factor: f64,
) -> Option<grafo::Renderer<'static>> {
    match block_on(grafo::Renderer::try_new_headless(
        physical_size,
        scale_factor,
    )) {
        Ok(r) => Some(r),
        Err(grafo::RendererCreationError::AdapterNotAvailable(_)) => {
            println!("Skipping test: no suitable GPU adapter available.");
            None
        }
        Err(e) => panic!("Failed to create headless renderer: {e}"),
    }
}

fn assert_pixels_match(pixel_buffer: &[u8], expectations: &[grafo_test_scenes::PixelExpectation]) {
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
    let Some(mut renderer) = create_headless_renderer_with_size_and_scale((64, 64), 1.0) else {
        return;
    };
    renderer.set_msaa_samples(4);
    renderer
        .load_effect(9_101, &[CACHED_SHAPE_EFFECT_BLUE_DROP])
        .expect("to load the MSAA shape effect");
    renderer
        .load_effect(9_102, &[CACHED_SHAPE_EFFECT_PASSTHROUGH])
        .expect("to load the MSAA backdrop effect");

    renderer
        .add_shape(
            grafo::Shape::rect([(0.0, 0.0), (64.0, 64.0)], grafo::Stroke::default()),
            None,
            None,
            grafo::ShapeDrawCommandOptions::new().color(grafo::Color::rgb(220, 200, 50)),
        )
        .unwrap();
    let panel_id = renderer
        .add_shape(
            grafo::Shape::rect([(16.0, 16.0), (48.0, 48.0)], grafo::Stroke::default()),
            None,
            Some(9_103),
            grafo::ShapeDrawCommandOptions::new(),
        )
        .unwrap();
    renderer
        .set_shape_effect(
            panel_id,
            9_101,
            &[],
            grafo::ShapeEffectConfig::new().outset(12.0),
        )
        .expect("to attach the MSAA shape effect");
    renderer
        .set_shape_backdrop_effect(panel_id, 9_102, &[], grafo::BackdropEffectConfig::default())
        .expect("to attach the MSAA backdrop effect");

    let mut pixel_buffer = Vec::new();
    renderer.render_to_buffer(&mut pixel_buffer);

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

const CACHED_SHAPE_EFFECT_PASSTHROUGH: &str = r#"
@fragment
fn effect_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    return textureSample(t_input, s_input, uv);
}
"#;

const CACHED_SHAPE_EFFECT_BLUE_DROP: &str = r#"
@fragment
fn effect_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let dimensions = vec2<f32>(textureDimensions(t_input));
    let coverage = textureSample(t_input, s_input, uv - vec2<f32>(8.0) / dimensions).a;
    return vec4<f32>(0.0, 0.0, coverage, coverage);
}
"#;

#[cfg(feature = "render_metrics")]
const CACHED_SHAPE_EFFECT_RED_MASK: &str = r#"
@fragment
fn effect_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let coverage = textureSample(t_input, s_input, uv).a;
    return vec4<f32>(coverage, 0.0, 0.0, coverage);
}
"#;

#[cfg(feature = "render_metrics")]
const PARAMETERIZED_CACHED_SHAPE_EFFECT: &str = r#"
struct Params {
    color: vec4<f32>,
}
@group(1) @binding(0) var<uniform> params: Params;

@fragment
fn effect_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    return params.color * textureSample(t_input, s_input, uv).a;
}
"#;

#[cfg(feature = "render_metrics")]
#[test]
fn unchanged_shape_effect_reuses_exact_gpu_result_and_collects_when_unused() {
    let Some(mut renderer) = create_headless_renderer_with_size_and_scale((64, 64), 1.0) else {
        return;
    };
    renderer
        .load_effect(8_001, &[CACHED_SHAPE_EFFECT_PASSTHROUGH])
        .unwrap();
    let shape_id = renderer
        .add_shape(
            grafo::Shape::rect([(16.0, 16.0), (48.0, 48.0)], grafo::Stroke::default()),
            None,
            Some(8_002),
            grafo::ShapeDrawCommandOptions::new().color(grafo::Color::rgb(220, 50, 50)),
        )
        .unwrap();
    renderer
        .set_shape_effect(
            shape_id,
            8_001,
            &[],
            grafo::ShapeEffectConfig::new().outset(4.0),
        )
        .unwrap();

    let mut pixels = Vec::new();
    renderer.render_to_buffer(&mut pixels);
    let first_frame = renderer.last_shape_effect_cache_metrics();
    assert_eq!(first_frame.misses, 1);
    assert_eq!(first_frame.hits, 0);
    assert_eq!(first_frame.generated_masks, 1);
    assert_eq!(first_frame.executed_passes, 1);

    renderer
        .load_effect(8_001, &[CACHED_SHAPE_EFFECT_PASSTHROUGH])
        .unwrap();
    renderer.render_to_buffer(&mut pixels);
    let second_frame = renderer.last_shape_effect_cache_metrics();
    assert_eq!(second_frame.hits, 1);
    assert_eq!(second_frame.misses, 0);
    assert_eq!(second_frame.generated_masks, 0);
    assert_eq!(second_frame.executed_passes, 0);

    renderer
        .load_effect(8_001, &[CACHED_SHAPE_EFFECT_RED_MASK])
        .unwrap();
    renderer.render_to_buffer(&mut pixels);
    let replaced_effect_frame = renderer.last_shape_effect_cache_metrics();
    assert_eq!(replaced_effect_frame.misses, 1);
    assert_eq!(replaced_effect_frame.hits, 0);

    renderer.remove_shape_effect(shape_id);
    renderer.render_to_buffer(&mut pixels);
    assert_eq!(
        renderer.last_shape_effect_cache_metrics().collected_results,
        1
    );
}

#[cfg(feature = "render_metrics")]
#[test]
fn cached_shape_effect_is_shared_by_instances_and_survives_queue_rebuild() {
    let Some(mut renderer) = create_headless_renderer_with_size_and_scale((96, 48), 1.0) else {
        return;
    };
    renderer
        .load_effect(8_101, &[CACHED_SHAPE_EFFECT_PASSTHROUGH])
        .unwrap();
    renderer.load_shape(
        grafo::Shape::rect([(0.0, 0.0), (24.0, 24.0)], grafo::Stroke::default()),
        8_102,
        Some(8_103),
    );

    let first_node = renderer
        .add_cached_shape_to_the_render_queue(
            8_102,
            None,
            grafo::ShapeDrawCommandOptions::new()
                .clips_children(false)
                .transform(grafo::TransformInstance::translation(4.0, 12.0)),
        )
        .unwrap();
    let second_node = renderer
        .add_cached_shape_to_the_render_queue(
            8_102,
            None,
            grafo::ShapeDrawCommandOptions::new()
                .transform(grafo::TransformInstance::translation(36.0, 12.0)),
        )
        .unwrap();
    for node_id in [first_node, second_node] {
        renderer
            .set_shape_effect(
                node_id,
                8_101,
                &[],
                grafo::ShapeEffectConfig::new().outset(3.0),
            )
            .unwrap();
    }

    let mut pixels = Vec::new();
    renderer.render_to_buffer(&mut pixels);
    let shared_frame = renderer.last_shape_effect_cache_metrics();
    assert_eq!(shared_frame.misses, 1);
    assert_eq!(shared_frame.hits, 1);

    renderer.clear_draw_queue();
    let rebuilt_node = renderer
        .add_cached_shape_to_the_render_queue(
            8_102,
            None,
            grafo::ShapeDrawCommandOptions::new()
                .transform(grafo::TransformInstance::translation(68.0, 12.0)),
        )
        .unwrap();
    renderer
        .set_shape_effect(
            rebuilt_node,
            8_101,
            &[],
            grafo::ShapeEffectConfig::new().outset(3.0),
        )
        .unwrap();
    renderer.render_to_buffer(&mut pixels);

    let rebuilt_frame = renderer.last_shape_effect_cache_metrics();
    assert_eq!(rebuilt_frame.hits, 1);
    assert_eq!(rebuilt_frame.misses, 0);
}

#[cfg(feature = "render_metrics")]
#[test]
fn cached_shape_effects_share_the_normal_texture_pipeline() {
    let Some(mut renderer) = create_headless_renderer_with_size_and_scale((96, 48), 1.0) else {
        return;
    };
    renderer
        .load_effect(8_151, &[CACHED_SHAPE_EFFECT_PASSTHROUGH])
        .unwrap();
    renderer.load_shape(
        grafo::Shape::rect([(0.0, 0.0), (24.0, 24.0)], grafo::Stroke::default()),
        8_152,
        Some(8_153),
    );

    for (translation_x, color) in [
        (8.0, grafo::Color::rgb(220, 50, 50)),
        (48.0, grafo::Color::rgb(50, 90, 220)),
    ] {
        let node_id = renderer
            .add_cached_shape_to_the_render_queue(
                8_152,
                None,
                grafo::ShapeDrawCommandOptions::new()
                    .color(color)
                    .transform(grafo::TransformInstance::translation(translation_x, 12.0)),
            )
            .unwrap();
        renderer
            .set_shape_effect(
                node_id,
                8_151,
                &[],
                grafo::ShapeEffectConfig::new().outset(3.0),
            )
            .unwrap();
    }

    let mut pixels = Vec::new();
    renderer.render_to_buffer(&mut pixels);
    renderer.render_to_buffer(&mut pixels);

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
fn cached_shape_effect_survives_normal_pipeline_recreation() {
    let Some(mut renderer) = create_headless_renderer_with_size_and_scale((64, 64), 1.0) else {
        return;
    };
    renderer
        .load_effect(8_161, &[CACHED_SHAPE_EFFECT_BLUE_DROP])
        .unwrap();
    let shape_id = renderer
        .add_shape(
            grafo::Shape::rect([(16.0, 16.0), (48.0, 48.0)], grafo::Stroke::default()),
            None,
            Some(8_162),
            grafo::ShapeDrawCommandOptions::new().color(grafo::Color::rgb(220, 200, 50)),
        )
        .unwrap();
    renderer
        .set_shape_effect(
            shape_id,
            8_161,
            &[],
            grafo::ShapeEffectConfig::new().outset(12.0),
        )
        .unwrap();

    let mut pixels = Vec::new();
    renderer.render_to_buffer(&mut pixels);
    renderer.set_msaa_samples(4);
    renderer.render_to_buffer(&mut pixels);

    let cache_metrics = renderer.last_shape_effect_cache_metrics();
    assert_eq!(cache_metrics.hits, 1);
    assert_eq!(cache_metrics.misses, 0);
    assert_eq!(read_pixel_rgba(&pixels, 64, 52, 32), [0, 0, 255, 255]);
}

#[cfg(feature = "render_metrics")]
#[test]
fn cached_shape_effect_uses_exact_parameter_bytes_on_transparent_shape() {
    let Some(mut renderer) = create_headless_renderer_with_size_and_scale((48, 48), 1.0) else {
        return;
    };
    renderer
        .load_effect(8_201, &[PARAMETERIZED_CACHED_SHAPE_EFFECT])
        .unwrap();
    let shape_id = renderer
        .add_shape(
            grafo::Shape::rect([(12.0, 12.0), (36.0, 36.0)], grafo::Stroke::default()),
            None,
            Some(8_202),
            grafo::ShapeDrawCommandOptions::new(),
        )
        .unwrap();
    let blue = [0.0f32, 0.0, 1.0, 1.0];
    renderer
        .set_shape_effect(
            shape_id,
            8_201,
            bytemuck::bytes_of(&blue),
            grafo::ShapeEffectConfig::default(),
        )
        .unwrap();

    let mut pixels = Vec::new();
    renderer.render_to_buffer(&mut pixels);
    assert_eq!(read_pixel_rgba(&pixels, 48, 24, 24), [0, 0, 255, 255]);

    let red = [1.0f32, 0.0, 0.0, 1.0];
    renderer
        .update_shape_effect_params(shape_id, bytemuck::bytes_of(&red))
        .unwrap();
    renderer.render_to_buffer(&mut pixels);
    assert_eq!(read_pixel_rgba(&pixels, 48, 24, 24), [255, 0, 0, 255]);
    let changed_parameter_frame = renderer.last_shape_effect_cache_metrics();
    assert_eq!(changed_parameter_frame.misses, 1);
    assert_eq!(changed_parameter_frame.hits, 0);
}

/// Main regression test — renders all shared visual-regression tiles.
#[test]
fn main_scene_pixel_expectations() {
    let Some(mut renderer) = create_headless_renderer() else {
        return;
    };

    let expectations = build_main_scene(&mut renderer);

    let mut pixel_buffer: Vec<u8> = Vec::new();
    renderer.render_to_buffer(&mut pixel_buffer);

    let failures = check_pixels(&pixel_buffer, CANVAS_WIDTH, CANVAS_HEIGHT, &expectations);
    if !failures.is_empty() {
        let message = format!(
            "{} pixel expectation(s) failed:\n{}",
            failures.len(),
            failures.join("\n"),
        );
        panic!("{message}");
    }
}

/// Regression test — empty draw queue should not crash.
#[test]
fn empty_draw_queue() {
    let Some(mut renderer) = create_headless_renderer() else {
        return;
    };

    // Render with nothing in the draw queue
    let mut pixel_buffer: Vec<u8> = Vec::new();
    renderer.render_to_buffer(&mut pixel_buffer);

    let bytes_per_pixel = 4;
    let expected_length = (CANVAS_WIDTH as usize) * (CANVAS_HEIGHT as usize) * bytes_per_pixel;
    assert_eq!(
        pixel_buffer.len(),
        expected_length,
        "Pixel buffer length should equal width * height * {bytes_per_pixel}",
    );

    // Every pixel should be fully transparent (all bytes zero)
    assert!(
        pixel_buffer.iter().all(|&byte| byte == 0),
        "Empty scene should produce a fully transparent (all-zero) buffer",
    );
}

/// Renderers created from the same context must keep independent draw queues while sharing GPU
/// resources such as textures.
#[test]
fn renderers_from_one_context_share_resources_and_keep_draw_queues_independent() {
    let context = match block_on(grafo::RendererContext::try_new()) {
        Ok(context) => context,
        Err(grafo::RendererCreationError::AdapterNotAvailable(_)) => {
            println!("Skipping test: no suitable GPU adapter available.");
            return;
        }
        Err(error) => panic!("Failed to create renderer context: {error}"),
    };

    let mut first = grafo::Renderer::try_new_headless_with_context(context.clone(), (16, 16), 1.0)
        .expect("to create first headless renderer");
    let mut second = grafo::Renderer::try_new_headless_with_context(context, (16, 16), 1.0)
        .expect("to create second headless renderer");

    first
        .texture_manager()
        .allocate_texture_with_data(42, (1, 1), &[255, 255, 255, 255]);
    assert!(second.texture_manager().is_texture_loaded(42));

    first.load_shape(
        grafo::Shape::rect([(0.0, 0.0), (16.0, 16.0)], grafo::Stroke::default()),
        99,
        Some(99),
    );
    second
        .add_cached_shape_to_the_render_queue(
            99,
            None,
            grafo::ShapeDrawCommandOptions::new().color(grafo::Color::rgb(0, 255, 0)),
        )
        .expect("to add shape loaded by first renderer");

    first
        .add_shape(
            grafo::Shape::rect([(0.0, 0.0), (16.0, 16.0)], grafo::Stroke::default()),
            None,
            None,
            grafo::ShapeDrawCommandOptions::new().color(grafo::Color::rgb(255, 0, 0)),
        )
        .expect("to add shape to first renderer");

    let mut first_pixels = Vec::new();
    let mut second_pixels = Vec::new();
    first.render_to_buffer(&mut first_pixels);
    second.render_to_buffer(&mut second_pixels);

    assert_eq!(read_pixel_rgba(&first_pixels, 16, 8, 8), [255, 0, 0, 255]);
    assert_eq!(read_pixel_rgba(&second_pixels, 16, 8, 8), [0, 255, 0, 255]);
}

/// Regression test — single root shape with no children should render correctly.
#[test]
fn single_root_no_children() {
    let Some(mut renderer) = create_headless_renderer() else {
        return;
    };

    let shape = grafo::Shape::rect([(10.0, 10.0), (100.0, 100.0)], grafo::Stroke::default());
    renderer
        .add_shape(
            shape,
            None,
            None,
            grafo::ShapeDrawCommandOptions::new().color(grafo::Color::rgb(200, 50, 50)),
        )
        .unwrap();

    let mut pixel_buffer: Vec<u8> = Vec::new();
    renderer.render_to_buffer(&mut pixel_buffer);

    let expectations = vec![
        grafo_test_scenes::PixelExpectation::opaque(55, 55, 200, 50, 50, "center_red"),
        grafo_test_scenes::PixelExpectation::transparent(5, 5, "outside_rect"),
    ];

    assert_pixels_match(&pixel_buffer, &expectations);
}

/// Regression test — OriginalSize texture fit uses physical pixels, not logical units.
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

    let shape = grafo::Shape::rect([(10.0, 10.0), (70.0, 70.0)], grafo::Stroke::default());
    renderer
        .add_shape(
            shape,
            None,
            None,
            grafo::ShapeDrawCommandOptions::new()
                .background_texture(
                    grafo::ShapeTextureOptions::new(green_texture_id)
                        .fit_mode(grafo::ShapeTextureFitMode::OriginalSize),
                )
                .color(grafo::Color::WHITE),
        )
        .unwrap();

    let mut pixel_buffer: Vec<u8> = Vec::new();
    renderer.render_to_buffer(&mut pixel_buffer);

    let expectations = vec![
        grafo_test_scenes::PixelExpectation::opaque(
            30,
            30,
            0,
            255,
            0,
            "inside_20px_physical_texture_region",
        ),
        grafo_test_scenes::PixelExpectation::opaque(
            60,
            30,
            255,
            255,
            255,
            "outside_texture_region_inside_shape",
        ),
        grafo_test_scenes::PixelExpectation::transparent(5, 5, "outside_shape"),
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

/// Regression test — scissor-only clipping rect clips children without drawing itself.
#[test]
fn clipping_rect_clips_child_without_visible_surface() {
    let Some(mut renderer) = create_headless_renderer() else {
        return;
    };

    let clip_rect_id = renderer
        .add_clipping_rect(
            [(20.0, 20.0), (80.0, 80.0)],
            None,
            None::<grafo::TransformInstance>,
            true,
        )
        .unwrap();
    let child = grafo::Shape::rect([(0.0, 0.0), (100.0, 100.0)], grafo::Stroke::default());
    renderer
        .add_shape(
            child,
            Some(clip_rect_id),
            None,
            grafo::ShapeDrawCommandOptions::new().color(grafo::Color::rgb(200, 50, 50)),
        )
        .unwrap();

    let mut pixel_buffer: Vec<u8> = Vec::new();
    renderer.render_to_buffer(&mut pixel_buffer);

    let expectations = vec![
        grafo_test_scenes::PixelExpectation::opaque(50, 50, 200, 50, 50, "inside_clip_rect"),
        grafo_test_scenes::PixelExpectation::transparent(10, 50, "left_of_clip_rect"),
        grafo_test_scenes::PixelExpectation::transparent(50, 10, "above_clip_rect"),
        grafo_test_scenes::PixelExpectation::transparent(90, 50, "right_of_clip_rect"),
        grafo_test_scenes::PixelExpectation::transparent(50, 90, "below_clip_rect"),
    ];

    assert_pixels_match(&pixel_buffer, &expectations);
}

/// Regression test — partially offscreen backdrop captures clear untouched pooled pixels.
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
    return 0.5 * (base + lookahead);
}
"#;

    renderer
        .load_effect(
            AVERAGE_WITH_RIGHT_NEIGHBOR_EFFECT_ID,
            &[AVERAGE_WITH_RIGHT_NEIGHBOR_WGSL],
        )
        .expect("Failed to compile deterministic backdrop test effect");

    let seeded_blue_panel =
        grafo::Shape::rect([(20.0, 20.0), (60.0, 60.0)], grafo::Stroke::default());
    renderer
        .add_shape(
            seeded_blue_panel.clone(),
            None,
            None,
            grafo::ShapeDrawCommandOptions::new().color(grafo::Color::rgb(40, 40, 220)),
        )
        .unwrap();
    let seeded_blue_panel_id = renderer
        .add_shape(
            seeded_blue_panel,
            None,
            None,
            grafo::ShapeDrawCommandOptions::new(),
        )
        .unwrap();
    renderer
        .set_shape_backdrop_effect(
            seeded_blue_panel_id,
            AVERAGE_WITH_RIGHT_NEIGHBOR_EFFECT_ID,
            &[],
            grafo::BackdropEffectConfig::new().capture_area(
                grafo::BackdropCaptureArea::ScreenRect([(20.0, 20.0), (60.0, 60.0)]),
            ),
        )
        .unwrap();

    let mut seeded_frame: Vec<u8> = Vec::new();
    renderer.render_to_buffer(&mut seeded_frame);

    renderer.clear_draw_queue();

    let visible_red_source =
        grafo::Shape::rect([(70.0, 20.0), (100.0, 60.0)], grafo::Stroke::default());
    renderer
        .add_shape(
            visible_red_source,
            None,
            None,
            grafo::ShapeDrawCommandOptions::new().color(grafo::Color::rgb(220, 40, 40)),
        )
        .unwrap();

    let partially_offscreen_panel =
        grafo::Shape::rect([(70.0, 20.0), (100.0, 60.0)], grafo::Stroke::default());
    let partially_offscreen_panel_id = renderer
        .add_shape(
            partially_offscreen_panel,
            None,
            None,
            grafo::ShapeDrawCommandOptions::new(),
        )
        .unwrap();
    renderer
        .set_shape_backdrop_effect(
            partially_offscreen_panel_id,
            AVERAGE_WITH_RIGHT_NEIGHBOR_EFFECT_ID,
            &[],
            grafo::BackdropEffectConfig::new().capture_area(
                grafo::BackdropCaptureArea::ScreenRect([(70.0, 20.0), (110.0, 60.0)]),
            ),
        )
        .unwrap();

    let mut pixel_buffer: Vec<u8> = Vec::new();
    renderer.render_to_buffer(&mut pixel_buffer);

    let sampled_pixel = read_pixel_rgba(&pixel_buffer, physical_size.0, 86, 40);
    assert!(
        sampled_pixel[0] > 80,
        "expected visible red contribution after clearing untouched capture space, got {:?}",
        sampled_pixel
    );
    assert!(
        sampled_pixel[2] <= 50,
        "expected offscreen capture space to stay transparent instead of leaking recycled blue, got {:?}",
        sampled_pixel
    );
    assert!(
        sampled_pixel[3] > 240,
        "expected the cleared backdrop sample to composite back over the visible red source, got {:?}",
        sampled_pixel
    );
}

/// Regression test — standalone clipping rect is a no-op and does not enter shape drawing.
#[test]
fn standalone_clipping_rect_does_not_panic() {
    let Some(mut renderer) = create_headless_renderer() else {
        return;
    };

    renderer
        .add_clipping_rect(
            [(20.0, 20.0), (80.0, 80.0)],
            None,
            None::<grafo::TransformInstance>,
            true,
        )
        .unwrap();

    let mut pixel_buffer: Vec<u8> = Vec::new();
    renderer.render_to_buffer(&mut pixel_buffer);

    assert!(
        pixel_buffer.iter().all(|&byte| byte == 0),
        "Standalone clipping rect should not draw any pixels",
    );
}

/// Regression test — unsupported clip-rect transforms are rejected instead of disabling clipping.
#[test]
fn clipping_rect_rejects_non_axis_aligned_transform() {
    let Some(mut renderer) = create_headless_renderer() else {
        return;
    };

    let clip_rect_id = renderer
        .add_clipping_rect(
            [(20.0, 20.0), (80.0, 80.0)],
            None,
            None::<grafo::TransformInstance>,
            true,
        )
        .unwrap();
    assert!(matches!(
        renderer.add_clipping_rect(
            [(20.0, 20.0), (80.0, 80.0)],
            None,
            Some(grafo::TransformInstance::rotation_z_deg(45.0)),
            true,
        ),
        Err(grafo::DrawCommandError::UnsupportedClipRectTransform)
    ));

    let child = grafo::Shape::rect([(0.0, 0.0), (100.0, 100.0)], grafo::Stroke::default());
    renderer
        .add_shape(
            child,
            Some(clip_rect_id),
            None,
            grafo::ShapeDrawCommandOptions::new().color(grafo::Color::rgb(200, 50, 50)),
        )
        .unwrap();

    let mut pixel_buffer: Vec<u8> = Vec::new();
    renderer.render_to_buffer(&mut pixel_buffer);

    let expectations = vec![
        grafo_test_scenes::PixelExpectation::opaque(
            50,
            50,
            200,
            50,
            50,
            "inside_unrotated_clip_rect",
        ),
        grafo_test_scenes::PixelExpectation::transparent(10, 50, "outside_unrotated_clip_rect"),
    ];

    assert_pixels_match(&pixel_buffer, &expectations);
}

/// Smoke test — gradient fill should produce non-transparent pixels.
#[test]
fn gradient_fill_basic() {
    use grafo::*;

    let Some(mut renderer) = create_headless_renderer() else {
        return;
    };

    // Root shape
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
    renderer.render_to_buffer(&mut pixel_buffer);

    // Canvas is CANVAS_WIDTH × CANVAS_HEIGHT
    let w = CANVAS_WIDTH;
    let center_x = 50u32;
    let center_y = 50u32;
    let offset = ((center_y * w + center_x) * 4) as usize;
    let b = pixel_buffer[offset];
    let g = pixel_buffer[offset + 1];
    let r = pixel_buffer[offset + 2];
    let a = pixel_buffer[offset + 3];
    // The center of a red-to-blue gradient should not be pure white
    assert!(
        !(r == 255 && g == 255 && b == 255),
        "Center pixel should not be white (got rgba({r},{g},{b},{a})). Gradient is not rendering."
    );
    // Should be opaque
    assert_eq!(a, 255, "Gradient pixel should be opaque");
}

/// Regression test — gradient bind groups must survive pipeline recreation
/// (e.g. MSAA sample count change) without producing validation errors or
/// rendering as white/transparent.
#[test]
fn gradient_survives_pipeline_recreation() {
    use grafo::*;

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

    // First render — populates and caches the gradient bind group.
    let mut buf = Vec::new();
    renderer.render_to_buffer(&mut buf);

    // Trigger pipeline recreation (swaps bind group layouts).
    renderer.set_msaa_samples(4);

    // Second render — stale bind groups must have been invalidated;
    // the gradient should render correctly against the new layout.
    buf.clear();
    renderer.render_to_buffer(&mut buf);

    let w = CANVAS_WIDTH;
    let cx = 50u32;
    let cy = 50u32;
    let off = ((cy * w + cx) * 4) as usize;
    let (b, g, r, a) = (buf[off], buf[off + 1], buf[off + 2], buf[off + 3]);

    assert_eq!(
        a, 255,
        "Gradient pixel should be opaque after pipeline recreation"
    );
    assert!(
        !(r == 255 && g == 255 && b == 255),
        "Gradient should not be white after pipeline recreation (got rgba({r},{g},{b},{a}))"
    );
    assert!(
        r < 200 && b < 200,
        "Center of red-to-blue gradient should be a purple-ish mix, got rgba({r},{g},{b},{a})"
    );
}

/// Regression test — a solid-colored non-leaf parent drawn immediately after a
/// gradient non-leaf parent on the same StencilIncrement pipeline must NOT
/// inherit the previous parent's gradient bind group.
///
/// We use rounded-rect parents so the renderer takes the stencil-increment path
/// instead of the scissor-optimization path (which only applies to axis-aligned
/// `Shape::Rect`).
///
/// Scene layout:
///
///   gradient_parent  (rounded rect, gradient fill, non-leaf)
///     └─ gradient_child
///   solid_parent     (rounded rect, green solid fill, non-leaf)
///     └─ solid_child
///
/// We check that the center of solid_child is green, not gradient-contaminated.
#[test]
fn stencil_increment_gradient_does_not_leak_to_solid_parent() {
    use grafo::*;

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

    // ── Gradient non-leaf parent (rounded rect → stencil path) ───────────
    let gradient_parent = renderer
        .add_shape(
            Shape::rounded_rect([(10.0, 10.0), (140.0, 90.0)], radii, Stroke::default()),
            Some(root),
            None,
            ShapeDrawCommandOptions::new().fill(Fill::from(gradient)),
        )
        .unwrap();

    // Child of gradient parent (makes it non-leaf → StencilIncrement).
    renderer
        .add_shape(
            Shape::rect([(20.0, 20.0), (130.0, 80.0)], Stroke::default()),
            Some(gradient_parent),
            None,
            ShapeDrawCommandOptions::new().color(Color::WHITE),
        )
        .unwrap();

    // ── Solid non-leaf parent (rounded rect → stencil path) ──────────────
    let solid_parent = renderer
        .add_shape(
            Shape::rounded_rect([(160.0, 10.0), (290.0, 90.0)], radii, Stroke::default()),
            Some(root),
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(0, 200, 0)),
        )
        .unwrap();

    // Child of solid parent (makes it non-leaf → StencilIncrement too).
    renderer
        .add_shape(
            Shape::rect([(170.0, 20.0), (280.0, 80.0)], Stroke::default()),
            Some(solid_parent),
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(0, 200, 0)),
        )
        .unwrap();

    // ── Render and verify ─────────────────────────────────────────────────
    let mut buf = Vec::new();
    renderer.render_to_buffer(&mut buf);

    // Sample the center of the solid_child rect.
    let w = CANVAS_WIDTH;
    let cx = 225u32; // midpoint of [170, 280]
    let cy = 50u32; // midpoint of [20, 80]
    let off = ((cy * w + cx) * 4) as usize;
    let (b, g, r, a) = (buf[off], buf[off + 1], buf[off + 2], buf[off + 3]);

    // Should be a solid green, not gradient-contaminated.
    assert_eq!(a, 255, "Solid child should be opaque, got alpha={a}");
    assert!(
        g >= 180 && r < 40 && b < 40,
        "Solid child should be green, got rgba({r},{g},{b},{a}). \
         If this is reddish/bluish the gradient leaked from the previous StencilIncrement parent."
    );
}

/// Regression test — touching triangle subpaths in one filled shape should not
/// show an internal AA seam along their shared diagonal.
#[test]
fn multi_subpath_fill_has_no_internal_seam() {
    let Some(mut renderer) = create_headless_renderer() else {
        return;
    };

    let canvas_root = grafo::Shape::rect(
        [(0.0, 0.0), (CANVAS_WIDTH as f32, CANVAS_HEIGHT as f32)],
        grafo::Stroke::default(),
    );
    let canvas_root_id = renderer
        .add_shape(
            canvas_root,
            None,
            None,
            grafo::ShapeDrawCommandOptions::new().color(grafo::Color::WHITE),
        )
        .unwrap();

    let shape = grafo::Shape::builder()
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
            grafo::ShapeDrawCommandOptions::new().color(grafo::Color::rgb(200, 50, 50)),
        )
        .unwrap();

    let rect = grafo::Shape::rect([(140.0, 10.0), (230.0, 100.0)], grafo::Stroke::default());
    renderer
        .add_shape(
            rect,
            Some(canvas_root_id),
            None,
            grafo::ShapeDrawCommandOptions::new().color(grafo::Color::rgb(200, 50, 50)),
        )
        .unwrap();

    let mut pixel_buffer: Vec<u8> = Vec::new();
    renderer.render_to_buffer(&mut pixel_buffer);

    let expectations = vec![
        grafo_test_scenes::PixelExpectation::opaque(30, 30, 200, 50, 50, "diag_top_left"),
        grafo_test_scenes::PixelExpectation::opaque(55, 55, 200, 50, 50, "diag_center"),
        grafo_test_scenes::PixelExpectation::opaque(80, 80, 200, 50, 50, "diag_bottom_right"),
        grafo_test_scenes::PixelExpectation::opaque(5, 5, 255, 255, 255, "outside_shape"),
        grafo_test_scenes::PixelExpectation::opaque(185, 55, 200, 50, 50, "rect_center"),
        grafo_test_scenes::PixelExpectation::opaque(145, 15, 200, 50, 50, "rect_near_corner"),
        grafo_test_scenes::PixelExpectation::opaque(235, 55, 255, 255, 255, "outside_rect"),
    ];

    assert_pixels_match(&pixel_buffer, &expectations);
}
