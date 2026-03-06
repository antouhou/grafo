/// Visual regression tests for the Grafo renderer.
///
/// These tests use the headless renderer to render scenes into a pixel buffer,
/// then validate specific pixel locations against expected colors.
///
/// Run with:   cargo test --test visual_regression
use futures::executor::block_on;
use grafo_test_scenes::{
    build_main_scene, check_pixels, PixelExpectation, CANVAS_HEIGHT, CANVAS_WIDTH,
};

const SOLID_RED_EFFECT: &str = r#"
@fragment
fn effect_main(@location(0) _uv: vec2<f32>) -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}
"#;

/// Creates a headless renderer, returning `None` (and printing a skip message)
/// when no suitable GPU adapter is available.
fn create_headless_renderer() -> Option<grafo::Renderer<'static>> {
    match block_on(grafo::Renderer::try_new_headless(
        (CANVAS_WIDTH, CANVAS_HEIGHT),
        1.0,
    )) {
        Ok(r) => Some(r),
        Err(grafo::RendererCreationError::AdapterNotAvailable(_)) => {
            println!("Skipping test: no suitable GPU adapter available.");
            None
        }
        Err(e) => panic!("Failed to create headless renderer: {e}"),
    }
}

fn assert_pixels_match(renderer: &mut grafo::Renderer<'static>, expectations: &[PixelExpectation]) {
    let mut pixel_buffer: Vec<u8> = Vec::new();
    renderer.render_to_buffer(&mut pixel_buffer);

    let failures = check_pixels(&pixel_buffer, CANVAS_WIDTH, CANVAS_HEIGHT, expectations);
    if !failures.is_empty() {
        let message = format!(
            "{} pixel expectation(s) failed:\n{}",
            failures.len(),
            failures.join("\n"),
        );
        panic!("{message}");
    }
}

/// Main regression test — renders all 34 tiles and validates pixel expectations.
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

/// Regression test — single root shape with no children should render correctly.
#[test]
fn single_root_no_children() {
    let Some(mut renderer) = create_headless_renderer() else {
        return;
    };

    let shape = grafo::Shape::rect([(10.0, 10.0), (100.0, 100.0)], grafo::Stroke::default());
    let id = renderer.add_shape(shape, None, None);
    renderer.set_shape_color(id, Some(grafo::Color::rgb(200, 50, 50)));

    let mut pixel_buffer: Vec<u8> = Vec::new();
    renderer.render_to_buffer(&mut pixel_buffer);

    let expectations = vec![
        grafo_test_scenes::PixelExpectation::opaque(55, 55, 200, 50, 50, "center_red"),
        grafo_test_scenes::PixelExpectation::transparent(5, 5, "outside_rect"),
    ];

    let failures = check_pixels(&pixel_buffer, CANVAS_WIDTH, CANVAS_HEIGHT, &expectations);
    if !failures.is_empty() {
        panic!(
            "{} pixel expectation(s) failed:\n{}",
            failures.len(),
            failures.join("\n"),
        );
    }
}

#[test]
fn transparent_scissor_parent_skips_self_but_still_clips_and_restores_state() {
    let Some(mut renderer) = create_headless_renderer() else {
        return;
    };

    let root = renderer.add_shape(
        grafo::Shape::rect(
            [(0.0, 0.0), (CANVAS_WIDTH as f32, CANVAS_HEIGHT as f32)],
            grafo::Stroke::default(),
        ),
        None,
        None,
    );
    renderer.set_shape_color(root, Some(grafo::Color::rgb(10, 10, 10)));

    let clip_parent = renderer.add_shape(
        grafo::Shape::rect([(10.0, 10.0), (70.0, 70.0)], grafo::Stroke::default()),
        Some(root),
        None,
    );
    renderer.set_shape_color(clip_parent, Some(grafo::Color::rgba(255, 255, 255, 0)));

    let clipped_child = renderer.add_shape(
        grafo::Shape::rect([(0.0, 0.0), (90.0, 90.0)], grafo::Stroke::default()),
        Some(clip_parent),
        None,
    );
    renderer.set_shape_color(clipped_child, Some(grafo::Color::rgb(220, 40, 40)));

    let sibling = renderer.add_shape(
        grafo::Shape::rect([(85.0, 15.0), (115.0, 45.0)], grafo::Stroke::default()),
        Some(root),
        None,
    );
    renderer.set_shape_color(sibling, Some(grafo::Color::rgb(40, 220, 40)));

    let expectations = [
        PixelExpectation::opaque(20, 20, 220, 40, 40, "clipped_child_inside"),
        PixelExpectation::opaque(80, 20, 10, 10, 10, "clip_parent_self_skipped"),
        PixelExpectation::opaque(95, 25, 40, 220, 40, "scissor_restored_for_sibling"),
    ];
    assert_pixels_match(&mut renderer, &expectations);
}

#[test]
fn transparent_trivial_rect_with_texture_still_draws() {
    let Some(mut renderer) = create_headless_renderer() else {
        return;
    };

    let texture_id = 7;
    renderer
        .texture_manager()
        .allocate_texture_with_data(texture_id, (1, 1), &[0, 255, 0, 255]);

    let rect = renderer.add_shape(
        grafo::Shape::rect([(10.0, 10.0), (50.0, 50.0)], grafo::Stroke::default()),
        None,
        None,
    );
    renderer.set_shape_color(rect, Some(grafo::Color::rgba(255, 255, 255, 0)));
    renderer.set_shape_texture(rect, Some(texture_id));

    let expectations = [
        PixelExpectation::opaque(25, 25, 0, 255, 0, "textured_rect_kept"),
        PixelExpectation::transparent(5, 5, "outside"),
    ];
    assert_pixels_match(&mut renderer, &expectations);
}

#[test]
fn transparent_rect_with_direct_group_effect_still_renders_effect_output() {
    let Some(mut renderer) = create_headless_renderer() else {
        return;
    };

    renderer
        .load_effect(1, &[SOLID_RED_EFFECT])
        .expect("effect shader should compile");

    let rect = renderer.add_shape(
        grafo::Shape::rect([(10.0, 10.0), (60.0, 60.0)], grafo::Stroke::default()),
        None,
        None,
    );
    renderer.set_shape_color(rect, Some(grafo::Color::rgba(255, 255, 255, 0)));
    renderer
        .set_group_effect(rect, 1, &[])
        .expect("group effect attachment should succeed");

    let expectations = [PixelExpectation::opaque(
        25,
        25,
        255,
        0,
        0,
        "group_effect_output",
    )];
    assert_pixels_match(&mut renderer, &expectations);
}

#[test]
fn transparent_rect_with_direct_backdrop_effect_still_renders_effect_output() {
    let Some(mut renderer) = create_headless_renderer() else {
        return;
    };

    renderer
        .load_effect(2, &[SOLID_RED_EFFECT])
        .expect("effect shader should compile");

    let root = renderer.add_shape(
        grafo::Shape::rect(
            [(0.0, 0.0), (CANVAS_WIDTH as f32, CANVAS_HEIGHT as f32)],
            grafo::Stroke::default(),
        ),
        None,
        None,
    );
    renderer.set_shape_color(root, Some(grafo::Color::rgb(30, 60, 200)));

    let rect = renderer.add_shape(
        grafo::Shape::rect([(15.0, 15.0), (70.0, 70.0)], grafo::Stroke::default()),
        Some(root),
        None,
    );
    renderer.set_shape_color(rect, Some(grafo::Color::rgba(255, 255, 255, 0)));
    renderer
        .set_shape_backdrop_effect(rect, 2, &[])
        .expect("backdrop effect attachment should succeed");

    let expectations = [
        PixelExpectation::opaque(30, 30, 255, 0, 0, "backdrop_effect_output"),
        PixelExpectation::opaque(90, 90, 30, 60, 200, "background_preserved"),
    ];
    assert_pixels_match(&mut renderer, &expectations);
}

#[test]
fn rotated_transparent_rect_keeps_stencil_fallback_behavior() {
    let Some(mut renderer) = create_headless_renderer() else {
        return;
    };

    let root = renderer.add_shape(
        grafo::Shape::rect(
            [(0.0, 0.0), (CANVAS_WIDTH as f32, CANVAS_HEIGHT as f32)],
            grafo::Stroke::default(),
        ),
        None,
        None,
    );
    renderer.set_shape_color(root, Some(grafo::Color::rgb(10, 10, 10)));

    let clip_parent = renderer.add_shape(
        grafo::Shape::rect([(-20.0, -20.0), (20.0, 20.0)], grafo::Stroke::default()),
        Some(root),
        None,
    );
    renderer.set_shape_color(clip_parent, Some(grafo::Color::rgba(255, 255, 255, 0)));
    renderer.set_shape_transform(
        clip_parent,
        grafo::TransformInstance::rotation_z_deg(45.0)
            .then(&grafo::TransformInstance::translation(80.0, 80.0)),
    );

    let clipped_child = renderer.add_shape(
        grafo::Shape::rect([(40.0, 40.0), (120.0, 120.0)], grafo::Stroke::default()),
        Some(clip_parent),
        None,
    );
    renderer.set_shape_color(clipped_child, Some(grafo::Color::rgb(220, 40, 40)));

    let expectations = [
        PixelExpectation::opaque(80, 80, 220, 40, 40, "inside_rotated_clip"),
        PixelExpectation::opaque(115, 80, 10, 10, 10, "outside_rotated_clip"),
    ];
    assert_pixels_match(&mut renderer, &expectations);
}
