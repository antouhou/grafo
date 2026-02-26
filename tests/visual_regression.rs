/// Visual regression tests for the Grafo renderer.
///
/// These tests use the headless renderer to render scenes into a pixel buffer,
/// then validate specific pixel locations against expected colors.
///
/// Run with:   cargo test --test visual_regression
use futures::executor::block_on;
use grafo_test_scenes::{build_main_scene, check_pixels, CANVAS_HEIGHT, CANVAS_WIDTH};

/// Main regression test — renders all 34 tiles and validates pixel expectations.
#[test]
fn main_scene_pixel_expectations() {
    let mut renderer = block_on(grafo::Renderer::new_headless(
        (CANVAS_WIDTH, CANVAS_HEIGHT),
        1.0,
    ));

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
    let mut renderer = block_on(grafo::Renderer::new_headless(
        (CANVAS_WIDTH, CANVAS_HEIGHT),
        1.0,
    ));

    // Render with nothing in the draw queue
    let mut pixel_buffer: Vec<u8> = Vec::new();
    renderer.render_to_buffer(&mut pixel_buffer);

    // Should produce a fully transparent/clear buffer
    assert!(
        !pixel_buffer.is_empty(),
        "Pixel buffer should not be empty after rendering an empty scene",
    );
}

/// Regression test — single root shape with no children should render correctly.
#[test]
fn single_root_no_children() {
    let mut renderer = block_on(grafo::Renderer::new_headless(
        (CANVAS_WIDTH, CANVAS_HEIGHT),
        1.0,
    ));

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
