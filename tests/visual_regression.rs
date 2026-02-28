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
    let mut renderer = match block_on(grafo::Renderer::try_new_headless(
        (CANVAS_WIDTH, CANVAS_HEIGHT),
        1.0,
    )) {
        Ok(r) => r,
        Err(grafo::RendererCreationError::AdapterNotAvailable(_)) => {
            println!("Skipping test: no suitable GPU adapter available.");
            return;
        }
        Err(e) => panic!("Failed to create headless renderer: {e}"),
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
    let mut renderer = match block_on(grafo::Renderer::try_new_headless(
        (CANVAS_WIDTH, CANVAS_HEIGHT),
        1.0,
    )) {
        Ok(r) => r,
        Err(grafo::RendererCreationError::AdapterNotAvailable(_)) => {
            println!("Skipping test: no suitable GPU adapter available.");
            return;
        }
        Err(e) => panic!("Failed to create headless renderer: {e}"),
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
    let mut renderer = match block_on(grafo::Renderer::try_new_headless(
        (CANVAS_WIDTH, CANVAS_HEIGHT),
        1.0,
    )) {
        Ok(r) => r,
        Err(grafo::RendererCreationError::AdapterNotAvailable(_)) => {
            println!("Skipping test: no suitable GPU adapter available.");
            return;
        }
        Err(e) => panic!("Failed to create headless renderer: {e}"),
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
