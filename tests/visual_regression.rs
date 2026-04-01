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

/// Main regression test — renders all 38 tiles and validates pixel expectations.
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
    let root_id = renderer.add_shape(root, None, None);
    renderer.set_shape_color(root_id, Some(Color::WHITE));

    // Gradient child
    let child = Shape::rect([(10.0, 10.0), (90.0, 90.0)], Stroke::default());
    let child_id = renderer.add_shape(child, Some(root_id), None);

    let gradient = Gradient::new(GradientDesc::Linear(LinearGradientDesc {
        common: GradientCommonDesc {
            units: GradientUnits::Local,
            spread: SpreadMode::Pad,
            interpolation: ColorInterpolation::Srgb,
            stops: vec![
                GradientStop {
                    positions: GradientStopPositions::Single(GradientStopOffset::LinearRadial(0.0)),
                    color: GradientColor::Srgb {
                        red: 1.0,
                        green: 0.0,
                        blue: 0.0,
                        alpha: 1.0,
                    },
                    hint_to_next_segment: None,
                },
                GradientStop {
                    positions: GradientStopPositions::Single(GradientStopOffset::LinearRadial(1.0)),
                    color: GradientColor::Srgb {
                        red: 0.0,
                        green: 0.0,
                        blue: 1.0,
                        alpha: 1.0,
                    },
                    hint_to_next_segment: None,
                },
            ]
            .into(),
        },
        line: LinearGradientLine {
            start: [10.0, 50.0],
            end: [90.0, 50.0],
        },
    }))
    .expect("valid gradient");

    renderer.set_shape_fill(child_id, Some(Fill::Gradient(gradient)));

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

    // Set up a gradient shape.
    let shape = Shape::rect([(10.0, 10.0), (90.0, 90.0)], Stroke::default());
    let id = renderer.add_shape(shape, None, None);

    let gradient = Gradient::new(GradientDesc::Linear(LinearGradientDesc {
        common: GradientCommonDesc {
            units: GradientUnits::Local,
            spread: SpreadMode::Pad,
            interpolation: ColorInterpolation::Srgb,
            stops: vec![
                GradientStop {
                    positions: GradientStopPositions::Single(GradientStopOffset::LinearRadial(0.0)),
                    color: GradientColor::Srgb {
                        red: 1.0,
                        green: 0.0,
                        blue: 0.0,
                        alpha: 1.0,
                    },
                    hint_to_next_segment: None,
                },
                GradientStop {
                    positions: GradientStopPositions::Single(GradientStopOffset::LinearRadial(1.0)),
                    color: GradientColor::Srgb {
                        red: 0.0,
                        green: 0.0,
                        blue: 1.0,
                        alpha: 1.0,
                    },
                    hint_to_next_segment: None,
                },
            ]
            .into(),
        },
        line: LinearGradientLine {
            start: [10.0, 50.0],
            end: [90.0, 50.0],
        },
    }))
    .expect("valid gradient");

    renderer.set_shape_fill(id, Some(Fill::Gradient(gradient)));

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
    let root = renderer.add_shape(
        Shape::rect(
            [(0.0, 0.0), (CANVAS_WIDTH as f32, CANVAS_HEIGHT as f32)],
            Stroke::default(),
        ),
        None,
        None,
    );
    renderer.set_shape_color(root, Some(Color::rgba(0, 0, 0, 0)));

    let radii = BorderRadii::new(8.0);

    // ── Gradient non-leaf parent (rounded rect → stencil path) ───────────
    let gradient_parent = renderer.add_shape(
        Shape::rounded_rect([(10.0, 10.0), (140.0, 90.0)], radii, Stroke::default()),
        Some(root),
        None,
    );

    let gradient = Gradient::new(GradientDesc::Linear(LinearGradientDesc {
        common: GradientCommonDesc {
            units: GradientUnits::Local,
            spread: SpreadMode::Pad,
            interpolation: ColorInterpolation::Srgb,
            stops: vec![
                GradientStop {
                    positions: GradientStopPositions::Single(GradientStopOffset::LinearRadial(0.0)),
                    color: GradientColor::Srgb {
                        red: 1.0,
                        green: 0.0,
                        blue: 0.0,
                        alpha: 1.0,
                    },
                    hint_to_next_segment: None,
                },
                GradientStop {
                    positions: GradientStopPositions::Single(GradientStopOffset::LinearRadial(1.0)),
                    color: GradientColor::Srgb {
                        red: 0.0,
                        green: 0.0,
                        blue: 1.0,
                        alpha: 1.0,
                    },
                    hint_to_next_segment: None,
                },
            ]
            .into(),
        },
        line: LinearGradientLine {
            start: [10.0, 50.0],
            end: [140.0, 50.0],
        },
    }))
    .expect("valid gradient");

    renderer.set_shape_fill(gradient_parent, Some(Fill::Gradient(gradient)));

    // Child of gradient parent (makes it non-leaf → StencilIncrement).
    let gradient_child = renderer.add_shape(
        Shape::rect([(20.0, 20.0), (130.0, 80.0)], Stroke::default()),
        Some(gradient_parent),
        None,
    );
    renderer.set_shape_color(gradient_child, Some(Color::WHITE));

    // ── Solid non-leaf parent (rounded rect → stencil path) ──────────────
    let solid_parent = renderer.add_shape(
        Shape::rounded_rect([(160.0, 10.0), (290.0, 90.0)], radii, Stroke::default()),
        Some(root),
        None,
    );
    renderer.set_shape_color(solid_parent, Some(Color::rgb(0, 200, 0)));

    // Child of solid parent (makes it non-leaf → StencilIncrement too).
    let solid_child = renderer.add_shape(
        Shape::rect([(170.0, 20.0), (280.0, 80.0)], Stroke::default()),
        Some(solid_parent),
        None,
    );
    renderer.set_shape_color(solid_child, Some(Color::rgb(0, 200, 0)));

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
    let canvas_root_id = renderer.add_shape(canvas_root, None, None);
    renderer.set_shape_color(canvas_root_id, Some(grafo::Color::WHITE));

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
    let id = renderer.add_shape(shape, Some(canvas_root_id), None);
    renderer.set_shape_color(id, Some(grafo::Color::rgb(200, 50, 50)));

    let rect = grafo::Shape::rect([(140.0, 10.0), (230.0, 100.0)], grafo::Stroke::default());
    let rect_id = renderer.add_shape(rect, Some(canvas_root_id), None);
    renderer.set_shape_color(rect_id, Some(grafo::Color::rgb(200, 50, 50)));

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
