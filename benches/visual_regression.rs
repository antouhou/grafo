//! Criterion benchmark for the visual-regression scene.
//!
//! Run with:
//! ```text
//! cargo bench --bench visual_regression
//! ```

use criterion::{criterion_group, criterion_main, Criterion};
use futures::executor::block_on;
use grafo_test_scenes::{build_main_scene, check_pixels, CANVAS_HEIGHT, CANVAS_WIDTH};
use std::hint::black_box;

fn create_renderer() -> grafo::Renderer<'static> {
    block_on(grafo::Renderer::try_new_headless(
        (CANVAS_WIDTH, CANVAS_HEIGHT),
        1.0,
    ))
    .expect("visual-regression benchmark requires a GPU adapter")
}

fn validate_scene(pixel_buffer: &[u8], expectations: &[grafo_test_scenes::PixelExpectation]) {
    let failures = check_pixels(pixel_buffer, CANVAS_WIDTH, CANVAS_HEIGHT, expectations);
    assert!(
        failures.is_empty(),
        "visual-regression benchmark rendered {} incorrect pixel expectation(s):\n{}",
        failures.len(),
        failures.join("\n"),
    );
}

fn benchmark_visual_regression_scene(criterion: &mut Criterion) {
    let mut renderer = create_renderer();
    let expectations = build_main_scene(&mut renderer);
    let mut pixel_buffer = Vec::new();

    renderer.render_to_buffer(&mut pixel_buffer);
    validate_scene(&pixel_buffer, &expectations);

    criterion.bench_function("visual_regression/end_to_end_readback", |bencher| {
        bencher.iter(|| renderer.render_to_buffer(black_box(&mut pixel_buffer)));
    });
}

criterion_group!(visual_regression_benches, benchmark_visual_regression_scene);
criterion_main!(visual_regression_benches);
