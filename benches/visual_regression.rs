//! Criterion benchmark for the visual-regression scene.
//!
//! Run with:
//! ```text
//! cargo bench --bench visual_regression
//! ```

use criterion::{criterion_group, criterion_main, Criterion};
use futures::executor::block_on;
use grafo::{PixelFormat, Pixmap, PixmapMut, Scene};
use grafo_test_scenes::{
    build_main_scene, check_pixels, PixelExpectation, CANVAS_HEIGHT, CANVAS_WIDTH,
};
use std::hint::black_box;

fn create_scene() -> Scene {
    block_on(Scene::try_new((CANVAS_WIDTH, CANVAS_HEIGHT), 1.0, 1))
        .expect("visual-regression benchmark requires a GPU adapter")
}

fn validate_scene(pixel_buffer: &[u8], expectations: &[PixelExpectation]) {
    let failures = check_pixels(pixel_buffer, CANVAS_WIDTH, CANVAS_HEIGHT, expectations);
    assert!(
        failures.is_empty(),
        "visual-regression benchmark rendered {} incorrect pixel expectation(s):\n{}",
        failures.len(),
        failures.join("\n"),
    );
}

fn benchmark_visual_regression_scene(criterion: &mut Criterion) {
    let mut scene = create_scene();
    let expectations = build_main_scene(&mut scene);
    let mut pixel_buffer = Pixmap::new(scene.size(), PixelFormat::Bgra8).unwrap();

    scene.render(&mut pixel_buffer).unwrap();
    validate_scene(pixel_buffer.pixels(), &expectations);

    criterion.bench_function("visual_regression/end_to_end_readback", |bencher| {
        bencher.iter(|| scene.render(black_box(&mut pixel_buffer)).unwrap());
    });

    let mut argb_pixels = vec![0; (CANVAS_WIDTH * CANVAS_HEIGHT) as usize];
    scene
        .render(PixmapMut::argb32(&mut argb_pixels, scene.size()).unwrap())
        .unwrap();
    let argb_bytes: Vec<u8> = argb_pixels
        .iter()
        .flat_map(|pixel| pixel.to_le_bytes())
        .collect();
    assert_eq!(argb_bytes, pixel_buffer.pixels());

    criterion.bench_function("visual_regression/argb_readback", |bencher| {
        bencher.iter(|| {
            scene
                .render(PixmapMut::argb32(black_box(&mut argb_pixels), scene.size()).unwrap())
                .unwrap()
        });
    });
}

criterion_group!(visual_regression_benches, benchmark_visual_regression_scene);
criterion_main!(visual_regression_benches);
