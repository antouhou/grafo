//! Measures shape loading and complete stroke replacement, drawing, and readback.
//! GPU setup and path construction happen before timing.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use futures::executor;
use grafo::{
    Color, LineCap, LineJoin, PixelFormat, Pixmap, Scene, Shape, ShapeDrawCommandOptions,
    StrokeOptions,
};
use std::{hint, time::Duration};

struct StrokeWorkload {
    name: &'static str,
    shape: Shape,
}

fn simple_line() -> Shape {
    Shape::builder()
        .begin((16.0, 32.0))
        .line_to((144.0, 32.0))
        .end()
        .build()
        .stroke(StrokeOptions::new(3.0))
}

fn curved_crossing() -> Shape {
    Shape::builder()
        .begin((32.0, 96.0))
        .cubic_bezier_to((64.0, 8.0), (176.0, 184.0), (224.0, 96.0))
        .cubic_bezier_to((176.0, 8.0), (64.0, 184.0), (32.0, 96.0))
        .cubic_bezier_to((8.0, 64.0), (80.0, 32.0), (128.0, 96.0))
        .cubic_bezier_to((176.0, 160.0), (208.0, 144.0), (240.0, 128.0))
        .end()
        .build()
        .stroke(
            StrokeOptions::new(5.0)
                .line_cap(LineCap::Round)
                .line_join(LineJoin::Round),
        )
}

fn dense_crossings() -> Shape {
    let mut builder = Shape::builder();
    for index in 0..32 {
        let position = 20.0 + index as f32 * 8.0;
        builder = builder
            .begin((16.0, position))
            .line_to((272.0, position))
            .end()
            .begin((position, 16.0))
            .line_to((position, 272.0))
            .end();
    }
    builder
        .build()
        .stroke(StrokeOptions::new(3.0).line_cap(LineCap::Round))
}

fn crossings_within_one_row() -> Shape {
    let mut builder = Shape::builder();
    for index in 0..64 {
        let start_y = 100.125 + index as f32 * 0.75 / 64.0;
        let end_y = 100.125 + ((index * 37) % 64) as f32 * 0.75 / 64.0;
        builder = builder.begin((16.0, start_y)).line_to((272.0, end_y)).end();
    }
    builder.build().stroke(StrokeOptions::new(0.004))
}

fn parallel_lines() -> Shape {
    let mut builder = Shape::builder();
    for index in 0..64 {
        let y = 20.0 + index as f32 * 4.0;
        builder = builder.begin((16.0, y)).line_to((272.0, y)).end();
    }
    builder
        .build()
        .stroke(StrokeOptions::new(3.0).line_cap(LineCap::Round))
}

fn benchmark_strokes(criterion: &mut Criterion) {
    let workloads = [
        StrokeWorkload {
            name: "simple_line",
            shape: simple_line(),
        },
        StrokeWorkload {
            name: "curved_crossing",
            shape: curved_crossing(),
        },
        StrokeWorkload {
            name: "dense_crossings_64_lines",
            shape: dense_crossings(),
        },
    ];
    let mut scene = executor::block_on(Scene::try_new((320, 240), 1.0, 1))
        .expect("stroke benchmark requires a GPU adapter for scene initialization");
    let mut group = criterion.benchmark_group("stroke_tessellation");
    group.sample_size(30);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));

    for (index, workload) in workloads.iter().enumerate() {
        // Replacement releases the previous buffers for a later load at this key.
        for _ in 0..3 {
            scene
                .load_shape(&workload.shape, 1, None)
                .expect("valid uncached stroke");
        }
        group.bench_function(BenchmarkId::new("uncached", workload.name), |bencher| {
            bencher.iter(|| {
                scene
                    .load_shape(hint::black_box(&workload.shape), 1, None)
                    .expect("valid uncached stroke");
            });
        });

        let geometry_id = index as u64 + 1;
        scene
            .load_shape(&workload.shape, 1, Some(geometry_id))
            .expect("valid cached stroke");
        group.bench_function(BenchmarkId::new("cached", workload.name), |bencher| {
            bencher.iter(|| {
                scene
                    .load_shape(hint::black_box(&workload.shape), 1, Some(geometry_id))
                    .expect("valid cached stroke");
            });
        });
    }
    group.finish();
}

fn benchmark_stroke_drawing(criterion: &mut Criterion) {
    let workloads = [
        StrokeWorkload {
            name: "simple_line",
            shape: simple_line(),
        },
        StrokeWorkload {
            name: "curved_crossing",
            shape: curved_crossing(),
        },
        StrokeWorkload {
            name: "dense_crossings_64_lines",
            shape: dense_crossings(),
        },
        StrokeWorkload {
            name: "crossings_within_one_row",
            shape: crossings_within_one_row(),
        },
        StrokeWorkload {
            name: "parallel_64_lines",
            shape: parallel_lines(),
        },
    ];
    let mut scene = executor::block_on(Scene::try_new((320, 320), 1.0, 1))
        .expect("stroke drawing benchmark requires a GPU adapter");
    let mut pixels = Pixmap::new((320, 320), PixelFormat::Bgra8).expect("valid size");
    let mut group = criterion.benchmark_group("stroke_drawing_with_readback");
    group.sample_size(30);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    for workload in workloads {
        scene.clear();
        let options = ShapeDrawCommandOptions::new().color(Color::rgba(0, 0, 0, 128));
        let shape_id = scene
            .add_shape(&workload.shape, None, None, options.clone())
            .expect("valid stroke");
        scene.render(&mut pixels).expect("initial draw");
        group.bench_function(workload.name, |bencher| {
            bencher.iter(|| {
                scene
                    .replace_with_shape(shape_id, &workload.shape, None, options.clone())
                    .expect("replace stroke");
                scene.render(&mut pixels).expect("draw stroke");
            });
        });
    }
    group.finish();
}

criterion_group!(
    stroke_tessellation_benches,
    benchmark_strokes,
    benchmark_stroke_drawing
);
criterion_main!(stroke_tessellation_benches);
