use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use futures::executor::block_on;
use grafo::{
    BorderRadii, Color, PixelFormat, Pixmap, Renderer, Shape, ShapeDrawCommandOptions,
    TransformInstance,
};

struct Workload {
    name: &'static str,
    size: (u32, u32),
    count: usize,
    side: f32,
    radius: f32,
}

fn background(size: (u32, u32)) -> Shape {
    Shape::rect([(0.0, 0.0), (size.0 as f32, size.1 as f32)])
}

fn populate(renderer: &mut Renderer, count: usize, translation_x: f32) -> usize {
    let root = renderer
        .add_shape(
            background(renderer.size()),
            None,
            None,
            ShapeDrawCommandOptions::new()
                .color(Color::BLACK)
                .clips_children(false),
        )
        .expect("valid background");
    for index in 0..count {
        let (x, y) = if count == 1 {
            (translation_x, 0.0)
        } else {
            (
                ((index % 20) * 90) as f32 + translation_x,
                ((index / 20) * 90) as f32,
            )
        };
        renderer
            .add_cached_shape(
                1,
                None,
                ShapeDrawCommandOptions::new()
                    .color(Color::rgb(210, 160, 80))
                    .transform(TransformInstance::translation(x, y)),
            )
            .expect("loaded path");
    }
    root
}

fn benchmark_paths(criterion: &mut Criterion) {
    let workloads = [
        Workload {
            name: "circle512",
            size: (512, 512),
            count: 1,
            side: 510.0,
            radius: 255.0,
        },
        Workload {
            name: "circle1080",
            size: (1920, 1080),
            count: 1,
            side: 1078.0,
            radius: 539.0,
        },
        Workload {
            name: "rounded240",
            size: (1920, 1080),
            count: 240,
            side: 80.0,
            radius: 12.0,
        },
    ];
    let mut group = criterion.benchmark_group("path_coverage_with_readback");
    group.sample_size(20);
    for workload in workloads {
        let mut renderer = block_on(Renderer::try_new(workload.size, 1.0, 1))
            .expect("path benchmark requires a GPU adapter");
        renderer.load_shape(
            Shape::rounded_rect(
                [(0.375, 0.625), (workload.side, workload.side)],
                BorderRadii::new(workload.radius),
            ),
            1,
            Some(1),
        );
        let mut pixels = Pixmap::new(workload.size, PixelFormat::Bgra8).expect("valid size");
        let root = populate(&mut renderer, workload.count, 0.0);
        renderer.render(&mut pixels).expect("initial render");
        group.bench_function(
            BenchmarkId::new(workload.name, "retained_redraw"),
            |bencher| {
                bencher.iter(|| {
                    renderer
                        .replace_with_shape(
                            root,
                            background(workload.size),
                            None,
                            ShapeDrawCommandOptions::new()
                                .color(Color::BLACK)
                                .clips_children(false),
                        )
                        .expect("background replacement");
                    renderer.render(&mut pixels).expect("redraw");
                });
            },
        );
        for should_move in [false, true] {
            let mut translation_x = 0.0;
            let mode = if should_move { "moving" } else { "rebuilt" };
            group.bench_function(BenchmarkId::new(workload.name, mode), |bencher| {
                bencher.iter(|| {
                    if should_move {
                        translation_x = (translation_x + 0.03125) % 1.0;
                    }
                    renderer.clear_draw_queue();
                    populate(&mut renderer, workload.count, translation_x);
                    renderer.render(&mut pixels).expect("rebuilt scene");
                });
            });
        }
    }
    group.finish();
}

criterion_group!(path_coverage_benches, benchmark_paths);
criterion_main!(path_coverage_benches);
