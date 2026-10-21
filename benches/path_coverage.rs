use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use futures::executor::block_on;
use grafo::{
    BorderRadii, Color, PixelFormat, Pixmap, Scene, Shape, ShapeDrawCommandOptions,
    TransformInstance,
};

#[derive(Clone, Copy)]
enum BenchmarkMode {
    Rebuilt,
    Moving,
    Resizing,
}

impl BenchmarkMode {
    fn name(self) -> &'static str {
        match self {
            Self::Rebuilt => "rebuilt",
            Self::Moving => "moving",
            Self::Resizing => "resizing",
        }
    }
}

struct Workload {
    name: &'static str,
    size: (u32, u32),
    count: usize,
    side: f32,
    radius: f32,
    should_use_distinct_geometry: bool,
}

fn background(size: (u32, u32)) -> Shape {
    Shape::rect([(0.0, 0.0), (size.0 as f32, size.1 as f32)])
}

fn populate(
    scene: &mut Scene,
    count: usize,
    translation_x: f32,
    should_use_distinct_geometry: bool,
) -> usize {
    let root = scene
        .add_shape(
            background(scene.size()),
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
        scene
            .add_cached_shape(
                if should_use_distinct_geometry {
                    index as u64 + 1
                } else {
                    1
                },
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
            should_use_distinct_geometry: false,
        },
        Workload {
            name: "circle1080",
            size: (1920, 1080),
            count: 1,
            side: 1078.0,
            radius: 539.0,
            should_use_distinct_geometry: false,
        },
        Workload {
            name: "rounded240",
            size: (1920, 1080),
            count: 240,
            side: 80.0,
            radius: 12.0,
            should_use_distinct_geometry: false,
        },
        Workload {
            name: "rounded240_distinct",
            size: (1920, 1080),
            count: 240,
            side: 80.0,
            radius: 5.0,
            should_use_distinct_geometry: true,
        },
    ];
    let mut group = criterion.benchmark_group("path_coverage_with_readback");
    group.sample_size(20);
    for workload in workloads {
        let mut scene = block_on(Scene::try_new(workload.size, 1.0, 1))
            .expect("path benchmark requires a GPU adapter");
        let geometry_count = if workload.should_use_distinct_geometry {
            workload.count
        } else {
            1
        };
        for index in 0..geometry_count {
            let key = index as u64 + 1;
            scene.load_shape(
                Shape::rounded_rect(
                    [(0.375, 0.625), (workload.side, workload.side)],
                    BorderRadii::new(workload.radius + index as f32 * 0.05),
                ),
                key,
                Some(key),
            );
        }
        let mut pixels = Pixmap::new(workload.size, PixelFormat::Bgra8).expect("valid size");
        let root = populate(
            &mut scene,
            workload.count,
            0.0,
            workload.should_use_distinct_geometry,
        );
        scene.render(&mut pixels).expect("initial render");
        group.bench_function(
            BenchmarkId::new(workload.name, "retained_redraw"),
            |bencher| {
                bencher.iter(|| {
                    scene
                        .replace_with_shape(
                            root,
                            background(workload.size),
                            None,
                            ShapeDrawCommandOptions::new()
                                .color(Color::BLACK)
                                .clips_children(false),
                        )
                        .expect("background replacement");
                    scene.render(&mut pixels).expect("redraw");
                });
            },
        );
        for mode in [
            BenchmarkMode::Rebuilt,
            BenchmarkMode::Moving,
            BenchmarkMode::Resizing,
        ] {
            let mut translation_x = 0.0;
            group.bench_function(BenchmarkId::new(workload.name, mode.name()), |bencher| {
                bencher.iter(|| {
                    if matches!(mode, BenchmarkMode::Moving | BenchmarkMode::Resizing) {
                        translation_x = (translation_x + 0.03125) % 1.0;
                    }
                    scene.clear();
                    if matches!(mode, BenchmarkMode::Resizing) {
                        for index in 0..geometry_count {
                            scene.load_shape(
                                Shape::rounded_rect(
                                    [
                                        (0.375, 0.625),
                                        (workload.side + translation_x, workload.side),
                                    ],
                                    BorderRadii::new(
                                        workload.radius
                                            + index as f32 * 0.05
                                            + translation_x * 0.25,
                                    ),
                                ),
                                index as u64 + 1,
                                None,
                            );
                        }
                    }
                    populate(
                        &mut scene,
                        workload.count,
                        translation_x,
                        workload.should_use_distinct_geometry,
                    );
                    scene.render(&mut pixels).expect("rebuilt scene");
                });
            });
        }
    }
    group.finish();
}

criterion_group!(path_coverage_benches, benchmark_paths);
criterion_main!(path_coverage_benches);
