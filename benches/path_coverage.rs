use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use futures::executor::block_on;
use grafo::{
    BorderRadii, Color, PixelFormat, Pixmap, Renderer, Shape, ShapeDrawCommandOptions,
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
    is_general_path: bool,
}

impl Workload {
    fn shape(&self, index: usize, offset: f32) -> Shape {
        let side = self.side + offset;
        if self.is_general_path {
            return Shape::builder()
                .begin((3.375 + index as f32 * 0.05, 0.625))
                .quadratic_bezier_to((side * 0.4, side * 0.55), (side - 0.125, 2.375))
                .line_to((side * 0.65, side * 0.6))
                .line_to((side - 0.375, self.side - 0.625))
                .quadratic_bezier_to((side * 0.45, self.side * 0.75), (2.625, self.side - 0.125))
                .line_to((side * 0.2, self.side * 0.4))
                .close()
                .build();
        }
        Shape::rounded_rect(
            [(0.375, 0.625), (side, self.side)],
            BorderRadii::new(self.radius + index as f32 * 0.05 + offset * 0.25),
        )
    }
}

fn background(size: (u32, u32)) -> Shape {
    Shape::rect([(0.0, 0.0), (size.0 as f32, size.1 as f32)])
}

fn populate(
    renderer: &mut Renderer,
    count: usize,
    translation_x: f32,
    should_use_distinct_geometry: bool,
) -> usize {
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
            is_general_path: false,
        },
        Workload {
            name: "circle1080",
            size: (1920, 1080),
            count: 1,
            side: 1078.0,
            radius: 539.0,
            should_use_distinct_geometry: false,
            is_general_path: false,
        },
        Workload {
            name: "rounded240",
            size: (1920, 1080),
            count: 240,
            side: 80.0,
            radius: 12.0,
            should_use_distinct_geometry: false,
            is_general_path: false,
        },
        Workload {
            name: "rounded240_distinct",
            size: (1920, 1080),
            count: 240,
            side: 80.0,
            radius: 5.0,
            should_use_distinct_geometry: true,
            is_general_path: false,
        },
        Workload {
            name: "general240_distinct",
            size: (1920, 1080),
            count: 240,
            side: 80.0,
            radius: 0.0,
            should_use_distinct_geometry: true,
            is_general_path: true,
        },
    ];
    let mut group = criterion.benchmark_group("path_coverage_with_readback");
    group.sample_size(20);
    for workload in workloads {
        let mut renderer = block_on(Renderer::try_new(workload.size, 1.0, 1))
            .expect("path benchmark requires a GPU adapter");
        let geometry_count = if workload.should_use_distinct_geometry {
            workload.count
        } else {
            1
        };
        for index in 0..geometry_count {
            let key = index as u64 + 1;
            renderer.load_shape(workload.shape(index, 0.0), key, Some(key));
        }
        let mut pixels = Pixmap::new(workload.size, PixelFormat::Bgra8).expect("valid size");
        let root = populate(
            &mut renderer,
            workload.count,
            0.0,
            workload.should_use_distinct_geometry,
        );
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
                    renderer.clear_draw_queue();
                    if matches!(mode, BenchmarkMode::Resizing) {
                        for index in 0..geometry_count {
                            renderer.load_shape(
                                workload.shape(index, translation_x),
                                index as u64 + 1,
                                None,
                            );
                        }
                    }
                    populate(
                        &mut renderer,
                        workload.count,
                        translation_x,
                        workload.should_use_distinct_geometry,
                    );
                    renderer.render(&mut pixels).expect("rebuilt scene");
                });
            });
        }
    }
    group.finish();
}

criterion_group!(path_coverage_benches, benchmark_paths);
criterion_main!(path_coverage_benches);
