use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use futures::executor::block_on;
use grafo::{Color, Renderer, RendererCreationError, Shape, Stroke, TransformInstance};

const CANVAS_WIDTH: u32 = 1024;
const CANVAS_HEIGHT: u32 = 1024;
const SCALE_FACTOR: f64 = 1.0;

const ROOT_CACHE_KEY: u64 = 1;
const LEAF_CACHE_KEY: u64 = 2;

const TOTAL_LEAF_COUNT: usize = 1024;
const MUTABLE_LEAF_COUNT: usize = TOTAL_LEAF_COUNT / 2;
const LEAF_COLUMNS_PER_HALF: usize = 16;
const SHALLOW_SUBTREE_COUNT: usize = 100;
const SHALLOW_SUBTREE_COLUMNS: usize = 10;
const SHALLOW_SUBTREE_CHILDREN_PER_ROW: usize = 4;

const ROOT_PADDING: f32 = 24.0;
const LEAF_SIZE: f32 = 24.0;
const LEAF_SPACING: f32 = 28.0;
const MUTABLE_VARIANT_OFFSET: f32 = 6.0;
const SHALLOW_SUBTREE_SPACING_X: f32 = 72.0;
const SHALLOW_SUBTREE_SPACING_Y: f32 = 72.0;

struct SceneBuildResult {
    root_node_id: usize,
    mutable_subtree_root_id: usize,
}

struct FullRebuildBenchmarkState {
    renderer: Renderer<'static>,
    mutable_variant: bool,
}

impl FullRebuildBenchmarkState {
    fn new() -> Option<Self> {
        let mut renderer = try_create_headless_renderer()?;
        load_benchmark_shapes(&mut renderer);

        Some(Self {
            renderer,
            mutable_variant: false,
        })
    }

    fn render_next_frame(&mut self) {
        self.renderer.clear_draw_queue();
        build_benchmark_scene(&mut self.renderer, self.mutable_variant);
        self.renderer.render_headless_frame();
        self.mutable_variant = !self.mutable_variant;
    }

    fn mutate_next_frame(&mut self) {
        self.renderer.clear_draw_queue();
        build_benchmark_scene(&mut self.renderer, self.mutable_variant);
        self.mutable_variant = !self.mutable_variant;
    }
}

struct PartialUpdateBenchmarkState {
    renderer: Renderer<'static>,
    root_node_id: usize,
    mutable_subtree_root_id: usize,
    mutable_variant: bool,
}

impl PartialUpdateBenchmarkState {
    fn new() -> Option<Self> {
        let mut renderer = try_create_headless_renderer()?;
        load_benchmark_shapes(&mut renderer);
        let scene = build_benchmark_scene(&mut renderer, false);

        Some(Self {
            renderer,
            root_node_id: scene.root_node_id,
            mutable_subtree_root_id: scene.mutable_subtree_root_id,
            mutable_variant: false,
        })
    }

    fn render_next_frame(&mut self) {
        self.renderer.remove_subtree(self.mutable_subtree_root_id);

        let next_mutable_variant = !self.mutable_variant;
        self.mutable_subtree_root_id =
            add_mutable_leaf_subtree(&mut self.renderer, self.root_node_id, next_mutable_variant);
        self.mutable_variant = next_mutable_variant;

        self.renderer.render_headless_frame();
    }

    fn mutate_next_frame(&mut self) {
        self.renderer.remove_subtree(self.mutable_subtree_root_id);

        let next_mutable_variant = !self.mutable_variant;
        self.mutable_subtree_root_id =
            add_mutable_leaf_subtree(&mut self.renderer, self.root_node_id, next_mutable_variant);
        self.mutable_variant = next_mutable_variant;
    }
}

struct ManyShallowPartialUpdateBenchmarkState {
    renderer: Renderer<'static>,
    root_node_id: usize,
    mutable_subtree_root_ids: Vec<usize>,
    replacement_subtree_root_ids: Vec<usize>,
    mutable_variant: bool,
}

impl ManyShallowPartialUpdateBenchmarkState {
    fn new() -> Option<Self> {
        let mut renderer = try_create_headless_renderer()?;
        load_benchmark_shapes(&mut renderer);

        let root_node_id = renderer.add_cached_shape_to_the_render_queue(ROOT_CACHE_KEY, None);
        renderer.set_shape_color(root_node_id, Some(Color::TRANSPARENT));
        add_stable_leaf_nodes(&mut renderer, root_node_id);

        let mut mutable_subtree_root_ids = Vec::with_capacity(SHALLOW_SUBTREE_COUNT);
        add_many_shallow_mutable_subtrees(
            &mut renderer,
            root_node_id,
            false,
            &mut mutable_subtree_root_ids,
        );

        Some(Self {
            renderer,
            root_node_id,
            mutable_subtree_root_ids,
            replacement_subtree_root_ids: Vec::with_capacity(SHALLOW_SUBTREE_COUNT),
            mutable_variant: false,
        })
    }

    fn render_next_frame(&mut self) {
        for &subtree_root_id in &self.mutable_subtree_root_ids {
            self.renderer.remove_subtree(subtree_root_id);
        }

        let next_mutable_variant = !self.mutable_variant;
        self.replacement_subtree_root_ids.clear();
        add_many_shallow_mutable_subtrees(
            &mut self.renderer,
            self.root_node_id,
            next_mutable_variant,
            &mut self.replacement_subtree_root_ids,
        );

        std::mem::swap(
            &mut self.mutable_subtree_root_ids,
            &mut self.replacement_subtree_root_ids,
        );
        self.mutable_variant = next_mutable_variant;

        self.renderer.render_headless_frame();
    }

    fn mutate_next_frame(&mut self) {
        for &subtree_root_id in &self.mutable_subtree_root_ids {
            self.renderer.remove_subtree(subtree_root_id);
        }

        let next_mutable_variant = !self.mutable_variant;
        self.replacement_subtree_root_ids.clear();
        add_many_shallow_mutable_subtrees(
            &mut self.renderer,
            self.root_node_id,
            next_mutable_variant,
            &mut self.replacement_subtree_root_ids,
        );

        std::mem::swap(
            &mut self.mutable_subtree_root_ids,
            &mut self.replacement_subtree_root_ids,
        );
        self.mutable_variant = next_mutable_variant;
    }
}

fn try_create_headless_renderer() -> Option<Renderer<'static>> {
    match block_on(Renderer::try_new_headless(
        (CANVAS_WIDTH, CANVAS_HEIGHT),
        SCALE_FACTOR,
    )) {
        Ok(renderer) => Some(renderer),
        Err(RendererCreationError::AdapterNotAvailable(_)) => None,
        Err(error) => panic!("Failed to create headless renderer for benchmark: {error}"),
    }
}

fn load_benchmark_shapes(renderer: &mut Renderer<'static>) {
    let root_shape = Shape::rect(
        [(0.0, 0.0), (CANVAS_WIDTH as f32, CANVAS_HEIGHT as f32)],
        Stroke::default(),
    );
    renderer.load_shape(root_shape, ROOT_CACHE_KEY, Some(ROOT_CACHE_KEY));

    let leaf_shape = Shape::rect([(0.0, 0.0), (LEAF_SIZE, LEAF_SIZE)], Stroke::default());
    renderer.load_shape(leaf_shape, LEAF_CACHE_KEY, Some(LEAF_CACHE_KEY));
}

fn build_benchmark_scene(
    renderer: &mut Renderer<'static>,
    mutable_variant: bool,
) -> SceneBuildResult {
    let root_node_id = renderer.add_cached_shape_to_the_render_queue(ROOT_CACHE_KEY, None);
    renderer.set_shape_color(root_node_id, Some(Color::TRANSPARENT));

    add_stable_leaf_nodes(renderer, root_node_id);
    let mutable_subtree_root_id = add_mutable_leaf_subtree(renderer, root_node_id, mutable_variant);

    SceneBuildResult {
        root_node_id,
        mutable_subtree_root_id,
    }
}

fn add_stable_leaf_nodes(renderer: &mut Renderer<'static>, root_node_id: usize) {
    for stable_leaf_index in 0..MUTABLE_LEAF_COUNT {
        let leaf_node_id =
            renderer.add_cached_shape_to_the_render_queue(LEAF_CACHE_KEY, Some(root_node_id));
        renderer.set_shape_color(leaf_node_id, Some(color_for_stable_leaf(stable_leaf_index)));
        renderer.set_shape_transform(leaf_node_id, stable_leaf_transform(stable_leaf_index));
    }
}

fn add_mutable_leaf_subtree(
    renderer: &mut Renderer<'static>,
    root_node_id: usize,
    mutable_variant: bool,
) -> usize {
    let mutable_subtree_root_id =
        renderer.add_cached_shape_to_the_render_queue(ROOT_CACHE_KEY, Some(root_node_id));
    renderer.set_shape_color(mutable_subtree_root_id, Some(Color::TRANSPARENT));

    for mutable_leaf_index in 0..MUTABLE_LEAF_COUNT {
        let leaf_node_id = renderer
            .add_cached_shape_to_the_render_queue(LEAF_CACHE_KEY, Some(mutable_subtree_root_id));
        renderer.set_shape_color(
            leaf_node_id,
            Some(color_for_mutable_leaf(mutable_leaf_index, mutable_variant)),
        );
        renderer.set_shape_transform(
            leaf_node_id,
            mutable_leaf_transform(mutable_leaf_index, mutable_variant),
        );
    }

    mutable_subtree_root_id
}

fn add_many_shallow_mutable_subtrees(
    renderer: &mut Renderer<'static>,
    root_node_id: usize,
    mutable_variant: bool,
    subtree_root_ids: &mut Vec<usize>,
) {
    subtree_root_ids.clear();

    for subtree_index in 0..SHALLOW_SUBTREE_COUNT {
        let subtree_root_id =
            renderer.add_cached_shape_to_the_render_queue(ROOT_CACHE_KEY, Some(root_node_id));
        renderer.set_shape_color(subtree_root_id, Some(Color::TRANSPARENT));
        renderer.set_shape_transform(
            subtree_root_id,
            shallow_subtree_root_transform(subtree_index, mutable_variant),
        );

        let first_leaf_index = shallow_subtree_leaf_start(subtree_index);
        let leaf_count = shallow_subtree_leaf_count(subtree_index);

        for local_leaf_index in 0..leaf_count {
            let mutable_leaf_index = first_leaf_index + local_leaf_index;
            let leaf_node_id = renderer
                .add_cached_shape_to_the_render_queue(LEAF_CACHE_KEY, Some(subtree_root_id));
            renderer.set_shape_color(
                leaf_node_id,
                Some(color_for_mutable_leaf(mutable_leaf_index, mutable_variant)),
            );
            renderer.set_shape_transform(
                leaf_node_id,
                shallow_subtree_leaf_transform(local_leaf_index, mutable_variant),
            );
        }

        subtree_root_ids.push(subtree_root_id);
    }
}

fn shallow_subtree_leaf_start(subtree_index: usize) -> usize {
    let base_leaf_count = MUTABLE_LEAF_COUNT / SHALLOW_SUBTREE_COUNT;
    let remainder_leaf_count = MUTABLE_LEAF_COUNT % SHALLOW_SUBTREE_COUNT;

    subtree_index * base_leaf_count + subtree_index.min(remainder_leaf_count)
}

fn shallow_subtree_leaf_count(subtree_index: usize) -> usize {
    let base_leaf_count = MUTABLE_LEAF_COUNT / SHALLOW_SUBTREE_COUNT;
    let remainder_leaf_count = MUTABLE_LEAF_COUNT % SHALLOW_SUBTREE_COUNT;

    base_leaf_count + usize::from(subtree_index < remainder_leaf_count)
}

fn shallow_subtree_root_transform(
    subtree_index: usize,
    mutable_variant: bool,
) -> TransformInstance {
    let row = subtree_index / SHALLOW_SUBTREE_COLUMNS;
    let column = subtree_index % SHALLOW_SUBTREE_COLUMNS;
    let variant_offset = if mutable_variant {
        MUTABLE_VARIANT_OFFSET
    } else {
        0.0
    };

    TransformInstance::translation(
        ROOT_PADDING + column as f32 * SHALLOW_SUBTREE_SPACING_X + variant_offset,
        ROOT_PADDING
            + 20.0 * LEAF_SPACING
            + row as f32 * SHALLOW_SUBTREE_SPACING_Y
            + variant_offset,
    )
}

fn shallow_subtree_leaf_transform(
    local_leaf_index: usize,
    mutable_variant: bool,
) -> TransformInstance {
    let row = local_leaf_index / SHALLOW_SUBTREE_CHILDREN_PER_ROW;
    let column = local_leaf_index % SHALLOW_SUBTREE_CHILDREN_PER_ROW;
    let variant_offset = if mutable_variant {
        MUTABLE_VARIANT_OFFSET
    } else {
        0.0
    };

    TransformInstance::translation(
        column as f32 * LEAF_SPACING * 0.6 + variant_offset,
        row as f32 * LEAF_SPACING * 0.6 + variant_offset,
    )
}

fn stable_leaf_transform(stable_leaf_index: usize) -> TransformInstance {
    let row = stable_leaf_index / LEAF_COLUMNS_PER_HALF;
    let column = stable_leaf_index % LEAF_COLUMNS_PER_HALF;

    TransformInstance::translation(
        ROOT_PADDING + column as f32 * LEAF_SPACING,
        ROOT_PADDING + row as f32 * LEAF_SPACING,
    )
}

fn mutable_leaf_transform(mutable_leaf_index: usize, mutable_variant: bool) -> TransformInstance {
    let row = mutable_leaf_index / LEAF_COLUMNS_PER_HALF;
    let column = mutable_leaf_index % LEAF_COLUMNS_PER_HALF;
    let variant_offset = if mutable_variant {
        MUTABLE_VARIANT_OFFSET
    } else {
        0.0
    };

    TransformInstance::translation(
        ROOT_PADDING + (column + LEAF_COLUMNS_PER_HALF) as f32 * LEAF_SPACING + variant_offset,
        ROOT_PADDING + row as f32 * LEAF_SPACING + variant_offset,
    )
}

fn color_for_stable_leaf(stable_leaf_index: usize) -> Color {
    let intensity = 80 + (stable_leaf_index % 5) as u8 * 20;
    Color::rgb(intensity, 140, 220)
}

fn color_for_mutable_leaf(mutable_leaf_index: usize, mutable_variant: bool) -> Color {
    let base_intensity = 70 + (mutable_leaf_index % 5) as u8 * 25;
    if mutable_variant {
        Color::rgb(220, base_intensity, 110)
    } else {
        Color::rgb(110, base_intensity, 220)
    }
}

fn benchmark_renderer_partial_updates(criterion: &mut Criterion) {
    let Some(mut full_rebuild_state) = FullRebuildBenchmarkState::new() else {
        eprintln!(
            "Skipping renderer_partial_updates benchmark: no suitable GPU adapter available."
        );
        return;
    };
    let Some(mut partial_update_state) = PartialUpdateBenchmarkState::new() else {
        eprintln!(
            "Skipping renderer_partial_updates benchmark: no suitable GPU adapter available."
        );
        return;
    };
    let Some(mut many_shallow_partial_update_state) = ManyShallowPartialUpdateBenchmarkState::new()
    else {
        eprintln!(
            "Skipping renderer_partial_updates benchmark: no suitable GPU adapter available."
        );
        return;
    };

    let mut benchmark_group = criterion.benchmark_group("renderer_full_frame");
    benchmark_group.measurement_time(Duration::from_secs(10));
    benchmark_group.sample_size(10);

    benchmark_group.bench_function("clear_tree_and_rebuild", |benchmark| {
        benchmark.iter(|| {
            full_rebuild_state.render_next_frame();
            black_box(full_rebuild_state.mutable_variant);
        });
    });

    benchmark_group.bench_function("remove_half_and_repopulate", |benchmark| {
        benchmark.iter(|| {
            partial_update_state.render_next_frame();
            black_box(partial_update_state.mutable_variant);
        });
    });

    benchmark_group.bench_function("remove_100_shallow_subtrees_and_repopulate", |benchmark| {
        benchmark.iter(|| {
            many_shallow_partial_update_state.render_next_frame();
            black_box(many_shallow_partial_update_state.mutable_variant);
        });
    });

    benchmark_group.finish();

    let Some(mut full_rebuild_prepare_state) = FullRebuildBenchmarkState::new() else {
        eprintln!("Skipping renderer_prepare_cpu benchmark: no suitable GPU adapter available.");
        return;
    };
    let Some(mut partial_update_prepare_state) = PartialUpdateBenchmarkState::new() else {
        eprintln!("Skipping renderer_prepare_cpu benchmark: no suitable GPU adapter available.");
        return;
    };
    let Some(mut many_shallow_prepare_state) = ManyShallowPartialUpdateBenchmarkState::new() else {
        eprintln!("Skipping renderer_prepare_cpu benchmark: no suitable GPU adapter available.");
        return;
    };

    let mut prepare_group = criterion.benchmark_group("renderer_prepare_cpu");
    prepare_group.measurement_time(Duration::from_secs(10));
    prepare_group.sample_size(10);

    prepare_group.bench_function("clear_tree_and_rebuild", |benchmark| {
        benchmark.iter_custom(|iterations| {
            let mut total_prepare_time = Duration::ZERO;
            for _ in 0..iterations {
                full_rebuild_prepare_state.render_next_frame();
                total_prepare_time += full_rebuild_prepare_state.renderer.last_prepare_cpu_time();
            }
            total_prepare_time
        });
    });

    prepare_group.bench_function("remove_half_and_repopulate", |benchmark| {
        benchmark.iter_custom(|iterations| {
            let mut total_prepare_time = Duration::ZERO;
            for _ in 0..iterations {
                partial_update_prepare_state.render_next_frame();
                total_prepare_time += partial_update_prepare_state
                    .renderer
                    .last_prepare_cpu_time();
            }
            total_prepare_time
        });
    });

    prepare_group.bench_function("remove_100_shallow_subtrees_and_repopulate", |benchmark| {
        benchmark.iter_custom(|iterations| {
            let mut total_prepare_time = Duration::ZERO;
            for _ in 0..iterations {
                many_shallow_prepare_state.render_next_frame();
                total_prepare_time += many_shallow_prepare_state.renderer.last_prepare_cpu_time();
            }
            total_prepare_time
        });
    });

    prepare_group.finish();

    let Some(mut full_rebuild_mutation_state) = FullRebuildBenchmarkState::new() else {
        eprintln!("Skipping renderer_mutation_cpu benchmark: no suitable GPU adapter available.");
        return;
    };
    let Some(mut partial_update_mutation_state) = PartialUpdateBenchmarkState::new() else {
        eprintln!("Skipping renderer_mutation_cpu benchmark: no suitable GPU adapter available.");
        return;
    };
    let Some(mut many_shallow_mutation_state) = ManyShallowPartialUpdateBenchmarkState::new()
    else {
        eprintln!("Skipping renderer_mutation_cpu benchmark: no suitable GPU adapter available.");
        return;
    };

    let mut mutation_group = criterion.benchmark_group("renderer_mutation_cpu");
    mutation_group.measurement_time(Duration::from_secs(10));
    mutation_group.sample_size(10);

    mutation_group.bench_function("clear_tree_and_rebuild", |benchmark| {
        benchmark.iter(|| {
            full_rebuild_mutation_state.mutate_next_frame();
            black_box(full_rebuild_mutation_state.mutable_variant);
        });
    });

    mutation_group.bench_function("remove_half_and_repopulate", |benchmark| {
        benchmark.iter(|| {
            partial_update_mutation_state.mutate_next_frame();
            black_box(partial_update_mutation_state.mutable_variant);
        });
    });

    mutation_group.bench_function("remove_100_shallow_subtrees_and_repopulate", |benchmark| {
        benchmark.iter(|| {
            many_shallow_mutation_state.mutate_next_frame();
            black_box(many_shallow_mutation_state.mutable_variant);
        });
    });

    mutation_group.finish();
}

criterion_group!(benches, benchmark_renderer_partial_updates);
criterion_main!(benches);
