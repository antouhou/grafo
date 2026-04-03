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

const ROOT_PADDING: f32 = 24.0;
const LEAF_SIZE: f32 = 24.0;
const LEAF_SPACING: f32 = 28.0;
const MUTABLE_VARIANT_OFFSET: f32 = 6.0;

struct SceneBuildResult {
    root_node_id: usize,
    mutable_leaf_node_ids: Vec<usize>,
}

struct FullRebuildBenchmarkState {
    renderer: Renderer<'static>,
    pixel_buffer: Vec<u8>,
    mutable_variant: bool,
}

impl FullRebuildBenchmarkState {
    fn new() -> Option<Self> {
        let mut renderer = try_create_headless_renderer()?;
        load_benchmark_shapes(&mut renderer);

        Some(Self {
            renderer,
            pixel_buffer: Vec::new(),
            mutable_variant: false,
        })
    }

    fn render_next_frame(&mut self) {
        self.renderer.clear_draw_queue();
        build_benchmark_scene(&mut self.renderer, self.mutable_variant);
        self.renderer.render_to_buffer(&mut self.pixel_buffer);
        self.mutable_variant = !self.mutable_variant;
    }
}

struct PartialUpdateBenchmarkState {
    renderer: Renderer<'static>,
    pixel_buffer: Vec<u8>,
    root_node_id: usize,
    mutable_leaf_node_ids: Vec<usize>,
    mutable_variant: bool,
}

impl PartialUpdateBenchmarkState {
    fn new() -> Option<Self> {
        let mut renderer = try_create_headless_renderer()?;
        load_benchmark_shapes(&mut renderer);
        let scene = build_benchmark_scene(&mut renderer, false);

        Some(Self {
            renderer,
            pixel_buffer: Vec::new(),
            root_node_id: scene.root_node_id,
            mutable_leaf_node_ids: scene.mutable_leaf_node_ids,
            mutable_variant: false,
        })
    }

    fn render_next_frame(&mut self) {
        for node_id in self.mutable_leaf_node_ids.drain(..) {
            self.renderer.remove_subtree(node_id);
        }

        let next_mutable_variant = !self.mutable_variant;
        self.mutable_leaf_node_ids =
            add_mutable_leaf_nodes(&mut self.renderer, self.root_node_id, next_mutable_variant);
        self.mutable_variant = next_mutable_variant;

        self.renderer.render_to_buffer(&mut self.pixel_buffer);
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
    let mutable_leaf_node_ids = add_mutable_leaf_nodes(renderer, root_node_id, mutable_variant);

    SceneBuildResult {
        root_node_id,
        mutable_leaf_node_ids,
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

fn add_mutable_leaf_nodes(
    renderer: &mut Renderer<'static>,
    root_node_id: usize,
    mutable_variant: bool,
) -> Vec<usize> {
    let mut mutable_leaf_node_ids = Vec::with_capacity(MUTABLE_LEAF_COUNT);

    for mutable_leaf_index in 0..MUTABLE_LEAF_COUNT {
        let leaf_node_id =
            renderer.add_cached_shape_to_the_render_queue(LEAF_CACHE_KEY, Some(root_node_id));
        renderer.set_shape_color(
            leaf_node_id,
            Some(color_for_mutable_leaf(mutable_leaf_index, mutable_variant)),
        );
        renderer.set_shape_transform(
            leaf_node_id,
            mutable_leaf_transform(mutable_leaf_index, mutable_variant),
        );
        mutable_leaf_node_ids.push(leaf_node_id);
    }

    mutable_leaf_node_ids
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

    let mut benchmark_group = criterion.benchmark_group("renderer_full_frame");
    benchmark_group.measurement_time(Duration::from_secs(10));
    benchmark_group.sample_size(10);

    benchmark_group.bench_function("clear_tree_and_rebuild", |benchmark| {
        benchmark.iter(|| {
            full_rebuild_state.render_next_frame();
            black_box(full_rebuild_state.pixel_buffer.first().copied());
        });
    });

    benchmark_group.bench_function("remove_half_and_repopulate", |benchmark| {
        benchmark.iter(|| {
            partial_update_state.render_next_frame();
            black_box(partial_update_state.pixel_buffer.first().copied());
        });
    });

    benchmark_group.finish();
}

criterion_group!(benches, benchmark_renderer_partial_updates);
criterion_main!(benches);
