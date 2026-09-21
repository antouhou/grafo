# Grafo

[![Grafo crate](https://img.shields.io/crates/v/grafo.svg)](https://crates.io/crates/grafo)
[![Grafo documentation](https://docs.rs/grafo/badge.svg)](https://docs.rs/grafo)
[![Build and test](https://github.com/antouhou/grafo/actions/workflows/rust.yml/badge.svg?branch=main)](https://github.com/antouhou/grafo/actions)

Grafo is a GPU-accelerated vector graphics library for Rust.

## Features

* Path rendering with cached tessellation
* Hierarchical path clipping
* Per-instance 3D and perspective transforms
* Solid fills and linear, radial, and conic gradients
* Custom WGSL shader effects on shape masks, groups, and backdrops
* Geometry-based antialiasing and MSAA

[Install Grafo from crates.io](https://crates.io/crates/grafo) or read the
[API documentation](https://docs.rs/grafo/).

## Getting started

Add the following to your `Cargo.toml`:

```toml
[dependencies]
grafo = "0.19"
winit = "0.30"
futures = "0.3"
env_logger = "0.11"
```

### Basic usage

Create a shape, set its fill and transform, then render it. For a complete window, run
`cargo run --example basic`. The same example appears in the [crate documentation](https://docs.rs/grafo/).

```rust
use grafo::{Color, Shape, ShapeDrawCommandOptions, Stroke};

// Set the fill when queueing the shape.
let rect = Shape::rect(
    [(0.0, 0.0), (200.0, 100.0)],
    Stroke::new(2.0_f32, Color::BLACK),
);
renderer
    .add_shape(
        rect,
        None,
        None,
        ShapeDrawCommandOptions::new()
            .color(Color::rgb(0, 128, 255))
            .transform(grafo::TransformInstance::translation(100.0, 100.0)),
    )
    .unwrap();

// Call this on RedrawRequested in a winit event loop
renderer.render().unwrap();
renderer.clear_draw_queue();
```

### Multiple independent windows

Create a `RendererContext` once, then create one renderer per window. Each renderer has its own
draw queue and render target, while sharing the WGPU device, queue, and texture storage:

```rust,no_run
use futures::executor::block_on;
use grafo::{Renderer, RendererContext};

let context = block_on(RendererContext::new());

let first_renderer = Renderer::new_with_context(
    context.clone(), first_window, first_size, first_scale_factor, true, false, 1,
);
let second_renderer = Renderer::new_with_context(
    context, second_window, second_size, second_scale_factor, true, false, 1,
);
```

Create a renderer when you open a window and drop it when you close the window.
Draw calls added to one renderer never appear in another renderer's draw queue.

Loaded shapes are also shared by the context. A `cache_key` passed to `load_shape` is scoped to
the `RendererContext`, so another renderer can reuse that shape through
`add_cached_shape`. Use a content-derived key when the same geometry should
be shared; loading a different shape with the same key replaces the shared entry, and
`remove_shape` removes it for every renderer using the context.

### Shape hierarchy and overflow

The second argument to `add_shape` and `add_clipping_rect` is the optional
`parent_shape_id`. A child is drawn inside that parent in the draw tree.
By default, parents clip their children:

```rust
use grafo::{Color, Shape, ShapeDrawCommandOptions, Stroke};

let clipping_parent_id = renderer
    .add_shape(
        Shape::rect([(0.0, 0.0), (120.0, 80.0)], Stroke::default()),
        None,
        None,
        ShapeDrawCommandOptions::new().color(Color::rgb(220, 220, 220)),
    )
    .unwrap();

renderer
    .add_shape(
        Shape::rect([(80.0, 20.0), (160.0, 60.0)], Stroke::default()),
        Some(clipping_parent_id),
        None,
        ShapeDrawCommandOptions::new().color(Color::rgb(220, 80, 80)),
    )
    .unwrap();

// Let children render outside `overflow_parent_id`, while still respecting any ancestor clip.
let overflow_parent_id = renderer
    .add_shape(
        Shape::rect([(0.0, 0.0), (120.0, 80.0)], Stroke::default()),
        None,
        None,
        ShapeDrawCommandOptions::new()
            .color(Color::rgb(220, 220, 220))
            .clips_children(false),
    )
    .unwrap();

renderer
    .add_shape(
        Shape::rect([(80.0, 20.0), (160.0, 60.0)], Stroke::default()),
        Some(overflow_parent_id),
        None,
        ShapeDrawCommandOptions::new().color(Color::rgb(220, 80, 80)),
    )
    .unwrap();

// The same call works for ids returned by `add_clipping_rect`, so clip rectangles
// can also be used as non-clipping containers.
```

## Examples

- `basic.rs` draws shapes using winit 0.30's `ApplicationHandler`.
- `transforms.rs` covers instance transforms, color, perspective, and hit-testing.
- `benches/visual_regression.rs` benchmarks the visual regression scene with Criterion.

Run the visual-regression benchmark in release mode:

```sh
cargo bench --bench visual_regression
```

The [examples](https://github.com/antouhou/grafo/tree/main/examples) directory includes
hierarchical clipping, texture layers, transforms, and shader effects.

### Background and foreground textures

Shapes composite up to two texture layers over the instance color using premultiplied alpha.

Set the layers with `ShapeDrawCommandOptions::background_texture` and
`ShapeDrawCommandOptions::foreground_texture`. Each accepts `ShapeTextureOptions`
with a texture ID and fit mode. Use `background_texture_id` and `foreground_texture_id`
to set just the IDs.

Composition from bottom to top:

`final = foreground + (background + color * (1 - background.a)) * (1 - foreground.a)`

API:

```rust
use grafo::{Color, Renderer, Shape, ShapeDrawCommandOptions, Stroke};

// After allocating textures via renderer.texture_manager()
renderer
    .add_shape(
        Shape::rect([(0.0, 0.0), (300.0, 200.0)], Stroke::new(1.0_f32, Color::BLACK)),
        None,
        None,
        ShapeDrawCommandOptions::new()
            .color(Color::rgb(40, 40, 40))
            .background_texture_id(bg_tex_id)
            .foreground_texture_id(fg_tex_id),
    )
    .unwrap();

// Transparent parts of the background texture reveal the white fill.
renderer
    .add_shape(
        Shape::rect([(0.0, 0.0), (300.0, 200.0)], Stroke::new(1.0_f32, Color::BLACK)),
        None,
        None,
        ShapeDrawCommandOptions::new()
            .color(Color::WHITE)
            .background_texture_id(bg_tex_id),
    )
    .unwrap();
```

See `examples/multi_texture.rs` for procedural background and foreground textures.

### Positioning shapes

Use per-shape transforms to position shapes. Common helpers:

- Translate: `TransformInstance::translation(tx, ty)`
- Scale: `TransformInstance::scale(sx, sy)`
- Rotate around Z: `TransformInstance::rotation_z_deg(deg)`
- Compose: `a.multiply(&b)` and `a.then(&b)` apply `a` first, then `b`.

Example:

```rust
use grafo::ShapeDrawCommandOptions;

let r = grafo::TransformInstance::rotation_z_deg(15.0);
let t = grafo::TransformInstance::translation(150.0, 80.0);
// Rotate first, then translate
renderer
    .add_shape(
        my_shape,
        None,
        None,
        ShapeDrawCommandOptions::new().transform(r.then(&t)),
    )
    .unwrap();
```

## Documentation

[Documentation is available on docs.rs](https://docs.rs/grafo/).

## Contributing

Read [CONTRIBUTING.md](./CONTRIBUTING.md) for coding conventions and required checks.

## Authors

- [Anton Suprunchuk](https://github.com/antouhou). [Website](https://antouhou.com).

## License

This project is licensed under the MIT License - see the [LICENSE.md](./LICENSE.md) file for details
