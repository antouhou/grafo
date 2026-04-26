# Grafo

[![Grafo crate](https://img.shields.io/crates/v/grafo.svg)](https://crates.io/crates/grafo)
[![Grafo documentation](https://docs.rs/grafo/badge.svg)](https://docs.rs/grafo)
[![Build and test](https://github.com/antouhou/grafo/actions/workflows/rust.yml/badge.svg?branch=main)](https://github.com/antouhou/grafo/actions)

Grafo is a GPU-accelerated rendering library for Rust. It’s a quick way to render shapes and images, with masking and hierarchical clipping built in.

The library is designed for flexibility and ease of use, making it suitable for a wide 
range of applications, from simple graphical interfaces to complex rendering engines.

## Features

* Shape Rendering: Create and render vector shapes (with optional texture layers).
* (Text rendering was previously integrated; it has now been extracted into a separate crate - https://crates.io/crates/protextinator)
* Stencil Operations: Advanced stencil operations for clipping and masking.
* Shape hierarchy: Attach shapes to parent nodes and choose whether each parent clips descendants.
* Per-instance data: Set transform and color per shape instance (no fill color stored on geometry).
* Antialiasing: You can choose between built-in support of inflated geometry or MSAA

Grafo is [available on crates.io](https://crates.io/crates/grafo), and
[API Documentation is available on docs.rs](https://docs.rs/grafo/).

## Getting Started

Add the following to your `Cargo.toml`:

```toml
[dependencies]
grafo = "0.10"
winit = "0.30"      # For window creation and event handling
image = "0.25"      # For image decoding (textures)
env_logger = "0.11" # For logging
log = "0.4"         # For logging
```

### Basic Usage

Below is a minimal snippet showing how to create a shape, set per-instance color and transform, and render. For a complete runnable window using `winit` 0.30 (ApplicationHandler API), see `examples/basic.rs`.

```rust
use grafo::{Color, Shape, ShapeDrawCommandOptions, Stroke};

// Create a rectangle shape (no fill color on the shape itself)
let rect = Shape::rect(
    [(0.0, 0.0), (200.0, 100.0)],
    Stroke::new(2.0, Color::BLACK),
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

// Render one frame (typical winit loop would call this on RedrawRequested)
renderer.render().unwrap();
renderer.clear_draw_queue();
```

### Shape hierarchy and overflow

The second argument to `add_shape` and `add_clipping_rect` is the optional
`parent_shape_id`. A child is drawn inside that parent in the draw tree.
By default, children are clipped to their parent:

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

- `basic.rs` – draw simple shapes (winit 0.30 ApplicationHandler)
- `transforms.rs` – demonstrates per-instance transform and color, perspective, and hit-testing

For a detailed example showcasing advanced features like hierarchical clipping and
multi-layer shape texturing, please refer to the 
[examples](https://github.com/antouhou/grafo/tree/main/examples) directory in the repository.

### Multi-texturing (Background + Foreground)

Shapes support up to two texture layers that are composited with per-instance color using premultiplied alpha:

1. Background layer (index 0 / `TextureLayer::Background`)
2. Foreground layer (index 1 / `TextureLayer::Foreground`)

Composition order (bottom to top):

`final = foreground + (background + color * (1 - background.a)) * (1 - foreground.a)`

API:

```rust
use grafo::{Color, Renderer, Shape, ShapeDrawCommandOptions, Stroke};

// After allocating textures via renderer.texture_manager()
renderer
    .add_shape(
        Shape::rect([(0.0, 0.0), (300.0, 200.0)], Stroke::new(1.0, Color::BLACK)),
        None,
        None,
        ShapeDrawCommandOptions::new()
            .color(Color::rgb(40, 40, 40))
            .background_texture_id(bg_tex_id)
            .foreground_texture_id(fg_tex_id),
    )
    .unwrap();

// Single-layer helper (Background):
renderer
    .add_shape(
        Shape::rect([(0.0, 0.0), (300.0, 200.0)], Stroke::new(1.0, Color::BLACK)),
        None,
        None,
        ShapeDrawCommandOptions::new()
            .color(Color::WHITE)
            .background_texture_id(bg_tex_id),
    )
    .unwrap(); // useful when texture transparency should reveal white
```

See `examples/multi_texture.rs` for a runnable demo that generates procedural background & foreground textures.

### Positioning shapes

Use per-shape transforms to position shapes. Common helpers:

- Translate: `TransformInstance::translation(tx, ty)`
- Scale: `TransformInstance::scale(sx, sy)`
- Rotate (Z): `TransformInstance::rotation_z_deg(deg)`
- Compose: `a.multiply(&b)` (or `a.then(&b)`) applies `a` first, then `b`

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
- [Renderer architecture notes](./docs/renderer-architecture.md)

## Contributing

Everyone is welcome to contribute in any way or form! For further details, please read [CONTRIBUTING.md](./CONTRIBUTING.md).

## Authors
- [Anton Suprunchuk](https://github.com/antouhou) - [Website](https://antouhou.com)

Also, see the list of contributors who participated in this project.

## License

This project is licensed under the MIT License - see the
[LICENSE.md](./LICENSE.md) file for details
