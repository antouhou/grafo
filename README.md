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
* Custom WGSL shader effects on shape masks, rendered content, and backdrops
* Pixel-area antialiasing, with optional MSAA

[Install Grafo from crates.io](https://crates.io/crates/grafo) or read the
[API documentation](https://docs.rs/grafo/).

## Getting started

Add the following to your `Cargo.toml`:

```toml
[dependencies]
grafo = "0.20"
winit = "0.30"
futures = "0.3"
env_logger = "0.11"
```

### Basic usage

Create a shape, set its fill and transform, then render it. For a complete window, run
`cargo run --example basic`. The same example appears in the [crate documentation](https://docs.rs/grafo/).

```rust
use grafo::{Color, Shape, ShapeDrawCommandOptions};

// Set the fill when queueing the shape.
let rect = Shape::rect([(0.0, 0.0), (200.0, 100.0)]);
scene
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
scene.render(&mut surface).unwrap();
scene.clear();
```

`Scene` retains its contents between renders. `clear()` removes those contents while keeping
loaded shapes, textures, and effect definitions available for reuse.

### Render targets

Create a surface and scene from the same context:

```rust,no_run
use grafo::{Scene, SceneContext, Surface};

let context = SceneContext::new().await;
let mut surface = Surface::new(&context, platform_target, size, true, false)?;
let mut scene = Scene::new_with_context(context, size, scale_factor, 1);
scene.render(&mut surface)?;
```

Pass a window or an object that provides native display and surface handles to `Surface::new`.
Use `surface.resize(size)` to resize it and `surface.set_vsync(enabled)` to change vsync.
`render` borrows the target and uses its dimensions.

For memory output, `Pixmap` owns the buffer and `PixmapMut` borrows a slice:

```rust
use grafo::{PixelFormat, PixelLayout, Pixmap, PixmapMut};

let size = (640, 480);
let mut image = Pixmap::new(size, PixelFormat::Bgra8)?;
scene.render(&mut image)?;
let bytes = image.pixels();

let mut pixels = vec![0_u32; 640 * 480];
scene.render(PixmapMut::argb32(&mut pixels, size)?)?;

// RGBA rows with 32 bytes of padding.
let layout = PixelLayout::new(size, PixelFormat::Rgba8, 640 * 4 + 32)?;
let mut storage = vec![0; layout.byte_len()];
let mut borrowed = PixmapMut::new(&mut storage, layout)?;
scene.render(&mut borrowed)?;
```

`Pixmap::resize` reuses buffer capacity when it can. A borrowed slice must fit its layout.
Rendering leaves row padding and trailing bytes untouched.

When rendering to memory, WGPU waits for readback. Errors leave the buffer unchanged.
Rendering to a surface submits and presents the frame.

BGRA8 and RGBA8 store channels in that byte order. ARGB32 stores native-endian `0xAARRGGBB` words.
RGB is premultiplied in linear space, then encoded as sRGB. Alpha stays linear.

### Multiple independent windows

Reuse a `SceneContext` to share the WGPU device, queue, and textures across windows:

```rust,no_run
use futures::executor::block_on;
use grafo::{Scene, SceneContext, Surface};

let context = block_on(SceneContext::new());

let first_surface = Surface::new(&context, first_window, first_size, true, false)?;
let second_surface = Surface::new(&context, second_window, second_size, true, false)?;
let first_scene = Scene::new_with_context(
    context.clone(), first_size, first_scale_factor, 1,
);
let second_scene = Scene::new_with_context(
    context, second_size, second_scale_factor, 1,
);
```

Create a scene when you open a window and drop it when you close the window.
Draw calls added to one scene never appear in another scene's draw queue.

Loaded shapes are also shared by the context. A `cache_key` passed to `load_shape` is scoped to
the `SceneContext`, so another scene can reuse that shape through
`add_cached_shape`. Use a content-derived key when the same geometry should
be shared; loading a different shape with the same key replaces the shared entry, and
`remove_shape` removes it for every scene using the context.

### Shape hierarchy and overflow

The second argument to `add_shape` and `add_clipping_rect` is the optional
`parent_shape_id`. A child is drawn inside that parent in the draw tree.
By default, parents clip their children:

```rust
use grafo::{Color, Shape, ShapeDrawCommandOptions};

let clipping_parent_id = scene
    .add_shape(
        Shape::rect([(0.0, 0.0), (120.0, 80.0)]),
        None,
        None,
        ShapeDrawCommandOptions::new().color(Color::rgb(220, 220, 220)),
    )
    .unwrap();

scene
    .add_shape(
        Shape::rect([(80.0, 20.0), (160.0, 60.0)]),
        Some(clipping_parent_id),
        None,
        ShapeDrawCommandOptions::new().color(Color::rgb(220, 80, 80)),
    )
    .unwrap();

// Let children render outside `overflow_parent_id`, while still respecting any ancestor clip.
let overflow_parent_id = scene
    .add_shape(
        Shape::rect([(0.0, 0.0), (120.0, 80.0)]),
        None,
        None,
        ShapeDrawCommandOptions::new()
            .color(Color::rgb(220, 220, 220))
            .clips_children(false),
    )
    .unwrap();

scene
    .add_shape(
        Shape::rect([(80.0, 20.0), (160.0, 60.0)]),
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

Run `cargo bench --bench path_coverage` to compare retained redraws, queue rebuilds,
moving paths, and paths whose outlines change. Each sample redraws the scene and
includes GPU readback.

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
use grafo::{Color, Scene, Shape, ShapeDrawCommandOptions};

// After allocating textures via scene.texture_manager()
scene
    .add_shape(
        Shape::rect([(0.0, 0.0), (300.0, 200.0)]),
        None,
        None,
        ShapeDrawCommandOptions::new()
            .color(Color::rgb(40, 40, 40))
            .background_texture_id(bg_tex_id)
            .foreground_texture_id(fg_tex_id),
    )
    .unwrap();

// Transparent parts of the background texture reveal the white fill.
scene
    .add_shape(
        Shape::rect([(0.0, 0.0), (300.0, 200.0)]),
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
scene
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
