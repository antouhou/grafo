# Grafo

[![Grafo crate](https://img.shields.io/crates/v/grafo.svg)](https://crates.io/crates/grafo)
[![Grafo documentation](https://docs.rs/grafo/badge.svg)](https://docs.rs/grapho)
[![Build and test](https://github.com/antouhou/grafo/actions/workflows/test.yml/badge.svg?branch=master)](https://github.com/antouhou/grafo/actions)

Grafo is a GPU-accelerated rendering library for Rust. 
Leveraging the power of the `wgpu` crate, Grafo provides efficient rendering of shapes, images, 
and text with support for advanced features like stencil operations and parallel processing.

The library is designed for flexibility and ease of use, making it suitable for a wide 
range of applications, from simple graphical interfaces to complex rendering engines.

Grafo [available on crates.io](https://crates.io/crates/grafo), and
[API Documentation is available on docs.rs](https://docs.rs/grafo/).

## Features

* Shape Rendering: Create and render complex shapes. 
* Image Rendering: Render images with support for clipping to shapes. 
* Text Rendering: Render text with customizable layout, alignment, and styling using the 
[glyphon](https://github.com/grovesNL/glyphon) crate. 
* Stencil Operations: Advanced stencil operations for clipping and masking. 
* Performance Optimization: Utilizes parallel processing with `rayon` to optimize rendering performance.

## Getting Started

Add the following to your `Cargo.toml`:

```toml
[dependencies]
grafo = "0.1.0" # Replace with the actual version
winit = "0.27"   # For window creation and event handling
image = "0.24"   # For image processing
env_logger = "0.10" # For logging
log = "0.4"      # For logging
```

### Basic Usage

Below is a simple example demonstrating how to initialize the `Renderer`, add shapes and text, 
and render a frame using `winit`. For a more comprehensive example, refer to the 
[examples](https://github.com/antouhou/grafo/tree/main/examples) folder in the repository.

```rust
use futures::executor::block_on;
use grafo::{BorderRadii, Shape};
use grafo::{Color, Stroke};
use std::sync::Arc;
use winit::event::{Event, WindowEvent};
use winit::event_loop::EventLoop;
use winit::window::WindowBuilder;

pub fn main() {
env_logger::init();
let event_loop = EventLoop::new();
let window = Arc::new(WindowBuilder::new().build(&event_loop).unwrap());

    let window_size = window.inner_size();
    let scale_factor = window.scale_factor();
    let physical_size = (window_size.width, window_size.height);

    // Initialize the renderer
    let mut renderer = block_on(grafo::Renderer::new(
        window.clone(),
        physical_size,
        scale_factor,
    ));

    // Define a simple rectangle shape
    let rect = Shape::rect(
        [(100.0, 100.0), (300.0, 200.0)],
        Color::rgb(0, 128, 255), // Blue fill
        Stroke::new(2.0, Color::BLACK), // Black stroke with width 2.0
    );
    renderer.add_shape(rect, None);

    // Start the event loop
    event_loop.run(move |event, event_loop_window_target| match event {
        Event::WindowEvent {
            ref event,
            window_id,
        } if window_id == window.id() => match event {
            WindowEvent::CloseRequested => event_loop_window_target.exit(),
            WindowEvent::Resized(physical_size) => {
                let new_size = (physical_size.width, physical_size.height);
                renderer.resize(new_size);
                window.request_redraw();
            }
            WindowEvent::RedrawRequested => {
                match renderer.render() {
                    Ok(_) => {
                        renderer.clear_draw_queue();
                    }
                    Err(wgpu::SurfaceError::Lost) => renderer.resize(renderer.size()),
                    Err(wgpu::SurfaceError::OutOfMemory) => event_loop_window_target.exit(),
                    Err(e) => eprintln!("{:?}", e),
                }
            }
            _ => {}
        },
        _ => {}
    });
}
```

## Examples

For a detailed example showcasing advanced features like hierarchical clipping, 
image rendering, and text rendering, please refer to the 
[examples](https://github.com/antouhou/grafo/tree/main/examples) directory in the repository.

## Documentation

[Documentation is available on docs.rs](https://docs.rs/grafo/).

## Contributing

Everyone is welcome to contribute in any way or form! For further details, please read [CONTRIBUTING.md](./CONTRIBUTING.md).

## Authors
- [Anton Suprunchuk](https://github.com/antouhou) - [Website](https://antouhou.com)

Also, see the list of contributors who participated in this project.

## License

This project is licensed under the MIT License - see the
[LICENSE.md](./LICENSE.md) file for details