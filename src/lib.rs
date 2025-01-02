//! # Grafo
//!
//! Grafo is a GPU-accelerated rendering library built in Rust. It leverages the
//! power of the [`wgpu`](https://crates.io/crates/wgpu) crate for rendering shapes, images, and
//! text efficiently. Designed with flexibility and ease of use in mind, Grafo integrates seamlessly
//! with other modules to provide a comprehensive rendering solution for Rust applications.
//!
//! ## Features
//!
//! - **Shape Rendering**: Create and render complex shapes
//! - **Image Rendering**: Render images with support for clipping to shapes.
//! - **Text Rendering**: Render text with customizable layout, alignment, and styling using the
//!   [`glyphon`](https://crates.io/crates/glyphon) crate.
//! - **Stencil Operations**: Advanced stencil operations for clipping and masking.
//! - **Performance Optimization**: Utilizes parallel processing with `rayon` to optimize rendering
//!   performance.
//!
//! ## Getting Started
//!
//! To get started with Grafo, add it as a dependency in your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! grafo = "0.1.0" # Replace with the actual version
//! winit = "0.27"   # For window creation and event handling
//! image = "0.24"   # For image processing
//! env_logger = "0.10" # For logging
//! log = "0.4"      # For logging
//! ```
//!
//! ### Basic Usage
//!
//! Below is a simple example demonstrating how to initialize the `Renderer`, add shapes and text,
//! and render a frame using `winit`. For a more comprehensive example, refer to the
//! [`examples`](https://github.com/antouhou/grafo/tree/main/examples) folder in the repository.
//!
//! ```rust,no_run
//! use futures::executor::block_on;
//! use grafo::{BorderRadii, Shape};
//! use grafo::{Color, Stroke};
//! use std::sync::Arc;
//! use winit::event::{Event, WindowEvent};
//! use winit::event_loop::EventLoop;
//! use winit::window::WindowBuilder;
//!
//!     env_logger::init();
//!     let event_loop = EventLoop::new().expect("To create the event loop");
//!     let window = Arc::new(WindowBuilder::new().build(&event_loop).unwrap());
//!
//!     let window_size = window.inner_size();
//!     let scale_factor = window.scale_factor();
//!     let physical_size = (window_size.width, window_size.height);
//!
//!     // Initialize the renderer
//!     let mut renderer = block_on(grafo::Renderer::new(
//!         window.clone(),
//!         physical_size,
//!         scale_factor,
//!     ));
//!
//!     // Define a simple rectangle shape
//!     let rect = Shape::rect(
//!         [(100.0, 100.0), (300.0, 200.0)],
//!         Color::rgb(0, 128, 255), // Blue fill
//!         Stroke::new(2.0, Color::BLACK), // Black stroke with width 2.0
//!     );
//!     renderer.add_shape(rect, None);
//!
//!     // Start the event loop
//!     let _ = event_loop.run(move |event, event_loop_window_target| match event {
//!         Event::WindowEvent {
//!             ref event,
//!             window_id,
//!         } if window_id == window.id() => match event {
//!             WindowEvent::CloseRequested => event_loop_window_target.exit(),
//!             WindowEvent::Resized(physical_size) => {
//!                 let new_size = (physical_size.width, physical_size.height);
//!                 renderer.resize(new_size);
//!                 window.request_redraw();
//!             }
//!             WindowEvent::RedrawRequested => {
//!                 match renderer.render() {
//!                     Ok(_) => {
//!                         renderer.clear_draw_queue();
//!                     }
//!                     Err(wgpu::SurfaceError::Lost) => renderer.resize(renderer.size()),
//!                     Err(wgpu::SurfaceError::OutOfMemory) => event_loop_window_target.exit(),
//!                     Err(e) => eprintln!("{:?}", e),
//!                 }
//!             }
//!             _ => {}
//!         },
//!         _ => {}
//!     });
//! ```
//!
//! ## Examples
//!
//! For a detailed example showcasing advanced features like hierarchical clipping, image rendering,
//! and text rendering, please refer to the [`examples`](https://github.com/your-repo/grafo/tree/main/examples)
//! directory in the repository.
//!

pub use lyon;
pub use wgpu;
pub use glyphon;
pub use glyphon::fontdb;

mod color;
mod debug_tools;
mod id;
mod image_draw_data;
mod pipeline;
mod renderer;
mod stroke;
mod util;
mod vertex;

mod shape;
mod text;

pub use color::Color;
pub use renderer::MathRect;
pub use renderer::Renderer;
pub use shape::*;
pub use stroke::Stroke;
pub use text::*;
