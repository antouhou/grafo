//! # Grafo
//!
//! [![Grafo crate](https://img.shields.io/crates/v/grafo.svg)](https://crates.io/crates/grafo)
//! [![Grafo documentation](https://docs.rs/grafo/badge.svg)](https://docs.rs/grafo)
//! [![Build and test](https://github.com/antouhou/grafo/actions/workflows/rust.yml/badge.svg?branch=main)](https://github.com/antouhou/grafo/actions)
//!
//! Grafo is a GPU-accelerated rendering library for Rust. It is a one-stop solution in case
//! you need a quick and simple way to render shapes (with optional texture layers) in your application. It
//! supports features such as masking, clipping, and font loading and rendering.
//!
//! The library is designed for flexibility and ease of use, making it suitable for a wide
//! range of applications, from simple graphical interfaces to complex rendering engines.
//!
//! ## Features
//!
//! * Shape Rendering: Create and render complex vector shapes.
//! * Shape Texturing: Apply up to two texture layers per shape with hierarchical clipping.
//! * Stencil Operations: Advanced stencil operations for clipping and masking.
//!
//! Grafo [available on crates.io](https://crates.io/crates/grafo), and
//! [API Documentation is available on docs.rs](https://docs.rs/grafo/).
//!
//! ## Getting Started
//!
//! Add the following to your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! grafo = "0.1.0"
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
//! [examples](https://github.com/antouhou/grafo/tree/main/examples) folder in the repository.
//!
//! ```rust,no_run
//! use futures::executor::block_on;
//! use grafo::{BorderRadii, Shape};
//! use grafo::{Color, Stroke};
//! use std::sync::Arc;
//! use winit::application::ApplicationHandler;
//! use winit::event::WindowEvent;
//! use winit::event_loop::{ActiveEventLoop, EventLoop};
//! use winit::window::{Window, WindowId};
//!
//! #[derive(Default)]
//! struct App<'a> {
//!     window: Option<Arc<Window>>,
//!     renderer: Option<grafo::Renderer<'a>>,
//! }
//!
//! impl<'a> ApplicationHandler for App<'a> {
//!     fn resumed(&mut self, event_loop: &ActiveEventLoop) {
//!         let window = Arc::new(
//!             event_loop
//!                 .create_window(Window::default_attributes())
//!                 .unwrap(),
//!         );
//!
//!         let window_size = window.inner_size();
//!         let scale_factor = window.scale_factor();
//!         let physical_size = (window_size.width, window_size.height);
//!
//!         // Initialize the renderer
//!         let mut renderer = block_on(grafo::Renderer::new(
//!             window.clone(),
//!             physical_size,
//!             scale_factor,
//!             true,  // vsync
//!             false, // transparent
//!         ));
//!
//!         // Define a simple rectangle shape
//!         let rect = Shape::rect(
//!             [(0.0, 0.0), (200.0, 100.0)],
//!             Color::rgb(0, 128, 255), // Blue fill
//!             Stroke::new(2.0, Color::BLACK), // Black stroke with width 2.0
//!         );
//!         let rect_id = renderer.add_shape(rect, None, None);
//!         renderer.set_shape_transform(rect_id, grafo::TransformInstance::identity());
//!
//!         self.window = Some(window);
//!         self.renderer = Some(renderer);
//!     }
//!
//!     fn window_event(
//!         &mut self,
//!         event_loop: &ActiveEventLoop,
//!         window_id: WindowId,
//!         event: WindowEvent,
//!     ) {
//!         if let Some(ref mut renderer) = self.renderer {
//!             match event {
//!                 WindowEvent::CloseRequested => event_loop.exit(),
//!                 WindowEvent::Resized(physical_size) => {
//!                     let new_size = (physical_size.width, physical_size.height);
//!                     renderer.resize(new_size);
//!                     if let Some(window) = &self.window {
//!                         window.request_redraw();
//!                     }
//!                 }
//!                 WindowEvent::RedrawRequested => {
//!                     match renderer.render() {
//!                         Ok(_) => {
//!                             renderer.clear_draw_queue();
//!                         }
//!                         Err(wgpu::SurfaceError::Lost) => renderer.resize(renderer.size()),
//!                         Err(wgpu::SurfaceError::OutOfMemory) => event_loop.exit(),
//!                         Err(e) => eprintln!("{:?}", e),
//!                     }
//!                 }
//!                 _ => {}
//!             }
//!         }
//!     }
//! }
//!
//!     env_logger::init();
//!     let event_loop = EventLoop::new().expect("to start an event loop");
//!     let mut app = App::default();
//!     event_loop.run_app(&mut app).unwrap();
//! ```
//!
//! ## Examples
//!
//! For a detailed example showcasing advanced features like hierarchical clipping,
//! multi-layer texturing, please refer to the
//! [examples](https://github.com/antouhou/grafo/tree/main/examples) directory in the repository.

pub use lyon;
pub use wgpu;

mod color;
mod debug_tools;
mod pipeline;
mod renderer;
mod stroke;
mod util;
mod vertex;

mod cache;
mod shape;
mod texture_manager;

pub use color::Color;
pub use renderer::MathRect;
pub use renderer::Renderer;
pub use renderer::TextureLayer;
pub use shape::*;
pub use stroke::Stroke;
pub use texture_manager::premultiply_rgba8_srgb_inplace;
pub use texture_manager::TextureManager;
pub use vertex::InstanceTransform as TransformInstance;
