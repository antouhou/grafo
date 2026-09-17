//! # Grafo
//!
//! [![Grafo crate](https://img.shields.io/crates/v/grafo.svg)](https://crates.io/crates/grafo)
//! [![Grafo documentation](https://docs.rs/grafo/badge.svg)](https://docs.rs/grafo)
//! [![Build and test](https://github.com/antouhou/grafo/actions/workflows/rust.yml/badge.svg?branch=main)](https://github.com/antouhou/grafo/actions)
//!
//! Grafo is a GPU-accelerated vector graphics library for Rust.
//!
//! ## Features
//!
//! * Path rendering with cached tessellation.
//! * Hierarchical path clipping.
//! * Per-instance 3D and perspective transforms.
//! * Solid fills and linear, radial, and conic gradients.
//! * Custom WGSL shader effects on shape masks, groups, and backdrops.
//! * Geometry-based antialiasing and MSAA.
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
//! grafo = "0.19"
//! winit = "0.30"   # For window creation and event handling
//! image = "0.24"   # For image processing
//! env_logger = "0.10" # For logging
//! log = "0.4"      # For logging
//! ```
//!
//! ### Basic Usage
//!
//! This example initializes the `Renderer`, adds a rectangle, and renders a frame using `winit`.
//! For runnable examples, see the
//! [examples](https://github.com/antouhou/grafo/tree/main/examples) folder in the repository.
//!
//! ```rust,no_run
//! use futures::executor::block_on;
//! use grafo::{BorderRadii, Shape};
//! use grafo::{Color, ShapeDrawCommandOptions, Stroke};
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
//!             1,     // msaa_samples (1 = off)
//!         ));
//!
//!         // Define a simple rectangle shape
//!         let rect = Shape::rect(
//!             [(0.0, 0.0), (200.0, 100.0)],
//!             Stroke::new(2.0, Color::BLACK), // Black stroke with width 2.0
//!         );
//!         renderer
//!             .add_shape(
//!                 rect,
//!                 None,
//!                 None,
//!                 ShapeDrawCommandOptions::new()
//!                     .color(Color::rgb(0, 128, 255))
//!                     .transform(grafo::TransformInstance::identity())
//!                     .clips_children(true),
//!             )
//!             .expect("to add shape to the renderer");
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
//!                         Err(
//!                             wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated,
//!                         ) => renderer.resize(renderer.size()),
//!                         Err(wgpu::SurfaceError::Timeout) => {
//!                             // The window is not visible yet (still appearing, minimized, or
//!                             // fully covered). Ask for another redraw instead of dropping the
//!                             // frame — winit does not request one when the window becomes
//!                             // visible again.
//!                             if let Some(window) = &self.window {
//!                                 window.request_redraw();
//!                             }
//!                         }
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
mod effect;
pub mod gradient;
mod pipeline;
mod renderer;
mod stroke;
mod util;
mod vertex;

mod cache;
mod shape;
mod texture_manager;

pub use color::Color;
pub use effect::{BackdropCaptureArea, BackdropEffectConfig, EffectError, ShapeEffectConfig};
pub use gradient::errors::GradientError;
pub use gradient::types::{
    ColorInterpolation, ConicGradientDesc, Fill, Gradient, GradientColor, GradientCommonDesc,
    GradientDesc, GradientStop, GradientStopOffset, GradientStopPositions, GradientSupport,
    GradientUnits, HueComponent, HueInterpolationMethod, LinearGradientDesc, LinearGradientLine,
    RadialGradientDesc, RadialGradientShape, RadialGradientSize, SpreadMode,
};
pub use renderer::{
    types::DrawCommandError, MathRect, Renderer, RendererContext, RendererCreationError,
    ShapeOverflow, TextureLayer,
};
pub use shape::*;
pub use stroke::Stroke;
pub use texture_manager::{premultiply_rgba8_srgb_inplace, TextureManager};
pub use vertex::InstanceTransform as TransformInstance;

#[cfg(feature = "render_metrics")]
pub use renderer::metrics::PhaseTimings;
#[cfg(feature = "render_metrics")]
pub use renderer::metrics::PipelineSwitchCounts;
#[cfg(feature = "render_metrics")]
pub use renderer::metrics::ShapeEffectCacheMetrics;
