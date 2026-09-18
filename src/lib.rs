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
//! Grafo is [available on crates.io](https://crates.io/crates/grafo), and
//! [API Documentation is available on docs.rs](https://docs.rs/grafo/).
//!
//! ## Getting started
//!
//! Add the following to your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! grafo = "0.19"
//! winit = "0.30"
//! futures = "0.3"
//! env_logger = "0.11"
//! ```
//!
//! ### Basic usage
//!
//! This is `examples/basic.rs`. It queues two rectangles once and renders them on each
//! redraw. Run it with `cargo run --example basic`.
//!
#![doc = concat!("```rust,no_run\n", include_str!("../examples/basic.rs"), "\n```\n")]
//!
//! ## Examples
//!
//! The [examples](https://github.com/antouhou/grafo/tree/main/examples) directory includes
//! hierarchical clipping, texture layers, transforms, and shader effects.

pub use lyon;
pub use wgpu;

mod color;
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
pub use effect::{
    BackdropCaptureArea, BackdropEffectConfig, EffectError, EffectShaderError, ShapeEffectConfig,
};
pub use gradient::errors::GradientError;
pub use gradient::types::{
    ColorInterpolation, ConicGradientDesc, Fill, Gradient, GradientColor, GradientCommonDesc,
    GradientDesc, GradientStop, GradientStopOffset, GradientStopPositions, GradientUnits,
    HueComponent, HueInterpolationMethod, LinearGradientDesc, LinearGradientLine,
    RadialGradientDesc, RadialGradientShape, RadialGradientSize, SpreadMode,
};
pub use renderer::{
    types::{DrawCommandError, GeometryBufferError, RenderError},
    MathRect, ReadbackError, Renderer, RendererContext, RendererCreationError, ShapeOverflow,
    TextureLayer,
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
