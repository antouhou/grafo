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
//! * Path rendering with cached tessellation
//! * Hierarchical path clipping
//! * Per-instance 3D and perspective transforms
//! * Solid fills and linear, radial, and conic gradients
//! * Custom WGSL shader effects on shape masks, groups, and backdrops
//! * Geometry-based antialiasing and MSAA
//!
//! [Install Grafo from crates.io](https://crates.io/crates/grafo) or read the
//! [API documentation](https://docs.rs/grafo/).
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

pub mod commands;
pub mod core;
mod pipeline;
mod renderer;
mod texture_manager;

pub use crate::core::*;
pub use commands::RenderPlan;
pub use renderer::{
    types::{DrawCommandError, GeometryBufferError, RenderError},
    EffectError, EffectShaderError, ReadbackError, RenderBackend, Renderer, RendererContext,
    RendererCreationError, WgpuBackend,
};
pub use texture_manager::TextureManager;

#[cfg(feature = "render_metrics")]
pub use renderer::metrics::PhaseTimings;
#[cfg(feature = "render_metrics")]
pub use renderer::metrics::PipelineSwitchCounts;
#[cfg(feature = "render_metrics")]
pub use renderer::metrics::ShapeEffectCacheMetrics;
