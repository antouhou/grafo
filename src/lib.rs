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
//! grafo = "0.20"
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

pub use crate::core::*;
pub use commands::RenderPlan;
pub use lyon;
pub use render_backend::{RenderBackend, TextureManager};
pub use scene::SceneError;
use std::sync::Arc;
pub use wgpu;
#[cfg(feature = "render_metrics")]
pub use wgpu_backend::metrics::{PhaseTimings, PipelineSwitchCounts, ShapeEffectCacheMetrics};
pub use wgpu_backend::texture_manager::WgpuTextureManager;
use wgpu_backend::WgpuContext;
pub use wgpu_backend::{
    BackendCreationError as RendererCreationError, EffectResourceError, EffectShaderError,
    GeometryBufferError, ReadbackError, WgpuBackend, WgpuBackendError,
};

pub mod commands;
pub mod core;
pub(crate) mod planner;
pub mod render_backend;
pub mod renderer;
pub mod scene;
pub mod wgpu_backend;
mod wgpu_renderer;

/// Coordinates scene construction, planning and execution, using WGPU by default.
pub type Renderer<'surface, B = WgpuBackend> = renderer::Renderer<'surface, B>;
/// Shared CPU shape storage and backend context, using WGPU by default.
pub type RendererContext<C = Arc<WgpuContext>> = renderer::RendererContext<C>;
/// Scene insertion or backend resource preparation failed.
pub type DrawCommandError<E = WgpuBackendError> = renderer::DrawCommandError<E>;
/// Scene attachment or backend effect validation failed.
pub type EffectError<E = WgpuBackendError> = renderer::EffectError<E>;
