//! Assembles the generic coordinator with WGPU resources.

use crate::renderer::{Renderer, RendererContext};
use crate::scene::SceneContext;
use crate::wgpu_backend::{BackendCreationError, WgpuBackend, WgpuContext};
use std::sync::Arc;
use wgpu::SurfaceTarget;

impl RendererContext<Arc<WgpuContext>> {
    /// Creates shared backend resources and CPU shape storage without an output surface.
    pub async fn try_new() -> Result<Self, BackendCreationError> {
        Ok(Self::from_parts(
            Arc::new(WgpuContext::try_new().await?),
            SceneContext::default(),
        ))
    }
    pub async fn new() -> Self {
        Self::try_new()
            .await
            .expect("Failed to create renderer context")
    }
}

impl<'a> Renderer<'a, WgpuBackend> {
    pub async fn new(
        window: impl Into<SurfaceTarget<'static>>,
        physical_size: (u32, u32),
        scale_factor: f64,
        vsync: bool,
        transparent: bool,
        msaa_samples: u32,
    ) -> Self {
        Self::new_with_context(
            RendererContext::new().await,
            window,
            physical_size,
            scale_factor,
            vsync,
            transparent,
            msaa_samples,
        )
    }

    /// Creates a renderer with an existing [`RendererContext`].
    ///
    /// Each renderer created through this method owns a distinct surface and draw queue while
    /// sharing the context's WGPU device, queue, and texture storage.
    pub fn new_with_context(
        context: RendererContext<Arc<WgpuContext>>,
        window: impl Into<SurfaceTarget<'static>>,
        physical_size: (u32, u32),
        scale_factor: f64,
        vsync: bool,
        transparent: bool,
        msaa_samples: u32,
    ) -> Self {
        Self::try_new_with_context(
            context,
            window,
            physical_size,
            scale_factor,
            vsync,
            transparent,
            msaa_samples,
        )
        .expect("Failed to build renderer from context")
    }

    /// Fallible version of [`Self::new_with_context`].
    pub fn try_new_with_context(
        context: RendererContext<Arc<WgpuContext>>,
        window: impl Into<SurfaceTarget<'static>>,
        physical_size: (u32, u32),
        scale_factor: f64,
        vsync: bool,
        transparent: bool,
        msaa_samples: u32,
    ) -> Result<Self, BackendCreationError> {
        let (backend_context, scene_context) = context.into_parts();
        let (backend, surface) = WgpuBackend::for_window(
            backend_context,
            window,
            physical_size,
            scale_factor,
            vsync,
            transparent,
            msaa_samples,
        )?;
        Ok(Self::from_backend(backend, surface, scene_context))
    }

    pub async fn new_transparent(
        window: impl Into<SurfaceTarget<'static>>,
        physical_size: (u32, u32),
        scale_factor: f64,
        vsync: bool,
        msaa_samples: u32,
    ) -> Self {
        Self::new(
            window,
            physical_size,
            scale_factor,
            vsync,
            true,
            msaa_samples,
        )
        .await
    }

    /// Creates a headless renderer without a window surface.
    ///
    /// Use `render_to_buffer()` or `render_to_argb32()` to read back rendered
    /// pixels. Calling `render()` on a headless renderer will panic.
    ///
    /// Returns an error if no suitable GPU adapter is available, the device
    /// cannot be created, or the `scale_factor` is invalid.
    pub async fn try_new_headless(
        physical_size: (u32, u32),
        scale_factor: f64,
    ) -> Result<Self, BackendCreationError> {
        Self::try_new_headless_with_context(
            RendererContext::try_new().await?,
            physical_size,
            scale_factor,
        )
    }

    /// Creates a headless renderer that shares an existing [`RendererContext`].
    pub fn try_new_headless_with_context(
        context: RendererContext<Arc<WgpuContext>>,
        physical_size: (u32, u32),
        scale_factor: f64,
    ) -> Result<Self, BackendCreationError> {
        let (backend_context, scene_context) = context.into_parts();
        let backend = WgpuBackend::headless(backend_context, physical_size, scale_factor)?;
        Ok(Self::from_backend(backend, None, scene_context))
    }

    /// Creates a headless renderer without a window surface.
    /// Panics if [`Self::try_new_headless`] returns an error.
    ///
    /// Use `render_to_buffer()` or `render_to_argb32()` to read back rendered
    /// pixels. Calling `render()` on a headless renderer will panic.
    ///
    /// Use [`Self::try_new_headless`] to handle creation errors.
    pub async fn new_headless(physical_size: (u32, u32), scale_factor: f64) -> Self {
        Self::try_new_headless(physical_size, scale_factor)
            .await
            .expect("Failed to create headless renderer")
    }
}
