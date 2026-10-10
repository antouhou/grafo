//! WGPU constructors for renderers, contexts, and surfaces.

use crate::render_backend::render_target::Surface;
use crate::renderer::{Renderer, RendererContext};
use crate::scene::SceneContext;
use crate::wgpu_backend::{BackendCreationError, WgpuBackend, WgpuContext, WgpuSurface};
use std::sync::Arc;
use wgpu::SurfaceTarget;

impl RendererContext<Arc<WgpuContext>> {
    /// Creates a WGPU context and shared CPU shape storage.
    pub async fn try_new() -> Result<Self, BackendCreationError> {
        Ok(Self {
            backend: Arc::new(WgpuContext::try_new().await?),
            scene: SceneContext::default(),
        })
    }

    pub async fn new() -> Self {
        Self::try_new()
            .await
            .expect("Failed to create renderer context")
    }
}

impl Surface<WgpuSurface> {
    /// Creates a surface from a window or another platform handle provider.
    /// Render to it with a renderer that uses the same context.
    pub fn new(
        context: &RendererContext<Arc<WgpuContext>>,
        target: impl Into<SurfaceTarget<'static>>,
        physical_size: (u32, u32),
        vsync: bool,
        transparent: bool,
    ) -> Result<Self, BackendCreationError> {
        context
            .backend
            .create_surface(target, physical_size, vsync, transparent)
    }
}

impl Renderer<WgpuBackend> {
    /// Creates a renderer with a new WGPU context.
    pub async fn new(physical_size: (u32, u32), scale_factor: f64, msaa_samples: u32) -> Self {
        Self::try_new(physical_size, scale_factor, msaa_samples)
            .await
            .expect("Failed to create renderer")
    }

    pub async fn try_new(
        physical_size: (u32, u32),
        scale_factor: f64,
        msaa_samples: u32,
    ) -> Result<Self, BackendCreationError> {
        Self::try_new_with_context(
            RendererContext::try_new().await?,
            physical_size,
            scale_factor,
            msaa_samples,
        )
    }

    /// Creates a renderer that shares GPU resources and loaded shapes through `context`.
    pub fn new_with_context(
        context: RendererContext<Arc<WgpuContext>>,
        physical_size: (u32, u32),
        scale_factor: f64,
        msaa_samples: u32,
    ) -> Self {
        Self::try_new_with_context(context, physical_size, scale_factor, msaa_samples)
            .expect("Failed to create renderer")
    }

    pub fn try_new_with_context(
        context: RendererContext<Arc<WgpuContext>>,
        physical_size: (u32, u32),
        scale_factor: f64,
        msaa_samples: u32,
    ) -> Result<Self, BackendCreationError> {
        let RendererContext { backend, scene } = context;
        let backend = WgpuBackend::new(backend, physical_size, scale_factor, msaa_samples)?;
        Ok(Self::from_parts(backend, scene))
    }
}
