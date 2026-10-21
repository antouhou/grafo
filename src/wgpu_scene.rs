//! WGPU constructors for scenes, contexts, and surfaces.

use crate::render_backend::render_target::Surface;
use crate::render_backend::texture_manager::TextureUpdatePublisher;
use crate::scene::{Scene, SceneContext};
use crate::scene_state::SceneStateContext;
use crate::wgpu_backend::{BackendCreationError, WgpuBackend, WgpuContext, WgpuSurface};
use std::sync::Arc;
use wgpu::SurfaceTarget;

impl SceneContext<Arc<WgpuContext>> {
    /// Creates a WGPU context and shared CPU shape storage.
    pub async fn try_new() -> Result<Self, BackendCreationError> {
        let texture_updates = TextureUpdatePublisher::default();
        Ok(Self {
            backend: Arc::new(WgpuContext::try_new(texture_updates.clone()).await?),
            state: SceneStateContext::new(texture_updates),
        })
    }

    pub async fn new() -> Self {
        Self::try_new()
            .await
            .expect("Failed to create scene context")
    }
}

impl Surface<WgpuSurface> {
    /// Creates a surface from a window or another platform handle provider.
    /// Render to it with a scene that uses the same context.
    pub fn new(
        context: &SceneContext<Arc<WgpuContext>>,
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

impl Scene<WgpuBackend> {
    /// Creates a scene with a new WGPU context.
    pub async fn new(physical_size: (u32, u32), scale_factor: f64, msaa_samples: u32) -> Self {
        Self::try_new(physical_size, scale_factor, msaa_samples)
            .await
            .expect("Failed to create scene")
    }

    pub async fn try_new(
        physical_size: (u32, u32),
        scale_factor: f64,
        msaa_samples: u32,
    ) -> Result<Self, BackendCreationError> {
        Self::try_new_with_context(
            SceneContext::try_new().await?,
            physical_size,
            scale_factor,
            msaa_samples,
        )
    }

    /// Creates a scene that shares GPU resources and loaded shapes through `context`.
    pub fn new_with_context(
        context: SceneContext<Arc<WgpuContext>>,
        physical_size: (u32, u32),
        scale_factor: f64,
        msaa_samples: u32,
    ) -> Self {
        Self::try_new_with_context(context, physical_size, scale_factor, msaa_samples)
            .expect("Failed to create scene")
    }

    pub fn try_new_with_context(
        context: SceneContext<Arc<WgpuContext>>,
        physical_size: (u32, u32),
        scale_factor: f64,
        msaa_samples: u32,
    ) -> Result<Self, BackendCreationError> {
        let SceneContext { backend, state } = context;
        let backend = WgpuBackend::new(backend, physical_size, scale_factor, msaa_samples)?;
        Ok(Self::from_parts(backend, state))
    }
}
