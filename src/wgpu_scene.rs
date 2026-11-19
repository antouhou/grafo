//! WGPU constructors for scenes, contexts, and surfaces.

use crate::render_backend::texture_manager::TextureUpdatePublisher;
use crate::scene::{Scene, SceneContext};
use crate::scene_state::SceneStateContext;
use crate::wgpu_backend::{BackendCreationError, WgpuBackend, WgpuContext};
use crate::Surface;
use std::sync::Arc;
use wgpu::wgt::WgpuHasDisplayHandle;
use wgpu::{Instance, InstanceDescriptor, Surface as WgpuSurface, SurfaceTarget};

impl SceneContext<Arc<WgpuContext>> {
    async fn from_instance(
        instance: Instance,
        surface: Option<&WgpuSurface<'_>>,
    ) -> Result<Self, BackendCreationError> {
        let texture_updates = TextureUpdatePublisher::default();
        let backend = WgpuContext::try_new(texture_updates.clone(), instance, surface).await?;
        Ok(Self {
            backend: Arc::new(backend),
            state: SceneStateContext::new(texture_updates),
        })
    }

    /// Creates a WGPU context without a surface.
    /// Use [`Self::try_new_with_surface`] to select an adapter for a window or canvas.
    pub async fn try_new() -> Result<Self, BackendCreationError> {
        Self::from_instance(Instance::default(), None).await
    }

    /// Creates a context and its first surface on an adapter that can present to the target.
    /// Pass the display that owns the target, such as `event_loop.owned_display_handle()` with winit.
    /// Set `transparent` when the native window is configured for compositor transparency.
    pub async fn try_new_with_surface(
        display: impl WgpuHasDisplayHandle,
        target: impl Into<SurfaceTarget<'static>>,
        physical_size: (u32, u32),
        vsync: bool,
        transparent: bool,
    ) -> Result<(Self, Surface), BackendCreationError> {
        display.display_handle()?;
        let descriptor = InstanceDescriptor::new_with_display_handle(Box::new(display));
        let instance = Instance::new(descriptor);
        let surface = instance.create_surface(target)?;
        let context = Self::from_instance(instance, Some(&surface)).await?;
        let surface = context
            .backend
            .wrap_surface(surface, physical_size, vsync, transparent)?;
        Ok((context, surface))
    }
}

impl Surface {
    /// Creates a surface for a window, canvas, or other supported target.
    /// Render to it with a scene that uses the same context.
    /// Set `transparent` when the native window is configured for compositor transparency.
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
    /// Creates a scene that shares GPU resources and loaded shapes through `context`.
    pub fn try_new(
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
