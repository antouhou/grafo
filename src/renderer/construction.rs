#[cfg(feature = "render_metrics")]
use super::metrics::RenderLoopMetricsTracker;
use super::plan::Planner;
use super::{Renderer, RendererContext, Viewport, WgpuBackend, WgpuContext, DEFAULT_FRINGE_WIDTH};
use crate::TextureManager;
use ahash::{HashMap, HashMapExt};
use std::sync::{Arc, RwLock};
#[cfg(feature = "render_metrics")]
use std::time::Duration;
use tracing::{error, info, warn};
use wgpu::{
    CompositeAlphaMode, DownlevelFlags, InstanceDescriptor, SurfaceConfiguration, SurfaceTarget,
};

fn pick_surface_format(surface_formats: &[wgpu::TextureFormat]) -> Option<wgpu::TextureFormat> {
    const PREFERRED_SURFACE_FORMATS: [wgpu::TextureFormat; 4] = [
        wgpu::TextureFormat::Bgra8UnormSrgb,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        wgpu::TextureFormat::Bgra8Unorm,
        wgpu::TextureFormat::Rgba8Unorm,
    ];

    PREFERRED_SURFACE_FORMATS
        .into_iter()
        .find(|surface_format| surface_formats.contains(surface_format))
        .or_else(|| surface_formats.first().copied())
}

fn pick_alpha_mode(alpha_modes: &[CompositeAlphaMode], transparent: bool) -> CompositeAlphaMode {
    if transparent && alpha_modes.contains(&CompositeAlphaMode::PreMultiplied) {
        info!("Using PreMultiplied alpha mode for transparency");
        CompositeAlphaMode::PreMultiplied
    } else if transparent && alpha_modes.contains(&CompositeAlphaMode::PostMultiplied) {
        info!("Using PostMultiplied alpha mode for transparency");
        CompositeAlphaMode::PostMultiplied
    } else {
        if transparent {
            warn!(
                "Transparency requested but no suitable alpha mode available, falling back to the surface default"
            );
        }

        alpha_modes
            .iter()
            .copied()
            .find(|alpha_mode| matches!(alpha_mode, CompositeAlphaMode::Opaque))
            .unwrap_or_else(|| {
                alpha_modes
                    .first()
                    .copied()
                    .unwrap_or(CompositeAlphaMode::Opaque)
            })
    }
}

/// Errors from creating a [`RendererContext`] or [`Renderer`].
#[derive(Debug, thiserror::Error)]
pub enum RendererCreationError {
    /// The `scale_factor` is not finite and positive.
    #[error("Invalid scale factor: {0} (must be finite and > 0.0)")]
    InvalidScaleFactor(f64),
    /// No suitable GPU adapter was found.
    #[error("No suitable GPU adapter available: {0}")]
    AdapterNotAvailable(#[from] wgpu::RequestAdapterError),
    /// The GPU device could not be created.
    #[error("GPU device creation failed: {0}")]
    DeviceCreationFailed(#[from] wgpu::RequestDeviceError),
    /// The window target could not be converted into a WGPU surface.
    #[error("Surface creation failed: {0}")]
    SurfaceCreationFailed(#[from] wgpu::CreateSurfaceError),
    /// The context's adapter cannot render to the provided surface.
    #[error("The renderer context adapter does not support the provided surface")]
    UnsupportedSurface,
}

impl RendererContext {
    /// Creates GPU resources that can be shared by any number of independent renderers.
    ///
    /// The context has no surface. A renderer created from it validates and
    /// configures its own surface, so windows can be added later without rebuilding the device.
    pub async fn try_new() -> Result<Self, RendererCreationError> {
        let instance = Arc::new(wgpu::Instance::new(&InstanceDescriptor::default()));
        let adapter = Arc::new(
            instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                })
                .await?,
        );

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                #[cfg(feature = "performance_measurement")]
                required_features: wgpu::Features::TIMESTAMP_QUERY
                    | wgpu::Features::DEPTH32FLOAT_STENCIL8,
                #[cfg(not(feature = "performance_measurement"))]
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: Default::default(),
                trace: Default::default(),
            })
            .await?;
        device.on_uncaptured_error(Box::new(|error| {
            error!(%error, "WGPU error");
        }));
        let device = Arc::new(device);
        let queue = Arc::new(queue);

        Ok(Self {
            loaded_shapes: Arc::new(RwLock::new(HashMap::new())),
            gpu: Arc::new(WgpuContext {
                instance,
                supports_base_vertex: adapter
                    .get_downlevel_capabilities()
                    .flags
                    .contains(DownlevelFlags::BASE_VERTEX),
                adapter,
                texture_manager: TextureManager::new(device.clone(), queue.clone()),
                device,
                queue,
            }),
        })
    }

    /// Creates a shared GPU context, panicking when no compatible device is available.
    pub async fn new() -> Self {
        Self::try_new()
            .await
            .expect("Failed to create renderer context")
    }
}

impl<'a> Renderer<'a> {
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
        context: RendererContext,
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
        context: RendererContext,
        window: impl Into<SurfaceTarget<'static>>,
        physical_size: (u32, u32),
        scale_factor: f64,
        vsync: bool,
        transparent: bool,
        msaa_samples: u32,
    ) -> Result<Self, RendererCreationError> {
        let surface = context.gpu.instance.create_surface(window)?;

        let surface_caps = surface.get_capabilities(&context.gpu.adapter);
        let swapchain_format = pick_surface_format(&surface_caps.formats)
            .ok_or(RendererCreationError::UnsupportedSurface)?;
        let alpha_mode = pick_alpha_mode(&surface_caps.alpha_modes, transparent);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            format: swapchain_format,
            width: physical_size.0,
            height: physical_size.1,
            present_mode: if vsync {
                wgpu::PresentMode::AutoVsync
            } else {
                wgpu::PresentMode::AutoNoVsync
            },
            desired_maximum_frame_latency: 2,
            alpha_mode,
            view_formats: vec![],
        };
        surface.configure(&context.gpu.device, &config);

        let msaa_sample_count = WgpuBackend::normalize_msaa_sample_count(msaa_samples);

        Self::build_from_context(
            context,
            Some(surface),
            config,
            physical_size,
            scale_factor,
            msaa_sample_count,
        )
    }

    fn build_from_context(
        context: RendererContext,
        surface: Option<wgpu::Surface<'a>>,
        config: SurfaceConfiguration,
        physical_size: (u32, u32),
        scale_factor: f64,
        msaa_sample_count: u32,
    ) -> Result<Self, RendererCreationError> {
        if !scale_factor.is_finite() || scale_factor <= 0.0 {
            return Err(RendererCreationError::InvalidScaleFactor(scale_factor));
        }
        let planner = Planner::new(
            context.loaded_shapes,
            context.gpu.device.limits().max_texture_dimension_2d,
            DEFAULT_FRINGE_WIDTH,
        );
        let backend = WgpuBackend::new(
            context.gpu,
            config,
            physical_size,
            scale_factor,
            msaa_sample_count,
        );
        Ok(Self {
            planner,
            backend,
            surface,
            viewport: Viewport {
                physical_size,
                scale_factor,
            },
            #[cfg(feature = "render_metrics")]
            last_planning_time: Duration::ZERO,
            #[cfg(feature = "render_metrics")]
            render_loop_metrics_tracker: RenderLoopMetricsTracker::default(),
        })
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
    ) -> Result<Self, RendererCreationError> {
        Self::try_new_headless_with_context(
            RendererContext::try_new().await?,
            physical_size,
            scale_factor,
        )
    }

    /// Creates a headless renderer that shares an existing [`RendererContext`].
    pub fn try_new_headless_with_context(
        context: RendererContext,
        physical_size: (u32, u32),
        scale_factor: f64,
    ) -> Result<Self, RendererCreationError> {
        let swapchain_format = wgpu::TextureFormat::Bgra8UnormSrgb;

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            format: swapchain_format,
            width: physical_size.0,
            height: physical_size.1,
            present_mode: wgpu::PresentMode::AutoVsync,
            desired_maximum_frame_latency: 2,
            alpha_mode: CompositeAlphaMode::Opaque,
            view_formats: vec![],
        };

        let msaa_sample_count = 1;

        Self::build_from_context(
            context,
            None,
            config,
            physical_size,
            scale_factor,
            msaa_sample_count,
        )
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
