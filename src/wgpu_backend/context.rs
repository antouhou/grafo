use super::texture_manager::WgpuTextureManager;
use super::{WgpuBackend, WgpuContext};
use std::sync::Arc;
use tracing::{error, info, warn};
use wgpu::{CompositeAlphaMode, DownlevelFlags, InstanceDescriptor, Surface, SurfaceTarget};

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

/// Errors from creating a WGPU context or output.
#[derive(Debug, thiserror::Error)]
pub enum BackendCreationError {
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

impl WgpuContext {
    /// Creates an unconfigured surface using this context's WGPU instance.
    /// The renderer configures it with its current size and presentation settings.
    pub fn create_surface<'surface>(
        &self,
        target: impl Into<SurfaceTarget<'surface>>,
    ) -> Result<Surface<'surface>, BackendCreationError> {
        Ok(self.instance.create_surface(target)?)
    }

    /// Creates GPU resources that can be shared by any number of independent renderers.
    ///
    /// The context has no surface. A renderer created from it validates and
    /// configures its own surface, so windows can be added later without rebuilding the device.
    pub async fn try_new() -> Result<Self, BackendCreationError> {
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
            instance,
            supports_base_vertex: adapter
                .get_downlevel_capabilities()
                .flags
                .contains(DownlevelFlags::BASE_VERTEX),
            adapter,
            texture_manager: WgpuTextureManager::new(device.clone(), queue.clone()),
            device,
            queue,
        })
    }
}

impl WgpuBackend {
    pub fn for_window(
        context: Arc<WgpuContext>,
        window: impl Into<SurfaceTarget<'static>>,
        physical_size: (u32, u32),
        scale_factor: f64,
        vsync: bool,
        transparent: bool,
        msaa_samples: u32,
    ) -> Result<(Self, Option<wgpu::Surface<'static>>), BackendCreationError> {
        validate_scale_factor(scale_factor)?;
        let surface = context.create_surface(window)?;

        let surface_caps = surface.get_capabilities(&context.adapter);
        let swapchain_format = pick_surface_format(&surface_caps.formats)
            .ok_or(BackendCreationError::UnsupportedSurface)?;
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
        surface.configure(&context.device, &config);

        let msaa_sample_count = WgpuBackend::normalize_msaa_sample_count(msaa_samples);

        let backend = Self::new(
            context,
            config,
            physical_size,
            scale_factor,
            msaa_sample_count,
        );
        Ok((backend, Some(surface)))
    }

    pub fn headless(
        context: Arc<WgpuContext>,
        physical_size: (u32, u32),
        scale_factor: f64,
    ) -> Result<Self, BackendCreationError> {
        validate_scale_factor(scale_factor)?;
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

        Ok(Self::new(
            context,
            config,
            physical_size,
            scale_factor,
            msaa_sample_count,
        ))
    }
}

fn validate_scale_factor(scale_factor: f64) -> Result<(), BackendCreationError> {
    if !scale_factor.is_finite() || scale_factor <= 0.0 {
        return Err(BackendCreationError::InvalidScaleFactor(scale_factor));
    }
    Ok(())
}
