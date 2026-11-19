use super::retained_output::{OutputColorFormat, RetainedOutputResources, SurfacePresentation};
use super::texture_manager::WgpuTextureManager;
use super::{WgpuBackend, WgpuBackendError, WgpuContext, WgpuSurfaceState};
use crate::render_backend::render_target::Surface;
use crate::render_backend::texture_manager::TextureUpdatePublisher;
use crate::render_backend::RenderContext;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use tracing::{error, info, warn};
use wgpu::rwh::HandleError;
#[cfg(feature = "performance_measurement")]
use wgpu::Features;
use wgpu::{
    Adapter, Backends, CompositeAlphaMode, Device, DeviceDescriptor, DeviceType, DownlevelFlags,
    Instance, Limits, PowerPreference, Queue, RequestAdapterOptions, Surface as WgpuSurface,
    SurfaceCapabilities, SurfaceColorSpace, SurfaceColorSpaces, SurfaceTarget, TextureFormat,
    TextureUsages,
};

pub(super) fn pick_surface_format(capabilities: &SurfaceCapabilities) -> Option<TextureFormat> {
    const PREFERRED_SURFACE_FORMATS: [TextureFormat; 6] = [
        TextureFormat::Bgra8UnormSrgb,
        TextureFormat::Rgba8UnormSrgb,
        TextureFormat::Bgra8Unorm,
        TextureFormat::Rgba8Unorm,
        TextureFormat::Rgb10a2Unorm,
        TextureFormat::Rgba16Float,
    ];

    PREFERRED_SURFACE_FORMATS.into_iter().find(|&format| {
        capabilities
            .color_spaces(format)
            .contains(SurfaceColorSpaces::SRGB)
    })
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

/// Errors from creating a WGPU scene, context, or surface.
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
    /// The display provider could not supply a native handle.
    #[error("Could not obtain the display handle: {0}")]
    DisplayHandle(#[from] HandleError),
    /// WGPU could not create a surface from the supplied target.
    #[error("Surface creation failed: {0}")]
    SurfaceCreationFailed(#[from] wgpu::CreateSurfaceError),
    /// The context's adapter cannot render to the provided surface.
    #[error("The scene context adapter does not support the provided surface")]
    UnsupportedSurface,
}

impl RenderContext for Arc<WgpuContext> {
    fn clear_caches(&self) {
        self.texture_manager.clear_caches();
    }
}

async fn open_render_device(
    adapter: &Adapter,
    surface: Option<&WgpuSurface<'_>>,
) -> Result<(Device, Queue), BackendCreationError> {
    if surface
        .is_some_and(|surface| pick_surface_format(&surface.get_capabilities(adapter)).is_none())
    {
        return Err(BackendCreationError::UnsupportedSurface);
    }
    let descriptor = DeviceDescriptor {
        required_limits: Limits {
            // The highest vertex-to-fragment location in Grafo's shaders is 9.
            max_inter_stage_shader_variables: 10,
            ..Limits::default()
        },
        #[cfg(feature = "performance_measurement")]
        required_features: Features::TIMESTAMP_QUERY | Features::DEPTH32FLOAT_STENCIL8,
        ..Default::default()
    };
    Ok(adapter.request_device(&descriptor).await?)
}

async fn request_render_device(
    instance: &Instance,
    surface: Option<&WgpuSurface<'_>>,
) -> Result<(Adapter, Device, Queue), BackendCreationError> {
    let preferred = instance
        .request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::HighPerformance,
            compatible_surface: surface,
            ..Default::default()
        })
        .await?;
    let preferred_error = match open_render_device(&preferred, surface).await {
        Ok((device, queue)) => return Ok((preferred, device, queue)),
        Err(error) => error,
    };

    let preferred_info = preferred.get_info();
    warn!(adapter = %preferred_info.name, backend = ?preferred_info.backend, %preferred_error, "Could not open preferred WGPU adapter");
    let mut alternatives = instance.enumerate_adapters(Backends::all()).await;
    // A new enumeration gives the same physical adapter a fresh WGPU handle.
    if let Some(index) = alternatives
        .iter()
        .position(|adapter| adapter.get_info() == preferred_info)
    {
        alternatives.remove(index);
    }
    alternatives.sort_by_cached_key(|adapter| adapter.get_info().device_type == DeviceType::Cpu);
    for adapter in alternatives {
        let info = adapter.get_info();
        match open_render_device(&adapter, surface).await {
            Ok((device, queue)) => {
                info!(adapter = %info.name, backend = ?info.backend, "Using fallback WGPU adapter");
                return Ok((adapter, device, queue));
            }
            Err(error) => {
                warn!(adapter = %info.name, backend = ?info.backend, %error, "Could not open WGPU adapter")
            }
        }
    }
    Err(preferred_error)
}

impl WgpuContext {
    pub(super) fn ensure_available(&self) -> Result<(), WgpuBackendError> {
        if self.is_device_lost.load(Ordering::Relaxed) {
            return Err(WgpuBackendError::DeviceLost);
        }
        Ok(())
    }

    /// Creates a device, command queue, and texture manager that scenes can share.
    pub(crate) async fn try_new(
        texture_updates: TextureUpdatePublisher,
        instance: Instance,
        compatible_surface: Option<&WgpuSurface<'_>>,
    ) -> Result<Self, BackendCreationError> {
        let (adapter, device, queue) = request_render_device(&instance, compatible_surface).await?;
        let instance = Arc::new(instance);
        let adapter = Arc::new(adapter);
        let is_device_lost = Arc::new(AtomicBool::new(false));
        let callback_device_lost = Arc::clone(&is_device_lost);
        device.set_device_lost_callback(move |reason, message| {
            callback_device_lost.store(true, Ordering::Relaxed);
            error!(?reason, %message, "WGPU device lost");
        });
        device.on_uncaptured_error(Arc::new(|error| {
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
            retained_output_resources: RetainedOutputResources::new(&device),
            texture_manager: WgpuTextureManager::new(
                device.clone(),
                queue.clone(),
                texture_updates,
            ),
            device,
            queue,
            is_device_lost,
        })
    }
}

impl WgpuContext {
    /// Creates a surface compatible with this context's adapter.
    /// The backend configures it on the first render.
    pub(crate) fn create_surface(
        self: &Arc<Self>,
        target: impl Into<SurfaceTarget<'static>>,
        physical_size: (u32, u32),
        vsync: bool,
        transparent: bool,
    ) -> Result<Surface<WgpuSurfaceState>, BackendCreationError> {
        let surface = self.instance.create_surface(target)?;
        self.wrap_surface(surface, physical_size, vsync, transparent)
    }

    pub(crate) fn wrap_surface(
        self: &Arc<Self>,
        surface: WgpuSurface<'static>,
        physical_size: (u32, u32),
        vsync: bool,
        transparent: bool,
    ) -> Result<Surface<WgpuSurfaceState>, BackendCreationError> {
        let capabilities = surface.get_capabilities(&self.adapter);
        let format =
            pick_surface_format(&capabilities).ok_or(BackendCreationError::UnsupportedSurface)?;
        let configuration = wgpu::SurfaceConfiguration {
            usage: TextureUsages::RENDER_ATTACHMENT,
            color_space: SurfaceColorSpace::Srgb,
            format,
            width: physical_size.0,
            height: physical_size.1,
            present_mode: if vsync {
                wgpu::PresentMode::AutoVsync
            } else {
                wgpu::PresentMode::AutoNoVsync
            },
            desired_maximum_frame_latency: 2,
            alpha_mode: pick_alpha_mode(&capabilities.alpha_modes, transparent),
            view_formats: vec![],
        };
        Ok(Surface::from_resource(
            WgpuSurfaceState {
                surface: Some(surface),
                presentation: SurfacePresentation::new(
                    &self.device,
                    &self.retained_output_resources,
                    OutputColorFormat::for_surface(
                        format,
                        configuration.alpha_mode,
                        self.adapter.get_info().backend,
                    ),
                ),
                configuration,
                context: Arc::clone(self),
                transparent,
            },
            physical_size,
            vsync,
        ))
    }
}

impl WgpuBackend {
    /// Creates rendering resources on the context's device.
    pub(crate) fn new(
        context: Arc<WgpuContext>,
        physical_size: (u32, u32),
        scale_factor: f64,
        msaa_samples: u32,
    ) -> Result<Self, BackendCreationError> {
        validate_scale_factor(scale_factor)?;
        Ok(Self::initialize(
            context,
            TextureFormat::Bgra8UnormSrgb,
            physical_size,
            scale_factor,
            Self::normalize_msaa_sample_count(msaa_samples),
        ))
    }
}

fn validate_scale_factor(scale_factor: f64) -> Result<(), BackendCreationError> {
    if !scale_factor.is_finite() || scale_factor <= 0.0 {
        return Err(BackendCreationError::InvalidScaleFactor(scale_factor));
    }
    Ok(())
}
