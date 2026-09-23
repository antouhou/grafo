use super::execution::effects;
use super::execution::effects::EffectExecutionResources;
use super::execution::shape_effects::ShapeEffectRendererResources;
use super::execution::shapes::ShapeExecutionResources;
use super::execution::textures::IntermediateTextureResources;
use super::state::{BackdropPipelineResources, Buffers, ShapePipelines};
use super::types::DrawTreeNode;
use super::*;
use crate::pipeline::{
    create_gradient_bind_group_layout, create_gradient_increment_pipeline,
    create_gradient_stencil_keep_color_pipeline, create_stencil_keep_color_pipeline,
    create_stencil_only_pipeline,
};
use crate::vertex::CustomVertex;
use std::mem;
use tracing::{error, info, warn};
use wgpu::{
    BindGroupLayout, Device, DownlevelFlags, InstanceDescriptor, SurfaceConfiguration,
    TextureFormat,
};

const INDEX_ELEMENT_SIZE: usize = mem::size_of::<u16>();

fn create_transparent_texture_view_and_sampler(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &'static str,
) -> (wgpu::TextureView, wgpu::Sampler) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let transparent: [u8; 4] = [0, 0, 0, 0];
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &transparent,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4),
            rows_per_image: Some(1),
        },
        wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
    );

    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });
    (view, sampler)
}

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

impl ShapePipelines {
    fn new(
        context: &RendererContext,
        config: &SurfaceConfiguration,
        physical_size: (u32, u32),
        scale_factor: f64,
        fringe_width: f32,
        msaa_sample_count: u32,
    ) -> Self {
        let device = &context.inner.device;
        let queue = &context.inner.queue;
        let canvas_logical_size = to_logical(physical_size, scale_factor);

        let (
            and_uniforms,
            and_uniform_buffer,
            and_bind_group,
            background_texture_layout,
            foreground_texture_layout,
            and_pipeline,
        ) = create_pipeline(
            canvas_logical_size,
            scale_factor,
            fringe_width,
            device,
            config,
            PipelineType::EqualIncrementStencil,
            msaa_sample_count,
        );

        let (
            decrementing_uniforms,
            decrementing_uniform_buffer,
            decrementing_bind_group,
            _,
            _,
            decrementing_pipeline,
        ) = create_pipeline(
            canvas_logical_size,
            scale_factor,
            fringe_width,
            device,
            config,
            PipelineType::EqualDecrementStencil,
            msaa_sample_count,
        );

        let gradient_bind_group_layout = create_gradient_bind_group_layout(device);
        let and_gradient_pipeline = create_gradient_increment_pipeline(
            device,
            config.format,
            msaa_sample_count,
            &and_pipeline.get_bind_group_layout(0),
            &background_texture_layout,
            &foreground_texture_layout,
            &gradient_bind_group_layout,
        );

        let leaf_draw_pipeline = create_stencil_keep_color_pipeline(
            device,
            config.format,
            msaa_sample_count,
            &and_pipeline.get_bind_group_layout(0),
            &background_texture_layout,
            &foreground_texture_layout,
        );
        let leaf_draw_gradient_pipeline = create_gradient_stencil_keep_color_pipeline(
            device,
            config.format,
            msaa_sample_count,
            &and_pipeline.get_bind_group_layout(0),
            &background_texture_layout,
            &foreground_texture_layout,
            &gradient_bind_group_layout,
        );

        let linear_clamp_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("linear_clamp_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let default_background_texture_bind_group =
            Renderer::create_default_shape_texture_bind_group(
                device,
                queue,
                &background_texture_layout,
            );
        let default_foreground_texture_bind_group =
            Renderer::create_default_shape_texture_bind_group(
                device,
                queue,
                &foreground_texture_layout,
            );
        let stencil_only_pipeline = create_stencil_only_pipeline(
            device,
            config.format,
            msaa_sample_count,
            &and_pipeline.get_bind_group_layout(0),
            &background_texture_layout,
            &foreground_texture_layout,
        );

        Self {
            and_pipeline: Arc::new(and_pipeline),
            and_gradient_pipeline: Arc::new(and_gradient_pipeline),
            and_bind_group,
            decrementing_pipeline: Arc::new(decrementing_pipeline),
            decrementing_bind_group,
            leaf_draw_pipeline: Arc::new(leaf_draw_pipeline),
            leaf_draw_gradient_pipeline: Arc::new(leaf_draw_gradient_pipeline),
            shape_texture_bind_group_layout_background: Arc::new(background_texture_layout),
            shape_texture_bind_group_layout_foreground: Arc::new(foreground_texture_layout),
            default_shape_texture_bind_groups: [
                Arc::new(default_background_texture_bind_group),
                Arc::new(default_foreground_texture_bind_group),
            ],
            texture_manager: context.inner.texture_manager.clone(),
            and_uniforms,
            and_uniform_buffer,
            decrementing_uniforms,
            decrementing_uniform_buffer,
            under_fill_pipelines: None,
            stencil_only_pipeline,
            gradient_bind_group_layout,
            linear_clamp_sampler,
        }
    }
}

impl BackdropPipelineResources {
    pub(super) fn new(
        device: &Device,
        format: TextureFormat,
        composite_layout: &BindGroupLayout,
    ) -> Self {
        Self {
            texture_blit_pipeline: effects::compile_texture_blit_pipeline(
                device,
                format,
                composite_layout,
            ),
            layer_composite_resources: effects::compile_backdrop_layer_composite_pipeline(
                device, format,
            ),
        }
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
            inner: Arc::new(RendererContextInner {
                instance,
                supports_base_vertex: adapter
                    .get_downlevel_capabilities()
                    .flags
                    .contains(DownlevelFlags::BASE_VERTEX),
                adapter,
                texture_manager: TextureManager::new(device.clone(), queue.clone()),
                shape_cache: RwLock::new(HashMap::new()),
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
        let surface = context.inner.instance.create_surface(window)?;

        let surface_caps = surface.get_capabilities(&context.inner.adapter);
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
        surface.configure(&context.inner.device, &config);

        let msaa_sample_count = Self::normalize_msaa_sample_count(msaa_samples);

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
        config: wgpu::SurfaceConfiguration,
        physical_size: (u32, u32),
        scale_factor: f64,
        msaa_sample_count: u32,
    ) -> Result<Self, RendererCreationError> {
        if !scale_factor.is_finite() || scale_factor <= 0.0 {
            return Err(RendererCreationError::InvalidScaleFactor(scale_factor));
        }

        let device = context.inner.device.clone();
        let resources = ShapePipelines::new(
            &context,
            &config,
            physical_size,
            scale_factor,
            Self::DEFAULT_FRINGE_WIDTH,
            msaa_sample_count,
        );
        let instance = context.inner.instance.clone();
        let queue = context.inner.queue.clone();
        let shape_effect_resources = ShapeEffectRendererResources::new(&device, config.format);
        let effect_registry = EffectRegistry::new(&device);

        let supports_base_vertex = context.inner.supports_base_vertex;
        let mut renderer = Self {
            context,
            instance,
            surface,
            device,
            queue,
            config,
            fringe_width: Self::DEFAULT_FRINGE_WIDTH,
            tessellator: FillTessellator::new(),
            pipeline_resources: RendererPipelineResources {
                shapes: resources,
                shape_effects: shape_effect_resources,
                effect_sampler: None,
                composite_resources: None,
                backdrops: None,
            },
            argb_readback: None,
            bgra_readback: None,
            msaa_sample_count,
            msaa_color_texture: None,
            msaa_color_texture_view: None,
            depth_stencil_texture: None,
            depth_stencil_view: None,
            effect_registry,
            #[cfg(feature = "render_metrics")]
            render_loop_metrics_tracker: RenderLoopMetricsTracker::default(),
            #[cfg(feature = "render_metrics")]
            last_phase_timings: Default::default(),
            last_render_to_texture_view_cpu_time: Default::default(),
            state: RendererState {
                draw_tree: easy_tree::Tree::new(),
                shape_resources: ShapeResources::new(),
                shape_execution: ShapeExecutionResources::new(),
                effect_execution: EffectExecutionResources::default(),
                group_effects: HashMap::new(),
                backdrop_effects: HashMap::new(),
                shape_effects: HashMap::new(),
                scratch: RendererScratch::new(),
                scale_factor,
                physical_size,
                textures: IntermediateTextureResources::new(),
                #[cfg(feature = "render_metrics")]
                pipeline_switch_counts: Default::default(),
                #[cfg(feature = "render_metrics")]
                shape_effect_cache_metrics: Default::default(),
                buffers: Buffers {
                    supports_base_vertex,
                    aggregated_vertex_buffer: None,
                    aggregated_index_buffer: None,
                    aggregated_instance_transform_buffer: None,
                    aggregated_instance_color_buffer: None,
                    aggregated_instance_metadata_buffer: None,
                },
            },
        };

        renderer.recreate_msaa_texture();
        renderer.recreate_depth_stencil_texture();
        Ok(renderer)
    }

    pub fn print_memory_usage_info(&self) {
        println!("=== Memory Usage Info ===");

        println!(
            "Cached shapes: {}",
            self.context
                .inner
                .shape_cache
                .read()
                .expect("shared shape cache lock poisoned")
                .len()
        );
        println!("Draw tree size: {}", self.state.draw_tree.len());

        println!("\n--- Temporary Vectors ---");
        println!(
            "Temp vertices: {} items, {} capacity, ~{} bytes",
            self.state.shape_execution.vertices.len(),
            self.state.shape_execution.vertices.capacity(),
            self.state.shape_execution.vertices.capacity() * CustomVertex::STRIDE as usize
        );
        println!(
            "Temp indices: {} items, {} capacity, ~{} bytes",
            self.state.shape_execution.indices.len(),
            self.state.shape_execution.indices.capacity(),
            self.state.shape_execution.indices.capacity() * INDEX_ELEMENT_SIZE
        );
        println!(
            "Temp instance transforms: {} items, {} capacity, ~{} bytes",
            self.state.shape_execution.instance_transforms.len(),
            self.state.shape_execution.instance_transforms.capacity(),
            self.state.shape_execution.instance_transforms.capacity()
                * InstanceTransform::STRIDE as usize
        );
        println!(
            "Temp instance colors: {} items, {} capacity, ~{} bytes",
            self.state.shape_execution.instance_colors.len(),
            self.state.shape_execution.instance_colors.capacity(),
            self.state.shape_execution.instance_colors.capacity() * InstanceColor::STRIDE as usize
        );
        println!(
            "Temp instance metadata: {} items, {} capacity, ~{} bytes",
            self.state.shape_execution.instance_metadata.len(),
            self.state.shape_execution.instance_metadata.capacity(),
            self.state.shape_execution.instance_metadata.capacity()
                * InstanceMetadata::STRIDE as usize
        );

        println!("\n--- GPU Buffers ---");
        let buffers = &self.state.buffers;
        if let Some(buf) = &buffers.aggregated_vertex_buffer {
            println!("Aggregated vertex buffer: {} bytes", buf.size());
        }
        if let Some(buf) = &buffers.aggregated_index_buffer {
            println!("Aggregated index buffer: {} bytes", buf.size());
        }
        if let Some(buf) = &buffers.aggregated_instance_transform_buffer {
            println!("Aggregated instance transform buffer: {} bytes", buf.size());
        }
        if let Some(buf) = &buffers.aggregated_instance_color_buffer {
            println!("Aggregated instance color buffer: {} bytes", buf.size());
        }
        if let Some(buf) = &buffers.aggregated_instance_metadata_buffer {
            println!("Aggregated instance metadata buffer: {} bytes", buf.size());
        }

        println!("\n--- ARGB Compute Buffers ---");
        if let Some(resources) = &self.argb_readback {
            let target = &resources.target;
            println!("ARGB input buffer: {} bytes", target.input_buffer.size());
            println!(
                "ARGB output storage buffer: {} bytes",
                target.output_buffer.size()
            );
            println!(
                "ARGB readback buffer: {} bytes",
                target.readback_buffer.size()
            );
            println!("ARGB params buffer: {} bytes", target.params_buffer.size());
            println!(
                "ARGB offscreen texture: {}x{}",
                target.texture.width(),
                target.texture.height()
            );
        }

        println!("\n--- Render-to-Buffer Caches ---");
        if let Some(resources) = &self.bgra_readback {
            println!(
                "RTB offscreen texture: {}x{}",
                resources.texture.width(),
                resources.texture.height()
            );
            println!("RTB readback buffer: {} bytes", resources.buffer.size());
        }

        println!("\n--- Uniform Buffers ---");
        println!(
            "AND uniform buffer: {} bytes",
            self.pipeline_resources.shapes.and_uniform_buffer.size()
        );
        println!(
            "Decrementing uniform buffer: {} bytes",
            self.pipeline_resources
                .shapes
                .decrementing_uniform_buffer
                .size()
        );

        println!("\n--- Texture Manager ---");
        println!(
            "{:?}",
            self.pipeline_resources.shapes.texture_manager.size()
        );

        println!("\n--- Shape Resources ---");
        self.state.shape_resources.print_sizes();
        self.state.shape_execution.gradient_cache.print_sizes();

        println!("=========================");
    }

    fn create_default_shape_texture_bind_group(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        shape_texture_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> wgpu::BindGroup {
        let (view, sampler) = create_transparent_texture_view_and_sampler(
            device,
            queue,
            "default_transparent_texture",
        );

        device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: shape_texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
            label: Some("default_shape_texture_bind_group_transparent"),
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

    pub(super) fn recreate_pipelines(&mut self) {
        let resources = ShapePipelines::new(
            &self.context,
            &self.config,
            self.state.physical_size,
            self.state.scale_factor,
            self.fringe_width,
            self.msaa_sample_count,
        );
        self.pipeline_resources.shapes = resources;
        self.state
            .shape_execution
            .texture_materials
            .invalidate_bindings();

        self.state.textures.clear_shape_effects();
        self.pipeline_resources.composite_resources = None;
        self.pipeline_resources
            .shape_effects
            .recreate_pipeline(&self.device, self.config.format);

        // Reset lazily-created pipelines so they pick up the new layout
        self.pipeline_resources.backdrops = None;

        self.state.shape_execution.gradient_cache.clear_materials();
        for (node_id, draw_tree_node) in self.state.draw_tree.iter_mut() {
            let DrawTreeNode::CachedShape(shape) = draw_tree_node else {
                continue;
            };
            let resources = self
                .state
                .shape_execution
                .draws
                .get_mut(&node_id)
                .expect("queued shapes have execution resources");
            resources.invalidate_material_bindings();
            resources.refresh_gradient_material(
                &mut shape.fill,
                &mut self.state.shape_execution.gradient_cache,
                &self.device,
                &self.queue,
                &self.pipeline_resources.shapes.gradient_bind_group_layout,
                &self.pipeline_resources.shapes.linear_clamp_sampler,
            );
        }

        self.state
            .effect_execution
            .backdrop_composites
            .invalidate_bindings();
    }
}
