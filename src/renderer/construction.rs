use std::collections::HashSet;

use super::*;
use crate::cache::{cached_tessellation_heap_bytes, CachedTessellation};
use crate::util::{
    hash_map_capacity_bytes, texture_memory_size, vector_capacity_bytes, MemoryUsage,
};
use tracing::{info, warn};

fn pick_surface_format(surface_formats: &[wgpu::TextureFormat]) -> wgpu::TextureFormat {
    const PREFERRED_SURFACE_FORMATS: [wgpu::TextureFormat; 4] = [
        wgpu::TextureFormat::Bgra8UnormSrgb,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        wgpu::TextureFormat::Bgra8Unorm,
        wgpu::TextureFormat::Rgba8Unorm,
    ];

    PREFERRED_SURFACE_FORMATS
        .into_iter()
        .find(|surface_format| surface_formats.contains(surface_format))
        .unwrap_or_else(|| {
            surface_formats
                .first()
                .copied()
                .unwrap_or(wgpu::TextureFormat::Bgra8UnormSrgb)
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

#[cfg(test)]
mod tests {
    use super::Renderer;

    #[test]
    fn format_memory_usage_size_uses_compact_binary_units() {
        assert_eq!(Renderer::format_memory_usage_size(0), "0B");
        assert_eq!(Renderer::format_memory_usage_size(1024), "1.00KB");
        assert_eq!(
            Renderer::format_memory_usage_size((18.91_f64 * 1024.0 * 1024.0).round() as u64),
            "18.91MB"
        );
    }
}

/// Errors that can occur when creating a [`Renderer`] via
/// [`Renderer::try_new_headless`].
#[derive(Debug, thiserror::Error)]
pub enum RendererCreationError {
    /// The provided `scale_factor` is not usable (must be finite and positive).
    #[error("Invalid scale factor: {0} (must be finite and > 0.0)")]
    InvalidScaleFactor(f64),
    /// No suitable GPU adapter was found.
    #[error("No suitable GPU adapter available: {0}")]
    AdapterNotAvailable(#[from] wgpu::RequestAdapterError),
    /// The GPU device could not be created.
    #[error("GPU device creation failed: {0}")]
    DeviceCreationFailed(#[from] wgpu::RequestDeviceError),
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
        let size = physical_size;

        let instance = wgpu::Instance::new(&InstanceDescriptor::default());
        let surface = instance
            .create_surface(window)
            .expect("Failed to create surface");

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

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
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        let swapchain_format = pick_surface_format(&surface_caps.formats);
        let alpha_mode = pick_alpha_mode(&surface_caps.alpha_modes, transparent);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            format: swapchain_format,
            width: size.0,
            height: size.1,
            present_mode: if vsync {
                wgpu::PresentMode::AutoVsync
            } else {
                wgpu::PresentMode::AutoNoVsync
            },
            desired_maximum_frame_latency: 2,
            alpha_mode,
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        let msaa_sample_count = Self::validate_sample_count_static(msaa_samples);

        Self::build_from_device(
            instance,
            Some(surface),
            device,
            queue,
            config,
            size,
            scale_factor,
            msaa_sample_count,
        )
        .expect("Failed to build renderer from device")
    }

    /// Shared constructor: takes the wgpu primitives produced by `new()` or
    /// `new_headless()` and builds the full `Renderer`.
    #[allow(clippy::too_many_arguments)]
    fn build_from_device(
        instance: wgpu::Instance,
        surface: Option<wgpu::Surface<'a>>,
        device: wgpu::Device,
        queue: wgpu::Queue,
        config: wgpu::SurfaceConfiguration,
        physical_size: (u32, u32),
        scale_factor: f64,
        msaa_sample_count: u32,
    ) -> Result<Self, RendererCreationError> {
        if !scale_factor.is_finite() || scale_factor <= 0.0 {
            return Err(RendererCreationError::InvalidScaleFactor(scale_factor));
        }

        let canvas_logical_size = to_logical(physical_size, scale_factor);

        let (
            and_uniforms,
            and_uniform_buffer,
            and_bind_group,
            and_texture_bgl_layer0,
            and_texture_bgl_layer1,
            and_pipeline,
        ) = create_pipeline(
            canvas_logical_size,
            scale_factor,
            Self::DEFAULT_FRINGE_WIDTH,
            &device,
            &config,
            PipelineType::EqualIncrementStencil,
            msaa_sample_count,
        );

        let (
            decrementing_uniforms,
            decrementing_uniform_buffer,
            decrementing_bind_group,
            _shape_texture_bind_group_layout_init0,
            _shape_texture_bind_group_layout_init1,
            decrementing_pipeline,
        ) = create_pipeline(
            canvas_logical_size,
            scale_factor,
            Self::DEFAULT_FRINGE_WIDTH,
            &device,
            &config,
            PipelineType::EqualDecrementStencil,
            msaa_sample_count,
        );

        let gradient_bind_group_layout =
            crate::pipeline::create_gradient_bind_group_layout(&device);
        let backdrop_texture_bind_group_layout =
            crate::pipeline::create_backdrop_texture_bind_group_layout(&device);
        let backdrop_gradient_bind_group_layout =
            crate::pipeline::create_backdrop_gradient_bind_group_layout(&device);
        let and_gradient_pipeline = crate::pipeline::create_gradient_increment_pipeline(
            &device,
            config.format,
            msaa_sample_count,
            &and_pipeline.get_bind_group_layout(0),
            &and_texture_bgl_layer0,
            &and_texture_bgl_layer1,
            &gradient_bind_group_layout,
        );

        let leaf_draw_pipeline = crate::pipeline::create_stencil_keep_color_pipeline(
            &device,
            config.format,
            msaa_sample_count,
            &and_pipeline.get_bind_group_layout(0),
            &and_texture_bgl_layer0,
            &and_texture_bgl_layer1,
        );
        let leaf_draw_gradient_pipeline =
            crate::pipeline::create_gradient_stencil_keep_color_pipeline(
                &device,
                config.format,
                msaa_sample_count,
                &and_pipeline.get_bind_group_layout(0),
                &and_texture_bgl_layer0,
                &and_texture_bgl_layer1,
                &gradient_bind_group_layout,
            );

        let gradient_ramp_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("gradient_ramp_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        let texture_manager = TextureManager::new(device.clone(), queue.clone());

        let (default_shape_texture_bind_group_layer0, shape_texture_bind_group_layout_layer0) =
            Self::create_default_shape_texture_bind_group(&device, &queue, &and_texture_bgl_layer0);
        let (default_shape_texture_bind_group_layer1, shape_texture_bind_group_layout_layer1) =
            Self::create_default_shape_texture_bind_group(&device, &queue, &and_texture_bgl_layer1);
        let default_backdrop_texture_bind_group = Self::create_default_backdrop_texture_bind_group(
            &device,
            &queue,
            &backdrop_texture_bind_group_layout,
        );

        let mut renderer = Self {
            instance,
            surface,
            device,
            queue,
            config,
            physical_size,
            scale_factor,
            fringe_width: Self::DEFAULT_FRINGE_WIDTH,
            tessellator: FillTessellator::new(),
            texture_manager,
            buffers_pool_manager: PoolManager::new(
                NonZeroUsize::new(MAX_CACHED_SHAPES).expect("Cache size to be greater than 0"),
            ),
            and_pipeline: Arc::new(and_pipeline),
            and_uniforms,
            and_uniform_buffer,
            and_bind_group,
            shape_texture_bind_group_layout_background: Arc::new(
                shape_texture_bind_group_layout_layer0,
            ),
            shape_texture_bind_group_layout_foreground: Arc::new(
                shape_texture_bind_group_layout_layer1,
            ),
            backdrop_texture_bind_group_layout: Arc::new(backdrop_texture_bind_group_layout),
            shape_texture_layout_epoch: 0,
            default_shape_texture_bind_groups: [
                Arc::new(default_shape_texture_bind_group_layer0),
                Arc::new(default_shape_texture_bind_group_layer1),
            ],
            default_backdrop_texture_bind_group: Arc::new(default_backdrop_texture_bind_group),
            decrementing_pipeline: Arc::new(decrementing_pipeline),
            decrementing_uniforms,
            decrementing_uniform_buffer,
            decrementing_bind_group,
            draw_tree: easy_tree::Tree::new(),
            metadata_to_clips: HashMap::new(),
            temp_vertices: Vec::new(),
            temp_indices: Vec::new(),
            geometry_dedup_map: HashMap::new(),
            temp_instance_transforms: Vec::new(),
            temp_instance_colors: Vec::new(),
            temp_instance_metadata: Vec::new(),
            aggregated_vertex_buffer: None,
            aggregated_index_buffer: None,
            aggregated_instance_transform_buffer: None,
            aggregated_instance_color_buffer: None,
            aggregated_instance_metadata_buffer: None,
            identity_instance_transform_buffer: None,
            identity_instance_color_buffer: None,
            identity_instance_metadata_buffer: None,
            shape_cache: HashMap::new(),
            argb_cs_bgl: None,
            argb_cs_pipeline: None,
            argb_swizzle_bind_group: None,
            argb_params_buffer: None,
            argb_input_buffer: None,
            argb_output_storage_buffer: None,
            argb_readback_buffer: None,
            argb_input_buffer_size: 0,
            argb_output_buffer_size: 0,
            argb_cached_width: 0,
            argb_cached_height: 0,
            argb_offscreen_texture: None,
            rtb_offscreen_texture: None,
            rtb_readback_buffer: None,
            rtb_cached_width: 0,
            rtb_cached_height: 0,
            msaa_sample_count,
            msaa_color_texture: None,
            msaa_color_texture_view: None,
            depth_stencil_texture: None,
            depth_stencil_view: None,
            loaded_effects: HashMap::new(),
            group_effects: HashMap::new(),
            backdrop_effects: HashMap::new(),
            offscreen_texture_pool: OffscreenTexturePool::new(),
            composite_pipeline: None,
            composite_bgl: None,
            effect_sampler: None,
            texture_blit_pipeline: None,
            stencil_only_pipeline: None,
            backdrop_color_pipeline: None,
            backdrop_color_gradient_pipeline: None,
            leaf_draw_pipeline: Arc::new(leaf_draw_pipeline),
            leaf_draw_gradient_pipeline: Arc::new(leaf_draw_gradient_pipeline),
            and_gradient_pipeline: Arc::new(and_gradient_pipeline),
            gradient_bind_group_layout,
            backdrop_gradient_bind_group_layout,
            gradient_bind_group_layout_epoch: 0,
            gradient_ramp_sampler,
            #[cfg(feature = "render_metrics")]
            render_loop_metrics_tracker: RenderLoopMetricsTracker::default(),
            #[cfg(feature = "render_metrics")]
            last_phase_timings: Default::default(),
            #[cfg(feature = "render_metrics")]
            last_pipeline_switch_counts: Default::default(),
            last_render_to_texture_view_cpu_time: Default::default(),
            scratch: RendererScratch::new(),
        };

        renderer.recreate_msaa_texture();
        renderer.recreate_depth_stencil_texture();
        Ok(renderer)
    }

    pub fn format_memory_usage_size(bytes: u64) -> String {
        const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
        let mut value = bytes as f64;
        let mut unit_index = 0;

        while value >= 1024.0 && unit_index + 1 < UNITS.len() {
            value /= 1024.0;
            unit_index += 1;
        }

        if unit_index == 0 {
            format!("{bytes}B")
        } else {
            format!("{value:.2}{}", UNITS[unit_index])
        }
    }

    pub fn print_memory_usage_info_human_readable(&self) {
        self.print_memory_usage_info_with_format(true);
    }

    pub fn print_total_memory_usage_info(&self) {
        let total_usage = self.total_memory_usage();
        let gpu_bytes = total_usage
            .gpu_buffer_bytes
            .saturating_add(total_usage.gpu_texture_bytes);

        println!(
            "CPU total RAM consumed by the renderer: {}, GPU total consumed {}",
            Self::format_memory_usage_size(total_usage.cpu_bytes),
            Self::format_memory_usage_size(gpu_bytes)
        );
    }

    pub fn print_memory_usage_info(&self) {
        self.print_memory_usage_info_with_format(false);
    }

    fn total_memory_usage(&self) -> MemoryUsage {
        let mut total_usage = MemoryUsage {
            cpu_bytes: std::mem::size_of_val(self) as u64,
            gpu_buffer_bytes: 0,
            gpu_texture_bytes: 0,
        };
        let mut visited_tessellations: HashSet<*const CachedTessellation> = HashSet::new();

        total_usage.cpu_bytes = total_usage
            .cpu_bytes
            .saturating_add(vector_capacity_bytes(&self.temp_vertices))
            .saturating_add(vector_capacity_bytes(&self.temp_indices))
            .saturating_add(vector_capacity_bytes(&self.temp_instance_transforms))
            .saturating_add(vector_capacity_bytes(&self.temp_instance_colors))
            .saturating_add(vector_capacity_bytes(&self.temp_instance_metadata));

        Self::add_optional_buffer_usage(&self.aggregated_vertex_buffer, &mut total_usage);
        Self::add_optional_buffer_usage(&self.aggregated_index_buffer, &mut total_usage);
        Self::add_optional_buffer_usage(
            &self.aggregated_instance_transform_buffer,
            &mut total_usage,
        );
        Self::add_optional_buffer_usage(&self.aggregated_instance_color_buffer, &mut total_usage);
        Self::add_optional_buffer_usage(
            &self.aggregated_instance_metadata_buffer,
            &mut total_usage,
        );
        Self::add_optional_buffer_usage(&self.identity_instance_transform_buffer, &mut total_usage);
        Self::add_optional_buffer_usage(&self.identity_instance_color_buffer, &mut total_usage);
        Self::add_optional_buffer_usage(&self.identity_instance_metadata_buffer, &mut total_usage);

        Self::add_optional_buffer_usage(&self.argb_input_buffer, &mut total_usage);
        Self::add_optional_buffer_usage(&self.argb_output_storage_buffer, &mut total_usage);
        Self::add_optional_buffer_usage(&self.argb_readback_buffer, &mut total_usage);
        Self::add_optional_buffer_usage(&self.argb_params_buffer, &mut total_usage);
        Self::add_optional_texture_usage(
            &self.argb_offscreen_texture,
            self.config.format,
            1,
            &mut total_usage,
        );

        Self::add_optional_texture_usage(
            &self.rtb_offscreen_texture,
            self.config.format,
            1,
            &mut total_usage,
        );
        Self::add_optional_buffer_usage(&self.rtb_readback_buffer, &mut total_usage);

        total_usage.gpu_buffer_bytes = total_usage
            .gpu_buffer_bytes
            .saturating_add(self.and_uniform_buffer.size())
            .saturating_add(self.decrementing_uniform_buffer.size());

        Self::add_optional_texture_usage(
            &self.msaa_color_texture,
            self.config.format,
            self.msaa_sample_count,
            &mut total_usage,
        );
        Self::add_optional_texture_usage(
            &self.depth_stencil_texture,
            wgpu::TextureFormat::Depth24PlusStencil8,
            self.msaa_sample_count,
            &mut total_usage,
        );

        if self.surface.is_some() {
            let surface_texture_bytes = texture_memory_size(
                wgpu::Extent3d {
                    width: self.config.width,
                    height: self.config.height,
                    depth_or_array_layers: 1,
                },
                self.config.format,
                1,
                1,
            )
            .saturating_mul(self.config.desired_maximum_frame_latency as u64);
            total_usage.gpu_texture_bytes = total_usage
                .gpu_texture_bytes
                .saturating_add(surface_texture_bytes);
        }

        let default_texture_bytes = texture_memory_size(
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            wgpu::TextureFormat::Rgba8UnormSrgb,
            1,
            1,
        )
        .saturating_mul(3);
        let default_backdrop_params_bytes =
            std::mem::size_of::<crate::gradient::gpu::GpuMaterialParams>() as u64;
        total_usage.gpu_texture_bytes = total_usage
            .gpu_texture_bytes
            .saturating_add(default_texture_bytes);
        total_usage.gpu_buffer_bytes = total_usage
            .gpu_buffer_bytes
            .saturating_add(default_backdrop_params_bytes);

        let draw_tree_bytes = (self.draw_tree.len() as u64).saturating_mul(
            std::mem::size_of::<DrawCommand>() as u64
                + std::mem::size_of::<Vec<usize>>() as u64
                + std::mem::size_of::<Option<usize>>() as u64,
        );
        total_usage.cpu_bytes = total_usage.cpu_bytes.saturating_add(draw_tree_bytes);

        for (_, draw_command) in self.draw_tree.iter() {
            if let DrawCommand::CachedShape(cached_shape) = draw_command {
                total_usage.cpu_bytes = total_usage
                    .cpu_bytes
                    .saturating_add(cached_shape.cpu_heap_bytes())
                    .saturating_add(cached_tessellation_heap_bytes(
                        &cached_shape.cached_shape.tessellation,
                        &mut visited_tessellations,
                    ));
                total_usage.gpu_buffer_bytes = total_usage
                    .gpu_buffer_bytes
                    .saturating_add(cached_shape.gpu_buffer_bytes());
            }
        }

        total_usage.cpu_bytes = total_usage
            .cpu_bytes
            .saturating_add(hash_map_capacity_bytes(&self.metadata_to_clips))
            .saturating_add(hash_map_capacity_bytes(&self.geometry_dedup_map))
            .saturating_add(hash_map_capacity_bytes(&self.shape_cache));

        for cached_shape in self.shape_cache.values() {
            total_usage.cpu_bytes =
                total_usage
                    .cpu_bytes
                    .saturating_add(cached_tessellation_heap_bytes(
                        &cached_shape.tessellation,
                        &mut visited_tessellations,
                    ));
        }

        total_usage.add(self.texture_manager.memory_usage());
        total_usage.add(
            self.buffers_pool_manager
                .memory_usage(&mut visited_tessellations),
        );

        let mut effects_usage = MemoryUsage {
            cpu_bytes: hash_map_capacity_bytes(&self.loaded_effects)
                .saturating_add(hash_map_capacity_bytes(&self.group_effects))
                .saturating_add(hash_map_capacity_bytes(&self.backdrop_effects)),
            gpu_buffer_bytes: 0,
            gpu_texture_bytes: 0,
        };
        for loaded_effect in self.loaded_effects.values() {
            effects_usage.add(loaded_effect.memory_usage());
        }
        for effect_instance in self
            .group_effects
            .values()
            .chain(self.backdrop_effects.values())
        {
            effects_usage.add(effect_instance.memory_usage());
        }
        total_usage.add(effects_usage);
        total_usage.add(self.offscreen_texture_pool.memory_usage());
        total_usage.add(self.scratch.memory_usage());

        total_usage
    }

    fn print_memory_usage_info_with_format(&self, human_readable_only: bool) {
        let memory_value = |bytes| Self::memory_value(bytes, human_readable_only);
        let mut total_usage = MemoryUsage {
            cpu_bytes: std::mem::size_of_val(self) as u64,
            gpu_buffer_bytes: 0,
            gpu_texture_bytes: 0,
        };
        let mut visited_tessellations: HashSet<*const CachedTessellation> = HashSet::new();

        println!("=== Memory Usage Info ===");

        println!("Cached shapes: {}", self.shape_cache.len());
        println!("Draw tree size: {}", self.draw_tree.len());
        println!(
            "Metadata to clips mappings: {}",
            self.metadata_to_clips.len()
        );
        println!(
            "Renderer CPU inline fields: {}",
            memory_value(std::mem::size_of_val(self) as u64)
        );

        println!("\n--- Temporary Vectors ---");
        let temp_vertices_bytes = vector_capacity_bytes(&self.temp_vertices);
        total_usage.cpu_bytes = total_usage.cpu_bytes.saturating_add(temp_vertices_bytes);
        println!(
            "Temp vertices: {} items, {} capacity, {}",
            self.temp_vertices.len(),
            self.temp_vertices.capacity(),
            memory_value(temp_vertices_bytes)
        );
        let temp_indices_bytes = vector_capacity_bytes(&self.temp_indices);
        total_usage.cpu_bytes = total_usage.cpu_bytes.saturating_add(temp_indices_bytes);
        println!(
            "Temp indices: {} items, {} capacity, {}",
            self.temp_indices.len(),
            self.temp_indices.capacity(),
            memory_value(temp_indices_bytes)
        );
        let temp_instance_transform_bytes = vector_capacity_bytes(&self.temp_instance_transforms);
        total_usage.cpu_bytes = total_usage
            .cpu_bytes
            .saturating_add(temp_instance_transform_bytes);
        println!(
            "Temp instance transforms: {} items, {} capacity, {}",
            self.temp_instance_transforms.len(),
            self.temp_instance_transforms.capacity(),
            memory_value(temp_instance_transform_bytes)
        );
        let temp_instance_color_bytes = vector_capacity_bytes(&self.temp_instance_colors);
        total_usage.cpu_bytes = total_usage
            .cpu_bytes
            .saturating_add(temp_instance_color_bytes);
        println!(
            "Temp instance colors: {} items, {} capacity, {}",
            self.temp_instance_colors.len(),
            self.temp_instance_colors.capacity(),
            memory_value(temp_instance_color_bytes)
        );
        let temp_instance_metadata_bytes = vector_capacity_bytes(&self.temp_instance_metadata);
        total_usage.cpu_bytes = total_usage
            .cpu_bytes
            .saturating_add(temp_instance_metadata_bytes);
        println!(
            "Temp instance metadata: {} items, {} capacity, {}",
            self.temp_instance_metadata.len(),
            self.temp_instance_metadata.capacity(),
            memory_value(temp_instance_metadata_bytes)
        );

        println!("\n--- GPU Buffers ---");
        Self::print_optional_buffer(
            "Aggregated vertex buffer",
            &self.aggregated_vertex_buffer,
            &mut total_usage,
            &memory_value,
        );
        Self::print_optional_buffer(
            "Aggregated index buffer",
            &self.aggregated_index_buffer,
            &mut total_usage,
            &memory_value,
        );
        Self::print_optional_buffer(
            "Aggregated instance transform buffer",
            &self.aggregated_instance_transform_buffer,
            &mut total_usage,
            &memory_value,
        );
        Self::print_optional_buffer(
            "Aggregated instance color buffer",
            &self.aggregated_instance_color_buffer,
            &mut total_usage,
            &memory_value,
        );
        Self::print_optional_buffer(
            "Aggregated instance metadata buffer",
            &self.aggregated_instance_metadata_buffer,
            &mut total_usage,
            &memory_value,
        );
        Self::print_optional_buffer(
            "Identity instance transform buffer",
            &self.identity_instance_transform_buffer,
            &mut total_usage,
            &memory_value,
        );
        Self::print_optional_buffer(
            "Identity instance color buffer",
            &self.identity_instance_color_buffer,
            &mut total_usage,
            &memory_value,
        );
        Self::print_optional_buffer(
            "Identity instance metadata buffer",
            &self.identity_instance_metadata_buffer,
            &mut total_usage,
            &memory_value,
        );

        println!("\n--- ARGB Compute Buffers ---");
        if let Some(buffer) = &self.argb_input_buffer {
            total_usage.gpu_buffer_bytes =
                total_usage.gpu_buffer_bytes.saturating_add(buffer.size());
            println!(
                "ARGB input buffer: {} (cached size: {})",
                memory_value(buffer.size()),
                memory_value(self.argb_input_buffer_size)
            );
        }
        if let Some(buffer) = &self.argb_output_storage_buffer {
            total_usage.gpu_buffer_bytes =
                total_usage.gpu_buffer_bytes.saturating_add(buffer.size());
            println!(
                "ARGB output storage buffer: {} (cached size: {})",
                memory_value(buffer.size()),
                memory_value(self.argb_output_buffer_size)
            );
        }
        Self::print_optional_buffer(
            "ARGB readback buffer",
            &self.argb_readback_buffer,
            &mut total_usage,
            &memory_value,
        );
        Self::print_optional_buffer(
            "ARGB params buffer",
            &self.argb_params_buffer,
            &mut total_usage,
            &memory_value,
        );
        if let Some(texture) = &self.argb_offscreen_texture {
            let texture_bytes = texture_memory_size(texture.size(), self.config.format, 1, 1);
            total_usage.gpu_texture_bytes =
                total_usage.gpu_texture_bytes.saturating_add(texture_bytes);
            let size = texture.size();
            println!(
                "ARGB offscreen texture: {}x{} (cached: {}x{}), {}",
                size.width,
                size.height,
                self.argb_cached_width,
                self.argb_cached_height,
                memory_value(texture_bytes)
            );
        }

        println!("\n--- Render-to-Buffer Caches ---");
        if let Some(texture) = &self.rtb_offscreen_texture {
            let texture_bytes = texture_memory_size(texture.size(), self.config.format, 1, 1);
            total_usage.gpu_texture_bytes =
                total_usage.gpu_texture_bytes.saturating_add(texture_bytes);
            let size = texture.size();
            println!(
                "RTB offscreen texture: {}x{} (cached: {}x{}), {}",
                size.width,
                size.height,
                self.rtb_cached_width,
                self.rtb_cached_height,
                memory_value(texture_bytes)
            );
        }
        Self::print_optional_buffer(
            "RTB readback buffer",
            &self.rtb_readback_buffer,
            &mut total_usage,
            &memory_value,
        );

        println!("\n--- Uniform Buffers ---");
        total_usage.gpu_buffer_bytes = total_usage
            .gpu_buffer_bytes
            .saturating_add(self.and_uniform_buffer.size())
            .saturating_add(self.decrementing_uniform_buffer.size());
        println!(
            "AND uniform buffer: {}",
            memory_value(self.and_uniform_buffer.size())
        );
        println!(
            "Decrementing uniform buffer: {}",
            memory_value(self.decrementing_uniform_buffer.size())
        );

        println!("\n--- Renderer Textures ---");
        if let Some(texture) = &self.msaa_color_texture {
            let texture_bytes = texture_memory_size(
                texture.size(),
                self.config.format,
                self.msaa_sample_count,
                1,
            );
            total_usage.gpu_texture_bytes =
                total_usage.gpu_texture_bytes.saturating_add(texture_bytes);
            println!(
                "MSAA color texture: {} samples, {}",
                self.msaa_sample_count,
                memory_value(texture_bytes)
            );
        }
        if let Some(texture) = &self.depth_stencil_texture {
            let texture_bytes = texture_memory_size(
                texture.size(),
                wgpu::TextureFormat::Depth24PlusStencil8,
                self.msaa_sample_count,
                1,
            );
            total_usage.gpu_texture_bytes =
                total_usage.gpu_texture_bytes.saturating_add(texture_bytes);
            println!("Depth/stencil texture: {}", memory_value(texture_bytes));
        }
        if self.surface.is_some() {
            let surface_texture_bytes = texture_memory_size(
                wgpu::Extent3d {
                    width: self.config.width,
                    height: self.config.height,
                    depth_or_array_layers: 1,
                },
                self.config.format,
                1,
                1,
            )
            .saturating_mul(self.config.desired_maximum_frame_latency as u64);
            total_usage.gpu_texture_bytes = total_usage
                .gpu_texture_bytes
                .saturating_add(surface_texture_bytes);
            println!(
                "Surface texture chain estimate: {}",
                memory_value(surface_texture_bytes)
            );
        }
        let default_texture_bytes = texture_memory_size(
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            wgpu::TextureFormat::Rgba8UnormSrgb,
            1,
            1,
        )
        .saturating_mul(3);
        let default_backdrop_params_bytes =
            std::mem::size_of::<crate::gradient::gpu::GpuMaterialParams>() as u64;
        total_usage.gpu_texture_bytes = total_usage
            .gpu_texture_bytes
            .saturating_add(default_texture_bytes);
        total_usage.gpu_buffer_bytes = total_usage
            .gpu_buffer_bytes
            .saturating_add(default_backdrop_params_bytes);
        println!(
            "Default bind-group textures: {}",
            memory_value(default_texture_bytes)
        );
        println!(
            "Default backdrop material buffer: {}",
            memory_value(default_backdrop_params_bytes)
        );

        println!("\n--- Draw Data and CPU Caches ---");
        let draw_tree_bytes = (self.draw_tree.len() as u64).saturating_mul(
            std::mem::size_of::<DrawCommand>() as u64
                + std::mem::size_of::<Vec<usize>>() as u64
                + std::mem::size_of::<Option<usize>>() as u64,
        );
        total_usage.cpu_bytes = total_usage.cpu_bytes.saturating_add(draw_tree_bytes);
        println!("Draw tree node payloads: {}", memory_value(draw_tree_bytes));

        let mut draw_command_heap_bytes = 0_u64;
        let mut draw_command_gpu_buffer_bytes = 0_u64;
        for (_, draw_command) in self.draw_tree.iter() {
            if let DrawCommand::CachedShape(cached_shape) = draw_command {
                draw_command_heap_bytes =
                    draw_command_heap_bytes.saturating_add(cached_shape.cpu_heap_bytes());
                draw_command_heap_bytes =
                    draw_command_heap_bytes.saturating_add(cached_tessellation_heap_bytes(
                        &cached_shape.cached_shape.tessellation,
                        &mut visited_tessellations,
                    ));
                draw_command_gpu_buffer_bytes =
                    draw_command_gpu_buffer_bytes.saturating_add(cached_shape.gpu_buffer_bytes());
            }
        }
        total_usage.cpu_bytes = total_usage
            .cpu_bytes
            .saturating_add(draw_command_heap_bytes);
        total_usage.gpu_buffer_bytes = total_usage
            .gpu_buffer_bytes
            .saturating_add(draw_command_gpu_buffer_bytes);
        println!(
            "Draw command heap payloads: {}",
            memory_value(draw_command_heap_bytes)
        );
        println!(
            "Draw command GPU buffers: {}",
            memory_value(draw_command_gpu_buffer_bytes)
        );

        let metadata_to_clips_bytes = hash_map_capacity_bytes(&self.metadata_to_clips);
        let geometry_dedup_map_bytes = hash_map_capacity_bytes(&self.geometry_dedup_map);
        let shape_cache_map_bytes = hash_map_capacity_bytes(&self.shape_cache);
        total_usage.cpu_bytes = total_usage
            .cpu_bytes
            .saturating_add(metadata_to_clips_bytes)
            .saturating_add(geometry_dedup_map_bytes)
            .saturating_add(shape_cache_map_bytes);
        println!(
            "Metadata to clips map: {}",
            memory_value(metadata_to_clips_bytes)
        );
        println!(
            "Geometry dedup map: {}",
            memory_value(geometry_dedup_map_bytes)
        );
        println!("Shape cache map: {}", memory_value(shape_cache_map_bytes));

        let mut shape_cache_tessellation_bytes = 0_u64;
        for cached_shape in self.shape_cache.values() {
            shape_cache_tessellation_bytes =
                shape_cache_tessellation_bytes.saturating_add(cached_tessellation_heap_bytes(
                    &cached_shape.tessellation,
                    &mut visited_tessellations,
                ));
        }
        total_usage.cpu_bytes = total_usage
            .cpu_bytes
            .saturating_add(shape_cache_tessellation_bytes);
        println!(
            "Unique cached tessellations: {}",
            memory_value(shape_cache_tessellation_bytes)
        );

        println!("\n--- Texture Manager ---");
        let texture_manager_usage = self.texture_manager.memory_usage();
        total_usage.add(texture_manager_usage);
        let (texture_count, texture_bind_group_count) = self.texture_manager.size();
        println!(
            "Textures: {}, bind groups: {}, CPU {}, GPU textures {}",
            texture_count,
            texture_bind_group_count,
            memory_value(texture_manager_usage.cpu_bytes),
            memory_value(texture_manager_usage.gpu_texture_bytes)
        );

        println!("\n--- Buffer Pool Manager ---");
        self.buffers_pool_manager.print_sizes();
        let buffer_pool_usage = self
            .buffers_pool_manager
            .memory_usage(&mut visited_tessellations);
        total_usage.add(buffer_pool_usage);
        println!(
            "Buffer pool CPU: {}, GPU buffers: {}, GPU textures: {}",
            memory_value(buffer_pool_usage.cpu_bytes),
            memory_value(buffer_pool_usage.gpu_buffer_bytes),
            memory_value(buffer_pool_usage.gpu_texture_bytes)
        );

        println!("\n--- Effects ---");
        let mut effects_usage = MemoryUsage {
            cpu_bytes: hash_map_capacity_bytes(&self.loaded_effects)
                .saturating_add(hash_map_capacity_bytes(&self.group_effects))
                .saturating_add(hash_map_capacity_bytes(&self.backdrop_effects)),
            gpu_buffer_bytes: 0,
            gpu_texture_bytes: 0,
        };
        for loaded_effect in self.loaded_effects.values() {
            effects_usage.add(loaded_effect.memory_usage());
        }
        for effect_instance in self
            .group_effects
            .values()
            .chain(self.backdrop_effects.values())
        {
            effects_usage.add(effect_instance.memory_usage());
        }
        total_usage.add(effects_usage);
        println!(
            "Loaded effects: {}, group instances: {}, backdrop instances: {}",
            self.loaded_effects.len(),
            self.group_effects.len(),
            self.backdrop_effects.len()
        );
        println!(
            "Effects CPU: {}, GPU buffers: {}",
            memory_value(effects_usage.cpu_bytes),
            memory_value(effects_usage.gpu_buffer_bytes)
        );

        let offscreen_pool_usage = self.offscreen_texture_pool.memory_usage();
        total_usage.add(offscreen_pool_usage);
        println!(
            "Offscreen texture pool CPU: {}, GPU textures: {}",
            memory_value(offscreen_pool_usage.cpu_bytes),
            memory_value(offscreen_pool_usage.gpu_texture_bytes)
        );

        println!("\n--- Renderer Scratch ---");
        let scratch_usage = self.scratch.memory_usage();
        total_usage.add(scratch_usage);
        println!(
            "Scratch CPU: {}, GPU textures: {}",
            memory_value(scratch_usage.cpu_bytes),
            memory_value(scratch_usage.gpu_texture_bytes)
        );

        println!("\n--- Totals ---");
        println!(
            "CPU-side tracked memory: {}",
            memory_value(total_usage.cpu_bytes)
        );
        println!(
            "GPU buffer memory: {}",
            memory_value(total_usage.gpu_buffer_bytes)
        );
        println!(
            "GPU texture memory: {}",
            memory_value(total_usage.gpu_texture_bytes)
        );
        println!("Tracked total: {}", memory_value(total_usage.total_bytes()));
        println!(
            "Note: WGPU driver-private allocations for pipelines, bind groups, samplers, device, queue, and surface internals are not exposed by wgpu."
        );

        println!("=========================");
    }

    fn memory_value(bytes: u64, human_readable_only: bool) -> String {
        let formatted = Self::format_memory_usage_size(bytes);
        if human_readable_only {
            formatted
        } else {
            format!("{bytes} bytes ({formatted})")
        }
    }

    fn print_optional_buffer(
        label: &str,
        buffer: &Option<wgpu::Buffer>,
        total_usage: &mut MemoryUsage,
        memory_value: &impl Fn(u64) -> String,
    ) {
        if let Some(buffer) = buffer {
            total_usage.gpu_buffer_bytes =
                total_usage.gpu_buffer_bytes.saturating_add(buffer.size());
            println!("{label}: {}", memory_value(buffer.size()));
        }
    }

    fn add_optional_buffer_usage(buffer: &Option<wgpu::Buffer>, total_usage: &mut MemoryUsage) {
        if let Some(buffer) = buffer {
            total_usage.gpu_buffer_bytes =
                total_usage.gpu_buffer_bytes.saturating_add(buffer.size());
        }
    }

    fn add_optional_texture_usage(
        texture: &Option<wgpu::Texture>,
        format: wgpu::TextureFormat,
        sample_count: u32,
        total_usage: &mut MemoryUsage,
    ) {
        if let Some(texture) = texture {
            total_usage.gpu_texture_bytes = total_usage
                .gpu_texture_bytes
                .saturating_add(texture_memory_size(texture.size(), format, sample_count, 1));
        }
    }

    fn create_default_shape_texture_bind_group(
        device: &Arc<wgpu::Device>,
        queue: &Arc<wgpu::Queue>,
        shape_texture_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> (wgpu::BindGroup, wgpu::BindGroupLayout) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("default_transparent_texture"),
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

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
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
        });

        (bind_group, shape_texture_bind_group_layout.clone())
    }

    fn create_default_backdrop_texture_bind_group(
        device: &Arc<wgpu::Device>,
        queue: &Arc<wgpu::Queue>,
        backdrop_texture_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> wgpu::BindGroup {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("default_transparent_backdrop_texture"),
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

        let material_params_buffer = crate::pipeline::create_buffer_init(
            device,
            Some("default_backdrop_material_params_buffer"),
            bytemuck::bytes_of(&crate::gradient::gpu::GpuMaterialParams::default()),
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );

        device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: backdrop_texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: material_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
            label: Some("default_backdrop_texture_bind_group_transparent"),
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
        let size = physical_size;

        let instance = wgpu::Instance::new(&InstanceDescriptor::default());

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await?;

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

        let swapchain_format = wgpu::TextureFormat::Bgra8UnormSrgb;

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            format: swapchain_format,
            width: size.0,
            height: size.1,
            present_mode: wgpu::PresentMode::AutoVsync,
            desired_maximum_frame_latency: 2,
            alpha_mode: CompositeAlphaMode::Opaque,
            view_formats: vec![],
        };

        let msaa_sample_count = 1;

        Self::build_from_device(
            instance,
            None,
            device,
            queue,
            config,
            size,
            scale_factor,
            msaa_sample_count,
        )
    }

    /// Creates a headless renderer without a window surface, panicking on
    /// any error from [`Self::try_new_headless`] (e.g. no suitable GPU adapter,
    /// invalid scale factor, device/queue creation failure).
    ///
    /// Use `render_to_buffer()` or `render_to_argb32()` to read back rendered
    /// pixels. Calling `render()` on a headless renderer will panic.
    ///
    /// For a non-panicking alternative (e.g. in tests), use
    /// [`Self::try_new_headless`] instead.
    pub async fn new_headless(physical_size: (u32, u32), scale_factor: f64) -> Self {
        Self::try_new_headless(physical_size, scale_factor)
            .await
            .expect("Failed to create headless renderer")
    }

    pub(super) fn recreate_pipelines(&mut self) {
        let canvas_logical_size = to_logical(self.physical_size, self.scale_factor);

        let (
            and_uniforms,
            and_uniform_buffer,
            and_bind_group,
            and_texture_bgl_layer0,
            and_texture_bgl_layer1,
            and_pipeline,
        ) = create_pipeline(
            canvas_logical_size,
            self.scale_factor,
            self.fringe_width,
            &self.device,
            &self.config,
            PipelineType::EqualIncrementStencil,
            self.msaa_sample_count,
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
            self.scale_factor,
            self.fringe_width,
            &self.device,
            &self.config,
            PipelineType::EqualDecrementStencil,
            self.msaa_sample_count,
        );

        self.and_pipeline = Arc::new(and_pipeline);
        self.and_uniforms = and_uniforms;
        self.and_uniform_buffer = and_uniform_buffer;
        self.and_bind_group = and_bind_group;

        self.decrementing_pipeline = Arc::new(decrementing_pipeline);
        self.decrementing_uniforms = decrementing_uniforms;
        self.decrementing_uniform_buffer = decrementing_uniform_buffer;
        self.decrementing_bind_group = decrementing_bind_group;

        self.shape_texture_bind_group_layout_background = Arc::new(and_texture_bgl_layer0);
        self.shape_texture_bind_group_layout_foreground = Arc::new(and_texture_bgl_layer1);
        self.backdrop_texture_bind_group_layout = Arc::new(
            crate::pipeline::create_backdrop_texture_bind_group_layout(&self.device),
        );
        self.shape_texture_layout_epoch += 1;

        self.gradient_bind_group_layout =
            crate::pipeline::create_gradient_bind_group_layout(&self.device);
        self.backdrop_gradient_bind_group_layout =
            crate::pipeline::create_backdrop_gradient_bind_group_layout(&self.device);
        self.gradient_bind_group_layout_epoch += 1;
        self.gradient_ramp_sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("gradient_ramp_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        self.and_gradient_pipeline = Arc::new(crate::pipeline::create_gradient_increment_pipeline(
            &self.device,
            self.config.format,
            self.msaa_sample_count,
            &self.and_pipeline.get_bind_group_layout(0),
            &self.shape_texture_bind_group_layout_background,
            &self.shape_texture_bind_group_layout_foreground,
            &self.gradient_bind_group_layout,
        ));

        let (default_shape_texture_bind_group_background, _) =
            Self::create_default_shape_texture_bind_group(
                &self.device,
                &self.queue,
                &self.shape_texture_bind_group_layout_background,
            );
        let (default_shape_texture_bind_group_foreground, _) =
            Self::create_default_shape_texture_bind_group(
                &self.device,
                &self.queue,
                &self.shape_texture_bind_group_layout_foreground,
            );
        let default_backdrop_texture_bind_group = Self::create_default_backdrop_texture_bind_group(
            &self.device,
            &self.queue,
            &self.backdrop_texture_bind_group_layout,
        );
        self.default_shape_texture_bind_groups = [
            Arc::new(default_shape_texture_bind_group_background),
            Arc::new(default_shape_texture_bind_group_foreground),
        ];
        self.default_backdrop_texture_bind_group = Arc::new(default_backdrop_texture_bind_group);

        self.composite_pipeline = None;
        self.composite_bgl = None;

        self.leaf_draw_pipeline = Arc::new(crate::pipeline::create_stencil_keep_color_pipeline(
            &self.device,
            self.config.format,
            self.msaa_sample_count,
            &self.and_pipeline.get_bind_group_layout(0),
            &self.shape_texture_bind_group_layout_background,
            &self.shape_texture_bind_group_layout_foreground,
        ));
        self.leaf_draw_gradient_pipeline = Arc::new(
            crate::pipeline::create_gradient_stencil_keep_color_pipeline(
                &self.device,
                self.config.format,
                self.msaa_sample_count,
                &self.and_pipeline.get_bind_group_layout(0),
                &self.shape_texture_bind_group_layout_background,
                &self.shape_texture_bind_group_layout_foreground,
                &self.gradient_bind_group_layout,
            ),
        );

        // Reset lazily-created pipelines so they pick up the new layout
        self.texture_blit_pipeline = None;
        self.stencil_only_pipeline = None;
        self.backdrop_color_pipeline = None;
        self.backdrop_color_gradient_pipeline = None;

        // Refresh per-shape gradient bind groups against the new layout so the
        // next render does not allocate gradient resources on the render path.
        self.buffers_pool_manager.gradient_cache.clear_bind_groups();
        for (_node_id, draw_command) in self.draw_tree.iter_mut() {
            draw_command.refresh_gradient_bind_group(
                &mut self.buffers_pool_manager.gradient_cache,
                &self.device,
                &self.queue,
                &self.gradient_bind_group_layout,
                &self.gradient_ramp_sampler,
                self.gradient_bind_group_layout_epoch,
            );

            if let crate::renderer::types::DrawCommand::CachedShape(cached_shape) = draw_command {
                cached_shape.backdrop_gradient_bind_group = None;
                cached_shape.backdrop_gradient_texture_id = None;
            }
        }

        for effect_instance in self.backdrop_effects.values_mut() {
            effect_instance.backdrop_texture_bind_group = None;
            effect_instance.backdrop_texture_id = None;
        }
    }
}
