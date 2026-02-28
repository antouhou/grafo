use super::*;

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

        let swapchain_format = wgpu::TextureFormat::Bgra8UnormSrgb;

        let surface_caps = surface.get_capabilities(&adapter);
        let alpha_mode = if transparent
            && surface_caps
                .alpha_modes
                .contains(&CompositeAlphaMode::PreMultiplied)
        {
            log::info!("Using PreMultiplied alpha mode for transparency");
            CompositeAlphaMode::PreMultiplied
        } else if transparent
            && surface_caps
                .alpha_modes
                .contains(&CompositeAlphaMode::PostMultiplied)
        {
            log::info!("Using PostMultiplied alpha mode for transparency");
            CompositeAlphaMode::PostMultiplied
        } else {
            if transparent {
                log::warn!("Transparency requested but no suitable alpha mode available, falling back to Opaque");
            }
            CompositeAlphaMode::Opaque
        };

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
    ) -> Self {
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

        let leaf_draw_pipeline = crate::pipeline::create_stencil_keep_color_pipeline(
            &device,
            config.format,
            msaa_sample_count,
            &and_pipeline.get_bind_group_layout(0),
            &and_texture_bgl_layer0,
            &and_texture_bgl_layer1,
        );

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        let texture_manager = TextureManager::new(device.clone(), queue.clone());

        let (default_shape_texture_bind_group_layer0, shape_texture_bind_group_layout_layer0) =
            Self::create_default_shape_texture_bind_group(&device, &queue, &and_texture_bgl_layer0);
        let (default_shape_texture_bind_group_layer1, shape_texture_bind_group_layout_layer1) =
            Self::create_default_shape_texture_bind_group(&device, &queue, &and_texture_bgl_layer1);

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
            shape_texture_layout_epoch: 0,
            default_shape_texture_bind_groups: [
                Arc::new(default_shape_texture_bind_group_layer0),
                Arc::new(default_shape_texture_bind_group_layer1),
            ],
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
            backdrop_snapshot_texture: None,
            backdrop_snapshot_view: None,
            stencil_only_pipeline: None,
            backdrop_color_pipeline: None,
            leaf_draw_pipeline: Arc::new(leaf_draw_pipeline),
            #[cfg(feature = "render_metrics")]
            render_loop_metrics_tracker: RenderLoopMetricsTracker::default(),
            #[cfg(feature = "render_metrics")]
            last_phase_timings: Default::default(),
            #[cfg(feature = "render_metrics")]
            last_pipeline_switch_counts: Default::default(),
            scratch: RendererScratch::new(),
        };

        renderer.recreate_msaa_texture();
        renderer.recreate_depth_stencil_texture();
        renderer
    }

    pub fn print_memory_usage_info(&self) {
        println!("=== Memory Usage Info ===");

        println!("Cached shapes: {}", self.shape_cache.len());
        println!("Draw tree size: {}", self.draw_tree.len());
        println!(
            "Metadata to clips mappings: {}",
            self.metadata_to_clips.len()
        );

        println!("\n--- Temporary Vectors ---");
        println!(
            "Temp vertices: {} items, {} capacity, ~{} bytes",
            self.temp_vertices.len(),
            self.temp_vertices.capacity(),
            self.temp_vertices.capacity() * std::mem::size_of::<crate::vertex::CustomVertex>()
        );
        println!(
            "Temp indices: {} items, {} capacity, ~{} bytes",
            self.temp_indices.len(),
            self.temp_indices.capacity(),
            self.temp_indices.capacity() * std::mem::size_of::<u16>()
        );
        println!(
            "Temp instance transforms: {} items, {} capacity, ~{} bytes",
            self.temp_instance_transforms.len(),
            self.temp_instance_transforms.capacity(),
            self.temp_instance_transforms.capacity() * std::mem::size_of::<InstanceTransform>()
        );
        println!(
            "Temp instance colors: {} items, {} capacity, ~{} bytes",
            self.temp_instance_colors.len(),
            self.temp_instance_colors.capacity(),
            self.temp_instance_colors.capacity() * std::mem::size_of::<InstanceColor>()
        );
        println!(
            "Temp instance metadata: {} items, {} capacity, ~{} bytes",
            self.temp_instance_metadata.len(),
            self.temp_instance_metadata.capacity(),
            self.temp_instance_metadata.capacity() * std::mem::size_of::<InstanceMetadata>()
        );

        println!("\n--- GPU Buffers ---");
        if let Some(buf) = &self.aggregated_vertex_buffer {
            println!("Aggregated vertex buffer: {} bytes", buf.size());
        }
        if let Some(buf) = &self.aggregated_index_buffer {
            println!("Aggregated index buffer: {} bytes", buf.size());
        }
        if let Some(buf) = &self.aggregated_instance_transform_buffer {
            println!("Aggregated instance transform buffer: {} bytes", buf.size());
        }
        if let Some(buf) = &self.aggregated_instance_color_buffer {
            println!("Aggregated instance color buffer: {} bytes", buf.size());
        }
        if let Some(buf) = &self.aggregated_instance_metadata_buffer {
            println!("Aggregated instance metadata buffer: {} bytes", buf.size());
        }
        if let Some(buf) = &self.identity_instance_transform_buffer {
            println!("Identity instance transform buffer: {} bytes", buf.size());
        }
        if let Some(buf) = &self.identity_instance_color_buffer {
            println!("Identity instance color buffer: {} bytes", buf.size());
        }
        if let Some(buf) = &self.identity_instance_metadata_buffer {
            println!("Identity instance metadata buffer: {} bytes", buf.size());
        }

        println!("\n--- ARGB Compute Buffers ---");
        if let Some(buf) = &self.argb_input_buffer {
            println!(
                "ARGB input buffer: {} bytes (cached size: {})",
                buf.size(),
                self.argb_input_buffer_size
            );
        }
        if let Some(buf) = &self.argb_output_storage_buffer {
            println!(
                "ARGB output storage buffer: {} bytes (cached size: {})",
                buf.size(),
                self.argb_output_buffer_size
            );
        }
        if let Some(buf) = &self.argb_readback_buffer {
            println!("ARGB readback buffer: {} bytes", buf.size());
        }
        if let Some(buf) = &self.argb_params_buffer {
            println!("ARGB params buffer: {} bytes", buf.size());
        }
        if let Some(tex) = &self.argb_offscreen_texture {
            let size = tex.size();
            println!(
                "ARGB offscreen texture: {}x{} (cached: {}x{})",
                size.width, size.height, self.argb_cached_width, self.argb_cached_height
            );
        }

        println!("\n--- Render-to-Buffer Caches ---");
        if let Some(tex) = &self.rtb_offscreen_texture {
            let size = tex.size();
            println!(
                "RTB offscreen texture: {}x{} (cached: {}x{})",
                size.width, size.height, self.rtb_cached_width, self.rtb_cached_height
            );
        }
        if let Some(buf) = &self.rtb_readback_buffer {
            println!("RTB readback buffer: {} bytes", buf.size());
        }

        println!("\n--- Uniform Buffers ---");
        println!(
            "AND uniform buffer: {} bytes",
            self.and_uniform_buffer.size()
        );
        println!(
            "Decrementing uniform buffer: {} bytes",
            self.decrementing_uniform_buffer.size()
        );

        println!("\n--- Texture Manager ---");
        println!("{:?}", self.texture_manager.size());

        println!("\n--- Buffer Pool Manager ---");
        self.buffers_pool_manager.print_sizes();

        println!("=========================");
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
    /// Returns `None` if no suitable GPU adapter is available. This is useful
    /// in environments without a GPU (e.g. CI), where tests can skip gracefully
    /// instead of panicking.
    pub async fn try_new_headless(physical_size: (u32, u32), scale_factor: f64) -> Option<Self> {
        let size = physical_size;

        let instance = wgpu::Instance::new(&InstanceDescriptor::default());

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok()?;

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
            .ok()?;

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

        Some(Self::build_from_device(
            instance,
            None,
            device,
            queue,
            config,
            size,
            scale_factor,
            msaa_sample_count,
        ))
    }

    /// Creates a headless renderer without a window surface, panicking if no
    /// suitable GPU adapter is available.
    ///
    /// Use `render_to_buffer()` or `render_to_argb32()` to read back rendered
    /// pixels. Calling `render()` on a headless renderer will panic.
    ///
    /// If you need a non-panicking variant (e.g. in tests), use
    /// [`Self::try_new_headless`] instead.
    pub async fn new_headless(physical_size: (u32, u32), scale_factor: f64) -> Self {
        Self::try_new_headless(physical_size, scale_factor)
            .await
            .expect("Failed to find a suitable GPU adapter for headless rendering")
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
        self.shape_texture_layout_epoch += 1;

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
        self.default_shape_texture_bind_groups = [
            Arc::new(default_shape_texture_bind_group_background),
            Arc::new(default_shape_texture_bind_group_foreground),
        ];

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
    }
}
