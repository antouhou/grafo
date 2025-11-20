//! Renderer for the Grafo library.
//!
//! This module provides the [`Renderer`] struct, which is responsible for rendering shapes,
//! images. It leverages the `wgpu` crate for GPU-accelerated rendering and integrates
//! with other modules like `shape` and `image_draw_data` to manage various rendering
//!     }
//!     fn window_event(&mut self, _: &ActiveEventLoop, _: winit::window::WindowId, _: winit::event::WindowEvent) {}
//! }
//! ```

use std::sync::Arc;

pub type MathRect = lyon::math::Box2D;

use crate::pipeline::{
    compute_padded_bytes_per_row, create_and_depth_texture, create_argb_swizzle_bind_group,
    create_argb_swizzle_pipeline, create_offscreen_color_texture, create_pipeline,
    create_readback_buffer, create_render_pass, create_storage_input_buffer,
    create_storage_output_buffer, encode_copy_texture_to_buffer, render_buffer_range_to_texture,
    ArgbParams, PipelineType, Uniforms,
};
use crate::shape::{CachedShapeDrawData, DrawShapeCommand, Shape, ShapeDrawData};
use crate::texture_manager::TextureManager;
use crate::util::{to_logical, PoolManager};
use crate::vertex::{InstanceColor, InstanceTransform, InstanceRenderParams};
use crate::{transformator, CachedShape};
use crate::Color;
use ahash::{HashMap, HashMapExt};
use log::warn;
use lyon::tessellation::FillTessellator;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{BindGroup, BufferUsages, CompositeAlphaMode, InstanceDescriptor, SurfaceTarget};

/// Represents different rendering pipelines used by the `Renderer`.
///
/// Each variant corresponds to a specific rendering pipeline configuration.
#[derive(Debug, Clone, Copy)]
pub enum Pipeline {
    /// No specific pipeline.
    None,
    /// Pipeline for incrementing stencil values.
    StencilIncrement,
    /// Pipeline for decrementing stencil values.
    StencilDecrement,
}

/// Semantic texture layers for a shape. Background is layer 0, Foreground is layer 1.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum TextureLayer {
    Background,
    Foreground,
}

impl From<TextureLayer> for usize {
    fn from(value: TextureLayer) -> Self {
        match value {
            TextureLayer::Background => 0,
            TextureLayer::Foreground => 1,
        }
    }
}

/// Calculates the depth value based on the draw command's ID and the total number of draw commands.
///
/// Depth values are clamped between `0.0000000001` and `0.9999999999` to avoid precision issues.
///
/// # Parameters
///
/// - `draw_command_id`: The identifier of the current draw command.
/// - `draw_commands_total`: The total number of draw commands.
///
/// # Returns
///
/// A `f32` representing the normalized depth value.
#[inline(always)]
pub fn order_value(draw_command_id: usize, draw_commands_total: usize) -> f32 {
    (1.0 - (draw_command_id as f32 / draw_commands_total as f32)).clamp(0.0000000001, 0.9999999999)
}

/// Represents a draw command, which can be either a shape or an image.
///
/// This enum is used internally by the `Renderer` to manage different types of draw operations.
#[derive(Debug)]
enum DrawCommand {
    Shape(ShapeDrawData),
    CachedShape(CachedShapeDrawData),
}

/// The renderer for the Grafo library. This is the main struct used to render shapes and images.
///
/// # Examples
///
/// ```rust,no_run
/// use grafo::Renderer;
/// use grafo::Shape;
/// use grafo::Color;
/// use grafo::Stroke;
/// use wgpu::Surface;
/// use winit::application::ApplicationHandler;
/// use winit::event_loop::{ActiveEventLoop, EventLoop};
/// use winit::window::Window;
/// use std::sync::Arc;
/// use futures::executor::block_on;
///
/// // This is for demonstration purposes only. If you want a working example with winit, please
/// // refer to the example in the "examples" folder.
/// struct App;
/// impl ApplicationHandler for App {
///     fn resumed(&mut self, event_loop: &ActiveEventLoop) {
///         let window_surface = Arc::new(
///             event_loop.create_window(Window::default_attributes()).unwrap()
///         );
///         let physical_size = (800, 600);
///         let scale_factor = 1.0;
///
///         // Initialize the renderer
///         let mut renderer = block_on(Renderer::new(window_surface, physical_size, scale_factor, true, false));
///
///         // Add a rectangle shape
///         let rect = Shape::rect(
///             [(0.0, 0.0), (200.0, 100.0)],
///             Stroke::new(2.0, Color::BLACK), // Black stroke with width 2.0
///         );
///         let rect_id = renderer.add_shape(rect, None, None);
///         // Blue fill
///         renderer.set_shape_color(rect_id, Some(Color::rgb(0, 128, 255)));
///         renderer.set_shape_transform(rect_id, grafo::TransformInstance::identity());
///
///         // Render the frame
///         match renderer.render() {
///             Ok(_) => println!("Frame rendered successfully."),
///             Err(e) => eprintln!("Rendering error: {:?}", e),
///         }
///
///         // Clear the draw queue after rendering
///         renderer.clear_draw_queue();
///     }
///     fn window_event(&mut self, _: &ActiveEventLoop, _: winit::window::WindowId, _: winit::event::WindowEvent) {}
/// }
/// ```
pub struct Renderer<'a> {
    // Window information
    /// Size of the window in pixels.
    pub(crate) physical_size: (u32, u32),
    /// Scale factor of the window (e.g., for high-DPI displays).
    scale_factor: f64,

    // WGPU components
    instance: wgpu::Instance,
    surface: wgpu::Surface<'a>,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    config: wgpu::SurfaceConfiguration,

    tessellator: FillTessellator,

    buffers_pool_manager: PoolManager,

    texture_manager: TextureManager,

    /// Tree structure holding shapes and images to be rendered.
    draw_tree: easy_tree::Tree<DrawCommand>,
    metadata_to_clips: HashMap<usize, usize>,

    /// Uniforms for the "And" rendering pipeline.
    and_uniforms: Uniforms,
    /// GPU buffer backing the and pipeline uniforms (for cheap updates).
    and_uniform_buffer: wgpu::Buffer,
    /// Bind group for the "And" rendering pipeline.
    and_bind_group: BindGroup,
    /// Render pipeline for the "And" operations.
    and_pipeline: Arc<wgpu::RenderPipeline>,
    /// Bind group layouts for shape texture layers (groups 1 and 2)
    shape_texture_bind_group_layout_background: Arc<wgpu::BindGroupLayout>,
    shape_texture_bind_group_layout_foreground: Arc<wgpu::BindGroupLayout>,
    /// Monotonic counter to invalidate cached shape texture bind groups when the layout changes
    shape_texture_layout_epoch: u64,
    /// Default transparent texture bind groups for both layers
    default_shape_texture_bind_groups: [Arc<wgpu::BindGroup>; 2], // [background, foreground]

    /// Render pipeline for decrementing stencil values.
    decrementing_pipeline: Arc<wgpu::RenderPipeline>,
    /// Uniforms for the decrementing pipeline.
    decrementing_uniforms: Uniforms,
    /// GPU buffer backing the decrementing pipeline uniforms.
    decrementing_uniform_buffer: wgpu::Buffer,
    /// Bind group for the decrementing pipeline.
    decrementing_bind_group: BindGroup,

    // #[cfg(feature = "performance_measurement")]
    // performance_query_set: wgpu::QuerySet,
    // #[cfg(feature = "performance_measurement")]
    // adapter: wgpu::Adapter,
    temp_vertices: Vec<crate::vertex::CustomVertex>,
    temp_indices: Vec<u16>,

    /// Per-frame instance transforms for shapes
    temp_instance_transforms: Vec<InstanceTransform>,
    /// Per-frame instance colors for shapes
    temp_instance_colors: Vec<InstanceColor>,
    /// Per-frame instance render params for shapes
    temp_instance_render_params: Vec<InstanceRenderParams>,

    /// Reusable aggregated vertex buffer
    aggregated_vertex_buffer: Option<wgpu::Buffer>,
    /// Reusable aggregated index buffer  
    aggregated_index_buffer: Option<wgpu::Buffer>,
    /// Per-frame instance transform GPU buffer
    aggregated_instance_transform_buffer: Option<wgpu::Buffer>,
    /// Per-frame instance color GPU buffer
    aggregated_instance_color_buffer: Option<wgpu::Buffer>,
    /// Per-frame instance render params GPU buffer
    aggregated_instance_render_params_buffer: Option<wgpu::Buffer>,

    /// Identity transform instance buffer
    identity_instance_transform_buffer: Option<wgpu::Buffer>,
    /// Identity color instance buffer (white)
    identity_instance_color_buffer: Option<wgpu::Buffer>,
    /// Identity render params instance buffer (default values)
    identity_instance_render_params_buffer: Option<wgpu::Buffer>,

    shape_cache: HashMap<u64, CachedShape>,

    // Cached resources for render_to_argb32 compute swizzle path
    argb_cs_bgl: Option<wgpu::BindGroupLayout>,
    argb_cs_pipeline: Option<wgpu::ComputePipeline>,
    argb_swizzle_bind_group: Option<wgpu::BindGroup>,
    argb_params_buffer: Option<wgpu::Buffer>,
    argb_input_buffer: Option<wgpu::Buffer>,
    argb_output_storage_buffer: Option<wgpu::Buffer>,
    argb_readback_buffer: Option<wgpu::Buffer>,
    argb_input_buffer_size: u64,
    argb_output_buffer_size: u64,
    argb_cached_width: u32,
    argb_cached_height: u32,
    argb_offscreen_texture: Option<wgpu::Texture>,
    // Cached resources for render_to_buffer (BGRA bytes) path
    rtb_offscreen_texture: Option<wgpu::Texture>,
    rtb_readback_buffer: Option<wgpu::Buffer>,
    rtb_cached_width: u32,
    rtb_cached_height: u32,
}

impl<'a> Renderer<'a> {
    /// Creates a new `Renderer` instance.
    ///
    /// # Parameters
    ///
    /// - `window`: The surface target (e.g., window) where rendering will occur.
    /// - `physical_size`: The physical size of the window in pixels.
    /// - `scale_factor`: The scale factor for high-DPI displays.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use std::sync::Arc;
    /// use futures::executor::block_on;
    /// use grafo::Renderer;
    /// use winit::application::ApplicationHandler;
    /// use winit::event_loop::{ActiveEventLoop, EventLoop};
    /// use winit::window::Window;
    ///
    /// // This is for demonstration purposes only. If you want a working example with winit, please
    /// // refer to the example in the "examples" folder.
    /// struct App;
    /// impl ApplicationHandler for App {
    ///     fn resumed(&mut self, event_loop: &ActiveEventLoop) {
    ///         let window_surface = Arc::new(
    ///             event_loop.create_window(Window::default_attributes()).unwrap()
    ///         );
    ///         let physical_size = (800, 600);
    ///         let scale_factor = 1.0;
    ///
    ///         // Initialize the renderer
    ///         let renderer = block_on(Renderer::new(window_surface, physical_size, scale_factor, true, false));
    ///     }
    ///     fn window_event(&mut self, _: &ActiveEventLoop, _: winit::window::WindowId, _: winit::event::WindowEvent) {}
    /// }
    /// ```
    pub async fn new(
        window: impl Into<SurfaceTarget<'static>>,
        physical_size: (u32, u32),
        scale_factor: f64,
        vsync: bool,
        transparent: bool,
    ) -> Self {
        let size = physical_size;
        let canvas_logical_size = to_logical(size, scale_factor);

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

        // #[cfg(feature = "performance_measurement")]
        // let frametime_query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
        //     label: wgpu::Label::Some("Frametime Query Set"),
        //     count: 2, // We will use two timestamps
        //     ty: wgpu::QueryType::Timestamp,
        // });

        // surface.get_preferred_format(&adapter).unwrap()

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
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
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

        let (
            and_uniforms,
            and_uniform_buffer,
            and_bind_group,
            and_texture_bgl_layer0,
            and_texture_bgl_layer1,
            and_pipeline,
        ) = create_pipeline(
            canvas_logical_size,
            &device,
            &config,
            PipelineType::EqualIncrementStencil,
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
            &device,
            &config,
            PipelineType::EqualDecrementStencil,
        );

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        let texture_manager = TextureManager::new(device.clone(), queue.clone());

        // Create default transparent texture and bind group for shapes
        let (default_shape_texture_bind_group_layer0, shape_texture_bind_group_layout_layer0) =
            Self::create_default_shape_texture_bind_group(&device, &queue, &and_texture_bgl_layer0);
        let (default_shape_texture_bind_group_layer1, shape_texture_bind_group_layout_layer1) =
            Self::create_default_shape_texture_bind_group(&device, &queue, &and_texture_bgl_layer1);

        Self {
            instance,
            surface,
            device,
            queue,
            config,
            physical_size: size,

            scale_factor,

            tessellator: FillTessellator::new(),

            texture_manager,

            buffers_pool_manager: PoolManager::new(),

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

            // #[cfg(feature = "performance_measurement")]
            // performance_query_set: frametime_query_set,
            // #[cfg(feature = "performance_measurement")]
            // adapter,
            temp_vertices: Vec::new(),
            temp_indices: Vec::new(),
            temp_instance_transforms: Vec::new(),
            temp_instance_colors: Vec::new(),
            temp_instance_render_params: Vec::new(),
            aggregated_vertex_buffer: None,
            aggregated_index_buffer: None,
            aggregated_instance_transform_buffer: None,
            aggregated_instance_color_buffer: None,
            aggregated_instance_render_params_buffer: None,
            identity_instance_transform_buffer: None,
            identity_instance_color_buffer: None,
            identity_instance_render_params_buffer: None,

            shape_cache: HashMap::new(),

            // ARGB compute path caches
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
            // render_to_buffer caches
            rtb_offscreen_texture: None,
            rtb_readback_buffer: None,
            rtb_cached_width: 0,
            rtb_cached_height: 0,
        }
    }

    fn create_default_shape_texture_bind_group(
        device: &Arc<wgpu::Device>,
        queue: &Arc<wgpu::Queue>,
        shape_texture_bgl: &wgpu::BindGroupLayout,
    ) -> (wgpu::BindGroup, wgpu::BindGroupLayout) {
        // Transparent 1x1 pixel to preserve shape transparency when no texture is set
        let tex = device.create_texture(&wgpu::TextureDescriptor {
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
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        // RGBA transparent pixel
        let transparent: [u8; 4] = [0, 0, 0, 0];
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &tex,
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
            layout: shape_texture_bgl,
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

        (bind_group, shape_texture_bgl.clone())
    }

    /// Creates a new transparent `Renderer` instance.
    ///
    /// This is a convenience method that calls [`Renderer::new`] with `transparent` set to `true`.
    ///
    /// # Parameters
    ///
    /// - `window`: The window surface target to render to.
    /// - `physical_size`: The physical size of the rendering surface in pixels.
    /// - `scale_factor`: The DPI scale factor of the window.
    /// - `vsync`: Whether to enable vertical synchronization.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use futures::executor::block_on;
    /// use std::sync::Arc;
    /// use winit::application::ApplicationHandler;
    /// use winit::event_loop::{ActiveEventLoop, EventLoop};
    /// use winit::window::{Window, WindowAttributes};
    ///
    /// struct App;
    /// impl ApplicationHandler for App {
    ///     fn resumed(&mut self, event_loop: &ActiveEventLoop) {
    ///         let window_attributes = Window::default_attributes().with_transparent(true);
    ///         let window = Arc::new(
    ///             event_loop.create_window(window_attributes).unwrap()
    ///         );
    ///         let physical_size = (800, 600);
    ///         let scale_factor = 1.0;
    ///
    ///         let mut renderer = block_on(grafo::Renderer::new_transparent(
    ///             window.clone(),
    ///             physical_size,
    ///             scale_factor,
    ///             true
    ///         ));
    ///     }
    ///     fn window_event(&mut self, _: &ActiveEventLoop, _: winit::window::WindowId, _: winit::event::WindowEvent) {}
    /// }
    /// ```
    pub async fn new_transparent(
        window: impl Into<SurfaceTarget<'static>>,
        physical_size: (u32, u32),
        scale_factor: f64,
        vsync: bool,
    ) -> Self {
        Self::new(window, physical_size, scale_factor, vsync, true).await
    }

    /// Adds a shape to the draw queue.
    ///
    /// # Parameters
    ///
    /// - `shape`: The shape to be rendered. It can be any type that implements `Into<Shape>`. If
    ///   you're going to render a lot of shapes with the same outline, it is
    ///   recommended to start shapes at 0.0, 0.0 where possible and use per-shape transforms to move them on
    ///   the screen. This way, it is going to be possible to cache tesselation results for
    ///   such shapes, which would increase rendering time. This is useful if you render a lot
    ///   of buttons with rounded corners, for example. Caching requires supplying a cache key
    ///   to cache tessellated shape.
    /// - `clip_to_shape`: Optional index of another shape to which this shape should be clipped.
    /// - `cache_key`: A key that is going to be used for tesselation caching.
    ///
    /// # Returns
    ///
    /// The unique identifier (`usize`) assigned to the added shape.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use std::sync::Arc;
    /// use futures::executor::block_on;
    /// use winit::application::ApplicationHandler;
    /// use winit::event_loop::{ActiveEventLoop, EventLoop};
    /// use winit::window::Window;
    /// use grafo::Renderer;
    /// use grafo::Shape;
    /// use grafo::Color;
    /// use grafo::Stroke;
    ///
    /// // This is for demonstration purposes only. If you want a working example with winit, please
    /// // refer to the example in the "examples" folder.
    /// struct App;
    /// impl ApplicationHandler for App {
    ///     fn resumed(&mut self, event_loop: &ActiveEventLoop) {
    ///         let window_surface = Arc::new(
    ///             event_loop.create_window(Window::default_attributes()).unwrap()
    ///         );
    ///         let physical_size = (800, 600);
    ///         let scale_factor = 1.0;
    ///
    ///         // Initialize the renderer
    ///         let mut renderer = block_on(Renderer::new(window_surface, physical_size, scale_factor, true, false));
    ///
    ///         let shape_id = renderer.add_shape(
    ///             Shape::rect(
    ///                 [(0.0, 100.0), (100.0, 100.0)],
    ///                 Stroke::new(2.0, Color::BLACK), // Black stroke with width 2.0
    ///             ),
    ///             None,
    ///             None,
    ///         );
    ///         // Blue fill
    ///         renderer.set_shape_color(shape_id, Some(Color::rgb(0, 128, 255)));
    ///         renderer.set_shape_transform(shape_id, grafo::TransformInstance::identity());
    ///     }
    ///     fn window_event(&mut self, _: &ActiveEventLoop, _: winit::window::WindowId, _: winit::event::WindowEvent) {}
    /// }
    /// ```
    pub fn add_shape(
        &mut self,
        shape: impl Into<Shape>,
        clip_to_shape: Option<usize>,
        cache_key: Option<u64>,
    ) -> usize {
        self.add_draw_command(
            DrawCommand::Shape(ShapeDrawData::new(shape, cache_key)),
            clip_to_shape,
        )
    }

    pub fn load_shape(
        &mut self,
        shape: impl AsRef<Shape>,
        cache_key: u64,
        tessellation_cache_key: Option<u64>,
    ) {
        let cached_shape = CachedShape::new(
            shape.as_ref(),
            0.0,
            &mut self.tessellator,
            &mut self.buffers_pool_manager,
            tessellation_cache_key,
        );
        self.shape_cache.insert(cache_key, cached_shape);
    }

    pub fn add_cached_shape_to_the_render_queue(
        &mut self,
        cache_key: u64,
        clip_to_shape: Option<usize>,
    ) -> usize {
        self.add_draw_command(
            DrawCommand::CachedShape(CachedShapeDrawData::new(cache_key)),
            clip_to_shape,
        )
    }

    /// A texture manager is a helper to allow more granular approach to drawing images. It can
    /// be cloned and passed to a different thread if you want to update texture
    pub fn texture_manager(&self) -> &TextureManager {
        &self.texture_manager
    }

    /// Prepares all draw commands for rendering by tessellating shapes and building GPU buffers.
    fn prepare_render(&mut self) {
        self.temp_vertices.clear();
        self.temp_indices.clear();
        self.temp_instance_transforms.clear();
        self.temp_instance_colors.clear();
        self.temp_instance_render_params.clear();

        let draw_tree_size = self.draw_tree.len();

        let tessellator = &mut self.tessellator;
        let buffers_pool_manager = &mut self.buffers_pool_manager;

        // First pass: tessellate all shapes and aggregate vertex/index and instance data
        for (node_id, draw_command) in self.draw_tree.iter_mut() {
            match draw_command {
                DrawCommand::Shape(ref mut shape) => {
                    let depth = order_value(node_id, draw_tree_size);
                    let vertex_buffers = shape.tessellate(depth, tessellator, buffers_pool_manager);

                    if vertex_buffers.vertices.is_empty() || vertex_buffers.indices.is_empty() {
                        shape.is_empty = true;
                        continue;
                    }

                    let vertex_start = self.temp_vertices.len();
                    let index_start = self.temp_indices.len();
                    let vertex_offset = vertex_start as u16;

                    self.temp_vertices
                        .extend_from_slice(&vertex_buffers.vertices);

                    // Offset indices by the current vertex count
                    for &index in &vertex_buffers.indices {
                        self.temp_indices.push(index + vertex_offset);
                    }

                    let index_count = vertex_buffers.indices.len();

                    shape.index_buffer_range = Some((index_start, index_count));

                    // Collect per-instance data (transform + fill color or override + render params)
                    let instance_idx = self.temp_instance_transforms.len();
                    let transform = shape
                        .transform()
                        .unwrap_or_else(InstanceTransform::identity);
                    let color = shape
                        .instance_color_override()
                        .unwrap_or([1.0, 1.0, 1.0, 1.0]);
                    let render_params = shape
                        .render_params()
                        .unwrap_or_default();
                    self.temp_instance_transforms.push(transform);
                    self.temp_instance_colors.push(InstanceColor { color });
                    self.temp_instance_render_params.push(render_params);
                    *shape.instance_index_mut() = Some(instance_idx);
                }
                DrawCommand::CachedShape(cached_shape_data) => {
                    let depth = order_value(node_id, draw_tree_size);

                    if let Some(cached_shape) = self.shape_cache.get_mut(&cached_shape_data.id) {
                        if cached_shape.vertex_buffers.vertices.is_empty()
                            || cached_shape.vertex_buffers.indices.is_empty()
                        {
                            cached_shape_data.is_empty = true;
                            continue;
                        }

                        if depth != cached_shape.depth {
                            cached_shape.set_depth(depth);
                        }

                        let vertex_start = self.temp_vertices.len();
                        let index_start = self.temp_indices.len();
                        let vertex_offset = vertex_start as u16;

                        let vertex_buffers = &cached_shape.vertex_buffers;

                        self.temp_vertices
                            .extend_from_slice(&vertex_buffers.vertices);

                        // Offset indices by the current vertex count
                        for &index in &vertex_buffers.indices {
                            self.temp_indices.push(index + vertex_offset);
                        }

                        let index_count = vertex_buffers.indices.len();

                        cached_shape_data.index_buffer_range = Some((index_start, index_count));

                        // Instance for cached shape (transform + default or override color + render params)
                        let instance_idx = self.temp_instance_transforms.len();
                        let transform = cached_shape_data
                            .transform()
                            .unwrap_or_else(InstanceTransform::identity);
                        let color = cached_shape_data
                            .instance_color_override()
                            .unwrap_or([1.0, 1.0, 1.0, 1.0]);
                        let render_params = cached_shape_data
                            .render_params()
                            .unwrap_or_default();
                        self.temp_instance_transforms.push(transform);
                        self.temp_instance_colors.push(InstanceColor { color });
                        self.temp_instance_render_params.push(render_params);
                        *cached_shape_data.instance_index_mut() = Some(instance_idx);
                    } else {
                        println!("Warning: Cached shape not found in cache");
                    }
                }
            }
        }

        // Create or update aggregated buffers only if needed
        if !self.temp_vertices.is_empty() {
            let required_vertex_size = std::mem::size_of_val(&self.temp_vertices[..]);

            // Check if we need to reallocate the vertex buffer
            let needs_realloc = self
                .aggregated_vertex_buffer
                .as_ref()
                .map(|buffer| buffer.size() < required_vertex_size as u64)
                .unwrap_or(true);

            if needs_realloc {
                self.aggregated_vertex_buffer =
                    Some(self.device.create_buffer_init(&BufferInitDescriptor {
                        label: Some("Aggregated Vertex Buffer"),
                        contents: bytemuck::cast_slice(&self.temp_vertices),
                        usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                    }));
            } else {
                // Update existing buffer content
                self.queue.write_buffer(
                    self.aggregated_vertex_buffer.as_ref().unwrap(),
                    0,
                    bytemuck::cast_slice(&self.temp_vertices),
                );
            }
        }

        if !self.temp_indices.is_empty() {
            let required_index_size = std::mem::size_of_val(&self.temp_indices[..]);

            // Check if we need to reallocate the index buffer
            let needs_realloc = self
                .aggregated_index_buffer
                .as_ref()
                .map(|buffer| buffer.size() < required_index_size as u64)
                .unwrap_or(true);

            if needs_realloc {
                self.aggregated_index_buffer =
                    Some(self.device.create_buffer_init(&BufferInitDescriptor {
                        label: Some("Aggregated Index Buffer"),
                        contents: bytemuck::cast_slice(&self.temp_indices),
                        usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                    }));
            } else {
                // Update existing buffer content
                self.queue.write_buffer(
                    self.aggregated_index_buffer.as_ref().unwrap(),
                    0,
                    bytemuck::cast_slice(&self.temp_indices),
                );
            }
        }

        // Ensure identity instance buffer exists before we start setting up passes
        if self.identity_instance_transform_buffer.is_none() {
            let identity = InstanceTransform::identity();
            self.identity_instance_transform_buffer =
                Some(self.device.create_buffer_init(&BufferInitDescriptor {
                    label: Some("Identity Instance Transform Buffer"),
                    contents: bytemuck::cast_slice(&[identity]),
                    usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                }));
        }
        if self.identity_instance_color_buffer.is_none() {
            let white = InstanceColor::white();
            self.identity_instance_color_buffer =
                Some(self.device.create_buffer_init(&BufferInitDescriptor {
                    label: Some("Identity Instance Color Buffer"),
                    contents: bytemuck::cast_slice(&[white]),
                    usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                }));
        }
        if self.identity_instance_render_params_buffer.is_none() {
            let default_params = InstanceRenderParams::default();
            self.identity_instance_render_params_buffer =
                Some(self.device.create_buffer_init(&BufferInitDescriptor {
                    label: Some("Identity Instance Render Params Buffer"),
                    contents: bytemuck::cast_slice(&[default_params]),
                    usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                }));
        }

        // Create/update aggregated instance transform buffer
        if !self.temp_instance_transforms.is_empty() {
            let required_instance_size = std::mem::size_of_val(&self.temp_instance_transforms[..]);
            let needs_realloc = self
                .aggregated_instance_transform_buffer
                .as_ref()
                .map(|buffer| buffer.size() < required_instance_size as u64)
                .unwrap_or(true);
            if needs_realloc {
                self.aggregated_instance_transform_buffer =
                    Some(self.device.create_buffer_init(&BufferInitDescriptor {
                        label: Some("Aggregated Instance Transform Buffer"),
                        contents: bytemuck::cast_slice(&self.temp_instance_transforms),
                        usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                    }));
            } else {
                self.queue.write_buffer(
                    self.aggregated_instance_transform_buffer.as_ref().unwrap(),
                    0,
                    bytemuck::cast_slice(&self.temp_instance_transforms),
                );
            }
        }
        // Create/update aggregated instance color buffer
        if !self.temp_instance_colors.is_empty() {
            let required_instance_size = std::mem::size_of_val(&self.temp_instance_colors[..]);
            let needs_realloc = self
                .aggregated_instance_color_buffer
                .as_ref()
                .map(|buffer| buffer.size() < required_instance_size as u64)
                .unwrap_or(true);
            if needs_realloc {
                self.aggregated_instance_color_buffer =
                    Some(self.device.create_buffer_init(&BufferInitDescriptor {
                        label: Some("Aggregated Instance Color Buffer"),
                        contents: bytemuck::cast_slice(&self.temp_instance_colors),
                        usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                    }));
            } else {
                self.queue.write_buffer(
                    self.aggregated_instance_color_buffer.as_ref().unwrap(),
                    0,
                    bytemuck::cast_slice(&self.temp_instance_colors),
                );
            }
        }
        
        // Create/update aggregated instance render params buffer
        if !self.temp_instance_render_params.is_empty() {
            let required_instance_size = std::mem::size_of_val(&self.temp_instance_render_params[..]);
            let needs_realloc = self
                .aggregated_instance_render_params_buffer
                .as_ref()
                .map(|buffer| buffer.size() < required_instance_size as u64)
                .unwrap_or(true);
            if needs_realloc {
                self.aggregated_instance_render_params_buffer =
                    Some(self.device.create_buffer_init(&BufferInitDescriptor {
                        label: Some("Aggregated Instance Render Params Buffer"),
                        contents: bytemuck::cast_slice(&self.temp_instance_render_params),
                        usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                    }));
            } else {
                self.queue.write_buffer(
                    self.aggregated_instance_render_params_buffer.as_ref().unwrap(),
                    0,
                    bytemuck::cast_slice(&self.temp_instance_render_params),
                );
            }
        }
    }

    /// Core rendering logic that renders to a texture view.
    fn render_to_texture_view(&mut self, texture_view: &wgpu::TextureView) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Command Encoder"),
            });

        let depth_texture =
            create_and_depth_texture(&self.device, (self.physical_size.0, self.physical_size.1));

        let pipelines = Pipelines {
            and_pipeline: &self.and_pipeline,
            and_bind_group: &self.and_bind_group,
            decrementing_pipeline: &self.decrementing_pipeline,
            decrementing_bind_group: &self.decrementing_bind_group,
            shape_texture_bgl_layer0: &self.shape_texture_bind_group_layout_background,
            shape_texture_bgl_layer1: &self.shape_texture_bind_group_layout_foreground,
            default_shape_texture_bgs: &self.default_shape_texture_bind_groups,
            shape_texture_layout_epoch: self.shape_texture_layout_epoch,
            texture_manager: &self.texture_manager,
        };

        let buffers = Buffers {
            aggregated_vertex_buffer: self.aggregated_vertex_buffer.as_ref().unwrap(),
            aggregated_index_buffer: self.aggregated_index_buffer.as_ref().unwrap(),
            identity_instance_transform_buffer: self
                .identity_instance_transform_buffer
                .as_ref()
                .unwrap(),
            identity_instance_color_buffer: self.identity_instance_color_buffer.as_ref().unwrap(),
            identity_instance_render_params_buffer: self
                .identity_instance_render_params_buffer
                .as_ref()
                .unwrap(),
            aggregated_instance_transform_buffer: self
                .aggregated_instance_transform_buffer
                .as_ref(),
            aggregated_instance_color_buffer: self.aggregated_instance_color_buffer.as_ref(),
            aggregated_instance_render_params_buffer: self
                .aggregated_instance_render_params_buffer
                .as_ref(),
        };

        {
            let depth_texture_view =
                depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

            let render_pass = create_render_pass(&mut encoder, texture_view, &depth_texture_view);

            // Use a simple stack of stencil references along the traversal path.
            // Top of the stack is the current parent stencil reference.
            let stencil_stack: Vec<u32> = Vec::new();
            let current_pipeline = Pipeline::None;

            let mut data = (render_pass, stencil_stack, current_pipeline);

            self.draw_tree.traverse_mut(
                |_shape_id, draw_command, data| {
                    // NOTE: this is destructured here and not above because we need to pass the
                    //  data to the closure below
                    let (render_pass, stencil_stack, currently_set_pipeline) = data;

                    match draw_command {
                        DrawCommand::Shape(shape) => {
                            handle_increment_pass(
                                render_pass,
                                currently_set_pipeline,
                                stencil_stack,
                                shape,
                                &pipelines,
                                &buffers,
                            );
                        }
                        DrawCommand::CachedShape(shape) => {
                            handle_increment_pass(
                                render_pass,
                                currently_set_pipeline,
                                stencil_stack,
                                shape,
                                &pipelines,
                                &buffers,
                            );
                        }
                    }
                },
                |_shape_id, draw_command, data| {
                    let (render_pass, stencil_stack, currently_set_pipeline) = data;

                    match draw_command {
                        DrawCommand::Shape(shape) => {
                            handle_decrement_pass(
                                render_pass,
                                currently_set_pipeline,
                                stencil_stack,
                                shape,
                                &pipelines,
                                &buffers,
                            );
                        }
                        DrawCommand::CachedShape(shape) => {
                            handle_decrement_pass(
                                render_pass,
                                currently_set_pipeline,
                                stencil_stack,
                                shape,
                                &pipelines,
                                &buffers,
                            );
                        }
                    }
                },
                &mut data,
            );
        }

        self.queue.submit(std::iter::once(encoder.finish()));

        // Clear shape buffer references and return other resources to the pool
        self.draw_tree
            .iter_mut()
            .for_each(|draw_command| match draw_command.1 {
                DrawCommand::Shape(ref mut shape) => {
                    // Clear the aggregated buffer references
                    shape.index_buffer_range = None;
                    shape.stencil_ref = None;
                }
                DrawCommand::CachedShape(ref mut shape) => {
                    // Clear the aggregated buffer references
                    shape.index_buffer_range = None;
                    shape.stencil_ref = None;
                }
            });

        // self.buffers_pool_manager.print_sizes();
    }

    /// Renders all items currently in the draw queue to the surface.
    ///
    /// This method processes the draw commands for shapes and images, tessellates them,
    /// prepares GPU buffers, and executes the rendering pipelines to produce the final frame.
    ///
    /// # Errors
    ///
    /// Returns a [`wgpu::SurfaceError`] if acquiring the next frame fails.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use std::sync::Arc;
    /// use futures::executor::block_on;
    /// use winit::application::ApplicationHandler;
    /// use winit::event_loop::{ActiveEventLoop, EventLoop};
    /// use winit::window::Window;
    /// use grafo::Renderer;
    /// use grafo::Shape;
    /// use grafo::Color;
    /// use grafo::Stroke;
    ///
    /// // This is for demonstration purposes only. If you want a working example with winit, please
    /// // refer to the example in the "examples" folder.
    /// struct App;
    /// impl ApplicationHandler for App {
    ///     fn resumed(&mut self, event_loop: &ActiveEventLoop) {
    ///         let window_surface = Arc::new(
    ///             event_loop.create_window(Window::default_attributes()).unwrap()
    ///         );
    ///         let physical_size = (800, 600);
    ///         let scale_factor = 1.0;
    ///
    ///         // Initialize the renderer
    ///         let mut renderer = block_on(Renderer::new(window_surface, physical_size, scale_factor, true, false));
    ///
    ///         // Add shapes and images...
    ///
    ///         // Render the frame
    ///         if let Err(e) = renderer.render() {
    ///             eprintln!("Rendering error: {:?}", e);
    ///         }
    ///     }
    ///     fn window_event(&mut self, _: &ActiveEventLoop, _: winit::window::WindowId, _: winit::event::WindowEvent) {}
    /// }
    /// ```
    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        self.prepare_render();

        let output = self.surface.get_current_texture()?;
        let output_texture_view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        self.render_to_texture_view(&output_texture_view);

        output.present();
        Ok(())
    }

    /// Renders all items currently in the draw queue to a buffer.
    ///
    /// This method creates an offscreen texture based on the renderer's current scale and size,
    /// renders to it, and then copies the result to the provided buffer.
    ///
    /// # Parameters
    ///
    /// - `buffer`: A mutable reference to a `Vec<u8>` where the BGRA pixel data will be written.
    ///   The buffer will be resized to fit `width * height * 4` bytes.
    ///   Format is BGRA8UnormSrgb (matching the surface format).
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use futures::executor::block_on;
    /// use grafo::{Renderer, Shape, Color, Stroke};
    ///
    /// // Create a renderer without a window surface
    /// // Note: You'll need to adjust initialization for offscreen rendering
    /// let mut buffer: Vec<u8> = Vec::new();
    /// // renderer.render_to_buffer(&mut buffer);
    /// // buffer now contains BGRA8 pixel data
    /// ```
    pub fn render_to_buffer(&mut self, buffer: &mut Vec<u8>) {
        self.prepare_render();

        let (width, height) = self.physical_size;

        // Recreate caches if size changed
        let size_changed = self.rtb_cached_width != width || self.rtb_cached_height != height;
        if size_changed {
            self.rtb_cached_width = width;
            self.rtb_cached_height = height;
        }

        // Create or reuse offscreen texture for BGRA readback
        if size_changed || self.rtb_offscreen_texture.is_none() {
            self.rtb_offscreen_texture = Some(create_offscreen_color_texture(
                &self.device,
                (width, height),
                self.config.format,
            ));
        }
        let texture_view = self
            .rtb_offscreen_texture
            .as_ref()
            .unwrap()
            .create_view(&wgpu::TextureViewDescriptor::default());

        self.render_to_texture_view(&texture_view);

        // Compute row strides (wgpu requires 256-byte alignment)
        let (unpadded_bytes_per_row, padded_bytes_per_row) = compute_padded_bytes_per_row(width, 4);

        // Create or reuse readback buffer to copy texture data into
        let buffer_size = (padded_bytes_per_row * height) as u64;
        if size_changed
            || self
                .rtb_readback_buffer
                .as_ref()
                .map(|b| b.size() < buffer_size)
                .unwrap_or(true)
        {
            self.rtb_readback_buffer = Some(create_readback_buffer(
                &self.device,
                Some("rtb_readback_buffer"),
                buffer_size,
            ));
        }
        let output_buffer = self.rtb_readback_buffer.as_ref().unwrap();

        // Copy texture to buffer
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("copy_texture_encoder"),
            });

        encode_copy_texture_to_buffer(
            &mut encoder,
            self.rtb_offscreen_texture.as_ref().unwrap(),
            output_buffer,
            width,
            height,
            padded_bytes_per_row,
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        // Map the buffer and read data
        let buffer_slice = output_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        let _ = self.device.poll(wgpu::MaintainBase::Wait);
        rx.recv().unwrap().unwrap();

        let data = buffer_slice.get_mapped_range();

        // Remove padding and copy to output buffer
        let output_size = (unpadded_bytes_per_row * height) as usize;
        buffer.resize(output_size, 0);

        if padded_bytes_per_row == unpadded_bytes_per_row {
            // No padding, direct copy
            buffer.copy_from_slice(&data);
        } else {
            // Remove padding from each row
            for row in 0..height {
                let padded_offset = (row * padded_bytes_per_row) as usize;
                let unpadded_offset = (row * unpadded_bytes_per_row) as usize;
                let row_data =
                    &data[padded_offset..padded_offset + unpadded_bytes_per_row as usize];
                buffer[unpadded_offset..unpadded_offset + unpadded_bytes_per_row as usize]
                    .copy_from_slice(row_data);
            }
        }

        drop(data);
        output_buffer.unmap();
    }

    /// Renders the current scene into the provided ARGB32 pixel slice.
    /// The slice length must be at least width*height; each u32 is 0xAARRGGBB.
    pub fn render_to_argb32(&mut self, out_pixels: &mut [u32]) {
        self.prepare_render();

        let (width, height) = self.physical_size;
        let needed_len = (width as usize) * (height as usize);
        if out_pixels.len() < needed_len {
            // TODO: make this method to return a result
            // Safety: avoid panic; just do nothing if insufficient space
            warn!(
                "render_to_argb32: output slice too small: {} < {}",
                out_pixels.len(),
                needed_len
            );
            return;
        }

        // Recreate caches if size changed
        let size_changed = self.argb_cached_width != width || self.argb_cached_height != height;
        if size_changed {
            self.argb_cached_width = width;
            self.argb_cached_height = height;
        }

        // 1) Offscreen BGRA8 texture (cache per size)
        if size_changed || self.argb_offscreen_texture.is_none() {
            self.argb_offscreen_texture = Some(create_offscreen_color_texture(
                &self.device,
                (width, height),
                self.config.format,
            ));
        }
        let texture_view = self
            .argb_offscreen_texture
            .as_ref()
            .unwrap()
            .create_view(&wgpu::TextureViewDescriptor::default());
        self.render_to_texture_view(&texture_view);

        // 2) Copy texture into a padded staging buffer (BGRA bytes)
        let (_, padded_bytes_per_row) = compute_padded_bytes_per_row(width, 4);
        let input_buffer_size = (padded_bytes_per_row as u64) * (height as u64);
        if size_changed
            || self.argb_input_buffer.is_none()
            || self.argb_input_buffer_size < input_buffer_size
        {
            self.argb_input_buffer = Some(create_storage_input_buffer(
                &self.device,
                Some("argb_input_padded_bytes"),
                input_buffer_size,
            ));
            self.argb_input_buffer_size = input_buffer_size;
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("argb_copy_encoder"),
            });

        encode_copy_texture_to_buffer(
            &mut encoder,
            self.argb_offscreen_texture.as_ref().unwrap(),
            self.argb_input_buffer.as_ref().unwrap(),
            width,
            height,
            padded_bytes_per_row,
        );

        // 3) Create an output buffer for ARGB32 u32 pixels
        let output_words = (width as u64) * (height as u64);
        let output_buffer_size = output_words * 4;
        if size_changed
            || self.argb_output_storage_buffer.is_none()
            || self.argb_output_buffer_size < output_buffer_size
        {
            self.argb_output_storage_buffer = Some(create_storage_output_buffer(
                &self.device,
                Some("argb_output_u32_storage"),
                output_buffer_size,
            ));
            self.argb_output_buffer_size = output_buffer_size;
            // readback buffer should match size too
            self.argb_readback_buffer = Some(create_readback_buffer(
                &self.device,
                Some("argb_output_u32_readback"),
                output_buffer_size,
            ));
        }

        // 4) Create compute pipeline for swizzle
        if self.argb_cs_pipeline.is_none() {
            let (bgl, pipeline) = create_argb_swizzle_pipeline(&self.device);
            self.argb_cs_bgl = Some(bgl);
            self.argb_cs_pipeline = Some(pipeline);
        }

        // Params buffer (recreate if size changed due to padding)
        let params = ArgbParams {
            width,
            height,
            padded_bpr: padded_bytes_per_row,
            _pad: 0,
        };
        let need_new_params = self.argb_params_buffer.is_none();
        if need_new_params {
            self.argb_params_buffer = Some(crate::pipeline::create_argb_params_buffer(
                &self.device,
                &params,
            ));
        } else {
            self.queue.write_buffer(
                self.argb_params_buffer.as_ref().unwrap(),
                0,
                bytemuck::bytes_of(&params),
            );
        }
        // Recreate bind group if any binding target changed
        if size_changed || self.argb_swizzle_bind_group.is_none() || need_new_params {
            self.argb_swizzle_bind_group = Some(create_argb_swizzle_bind_group(
                &self.device,
                self.argb_cs_bgl.as_ref().unwrap(),
                self.argb_input_buffer.as_ref().unwrap(),
                self.argb_output_storage_buffer.as_ref().unwrap(),
                self.argb_params_buffer.as_ref().unwrap(),
            ));
        }

        // Submit the copy first
        self.queue.submit(std::iter::once(encoder.finish()));

        // 5) Dispatch compute to swizzle into ARGB32
        let mut cenc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("argb_compute_encoder"),
            });
        {
            let mut pass = cenc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("argb_swizzle_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(self.argb_cs_pipeline.as_ref().unwrap());
            pass.set_bind_group(0, self.argb_swizzle_bind_group.as_ref().unwrap(), &[]);
            let wg_x = width.div_ceil(16);
            let wg_y = height.div_ceil(16);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }
        self.queue.submit(std::iter::once(cenc.finish()));

        // 6) Copy compute output to a mappable readback buffer and map
        let mut enc3 = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("argb_readback_copy_encoder"),
            });
        enc3.copy_buffer_to_buffer(
            self.argb_output_storage_buffer.as_ref().unwrap(),
            0,
            self.argb_readback_buffer.as_ref().unwrap(),
            0,
            output_buffer_size,
        );
        self.queue.submit(std::iter::once(enc3.finish()));

        let slice = self.argb_readback_buffer.as_ref().unwrap().slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        let _ = self.device.poll(wgpu::MaintainBase::Wait);
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        // Copy exactly width*height u32s
        let src_words: &[u32] = bytemuck::cast_slice(&data);
        out_pixels[..needed_len].copy_from_slice(&src_words[..needed_len]);
        drop(data);
        self.argb_readback_buffer.as_ref().unwrap().unmap();
    }

    /// Clears all items currently in the draw queue.
    ///
    /// This method removes all shapes and images from the draw queue,
    /// preparing the renderer for the next frame.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use std::sync::Arc;
    /// use futures::executor::block_on;
    /// use winit::application::ApplicationHandler;
    /// use winit::event_loop::{ActiveEventLoop, EventLoop};
    /// use winit::window::Window;
    /// use grafo::Renderer;
    /// use grafo::Shape;
    /// use grafo::Color;
    /// use grafo::Stroke;
    ///
    /// // This is for demonstration purposes only. If you want a working example with winit, please
    /// // refer to the example in the "examples" folder.
    ///
    /// struct App;
    /// impl ApplicationHandler for App {
    ///     fn resumed(&mut self, event_loop: &ActiveEventLoop) {
    ///         let window_surface = Arc::new(
    ///             event_loop.create_window(Window::default_attributes()).unwrap()
    ///         );
    ///         let physical_size = (800, 600);
    ///         let scale_factor = 1.0;
    ///
    ///         // Initialize the renderer
    ///         let mut renderer = block_on(Renderer::new(window_surface, physical_size, scale_factor, true, false));
    ///
    ///         // Add shapes and images...
    ///
    ///         // Clear the draw queue
    ///         renderer.clear_draw_queue();
    ///     }
    ///
    ///     fn window_event(&mut self, _: &ActiveEventLoop, _: winit::window::WindowId, _: winit::event::WindowEvent) {
    ///         // Handle window events (stub for doc test)
    ///     }
    /// }
    /// ```
    pub fn clear_draw_queue(&mut self) {
        self.draw_tree.clear();
        self.metadata_to_clips.clear();
    }

    /// Adds a generic draw command to the draw tree.
    ///
    /// # Parameters
    ///
    /// - `draw_command`: The draw command to be added (`ShapeDrawData` or `ImageDrawData`).
    /// - `clip_to_shape`: Optional index of a shape to which this draw command should be clipped.
    ///
    /// # Returns
    ///
    /// The unique identifier (`usize`) assigned to the added draw command.
    fn add_draw_command(
        &mut self,
        draw_command: DrawCommand,
        clip_to_shape: Option<usize>,
    ) -> usize {
        if self.draw_tree.is_empty() {
            self.draw_tree.add_node(draw_command)
        } else if let Some(clip_to_shape) = clip_to_shape {
            self.draw_tree.add_child(clip_to_shape, draw_command)
        } else {
            self.draw_tree.add_child_to_root(draw_command)
        }
    }

    /// Sets a 4x4 column-major transform for a shape or cached shape.
    /// The transform is applied in clip space AFTER pixel-to-NDC normalization in the shader.
    pub fn set_shape_transform_cols(&mut self, node_id: usize, cols: [[f32; 4]; 4]) {
        let t = InstanceTransform {
            col0: cols[0],
            col1: cols[1],
            col2: cols[2],
            col3: cols[3],
        };
        if let Some(draw_command) = self.draw_tree.get_mut(node_id) {
            match draw_command {
                DrawCommand::Shape(shape) => shape.set_transform(t),
                DrawCommand::CachedShape(cached) => cached.set_transform(t),
            }
        }
    }

    /// Sets a transform for a shape or cached shape using any type that can be converted
    /// into a GPU `TransformInstance`.
    pub fn set_shape_transform(&mut self, node_id: usize, m: impl Into<InstanceTransform>) {
        let t: InstanceTransform = m.into();
        if let Some(draw_command) = self.draw_tree.get_mut(node_id) {
            match draw_command {
                DrawCommand::Shape(shape) => shape.set_transform(t),
                DrawCommand::CachedShape(cached) => cached.set_transform(t),
            }
        }
    }

    pub fn set_transformator(&mut self, node_id: usize, transformator: &transformator::Transform) {
        let exists = self.draw_tree.get(node_id).is_some();
        if !exists {
            return;
        }
        self.set_shape_transform_cols(node_id, transformator.cols_world());
        self.set_shape_camera_perspective(node_id, transformator.perspective_distance);
        self.set_shape_viewport_position(
            node_id,
            transformator.camera_perspective_origin.0,
            transformator.camera_perspective_origin.1,
        );
    }

    /// Associates a texture with a shape or cached shape by node id.
    /// Pass `None` to remove texture and fall back to solid fill color.
    pub fn set_shape_texture(&mut self, node_id: usize, texture_id: Option<u64>) {
        self.set_shape_texture_layer(node_id, 0, texture_id);
    }

    /// Associates a texture with a shape/cached shape for a specific layer (0 or 1)
    pub fn set_shape_texture_layer(
        &mut self,
        node_id: usize,
        layer: usize,
        texture_id: Option<u64>,
    ) {
        if layer > 1 {
            return;
        }
        if let Some(draw_command) = self.draw_tree.get_mut(node_id) {
            match draw_command {
                DrawCommand::Shape(shape) => shape.set_texture_id(layer, texture_id),
                DrawCommand::CachedShape(cached) => cached.set_texture_id(layer, texture_id),
            }
        }
    }

    /// Same as `set_shape_texture_layer` but takes a `TextureLayer` enum.
    pub fn set_shape_texture_on(
        &mut self,
        node_id: usize,
        layer: TextureLayer,
        texture_id: Option<u64>,
    ) {
        self.set_shape_texture_layer(node_id, layer.into(), texture_id);
    }

    /// Sets a per-instance color for a shape/cached shape by node id. Pass None to use shape's fill.
    pub fn set_shape_color(&mut self, node_id: usize, color: Option<Color>) {
        if let Some(draw_command) = self.draw_tree.get_mut(node_id) {
            let color_norm = color.map(|c| c.normalize());
            match draw_command {
                DrawCommand::Shape(shape) => shape.set_instance_color_override(color_norm),
                DrawCommand::CachedShape(cached) => cached.set_instance_color_override(color_norm),
            }
        }
    }

    /// Sets the camera perspective distance for a shape. Smaller values create stronger perspective effect.
    /// Pass 0.0 to disable perspective.
    pub fn set_shape_camera_perspective(&mut self, node_id: usize, distance: f32) {
        if let Some(draw_command) = self.draw_tree.get_mut(node_id) {
            let current_params = match draw_command {
                DrawCommand::Shape(shape) => shape.render_params().unwrap_or_default(),
                DrawCommand::CachedShape(cached) => cached.render_params().unwrap_or_default(),
            };
            
            let new_params = InstanceRenderParams {
                camera_perspective: distance,
                ..current_params
            };
            
            match draw_command {
                DrawCommand::Shape(shape) => shape.set_render_params(new_params),
                DrawCommand::CachedShape(cached) => cached.set_render_params(new_params),
            }
        }
    }

    /// Sets the viewport position offset for a shape in pixel space.
    pub fn set_shape_viewport_position(&mut self, node_id: usize, x: f32, y: f32) {
        if let Some(draw_command) = self.draw_tree.get_mut(node_id) {
            let current_params = match draw_command {
                DrawCommand::Shape(shape) => shape.render_params().unwrap_or_default(),
                DrawCommand::CachedShape(cached) => cached.render_params().unwrap_or_default(),
            };
            
            let new_params = InstanceRenderParams {
                camera_perspective_origin: [x, y],
                ..current_params
            };
            
            match draw_command {
                DrawCommand::Shape(shape) => shape.set_render_params(new_params),
                DrawCommand::CachedShape(cached) => cached.set_render_params(new_params),
            }
        }
    }

    /// Sets both camera perspective and viewport position for a shape.
    pub fn set_shape_render_params(&mut self, node_id: usize, params: InstanceRenderParams) {
        if let Some(draw_command) = self.draw_tree.get_mut(node_id) {
            match draw_command {
                DrawCommand::Shape(shape) => shape.set_render_params(params),
                DrawCommand::CachedShape(cached) => cached.set_render_params(params),
            }
        }
    }

    /// Retrieves the current physical size of the rendering surface.
    ///
    /// # Returns
    ///
    /// A tuple `(u32, u32)` representing the width and height of the window in pixels.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use std::sync::Arc;
    /// use futures::executor::block_on;
    /// use winit::application::ApplicationHandler;
    /// use winit::event_loop::{ActiveEventLoop, EventLoop};
    /// use winit::window::Window;
    /// use grafo::Renderer;
    /// use grafo::Shape;
    /// use grafo::Color;
    /// use grafo::Stroke;
    ///
    /// // This is for demonstration purposes only. If you want a working example with winit, please
    /// // refer to the example in the "examples" folder.
    ///
    /// struct App;
    /// impl ApplicationHandler for App {
    ///     fn resumed(&mut self, event_loop: &ActiveEventLoop) {
    ///         let window_surface = Arc::new(
    ///             event_loop.create_window(Window::default_attributes()).unwrap()
    ///         );
    ///         let physical_size = (800, 600);
    ///         let scale_factor = 1.0;
    ///
    ///         // Initialize the renderer
    ///         let mut renderer = block_on(Renderer::new(window_surface, physical_size, scale_factor, true, false));
    ///
    ///         let size = renderer.size();
    ///         println!("Rendering surface size: {}x{}", size.0, size.1);
    ///     }
    ///
    ///     fn window_event(&mut self, _: &ActiveEventLoop, _: winit::window::WindowId, _: winit::event::WindowEvent) {
    ///         // Handle window events (stub for doc test)
    ///     }
    /// }
    /// ```
    pub fn size(&self) -> (u32, u32) {
        self.physical_size
    }

    /// Changes the scale factor of the renderer (e.g., for DPI scaling).
    ///
    /// This method updates the scale factor and resizes the renderer accordingly.
    ///
    /// # Parameters
    ///
    /// - `new_scale_factor`: The new scale factor to apply.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use std::sync::Arc;
    /// use futures::executor::block_on;
    /// use winit::application::ApplicationHandler;
    /// use winit::event_loop::{ActiveEventLoop, EventLoop};
    /// use winit::window::Window;
    /// use grafo::Renderer;
    /// use grafo::Shape;
    /// use grafo::Color;
    /// use grafo::Stroke;
    ///
    /// // This is for demonstration purposes only. If you want a working example with winit, please
    /// // refer to the example in the "examples" folder.
    ///
    /// struct App;
    /// impl ApplicationHandler for App {
    ///     fn resumed(&mut self, event_loop: &ActiveEventLoop) {
    ///         let window_surface = Arc::new(
    ///             event_loop.create_window(Window::default_attributes()).unwrap()
    ///         );
    ///         let physical_size = (800, 600);
    ///         let scale_factor = 1.0;
    ///
    ///         // Initialize the renderer
    ///         let mut renderer = block_on(Renderer::new(window_surface, physical_size, scale_factor, true, false));
    ///
    ///         // Change the scale factor to 2.0 for high-DPI rendering
    ///         renderer.change_scale_factor(2.0);
    ///     }
    ///
    ///     fn window_event(&mut self, _: &ActiveEventLoop, _: winit::window::WindowId, _: winit::event::WindowEvent) {
    ///         // Handle window events (stub for doc test)
    ///     }
    /// }
    pub fn change_scale_factor(&mut self, new_scale_factor: f64) {
        self.scale_factor = new_scale_factor;
        self.resize(self.physical_size)
    }

    pub fn scale_factor(&self) -> f64 {
        self.scale_factor
    }

    /// Resizes the renderer to the specified physical size.
    ///
    /// This method updates the renderer's configuration to match the new window size,
    /// reconfigures the surface, and updates all relevant pipelines and bind groups.
    ///
    /// # Parameters
    ///
    /// - `new_physical_size`: A tuple `(u32, u32)` representing the new width and height in pixels.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use std::sync::Arc;
    /// use futures::executor::block_on;
    /// use winit::application::ApplicationHandler;
    /// use winit::event_loop::{ActiveEventLoop, EventLoop};
    /// use winit::window::Window;
    /// use grafo::Renderer;
    /// use grafo::Shape;
    /// use grafo::Color;
    /// use grafo::Stroke;
    ///
    /// // This is for demonstration purposes only. If you want a working example with winit, please
    /// // refer to the example in the "examples" folder.
    ///
    /// struct App;
    /// impl ApplicationHandler for App {
    ///     fn resumed(&mut self, event_loop: &ActiveEventLoop) {
    ///         let window_surface = Arc::new(
    ///             event_loop.create_window(Window::default_attributes()).unwrap()
    ///         );
    ///         let physical_size = (800, 600);
    ///         let scale_factor = 1.0;
    ///
    ///         // Initialize the renderer
    ///         let mut renderer = block_on(Renderer::new(window_surface, physical_size, scale_factor, true, false));
    ///
    ///         // Resize the renderer to 1024x768 pixels
    ///         renderer.resize((1024, 768));
    ///     }
    ///
    ///     fn window_event(&mut self, _: &ActiveEventLoop, _: winit::window::WindowId, _: winit::event::WindowEvent) {
    ///         // Handle window events (stub for doc test)
    ///     }
    /// }
    /// ```
    pub fn resize(&mut self, new_physical_size: (u32, u32)) {
        self.physical_size = new_physical_size;
        self.config.width = new_physical_size.0;
        self.config.height = new_physical_size.1;
        // Cheap uniform buffer update instead of full pipeline recreation.
        let logical = to_logical(new_physical_size, self.scale_factor);
        self.and_uniforms.canvas_size = [logical.0, logical.1];
        self.decrementing_uniforms.canvas_size = [logical.0, logical.1];
        // Write both uniform buffers.
        self.queue.write_buffer(
            &self.and_uniform_buffer,
            0,
            bytemuck::cast_slice(&[self.and_uniforms]),
        );
        self.queue.write_buffer(
            &self.decrementing_uniform_buffer,
            0,
            bytemuck::cast_slice(&[self.decrementing_uniforms]),
        );
        self.surface.configure(&self.device, &self.config);
    }

    /// Sets a new surface for the renderer.
    ///
    /// This method allows you to change the rendering target (e.g., switch windows).
    /// The new surface will be configured with the renderer's current settings.
    ///
    /// # Parameters
    ///
    /// - `window`: The new surface target to render to.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use std::sync::Arc;
    /// use futures::executor::block_on;
    /// use winit::application::ApplicationHandler;
    /// use winit::event_loop::{ActiveEventLoop, EventLoop};
    /// use winit::window::Window;
    /// use grafo::Renderer;
    ///
    /// struct App;
    /// impl ApplicationHandler for App {
    ///     fn resumed(&mut self, event_loop: &ActiveEventLoop) {
    ///         let window1 = Arc::new(
    ///             event_loop.create_window(Window::default_attributes()).unwrap()
    ///         );
    ///         let mut renderer = block_on(Renderer::new(window1, (800, 600), 1.0, true, false));
    ///
    ///         // Later, switch to a different window
    ///         let window2 = Arc::new(
    ///             event_loop.create_window(Window::default_attributes()).unwrap()
    ///         );
    ///         renderer.set_surface(window2);
    ///     }
    ///     fn window_event(&mut self, _: &ActiveEventLoop, _: winit::window::WindowId, _: winit::event::WindowEvent) {}
    /// }
    /// ```
    pub fn set_surface(&mut self, window: impl Into<SurfaceTarget<'static>>) {
        self.surface = self
            .instance
            .create_surface(window)
            .expect("Failed to create surface");
        self.surface.configure(&self.device, &self.config);
    }

    pub fn set_vsync(&mut self, vsync: bool) {
        self.config.present_mode = if vsync {
            wgpu::PresentMode::Fifo
        } else {
            wgpu::PresentMode::Immediate
        };
        self.surface.configure(&self.device, &self.config);
    }
}

struct Buffers<'a> {
    aggregated_vertex_buffer: &'a wgpu::Buffer,
    aggregated_index_buffer: &'a wgpu::Buffer,
    identity_instance_transform_buffer: &'a wgpu::Buffer,
    identity_instance_color_buffer: &'a wgpu::Buffer,
    identity_instance_render_params_buffer: &'a wgpu::Buffer,
    aggregated_instance_transform_buffer: Option<&'a wgpu::Buffer>,
    aggregated_instance_color_buffer: Option<&'a wgpu::Buffer>,
    aggregated_instance_render_params_buffer: Option<&'a wgpu::Buffer>,
}

struct Pipelines<'a> {
    and_pipeline: &'a wgpu::RenderPipeline,
    and_bind_group: &'a wgpu::BindGroup,
    decrementing_pipeline: &'a wgpu::RenderPipeline,
    decrementing_bind_group: &'a wgpu::BindGroup,
    shape_texture_bgl_layer0: &'a wgpu::BindGroupLayout,
    shape_texture_bgl_layer1: &'a wgpu::BindGroupLayout,
    default_shape_texture_bgs: &'a [Arc<wgpu::BindGroup>; 2],
    shape_texture_layout_epoch: u64,
    texture_manager: &'a TextureManager,
}

// Helper to handle stencil increment pass for any shape-like data
fn handle_increment_pass<'rp>(
    render_pass: &mut wgpu::RenderPass<'rp>,
    currently_set_pipeline: &mut Pipeline,
    stencil_stack: &mut Vec<u32>,
    shape: &mut impl DrawShapeCommand,
    pipelines: &Pipelines,
    buffers: &Buffers,
) {
    if let Some(index_range) = shape.index_buffer_range() {
        if shape.is_empty() {
            return;
        }

        if !matches!(currently_set_pipeline, Pipeline::StencilIncrement) {
            render_pass.set_pipeline(pipelines.and_pipeline);
            render_pass.set_bind_group(0, pipelines.and_bind_group, &[]);
            // Bind default textures for both layers
            render_pass.set_bind_group(1, &*pipelines.default_shape_texture_bgs[0], &[]);
            render_pass.set_bind_group(2, &*pipelines.default_shape_texture_bgs[1], &[]);

            // Those pipelines use the same vertex buffers
            if !matches!(currently_set_pipeline, Pipeline::StencilDecrement) {
                render_pass.set_vertex_buffer(0, buffers.aggregated_vertex_buffer.slice(..));
                // Bind identity instance buffers so shader inputs are valid
                render_pass
                    .set_vertex_buffer(1, buffers.identity_instance_transform_buffer.slice(..));
                render_pass.set_vertex_buffer(2, buffers.identity_instance_color_buffer.slice(..));
                render_pass.set_vertex_buffer(3, buffers.identity_instance_render_params_buffer.slice(..));
                render_pass.set_index_buffer(
                    buffers.aggregated_index_buffer.slice(..),
                    wgpu::IndexFormat::Uint16,
                );
            }

            *currently_set_pipeline = Pipeline::StencilIncrement;
        }

        // Determine the parent stencil from the stack (0 if root)
        let parent_stencil = stencil_stack.last().copied().unwrap_or(0);

        // Bind per-layer textures if present
        for layer in 0..2 {
            if let Some(tex_id) = shape.texture_id(layer) {
                if pipelines.texture_manager.is_texture_loaded(tex_id) {
                    if let Ok(bg_arc) = pipelines.texture_manager.get_or_create_shape_bind_group(
                        if layer == 0 {
                            pipelines.shape_texture_bgl_layer0
                        } else {
                            pipelines.shape_texture_bgl_layer1
                        },
                        pipelines.shape_texture_layout_epoch,
                        tex_id,
                    ) {
                        render_pass.set_bind_group(1 + layer as u32, &*bg_arc, &[]);
                    }
                }
            } else {
                render_pass.set_bind_group(
                    1 + layer as u32,
                    &*pipelines.default_shape_texture_bgs[layer],
                    &[],
                );
            }
        }

        // Render increment pass with parent stencil
        // Bind per-instance transform slice if available, else fall back to identity
        if let Some(instance_idx) = shape.instance_index() {
            // Transform slice
            if let Some(inst_t_buf) = buffers.aggregated_instance_transform_buffer {
                let stride_t = std::mem::size_of::<InstanceTransform>() as u64;
                let offset_t = instance_idx as u64 * stride_t;
                render_pass.set_vertex_buffer(1, inst_t_buf.slice(offset_t..offset_t + stride_t));
            } else {
                render_pass
                    .set_vertex_buffer(1, buffers.identity_instance_transform_buffer.slice(..));
            }
            // Color slice
            if let Some(inst_c_buf) = buffers.aggregated_instance_color_buffer {
                let stride_c = std::mem::size_of::<InstanceColor>() as u64;
                let offset_c = instance_idx as u64 * stride_c;
                render_pass.set_vertex_buffer(2, inst_c_buf.slice(offset_c..offset_c + stride_c));
            } else {
                render_pass.set_vertex_buffer(2, buffers.identity_instance_color_buffer.slice(..));
            }
            // Render params slice
            if let Some(inst_r_buf) = buffers.aggregated_instance_render_params_buffer {
                let stride_r = std::mem::size_of::<InstanceRenderParams>() as u64;
                let offset_r = instance_idx as u64 * stride_r;
                render_pass.set_vertex_buffer(3, inst_r_buf.slice(offset_r..offset_r + stride_r));
            } else {
                render_pass.set_vertex_buffer(3, buffers.identity_instance_render_params_buffer.slice(..));
            }
        } else {
            render_pass.set_vertex_buffer(1, buffers.identity_instance_transform_buffer.slice(..));
            render_pass.set_vertex_buffer(2, buffers.identity_instance_color_buffer.slice(..));
            render_pass.set_vertex_buffer(3, buffers.identity_instance_render_params_buffer.slice(..));
        }
        render_buffer_range_to_texture(index_range, render_pass, parent_stencil);

        // Assign and push this node's stencil reference (parent + 1)
        let this_stencil = parent_stencil + 1;
        *shape.stencil_ref_mut() = Some(this_stencil);
        stencil_stack.push(this_stencil);
    } else {
        warn!("Shape with no index buffer range found, skipping increment pass");
    }
}

// Helper to handle stencil decrement pass for any shape-like data
fn handle_decrement_pass<'rp>(
    render_pass: &mut wgpu::RenderPass<'rp>,
    currently_set_pipeline: &mut Pipeline,
    stencil_stack: &mut Vec<u32>,
    shape: &mut impl DrawShapeCommand,
    pipelines: &Pipelines,
    buffers: &Buffers,
) {
    if let Some(index_range) = shape.index_buffer_range() {
        if shape.is_empty() {
            return;
        }

        if !matches!(currently_set_pipeline, Pipeline::StencilDecrement) {
            render_pass.set_pipeline(pipelines.decrementing_pipeline);
            render_pass.set_bind_group(0, pipelines.decrementing_bind_group, &[]);
            // Bind default textures for both layers to satisfy layout
            render_pass.set_bind_group(1, &*pipelines.default_shape_texture_bgs[0], &[]);
            render_pass.set_bind_group(2, &*pipelines.default_shape_texture_bgs[1], &[]);

            if !matches!(currently_set_pipeline, Pipeline::StencilIncrement) {
                render_pass.set_vertex_buffer(0, buffers.aggregated_vertex_buffer.slice(..));
                // Bind identity instance buffers so shader inputs are valid
                render_pass
                    .set_vertex_buffer(1, buffers.identity_instance_transform_buffer.slice(..));
                render_pass.set_vertex_buffer(2, buffers.identity_instance_color_buffer.slice(..));
                render_pass.set_vertex_buffer(3, buffers.identity_instance_render_params_buffer.slice(..));
                render_pass.set_index_buffer(
                    buffers.aggregated_index_buffer.slice(..),
                    wgpu::IndexFormat::Uint16,
                );
            }

            *currently_set_pipeline = Pipeline::StencilDecrement;
        }

        // Use this node's stored stencil reference and then pop the stack
        let this_shape_stencil = shape.stencil_ref_mut().unwrap_or(0);

        // Bind per-instance transform slice if available, else fall back to identity
        if let Some(instance_idx) = shape.instance_index() {
            // Transform slice
            if let Some(inst_t_buf) = buffers.aggregated_instance_transform_buffer {
                let stride_t = std::mem::size_of::<InstanceTransform>() as u64;
                let offset_t = instance_idx as u64 * stride_t;
                render_pass.set_vertex_buffer(1, inst_t_buf.slice(offset_t..offset_t + stride_t));
            } else {
                render_pass
                    .set_vertex_buffer(1, buffers.identity_instance_transform_buffer.slice(..));
            }
            // Color slice
            if let Some(inst_c_buf) = buffers.aggregated_instance_color_buffer {
                let stride_c = std::mem::size_of::<InstanceColor>() as u64;
                let offset_c = instance_idx as u64 * stride_c;
                render_pass.set_vertex_buffer(2, inst_c_buf.slice(offset_c..offset_c + stride_c));
            } else {
                render_pass.set_vertex_buffer(2, buffers.identity_instance_color_buffer.slice(..));
            }
            // Render params slice
            if let Some(inst_r_buf) = buffers.aggregated_instance_render_params_buffer {
                let stride_r = std::mem::size_of::<InstanceRenderParams>() as u64;
                let offset_r = instance_idx as u64 * stride_r;
                render_pass.set_vertex_buffer(3, inst_r_buf.slice(offset_r..offset_r + stride_r));
            } else {
                render_pass.set_vertex_buffer(3, buffers.identity_instance_render_params_buffer.slice(..));
            }
        } else {
            render_pass.set_vertex_buffer(1, buffers.identity_instance_transform_buffer.slice(..));
            render_pass.set_vertex_buffer(2, buffers.identity_instance_color_buffer.slice(..));
            render_pass.set_vertex_buffer(3, buffers.identity_instance_render_params_buffer.slice(..));
        }
        render_buffer_range_to_texture(index_range, render_pass, this_shape_stencil);

        if shape.stencil_ref_mut().is_some() {
            stencil_stack.pop();
        }
    } else {
        warn!("Shape with no index buffer range found, skipping decrement pass");
    }
}
