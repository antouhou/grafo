//! Renderer for the Grafo library.
//!
//! This module provides the [`Renderer`] struct, which is responsible for rendering shapes,
//! images, and text. It leverages the `wgpu` crate for GPU-accelerated rendering and integrates
//! with other modules like `shape`, `text`, and `image_draw_data` to manage various rendering
//! components.
//!
//! # Examples
//!
//! Initializing and using the `Renderer`:
//!
//! ```rust,no_run
//! use std::sync::Arc;
//! use grafo::{FontFamily, Renderer};
//! use grafo::Shape;
//! use grafo::Color;
//! use grafo::Stroke;
//! use grafo::{TextAlignment, TextLayout};
//! use grafo::MathRect;
//! use winit::application::ApplicationHandler;
//! use winit::event_loop::{ActiveEventLoop, EventLoop};
//! use winit::window::Window;
//! use futures::executor::block_on;
//!
//! // This is for demonstration purposes only. If you want a working example with winit, please
//! // refer to the example in the "examples" folder.
//!
//! struct App;
//! impl ApplicationHandler for App {
//!     fn resumed(&mut self, event_loop: &ActiveEventLoop) {
//!         let window_surface = Arc::new(
//!             event_loop.create_window(Window::default_attributes()).unwrap()
//!         );
//!         let physical_size = (800, 600);
//!         let scale_factor = 1.0;
//!
//!         // Initialize the renderer
//!         let mut renderer = block_on(Renderer::new(window_surface, physical_size, scale_factor, true, false));
//!
//!         // Add a rectangle shape
//!         let rect = Shape::rect(
//!             [(0.0, 0.0), (200.0, 100.0)],
//!             Color::rgb(0, 128, 255), // Blue fill
//!             Stroke::new(2.0, Color::BLACK), // Black stroke with width 2.0
//!         );
//!         renderer.add_shape(rect, None, (100.0, 100.0), None);
//!
//!         // Add some text
//!         let layout = TextLayout {
//!             font_size: 24.0,
//!             line_height: 30.0,
//!             color: Color::rgb(255, 255, 255), // White text
//!             area: MathRect {
//!                 min: (50.0, 50.0).into(),
//!                 max: (400.0, 100.0).into(),
//!             },
//!             horizontal_alignment: TextAlignment::Center,
//!             vertical_alignment: TextAlignment::Center,
//!         };
//!         renderer.add_text("Hello, Grafo!", layout, FontFamily::SansSerif, None, 0);
//!
//!         // Render the frame
//!         match renderer.render() {
//!             Ok(_) => println!("Frame rendered successfully."),
//!             Err(e) => eprintln!("Rendering error: {:?}", e),
//!         }
//!
//!         // Clear the draw queue after rendering
//!         renderer.clear_draw_queue();
//!     }
//!
//!     fn window_event(&mut self, _: &ActiveEventLoop, _: winit::window::WindowId, _: winit::event::WindowEvent) {
//!         // Handle window events (stub for doc test)
//!     }
//! }
//! ```

use std::hash::{Hash, Hasher};
use std::sync::Arc;

pub type MathRect = lyon::math::Box2D;

use crate::image_draw_data::ImageDrawData;
use crate::pipeline::{
    create_and_depth_texture, create_depth_stencil_state_for_text, create_pipeline,
    create_render_pass, create_texture_pipeline, render_buffer_range_to_texture, PipelineType,
    Uniforms,
};
use crate::shape::{CachedShapeDrawData, Shape, ShapeDrawData};
use crate::text::{TextDrawData, TextLayout, TextRendererWrapper};
use crate::texture_manager::TextureManager;
use crate::util::{to_logical, PoolManager};
use crate::{CachedShape, Color, FontFamily, IntoCowBuffer, NoopTextDataIter};
use ahash::{HashMap, HashMapExt};
use glyphon::{fontdb, FontSystem, Resolution, SwashCache};
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
    /// Pipeline for cropping textures based on stencil.
    TextureCrop,
    /// Pipeline for always rendering textures without clipping.
    TextureAlways,
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
pub fn depth(draw_command_id: usize, draw_commands_total: usize) -> f32 {
    (1.0 - (draw_command_id as f32 / draw_commands_total as f32)).clamp(0.0000000001, 0.9999999999)
}

/// Represents a draw command, which can be either a shape or an image.
///
/// This enum is used internally by the `Renderer` to manage different types of draw operations.
enum DrawCommand {
    Shape(ShapeDrawData),
    CachedShape(CachedShapeDrawData),
    Image(ImageDrawData),
}

/// The renderer for the Grafo library. This is the main struct that is used to render shapes,
/// images, and text.
///
/// # Examples
///
/// ```rust,no_run
/// use grafo::{FontFamily, Renderer};
/// use grafo::Shape;
/// use grafo::Color;
/// use grafo::Stroke;
/// use grafo::{TextAlignment, TextLayout};
/// use grafo::MathRect;
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
///             Color::rgb(0, 128, 255), // Blue fill
///             Stroke::new(2.0, Color::BLACK), // Black stroke with width 2.0
///         );
///         renderer.add_shape(rect, None, (100.0, 100.0), None);
///
///         // Add some text
///         let layout = TextLayout {
///             font_size: 24.0,
///             line_height: 30.0,
///             color: Color::rgb(255, 255, 255), // White text
///             area: MathRect {
///                 min: (50.0, 50.0).into(),
///                 max: (400.0, 100.0).into(),
///             },
///             horizontal_alignment: TextAlignment::Center,
///             vertical_alignment: TextAlignment::Center,
///         };
///         renderer.add_text("Hello, Grafo!", layout, FontFamily::SansSerif , None, 0);
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
    surface: wgpu::Surface<'a>,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    config: wgpu::SurfaceConfiguration,

    tessellator: FillTessellator,

    buffers_pool_manager: PoolManager,

    texture_manager: TextureManager,

    font_system: FontSystem,
    swash_cache: SwashCache,
    /// Text instances to be rendered
    text_instances: Vec<TextDrawData<'a>>,
    /// Internal wrapper for text rendering components.
    text_renderer_wrapper: TextRendererWrapper,
    glyphon_viewport: glyphon::Viewport,

    /// Tree structure holding shapes and images to be rendered.
    draw_tree: easy_tree::Tree<DrawCommand>,
    metadata_to_clips: HashMap<usize, usize>,

    /// Uniforms for the "And" rendering pipeline.
    and_uniforms: Uniforms,
    /// Bind group for the "And" rendering pipeline.
    and_bind_group: BindGroup,
    /// Render pipeline for the "And" operations.
    and_pipeline: Arc<wgpu::RenderPipeline>,

    /// Render pipeline for decrementing stencil values.
    decrementing_pipeline: Arc<wgpu::RenderPipeline>,
    /// Uniforms for the decrementing pipeline.
    decrementing_uniforms: Uniforms,
    /// Bind group for the decrementing pipeline.
    decrementing_bind_group: BindGroup,

    // #[cfg(feature = "performance_measurement")]
    // performance_query_set: wgpu::QuerySet,
    // #[cfg(feature = "performance_measurement")]
    // adapter: wgpu::Adapter,
    /// Render pipeline for cropping textures based on stencil.
    texture_crop_render_pipeline: Arc<wgpu::RenderPipeline>,
    /// Render pipeline for always rendering textures without clipping.
    texture_always_render_pipeline: Arc<wgpu::RenderPipeline>,

    temp_vertices: Vec<crate::vertex::CustomVertex>,
    temp_indices: Vec<u16>,
    
    /// Reusable aggregated vertex buffer
    aggregated_vertex_buffer: Option<wgpu::Buffer>,
    /// Reusable aggregated index buffer  
    aggregated_index_buffer: Option<wgpu::Buffer>,

    shape_cache: HashMap<u64, CachedShape>
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

        let text_instances = Vec::new();

        let (and_uniforms, and_bind_group, and_pipeline) = create_pipeline(
            canvas_logical_size,
            &device,
            &config,
            PipelineType::EqualIncrementStencil,
        );

        let (decrementing_uniforms, decrementing_bind_group, decrementing_pipeline) =
            create_pipeline(
                canvas_logical_size,
                &device,
                &config,
                PipelineType::EqualDecrementStencil,
            );

        let text_renderer_wrapper = TextRendererWrapper::new(
            &device,
            &queue,
            swapchain_format,
            Some(create_depth_stencil_state_for_text()),
        );

        let font_system = FontSystem::new();
        let swash_cache = SwashCache::new();

        let mut glyphon_viewport =
            glyphon::Viewport::new(&device, &text_renderer_wrapper.glyphon_cache);

        {
            glyphon_viewport.update(
                &queue,
                Resolution {
                    width: size.0,
                    height: size.1,
                },
            );
        }

        let (
            texture_bind_group_layout,
            texture_crop_render_pipeline,
            texture_always_render_pipeline,
        ) = create_texture_pipeline(&device, &config);

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        let texture_manager =
            TextureManager::new(device.clone(), queue.clone(), texture_bind_group_layout);

        Self {
            surface,
            device,
            queue,
            config,
            physical_size: size,

            scale_factor,
            text_instances,
            // draw_queue: BTreeMap::new(),
            font_system,
            swash_cache,
            text_renderer_wrapper,
            glyphon_viewport,

            tessellator: FillTessellator::new(),

            texture_manager,

            buffers_pool_manager: PoolManager::new(),

            and_pipeline: Arc::new(and_pipeline),
            and_uniforms,
            and_bind_group,

            decrementing_pipeline: Arc::new(decrementing_pipeline),
            decrementing_uniforms,
            decrementing_bind_group,

            draw_tree: easy_tree::Tree::new(),
            metadata_to_clips: HashMap::new(),

            // #[cfg(feature = "performance_measurement")]
            // performance_query_set: frametime_query_set,
            // #[cfg(feature = "performance_measurement")]
            // adapter,
            texture_crop_render_pipeline: Arc::new(texture_crop_render_pipeline),
            texture_always_render_pipeline: Arc::new(texture_always_render_pipeline),

            temp_vertices: Vec::new(),
            temp_indices: Vec::new(),
            aggregated_vertex_buffer: None,
            aggregated_index_buffer: None,

            shape_cache: HashMap::new(),
        }
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
    ///   recommended to start shapes at 0.0, 0.0 where possible and use offset to move them on
    ///   the screen. This way, it is going to be possible to cache tesselation results for
    ///   such shapes, which would increase rendering time. This is useful if you render a lot
    ///   of buttons with rounded corners, for example. Caching requires supplying a cache key
    ///   to cache tessellated shape.
    /// - `clip_to_shape`: Optional index of another shape to which this shape should be clipped.
    /// - `offset`: Offset to render the shape at.
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
    ///                 Color::rgb(0, 128, 255), // Blue fill
    ///                 Stroke::new(2.0, Color::BLACK), // Black stroke with width 2.0
    ///             ),
    ///             None,
    ///             // Shift rectangle by 100.0, 100.0
    ///             (100.0, 100.0),
    ///             None
    ///         );
    ///     }
    ///     fn window_event(&mut self, _: &ActiveEventLoop, _: winit::window::WindowId, _: winit::event::WindowEvent) {}
    /// }
    /// ```
    pub fn add_shape(
        &mut self,
        shape: impl Into<Shape>,
        clip_to_shape: Option<usize>,
        offset: (f32, f32),
        cache_key: Option<u64>,
    ) -> usize {
        self.add_draw_command(
            DrawCommand::Shape(ShapeDrawData::new(shape, clip_to_shape, offset, cache_key)),
            clip_to_shape,
        )
    }

    pub fn load_shape(
        &mut self,
        shape: impl AsRef<Shape>,
        offset: (f32, f32),
        cache_key: u64,
        tessellation_cache_key: Option<u64>,
    ) {
        let cached_shape = CachedShape::new(shape.as_ref(), offset, 0.0, &mut self.tessellator, &mut self.buffers_pool_manager, tessellation_cache_key);
        self.shape_cache.insert(cache_key, cached_shape);
    }

    pub fn add_cached_shape_to_the_render_queue(
        &mut self,
        cache_key: u64,
        clip_to_shape: Option<usize>,
        offset: (f32, f32),
    ) -> usize {
        self.add_draw_command(
            DrawCommand::CachedShape(CachedShapeDrawData::new(cache_key, offset, clip_to_shape)),
            clip_to_shape,
        )
    }

    /// Adds an image to the draw queue. This is a shorthand method if you quickly want to render
    /// an image from the main thread; If you want to update texture data from a different thread,
    /// or want more control over allocating and updating texture data in general, you should use
    /// [Renderer::texture_manager].
    ///
    /// # Parameters
    ///
    /// - `image`: A byte slice representing the image data.
    /// - `physical_image_dimensions`: A tuple representing the image's width and height in pixels.
    /// - `area`: An array containing two tuples representing the top-left and bottom-right
    ///   coordinates where the image should be rendered.
    /// - `clip_to_shape`: Optional index of a shape to which this image should be clipped.
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
    ///         let image_data = vec![0; 16]; // A 2x2 black image
    ///         renderer.add_rgba_image(
    ///             &image_data,
    ///             (2, 2), // Image dimensions
    ///             [(50.0, 50.0), (150.0, 150.0)], // Rendering rectangle
    ///             Some(0), // Clip to shape with ID 0
    ///         );
    ///     }
    ///     fn window_event(&mut self, _: &ActiveEventLoop, _: winit::window::WindowId, _: winit::event::WindowEvent) {}
    /// }
    /// ```
    pub fn add_rgba_image(
        &mut self,
        texture_rgba_data: &[u8],
        physical_image_dimensions: (u32, u32),
        draw_at: [(f32, f32); 2],
        clip_to_shape: Option<usize>,
    ) {
        // Creating naive texture ID based on the image data
        let mut default_hasher = std::collections::hash_map::DefaultHasher::new();
        texture_rgba_data.hash(&mut default_hasher);
        let texture_id = default_hasher.finish();

        let is_already_loaded = self.texture_manager.is_texture_loaded(texture_id);

        if !is_already_loaded {
            self.texture_manager.allocate_texture_with_data(
                texture_id,
                physical_image_dimensions,
                texture_rgba_data,
            );
        }

        let draw_command =
            DrawCommand::Image(ImageDrawData::new(texture_id, draw_at, clip_to_shape));

        self.add_draw_command(draw_command, clip_to_shape);
    }

    /// Adds a texture to the draw queue. The texture must be loaded with the [Renderer::texture_manager]
    /// first.
    pub fn add_texture_draw_to_queue(
        &mut self,
        texture_id: u64,
        draw_at: [(f32, f32); 2],
        clip_to_shape: Option<usize>,
    ) {
        let draw_command =
            DrawCommand::Image(ImageDrawData::new(texture_id, draw_at, clip_to_shape));

        self.add_draw_command(draw_command, clip_to_shape);
    }

    /// A texture manager is a helper to allow more granular approach to drawing images. It can
    /// be cloned and passed to a different thread if you want to update texture
    pub fn texture_manager(&self) -> &TextureManager {
        &self.texture_manager
    }

    /// Adds text to the draw queue. If you want to use a custom font, you need to load it first
    /// using the [Renderer::load_fonts] or [Renderer::load_font_from_bytes] methods.
    ///
    /// # Parameters
    ///
    /// - `text`: The string of text to be rendered.
    /// - `layout`: The layout configuration for the text.
    /// - `font_family`: The font family to be used for rendering the text.
    /// - `clip_to_shape`: Optional index of a shape to which this text should be clipped.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use std::sync::Arc;
    /// use futures::executor::block_on;
    /// use winit::application::ApplicationHandler;
    /// use winit::event_loop::{ActiveEventLoop, EventLoop};
    /// use winit::window::Window;
    /// use grafo::{FontFamily, MathRect, Renderer, TextAlignment, TextLayout};
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
    ///         let layout = TextLayout {
    ///             font_size: 24.0,
    ///             line_height: 30.0,
    ///             color: Color::rgb(255, 255, 255), // White text
    ///             area: MathRect {
    ///                 min: (50.0, 50.0).into(),
    ///                 max: (400.0, 100.0).into(),
    ///             },
    ///             horizontal_alignment: TextAlignment::Center,
    ///             vertical_alignment: TextAlignment::Center,
    ///         };
    ///         renderer.add_text("Hello, Grafo!", layout, FontFamily::SansSerif, None, 0);
    ///     }
    ///     fn window_event(&mut self, _: &ActiveEventLoop, _: winit::window::WindowId, _: winit::event::WindowEvent) {}
    /// }
    /// ```
    pub fn add_text(
        &mut self,
        text: &str,
        layout: impl Into<TextLayout>,
        font_family: FontFamily,
        clip_to_shape: Option<usize>,
        buffer_id: usize,
    ) {
        self.add_text_internal(text, layout, font_family, clip_to_shape, None, buffer_id)
    }

    /// Same as [Renderer::add_text], but allows to use a custom font system
    pub fn add_text_with_custom_font_system(
        &mut self,
        text: &str,
        layout: impl Into<TextLayout>,
        font_family: FontFamily,
        clip_to_shape: Option<usize>,
        font_system: &mut FontSystem,
        buffer_id: usize,
    ) {
        self.add_text_internal(
            text,
            layout,
            font_family,
            clip_to_shape,
            Some(font_system),
            buffer_id,
        )
    }

    fn add_text_internal(
        &mut self,
        text: &str,
        layout: impl Into<TextLayout>,
        font_family: FontFamily,
        clip_to_shape: Option<usize>,
        font_system: Option<&mut FontSystem>,
        buffer_id: usize,
    ) {
        let font_system = font_system.unwrap_or(&mut self.font_system);

        self.text_instances.push(TextDrawData::new(
            text,
            layout,
            self.scale_factor as f32,
            font_system,
            font_family,
            clip_to_shape,
            buffer_id,
        ));
    }

    /// This method  adds the text buffer to the draw queue. This is useful when you want to use
    /// the text buffer somewhere else, for example to detect clicks on the text.
    ///
    /// # Parameters
    ///
    /// - `text_buffer`: Text buffer to be rendered.
    /// - `area`: Where to render the text.
    /// - `fallback_color`: Color used as a fallback color.
    /// - `vertical_offset`: Vertical offset from the top of the canvas where to start rendering the text.
    ///
    /// # NOTE
    /// It is very important to set the metadata of the text buffer to be equal to the id of the
    /// shape that is going to be used for clipping, if you want the text to be clipped by a shape.
    pub fn add_text_buffer(
        &mut self,
        text_buffer: impl IntoCowBuffer<'a>,
        area: MathRect,
        fallback_color: Color,
        vertical_offset: f32,
        buffer_metadata: usize,
        clip_to_shape: Option<usize>,
    ) {
        self.text_instances.push(TextDrawData::with_buffer(
            text_buffer,
            area,
            fallback_color,
            vertical_offset,
            clip_to_shape,
            buffer_metadata,
        ));
    }

    /// Renders all items currently in the draw queue.
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
    /// use grafo::{MathRect, Renderer, TextAlignment, TextLayout};
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
    ///         // Add shapes, images, and text...
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
        self.render_internal(None, None, None::<NoopTextDataIter>)
    }

    pub fn render_with_custom_font_system<'b>(
        &mut self,
        font_system: &mut FontSystem,
        swash_cache: &mut SwashCache,
        text_instances: Option<impl Iterator<Item = TextDrawData<'b>>>,
    ) -> Result<(), wgpu::SurfaceError> {
        self.render_internal(Some(font_system), Some(swash_cache), text_instances)
    }

    fn render_internal<'b>(
        &mut self,
        custom_font_system: Option<&mut FontSystem>,
        custom_swash_cache: Option<&mut SwashCache>,
        custom_text_instances: Option<impl Iterator<Item = TextDrawData<'b>>>,
    ) -> Result<(), wgpu::SurfaceError> {
        self.temp_vertices.clear();
        self.temp_indices.clear();

        let draw_tree_size = self.draw_tree.len();

        let tessellator = &mut self.tessellator;
        let buffers_pool_manager = &mut self.buffers_pool_manager;
        let texture_manager = &self.texture_manager;
        let physical_size = self.physical_size;
        let scale_factor = self.scale_factor;

        // First pass: tessellate all shapes and aggregate vertex/index data
        for (node_id, draw_command) in self.draw_tree.iter_mut() {
            match draw_command {
                DrawCommand::Shape(ref mut shape) => {
                    let depth = depth(node_id, draw_tree_size);

                    // Get tessellated buffers using the optimized cache approach
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
                }
                DrawCommand::Image(ref mut image) => {
                    image.prepare(
                        texture_manager,
                        physical_size,
                        scale_factor as f32,
                        buffers_pool_manager,
                    );
                }
                DrawCommand::CachedShape(cached_shape_data) => {
                    let depth = depth(node_id, draw_tree_size);

                    if let Some(cached_shape) = self.shape_cache.get_mut(&cached_shape_data.id) {
                        if cached_shape.vertex_buffers.vertices.is_empty() || cached_shape.vertex_buffers.indices.is_empty() {
                            cached_shape_data.is_empty = true;
                            continue;
                        }

                        if depth != cached_shape.depth && cached_shape.offset != cached_shape_data.offset {
                            cached_shape.set_offset_and_depth(cached_shape_data.offset, depth);
                        } else if depth != cached_shape.depth {
                            cached_shape.set_depth(depth);
                        } else if cached_shape_data.offset != cached_shape.offset {
                            cached_shape.set_offset(cached_shape_data.offset);
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
            let needs_realloc = self.aggregated_vertex_buffer.as_ref()
                .map(|buffer| buffer.size() < required_vertex_size as u64)
                .unwrap_or(true);
                
            if needs_realloc {
                self.aggregated_vertex_buffer = Some(self.device.create_buffer_init(&BufferInitDescriptor {
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
            let needs_realloc = self.aggregated_index_buffer.as_ref()
                .map(|buffer| buffer.size() < required_index_size as u64)
                .unwrap_or(true);
                
            if needs_realloc {
                self.aggregated_index_buffer = Some(self.device.create_buffer_init(&BufferInitDescriptor {
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

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("And Command Encoder"),
            });

        let output = self.surface.get_current_texture()?;
        let output_texture_view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let depth_texture =
            create_and_depth_texture(&self.device, (self.physical_size.0, self.physical_size.1));

        {
            let depth_texture_view =
                depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

            let render_pass =
                create_render_pass(&mut encoder, &output_texture_view, &depth_texture_view);

            let shape_ids_to_stencil_references = HashMap::<usize, u32>::new();
            let current_pipeline = Pipeline::None;

            let mut data = (
                render_pass,
                shape_ids_to_stencil_references,
                current_pipeline,
            );

            self.draw_tree.traverse(
                |shape_id, draw_command, data| {
                    // NOTE: this is destructured here and not above because we need to pass the
                    //  data to the closure below
                    let (render_pass, stencil_references, currently_set_pipeline) = data;

                    match draw_command {
                        DrawCommand::Shape(shape) => {
                            if let Some(index_range) = shape.index_buffer_range
                            {
                                if shape.is_empty {
                                    return;
                                }

                                if !matches!(currently_set_pipeline, Pipeline::StencilIncrement) {
                                    render_pass.set_pipeline(&self.and_pipeline);
                                    render_pass.set_bind_group(0, &self.and_bind_group, &[]);
                                    *currently_set_pipeline = Pipeline::StencilIncrement;
                                }

                                if let Some(clip_to_shape) = shape.clip_to_shape {
                                    let parent_stencil =
                                        *stencil_references.get(&clip_to_shape).unwrap_or(&0);

                                    render_buffer_range_to_texture(
                                        self.aggregated_vertex_buffer.as_ref().unwrap(),
                                        self.aggregated_index_buffer.as_ref().unwrap(),
                                        index_range,
                                        render_pass,
                                        parent_stencil,
                                    );

                                    stencil_references.insert(shape_id, parent_stencil + 1);
                                } else {
                                    // TODO: every no clip should have its own tree, and before
                                    //  rendering them we need to reset stencil texture, and
                                    //  they should be rendered in a separate step

                                    render_buffer_range_to_texture(
                                        self.aggregated_vertex_buffer.as_ref().unwrap(),
                                        self.aggregated_index_buffer.as_ref().unwrap(),
                                        index_range,
                                        render_pass,
                                        0,
                                    );

                                    stencil_references.insert(shape_id, 1);
                                }
                            } else {
                                warn!(
                                    "Missing vertex or index buffer or ranges for shape {}",
                                    shape_id
                                );
                            }
                        },
                        DrawCommand::CachedShape(shape) => {
                            // TODO: this code is exactly the same as the above, think about how to
                            //  merge those branches into one
                            if let Some(index_range) =
                                shape.index_buffer_range
                            {
                                if shape.is_empty {
                                    return;
                                }

                                if !matches!(currently_set_pipeline, Pipeline::StencilIncrement) {
                                    render_pass.set_pipeline(&self.and_pipeline);
                                    render_pass.set_bind_group(0, &self.and_bind_group, &[]);
                                    *currently_set_pipeline = Pipeline::StencilIncrement;
                                }

                                if let Some(clip_to_shape) = shape.clip_to_shape {
                                    let parent_stencil =
                                        *stencil_references.get(&clip_to_shape).unwrap_or(&0);

                                    render_buffer_range_to_texture(
                                        self.aggregated_vertex_buffer.as_ref().unwrap(),
                                        self.aggregated_index_buffer.as_ref().unwrap(),
                                        index_range,
                                        render_pass,
                                        parent_stencil,
                                    );

                                    stencil_references.insert(shape_id, parent_stencil + 1);
                                } else {
                                    // TODO: every no clip should have its own tree, and before
                                    //  rendering them we need to reset stencil texture, and
                                    //  they should be rendered in a separate step

                                    render_buffer_range_to_texture(
                                        self.aggregated_vertex_buffer.as_ref().unwrap(),
                                        self.aggregated_index_buffer.as_ref().unwrap(),
                                        index_range,
                                        render_pass,
                                        0,
                                    );

                                    stencil_references.insert(shape_id, 1);
                                }
                            } else {
                                warn!(
                                    "Missing vertex or index buffer or ranges for shape {}",
                                    shape_id
                                );
                            }
                        }
                        DrawCommand::Image(image) => {
                            if let Some(clip_to_shape) = image.clip_to_shape {
                                let parent_stencil =
                                    *stencil_references.get(&clip_to_shape).unwrap_or(&0);
                                // If the image is clipped to a shape, we set up the pipeline
                                // to crop the image to the shape using the stencil texture
                                render_pass.set_pipeline(&self.texture_crop_render_pipeline);
                                render_pass.set_stencil_reference(parent_stencil);
                                *currently_set_pipeline = Pipeline::TextureCrop;
                            } else {
                                // If the image is not clipped to a shape, we set up the pipeline
                                // to always render the image
                                render_pass.set_pipeline(&self.texture_always_render_pipeline);
                                *currently_set_pipeline = Pipeline::TextureAlways;
                            };

                            render_pass.set_vertex_buffer(0, image.vertex_buffer());
                            render_pass
                                .set_index_buffer(image.index_buffer(), wgpu::IndexFormat::Uint16);
                            render_pass.set_bind_group(0, image.bind_group(), &[]);
                            render_pass.draw_indexed(0..image.num_indices(), 0, 0..1);
                        }
                    }
                },
                |shape_id, draw_command, data| {
                    let (render_pass, _stencil_references, currently_set_pipeline) = data;

                    match draw_command {
                        DrawCommand::Shape(shape) => {
                            if let Some(index_range) = shape.index_buffer_range
                            {
                                if shape.is_empty {
                                    return;
                                }

                                if !matches!(currently_set_pipeline, Pipeline::StencilDecrement) {
                                    render_pass.set_pipeline(&self.decrementing_pipeline);
                                    render_pass.set_bind_group(
                                        0,
                                        &self.decrementing_bind_group,
                                        &[],
                                    );
                                    *currently_set_pipeline = Pipeline::StencilDecrement;
                                }

                                let this_shape_stencil = { *data.1.get(&shape_id).unwrap_or(&0) };

                                render_buffer_range_to_texture(
                                    self.aggregated_vertex_buffer.as_ref().unwrap(),
                                    self.aggregated_index_buffer.as_ref().unwrap(),
                                    index_range,
                                    render_pass,
                                    this_shape_stencil,
                                );
                            } else {
                                warn!(
                                    "No vertex or index buffer or ranges found for shape {}",
                                    shape_id
                                );
                                // TODO: no vertex or index buffer found - it is an error,
                                //  so probably should panic
                            }
                        },
                        DrawCommand::CachedShape(shape) => {
                            // TODO: this is a copy-paste from the above, think about how to combine
                            //  them into one branch
                            if let Some(index_range) =
                                shape.index_buffer_range
                            {
                                if shape.is_empty {
                                    return;
                                }

                                if !matches!(currently_set_pipeline, Pipeline::StencilDecrement) {
                                    render_pass.set_pipeline(&self.decrementing_pipeline);
                                    render_pass.set_bind_group(
                                        0,
                                        &self.decrementing_bind_group,
                                        &[],
                                    );
                                    *currently_set_pipeline = Pipeline::StencilDecrement;
                                }

                                let this_shape_stencil = { *data.1.get(&shape_id).unwrap_or(&0) };

                                render_buffer_range_to_texture(
                                    self.aggregated_vertex_buffer.as_ref().unwrap(),
                                    self.aggregated_index_buffer.as_ref().unwrap(),
                                    index_range,
                                    render_pass,
                                    this_shape_stencil,
                                );
                            } else {
                                warn!(
                                    "No vertex or index buffer or ranges found for shape {}",
                                    shape_id
                                );
                                // TODO: no vertex or index buffer found - it is an error,
                                //  so probably should panic
                            }
                        }
                        DrawCommand::Image(_) => {
                            // nothing to do here
                        }
                    }
                },
                &mut data,
            );

            // // Preparing text renderer
            self.prepare_text_buffers(
                custom_text_instances,
                custom_font_system,
                custom_swash_cache,
            );

            self.text_renderer_wrapper
                .text_renderer
                .render(
                    &self.text_renderer_wrapper.atlas,
                    &self.glyphon_viewport,
                    &mut data.0,
                )
                .unwrap();
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        // Clear shape buffer references and return other resources to the pool
        self.draw_tree
            .iter_mut()
            .for_each(|draw_command| match draw_command.1 {
                DrawCommand::Shape(ref mut shape) => {
                    // Clear the aggregated buffer references
                    shape.index_buffer_range = None;
                }
                DrawCommand::CachedShape(ref mut shape) => {
                    // Clear the aggregated buffer references
                    shape.index_buffer_range = None;
                }
                DrawCommand::Image(ref mut image) => {
                    image.return_buffers_to_pool(&mut self.buffers_pool_manager);
                }
            });

        // self.buffers_pool_manager.print_sizes();

        Ok(())
    }

    pub fn prepare_text_buffers<'b>(
        &mut self,
        custom_text_instances_iter: Option<impl Iterator<Item = TextDrawData<'b>>>,
        custom_font_system: Option<&mut FontSystem>,
        custom_swash_cache: Option<&mut SwashCache>,
    ) {
        let font_system = custom_font_system.unwrap_or(&mut self.font_system);
        let swash_cache = custom_swash_cache.unwrap_or(&mut self.swash_cache);

        if let Some(text_instances_iter) = custom_text_instances_iter {
            // TODO: make vec a pool
            let text_areas = text_instances_iter
                .map(|text_instance| {
                    self.metadata_to_clips.insert(
                        text_instance.buffer_metadata,
                        text_instance.clip_to_shape.unwrap_or_default(),
                    );
                    text_instance.into_text_area(self.scale_factor as f32)
                })
                .collect::<Vec<_>>();

            self.text_renderer_wrapper
                .text_renderer
                .prepare_with_depth(
                    &self.device,
                    &self.queue,
                    font_system,
                    &mut self.text_renderer_wrapper.atlas,
                    &self.glyphon_viewport,
                    text_areas,
                    swash_cache,
                    |metadata| {
                        depth(
                            self.metadata_to_clips.get(&metadata).copied().unwrap_or(0),
                            self.draw_tree.len(),
                        )
                    },
                )
                .unwrap();
        } else {
            // TODO: make vec a pool
            let text_areas = self
                .text_instances
                .iter()
                .map(|text_instance| {
                    self.metadata_to_clips.insert(
                        text_instance.buffer_metadata,
                        text_instance.clip_to_shape.unwrap_or_default(),
                    );
                    text_instance.to_text_area(self.scale_factor as f32)
                })
                .collect::<Vec<_>>();

            self.text_renderer_wrapper
                .text_renderer
                .prepare_with_depth(
                    &self.device,
                    &self.queue,
                    font_system,
                    &mut self.text_renderer_wrapper.atlas,
                    &self.glyphon_viewport,
                    text_areas,
                    swash_cache,
                    |metadata| {
                        depth(
                            self.metadata_to_clips.get(&metadata).copied().unwrap_or(0),
                            self.draw_tree.len(),
                        )
                    },
                )
                .unwrap();

            self.text_instances.clear();
        }
    }

    /// Clears all items currently in the draw queue.
    ///
    /// This method removes all shapes, images, and text instances from the draw queue,
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
    /// use grafo::{MathRect, Renderer, TextAlignment, TextLayout};
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
    ///         // Add shapes, images, and text...
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
        self.text_instances.clear();
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

    /// Retrieves the current size of the rendering surface.
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
    /// use grafo::{MathRect, Renderer, TextAlignment, TextLayout};
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
    /// use grafo::{MathRect, Renderer, TextAlignment, TextLayout};
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
    /// ```
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
    /// use grafo::{MathRect, Renderer, TextAlignment, TextLayout};
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
        self.surface.configure(&self.device, &self.config);
        // Update the render pipeline to match the new size with new uniforms. Uniforms are
        // needed to normalize the coordinates of the shapes

        let (and_uniforms, and_bind_group, and_pipeline) = create_pipeline(
            to_logical(new_physical_size, self.scale_factor),
            &self.device,
            &self.config,
            PipelineType::EqualIncrementStencil,
        );
        self.and_uniforms = and_uniforms;
        self.and_bind_group = and_bind_group;
        self.and_pipeline = Arc::new(and_pipeline);

        // Update the always decrement pipeline
        let (decrementing_uniforms, decrementing_bind_group, decrementing_pipeline) =
            create_pipeline(
                to_logical(new_physical_size, self.scale_factor),
                &self.device,
                &self.config,
                PipelineType::EqualDecrementStencil,
            );
        self.decrementing_uniforms = decrementing_uniforms;
        self.decrementing_bind_group = decrementing_bind_group;
        self.decrementing_pipeline = Arc::new(decrementing_pipeline);

        let (
            texture_bind_group_layout,
            texture_crop_render_pipeline,
            texture_always_render_pipeline,
        ) = create_texture_pipeline(&self.device, &self.config);
        self.texture_manager
            .set_bind_group_layout(texture_bind_group_layout);
        self.texture_crop_render_pipeline = Arc::new(texture_crop_render_pipeline);
        self.texture_always_render_pipeline = Arc::new(texture_always_render_pipeline);

        self.glyphon_viewport.update(
            &self.queue,
            Resolution {
                width: new_physical_size.0,
                height: new_physical_size.1,
            },
        );
    }

    pub fn set_vsync(&mut self, vsync: bool) {
        self.config.present_mode = if vsync {
            wgpu::PresentMode::Fifo
        } else {
            wgpu::PresentMode::Immediate
        };
        self.surface.configure(&self.device, &self.config);
    }

    /// Loads fonts from the specified sources.
    ///
    /// Loaded fonts can be later used to render text using the [Renderer::add_text] method.
    ///
    /// # Parameters
    ///
    /// - `fonts`: An iterator of [fontdb::Source] objects representing the font sources to load.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use std::sync::Arc;
    /// use futures::executor::block_on;
    /// use winit::application::ApplicationHandler;
    /// use winit::event_loop::{ActiveEventLoop, EventLoop};
    /// use winit::window::Window;
    /// use grafo::{MathRect, Renderer, TextAlignment, TextLayout};
    /// use grafo::Shape;
    /// use grafo::Color;
    /// use grafo::Stroke;
    /// use grafo::fontdb;
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
    ///         let roboto_font_ttf = include_bytes!("../examples/assets/Roboto-Regular.ttf").to_vec();
    ///         let roboto_font_source = fontdb::Source::Binary(Arc::new(roboto_font_ttf));
    ///         renderer.load_fonts([roboto_font_source].into_iter());
    ///     }
    ///
    ///     fn window_event(&mut self, _: &ActiveEventLoop, _: winit::window::WindowId, _: winit::event::WindowEvent) {
    ///         // Handle window events (stub for doc test)
    ///     }
    /// }
    /// ```
    pub fn load_fonts(&mut self, fonts: impl Iterator<Item = fontdb::Source>) {
        self.load_fonts_internal(fonts, None)
    }

    /// Same as [Renderer::load_fonts], but allows to use a custom font system.
    pub fn load_fonts_with_custom_font_system(
        &mut self,
        fonts: impl Iterator<Item = fontdb::Source>,
        font_system: &mut FontSystem,
    ) {
        self.load_fonts_internal(fonts, Some(font_system))
    }

    fn load_fonts_internal(
        &mut self,
        fonts: impl Iterator<Item = fontdb::Source>,
        font_system: Option<&mut FontSystem>,
    ) {
        let font_system = font_system.unwrap_or(&mut self.font_system);
        let db = font_system.db_mut();

        for source in fonts {
            db.load_font_source(source);
        }
    }

    /// Loads a font from a byte slice.
    ///
    /// Loaded fonts can be later used to render text using the [Renderer::add_text] method.
    ///
    /// # Parameters
    ///
    /// - `font_bytes`: A slice of bytes representing the font file.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use std::sync::Arc;
    /// use futures::executor::block_on;
    /// use winit::application::ApplicationHandler;
    /// use winit::event_loop::{ActiveEventLoop, EventLoop};
    /// use winit::window::Window;
    /// use grafo::{MathRect, Renderer, TextAlignment, TextLayout};
    /// use grafo::Shape;
    /// use grafo::Color;
    /// use grafo::Stroke;
    /// use grafo::fontdb;
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
    ///         let roboto_font_ttf = include_bytes!("../examples/assets/Roboto-Regular.ttf");
    ///         renderer.load_font_from_bytes(roboto_font_ttf);
    ///     }
    ///
    ///     fn window_event(&mut self, _: &ActiveEventLoop, _: winit::window::WindowId, _: winit::event::WindowEvent) {
    ///         // Handle window events (stub for doc test)
    ///     }
    /// }
    /// ```
    pub fn load_font_from_bytes(&mut self, font_bytes: &[u8]) {
        self.load_font_from_bytes_internal(font_bytes, None)
    }

    /// Same as [Renderer::load_font_from_bytes], but allows to use a custom font system.
    pub fn load_font_from_bytes_with_custom_font_system(
        &mut self,
        font_bytes: &[u8],
        font_system: &mut FontSystem,
    ) {
        self.load_font_from_bytes_internal(font_bytes, Some(font_system))
    }

    fn load_font_from_bytes_internal(
        &mut self,
        font_bytes: &[u8],
        font_system: Option<&mut FontSystem>,
    ) {
        let font_system = font_system.unwrap_or(&mut self.font_system);
        let db = font_system.db_mut();
        let source = fontdb::Source::Binary(Arc::new(font_bytes.to_vec()));
        db.load_font_source(source);
    }
}
