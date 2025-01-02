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
//! use grafo::Renderer;
//! use grafo::Shape;
//! use grafo::Color;
//! use grafo::Stroke;
//! use grafo::{TextAlignment, TextLayout};
//! use grafo::MathRect;
//! use wgpu::Surface;
//! use winit::event_loop::EventLoop;
//! use winit::window::WindowBuilder;
//! use futures::executor::block_on;
//!
//! // This is for demonstration purposes only. If you want a working example with winit, please
//! // refer to the example in the "examples" folder.
//! let event_loop = EventLoop::new().expect("To create the event loop");
//! let window_surface = Arc::new(WindowBuilder::new().build(&event_loop).unwrap());
//! let physical_size = (800, 600);
//! let scale_factor = 1.0;
//!
//! // Initialize the renderer
//! let mut renderer = block_on(Renderer::new(window_surface, physical_size, scale_factor));
//!
//! // Add a rectangle shape
//! let rect = Shape::rect(
//!     [(100.0, 100.0), (300.0, 200.0)],
//!     Color::rgb(0, 128, 255), // Blue fill
//!     Stroke::new(2.0, Color::BLACK), // Black stroke with width 2.0
//! );
//! renderer.add_shape(rect, None);
//!
//! // Add some text
//! let layout = TextLayout {
//!     font_size: 24.0,
//!     line_height: 30.0,
//!     color: Color::rgb(255, 255, 255), // White text
//!     area: MathRect {
//!         min: (50.0, 50.0).into(),
//!         max: (400.0, 100.0).into(),
//!     },
//!     horizontal_alignment: TextAlignment::Center,
//!     vertical_alignment: TextAlignment::Center,
//! };
//! renderer.add_text("Hello, Grafo!", layout, None, None);
//!
//! // Render the frame
//! match renderer.render() {
//!     Ok(_) => println!("Frame rendered successfully."),
//!     Err(e) => eprintln!("Rendering error: {:?}", e),
//! }
//!
//! // Clear the draw queue after rendering
//! renderer.clear_draw_queue();
//! ```

use std::sync::Arc;
use std::time::Instant;

use easy_tree::rayon::iter::ParallelIterator;

pub type MathRect = lyon::math::Box2D;

use crate::pipeline::{
    create_and_depth_texture, create_depth_stencil_state_for_text, create_pipeline,
    create_render_pass, create_texture_pipeline, render_buffer_to_texture2, PipelineType, Uniforms,
};
use crate::util::to_logical;
use ahash::{HashMap, HashMapExt};
use glyphon::{fontdb, Family, Resolution};

use crate::image_draw_data::ImageDrawData;
use crate::shape::{Shape, ShapeDrawData};

use crate::text::{TextDrawData, TextLayout, TextRendererWrapper};
use wgpu::{BindGroup, CompositeAlphaMode, InstanceDescriptor, SurfaceTarget};

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
    Image(ImageDrawData),
}

/// The renderer for the Grafo library. This is the main struct that is used to render shapes,
/// images, and text.
///
/// # Examples
///
/// ```rust,no_run
/// use grafo::Renderer;
/// use grafo::Shape;
/// use grafo::Color;
/// use grafo::Stroke;
/// use grafo::{TextAlignment, TextLayout};
/// use grafo::MathRect;
/// use wgpu::Surface;
/// use winit::event_loop::EventLoop;
/// use winit::window::WindowBuilder;
/// use std::sync::Arc;
/// use futures::executor::block_on;
///
/// // This is for demonstration purposes only. If you want a working example with winit, please
/// // refer to the example in the "examples" folder.
/// let event_loop = EventLoop::new().expect("To create the event loop");
/// let window_surface = Arc::new(WindowBuilder::new().build(&event_loop).unwrap());
/// let physical_size = (800, 600);
/// let scale_factor = 1.0;
///
///
/// // Initialize the renderer
/// let mut renderer = block_on(Renderer::new(window_surface, physical_size, scale_factor));
///
/// // Add a rectangle shape
/// let rect = Shape::rect(
///     [(100.0, 100.0), (300.0, 200.0)],
///     Color::rgb(0, 128, 255), // Blue fill
///     Stroke::new(2.0, Color::BLACK), // Black stroke with width 2.0
/// );
/// renderer.add_shape(rect, None);
///
/// // Add some text
/// let layout = TextLayout {
///     font_size: 24.0,
///     line_height: 30.0,
///     color: Color::rgb(255, 255, 255), // White text
///     area: MathRect {
///         min: (50.0, 50.0).into(),
///         max: (400.0, 100.0).into(),
///     },
///     horizontal_alignment: TextAlignment::Center,
///     vertical_alignment: TextAlignment::Center,
/// };
/// renderer.add_text("Hello, Grafo!", layout, None, None);
///
/// // Render the frame
/// match renderer.render() {
///     Ok(_) => println!("Frame rendered successfully."),
///     Err(e) => eprintln!("Rendering error: {:?}", e),
/// }
///
/// // Clear the draw queue after rendering
/// renderer.clear_draw_queue();
/// ```
pub struct Renderer<'a> {
    // Window information
    /// Size of the window in pixels.
    pub(crate) physical_size: (u32, u32),
    /// Scale factor of the window (e.g., for high-DPI displays).
    scale_factor: f64,

    // WGPU components
    surface: wgpu::Surface<'a>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,

    /// Text instances to be rendered
    text_instances: Vec<TextDrawData>,
    /// Internal wrapper for text rendering components.
    text_renderer_wrapper: TextRendererWrapper,
    glyphon_viewport: glyphon::Viewport,

    /// Tree structure holding shapes and images to be rendered.
    draw_tree: easy_tree::Tree<DrawCommand>,

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
    /// Bind group layout for texture rendering pipelines.
    texture_bind_group_layout: wgpu::BindGroupLayout,
}

impl Renderer<'_> {
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
    /// use wgpu::Surface;
    /// use winit::event_loop::EventLoop;
    /// use winit::window::WindowBuilder;
    ///
    /// // This is for demonstration purposes only. If you want a working example with winit, please
    /// // refer to the example in the "examples" folder.
    /// let event_loop = EventLoop::new().expect("To create the event loop");
    /// let window_surface = Arc::new(WindowBuilder::new().build(&event_loop).unwrap());
    /// let physical_size = (800, 600);
    /// let scale_factor = 1.0;
    ///
    /// // Initialize the renderer
    /// let renderer = block_on(Renderer::new(window_surface, physical_size, scale_factor));
    /// ```
    pub async fn new(
        window: impl Into<SurfaceTarget<'static>>,
        physical_size: (u32, u32),
        scale_factor: f64,
    ) -> Self {
        let size = physical_size;
        let canvas_logical_size = to_logical(size, scale_factor);

        let instance = wgpu::Instance::new(InstanceDescriptor::default());
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
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    #[cfg(feature = "performance_measurement")]
                    required_features: wgpu::Features::TIMESTAMP_QUERY
                        | wgpu::Features::DEPTH32FLOAT_STENCIL8,
                    #[cfg(not(feature = "performance_measurement"))]
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: Default::default(),
                },
                None,
            )
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

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: swapchain_format,
            width: size.0,
            height: size.1,
            present_mode: wgpu::PresentMode::Fifo,
            desired_maximum_frame_latency: 2,
            // TODO: Check if this is the correct alpha mode
            alpha_mode: CompositeAlphaMode::Opaque,
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

        let glyphon_cache = glyphon::Cache::new(&device);
        let mut glyphon_viewport = glyphon::Viewport::new(&device, &glyphon_cache);

        {
            glyphon_viewport.update(
                &queue,
                Resolution {
                    width: size.0,
                    height: size.1,
                },
            );
        }

        let text_renderer_wrapper = TextRendererWrapper::new(
            &device,
            &queue,
            swapchain_format,
            Some(create_depth_stencil_state_for_text()),
            &glyphon_cache,
        );

        let (
            texture_bind_group_layout,
            texture_crop_render_pipeline,
            texture_always_render_pipeline,
        ) = create_texture_pipeline(&device, &config);

        Self {
            surface,
            device,
            queue,
            config,
            physical_size: size,

            scale_factor,
            text_instances,
            // draw_queue: BTreeMap::new(),
            text_renderer_wrapper,
            glyphon_viewport,

            and_pipeline: Arc::new(and_pipeline),
            and_uniforms,
            and_bind_group,

            decrementing_pipeline: Arc::new(decrementing_pipeline),
            decrementing_uniforms,
            decrementing_bind_group,

            draw_tree: easy_tree::Tree::new(),

            // #[cfg(feature = "performance_measurement")]
            // performance_query_set: frametime_query_set,
            // #[cfg(feature = "performance_measurement")]
            // adapter,
            texture_crop_render_pipeline: Arc::new(texture_crop_render_pipeline),
            texture_always_render_pipeline: Arc::new(texture_always_render_pipeline),
            texture_bind_group_layout,
        }
    }

    /// Adds a shape to the draw queue.
    ///
    /// # Parameters
    ///
    /// - `shape`: The shape to be rendered. It can be any type that implements `Into<Shape>`.
    /// - `clip_to_shape`: Optional index of another shape to which this shape should be clipped.
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
    /// use winit::event_loop::EventLoop;
    /// use winit::window::WindowBuilder;
    /// use grafo::Renderer;
    /// use grafo::Shape;
    /// use grafo::Color;
    /// use grafo::Stroke;
    ///
    /// // This is for demonstration purposes only. If you want a working example with winit, please
    /// // refer to the example in the "examples" folder.
    /// let event_loop = EventLoop::new().expect("To create the event loop");
    /// let window_surface = Arc::new(WindowBuilder::new().build(&event_loop).unwrap());
    /// let physical_size = (800, 600);
    /// let scale_factor = 1.0;
    ///
    /// // Initialize the renderer
    /// let mut renderer = block_on(Renderer::new(window_surface, physical_size, scale_factor));
    ///
    /// let shape_id = renderer.add_shape(
    ///     Shape::rect(
    ///         [(100.0, 100.0), (300.0, 200.0)],
    ///         Color::rgb(0, 128, 255), // Blue fill
    ///         Stroke::new(2.0, Color::BLACK), // Black stroke with width 2.0
    ///     ),
    ///     None,
    /// );
    /// ```
    pub fn add_shape(&mut self, shape: impl Into<Shape>, clip_to_shape: Option<usize>) -> usize {
        self.add_draw_command(
            DrawCommand::Shape(ShapeDrawData::new(shape, clip_to_shape)),
            clip_to_shape,
        )
    }

    /// Adds an image to the draw queue.
    ///
    /// # Parameters
    ///
    /// - `image`: A byte slice representing the image data.
    /// - `physical_image_dimensions`: A tuple representing the image's width and height in pixels.
    /// - `rect`: An array containing two tuples representing the top-left and bottom-right
    ///   coordinates where the image should be rendered.
    /// - `clip_to_shape`: Optional index of a shape to which this image should be clipped.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use std::sync::Arc;
    /// use futures::executor::block_on;
    /// use winit::event_loop::EventLoop;
    /// use winit::window::WindowBuilder;
    /// use grafo::Renderer;
    /// use grafo::Shape;
    /// use grafo::Color;
    /// use grafo::Stroke;
    ///
    /// // This is for demonstration purposes only. If you want a working example with winit, please
    /// // refer to the example in the "examples" folder.
    /// let event_loop = EventLoop::new().expect("To create the event loop");
    /// let window_surface = Arc::new(WindowBuilder::new().build(&event_loop).unwrap());
    /// let physical_size = (800, 600);
    /// let scale_factor = 1.0;
    ///
    /// // Initialize the renderer
    /// let mut renderer = block_on(Renderer::new(window_surface, physical_size, scale_factor));
    ///
    /// let image_data = vec![0; 16]; // A 2x2 black image
    /// renderer.add_image(
    ///     &image_data,
    ///     (2, 2), // Image dimensions
    ///     [(50.0, 50.0), (150.0, 150.0)], // Rendering rectangle
    ///     Some(0), // Clip to shape with ID 0
    /// );
    /// ```
    pub fn add_image(
        &mut self,
        image: &[u8],
        physical_image_dimensions: (u32, u32),
        rect: [(f32, f32); 2],
        clip_to_shape: Option<usize>,
    ) {
        // TODO: cache image data
        // let texture_id = TextureId::new(image);
        let image_data = image.to_vec();

        let draw_command = DrawCommand::Image(ImageDrawData::new(
            image_data,
            physical_image_dimensions,
            rect,
            clip_to_shape,
        ));

        self.add_draw_command(draw_command, clip_to_shape);
    }

    /// Adds text to the draw queue.
    ///
    /// # Parameters
    ///
    /// - `text`: The string of text to be rendered.
    /// - `layout`: The layout configuration for the text.
    /// - `clip_to_shape`: Optional index of a shape to which this text should be clipped.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use std::sync::Arc;
    /// use futures::executor::block_on;
    /// use winit::event_loop::EventLoop;
    /// use winit::window::WindowBuilder;
    /// use grafo::{MathRect, Renderer, TextAlignment, TextLayout};
    /// use grafo::Shape;
    /// use grafo::Color;
    /// use grafo::Stroke;
    ///
    /// // This is for demonstration purposes only. If you want a working example with winit, please
    /// // refer to the example in the "examples" folder.
    /// let event_loop = EventLoop::new().expect("To create the event loop");
    /// let window_surface = Arc::new(WindowBuilder::new().build(&event_loop).unwrap());
    /// let physical_size = (800, 600);
    /// let scale_factor = 1.0;
    ///
    /// // Initialize the renderer
    /// let mut renderer = block_on(Renderer::new(window_surface, physical_size, scale_factor));
    ///
    /// let layout = TextLayout {
    ///     font_size: 24.0,
    ///     line_height: 30.0,
    ///     color: Color::rgb(255, 255, 255), // White text
    ///     area: MathRect {
    ///         min: (50.0, 50.0).into(),
    ///         max: (400.0, 100.0).into(),
    ///     },
    ///     horizontal_alignment: TextAlignment::Center,
    ///     vertical_alignment: TextAlignment::Center,
    /// };
    /// renderer.add_text("Hello, Grafo!", layout, None, None);
    /// ```
    pub fn add_text(
        &mut self,
        text: &str,
        layout: impl Into<TextLayout>,
        clip_to_shape: Option<usize>,
        font_family: Option<&str>,
    ) {
        self.text_instances.push(TextDrawData::new(
            text,
            layout,
            clip_to_shape,
            self.scale_factor as f32,
            &mut self.text_renderer_wrapper.font_system,
            Some(Family::Name(font_family.unwrap_or("sans-serif"))),
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
    /// use winit::event_loop::EventLoop;
    /// use winit::window::WindowBuilder;
    /// use grafo::{MathRect, Renderer, TextAlignment, TextLayout};
    /// use grafo::Shape;
    /// use grafo::Color;
    /// use grafo::Stroke;
    ///
    /// // This is for demonstration purposes only. If you want a working example with winit, please
    /// // refer to the example in the "examples" folder.
    /// let event_loop = EventLoop::new().expect("To create the event loop");
    /// let window_surface = Arc::new(WindowBuilder::new().build(&event_loop).unwrap());
    /// let physical_size = (800, 600);
    /// let scale_factor = 1.0;
    ///
    /// // Initialize the renderer
    /// let mut renderer = block_on(Renderer::new(window_surface, physical_size, scale_factor));
    ///
    /// // Add shapes, images, and text...
    ///
    /// // Render the frame
    /// if let Err(e) = renderer.render() {
    ///     eprintln!("Rendering error: {:?}", e);
    /// }
    /// ```
    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        println!("===============");
        let first_timer = Instant::now();
        let draw_tree_size = self.draw_tree.len();
        let iter = self.draw_tree.par_iter_mut();

        iter.for_each(|draw_command| match draw_command.1 {
            DrawCommand::Shape(ref mut shape) => {
                shape.prepare_buffers(&self.device, draw_command.0, draw_tree_size);
            }
            DrawCommand::Image(ref mut image) => {
                image.prepare(
                    &self.device,
                    &self.queue,
                    &self.texture_bind_group_layout,
                    self.physical_size,
                    self.scale_factor as f32,
                );
            }
        });

        println!(
            "Iterating over the draw tree of {} elements took: {:?}",
            self.draw_tree.len(),
            first_timer.elapsed()
        );

        // TODO: this is for debugging purposes
        // let query_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
        //     label: Some("Query Buffer"),
        //     size: 16,
        //     usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        //     mapped_at_creation: false,
        // });
        //
        // let read_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
        //     label: Some("Read Buffer"),
        //     size: 16,
        //     usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        //     mapped_at_creation: false,
        // });
        println!("Creating buffers took: {:?}", first_timer.elapsed());

        // TODO: this is for debugging purposes
        // let encoder_timer = Instant::now();
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("And Command Encoder"),
            });
        // #[cfg(feature = "performance_measurement")]
        // let timer = Instant::now();
        // #[cfg(feature = "performance_measurement")]
        // encoder.write_timestamp(&self.performance_query_set, 0);
        println!("Creating encoder took: {:?}", first_timer.elapsed());

        let output = self.surface.get_current_texture()?;
        let output_texture_view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        println!("Getting output texture took: {:?}", first_timer.elapsed());

        let depth_texture =
            create_and_depth_texture(&self.device, (self.physical_size.0, self.physical_size.1));
        // let iter_timer = Instant::now();

        {
            // TODO: this is for debugging purposes
            // let pass_timer = Instant::now();
            let depth_texture_view =
                depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

            let render_pass =
                create_render_pass(&mut encoder, &output_texture_view, &depth_texture_view);

            println!("Creating pass took: {:?}", first_timer.elapsed());

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
                            if let (Some(vertex_buffer), Some(index_buffer)) =
                                (shape.vertex_buffer.as_ref(), shape.index_buffer.as_ref())
                            {
                                if !matches!(currently_set_pipeline, Pipeline::StencilIncrement) {
                                    render_pass.set_pipeline(&self.and_pipeline);
                                    render_pass.set_bind_group(0, &self.and_bind_group, &[]);
                                    *currently_set_pipeline = Pipeline::StencilIncrement;
                                }

                                if let Some(clip_to_shape) = shape.clip_to_shape {
                                    let parent_stencil =
                                        *stencil_references.get(&clip_to_shape).unwrap_or(&0);

                                    render_buffer_to_texture2(
                                        vertex_buffer,
                                        index_buffer,
                                        shape.num_indices.unwrap(),
                                        render_pass,
                                        parent_stencil,
                                    );

                                    stencil_references.insert(shape_id, parent_stencil + 1);
                                } else {
                                    // TODO: every no clip should have its own tree, and before
                                    //  rendering them we need to reset stencil texture, and
                                    //  they should be rendered in a separate step

                                    render_buffer_to_texture2(
                                        vertex_buffer,
                                        index_buffer,
                                        shape.num_indices.unwrap(),
                                        render_pass,
                                        0,
                                    );

                                    stencil_references.insert(shape_id, 1);
                                }
                            } else {
                                println!("Missing vertex or index buffer for shape {}", shape_id);
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
                            if let (Some(vertex_buffer), Some(index_buffer)) =
                                (shape.vertex_buffer.as_ref(), shape.index_buffer.as_ref())
                            {
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

                                render_buffer_to_texture2(
                                    vertex_buffer,
                                    index_buffer,
                                    shape.num_indices.unwrap(),
                                    render_pass,
                                    this_shape_stencil,
                                );
                            } else {
                                println!("No vertex or index buffer found for shape {}", shape_id);
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
            println!("Walking the tree took: {:?}", first_timer.elapsed());

            // println!("{}", self.text_instances.len());
            // TODO: cache the text rendering
            let text_instances = std::mem::take(&mut self.text_instances);
            let text_areas = text_instances
                .iter()
                .map(|text_instance| text_instance.to_text_area(self.scale_factor as f32));

            self.text_renderer_wrapper
                .text_renderer
                .prepare_with_depth(
                    &self.device,
                    &self.queue,
                    &mut self.text_renderer_wrapper.font_system,
                    &mut self.text_renderer_wrapper.atlas,
                    &self.glyphon_viewport,
                    text_areas,
                    &mut self.text_renderer_wrapper.swash_cache,
                    |index| depth(index, draw_tree_size),
                )
                .unwrap();

            self.text_renderer_wrapper
                .text_renderer
                .render(
                    &self.text_renderer_wrapper.atlas,
                    &self.glyphon_viewport,
                    &mut data.0,
                )
                .unwrap();
        }

        // /// TODO: this is for debugging purposes
        // self.depth_stencil_texture_viewer.copy_depth_stencil_texture_to_buffer(&mut encoder, &depth_texture);

        // #[cfg(feature = "performance_measurement")]
        // {
        //     encoder.write_timestamp(&self.performance_query_set, 1);
        //
        //     // encoder.resolve_query_set(&self.performance_query_set, 0..2, &query_buffer, 0);
        //     // // Copy the query buffer data to the read buffer
        //     // encoder.copy_buffer_to_buffer(&query_buffer, 0, &read_buffer, 0, 16);
        // };

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        println!("Queue submit took: {:?}", first_timer.elapsed());

        #[cfg(feature = "performance_measurement")]
        {
            self.device.poll(wgpu::Maintain::Poll);

            // let buffer_slice = read_buffer.slice(..);
            // buffer_slice.map_async(wgpu::MapMode::Read, |result| {
            //     assert!(result.is_ok());
            // });
            //
            // self.device.poll(wgpu::Maintain::Wait);
            //
            // let data = buffer_slice.get_mapped_range();
            // let timestamps: &[u64] = bytemuck::cast_slice(&data);
            // let timestamp_start = timestamps[0];
            // let timestamp_end = timestamps[1];
            // drop(data);
            // read_buffer.unmap();

            // let timestamp_period = self.queue.get_timestamp_period(); // Duration of a single timestamp in nanoseconds
            //                                                           // println!("Timestamp end: {:?}", timestamp_end);
            //                                                           // println!("Timestamp start: {:?}", timestamp_start);
            //                                                           // if (timestamp_end != 0) {
            //                                                           //     let frame_time_ns = (timestamp_end - timestamp_start) as f32 * timestamp_period;
            //                                                           //     println!("Frame time: {} ns", frame_time_ns);
            //                                                           // }
            // if timestamp_start > timestamp_end {
            //     // let frame_time_ns = (timestamp_start - timestamp_end) as f32 * timestamp_period;
            //     // println!("Frame time: {} ns", frame_time_ns);
            // } else {
            //     // let frame_time_ns = (timestamp_end - timestamp_start) as f32 * timestamp_period;
            //     // println!("Frame time: {} ns", frame_time_ns);
            // }
        }

        // self.depth_stencil_texture_viewer.save_depth_stencil_texture(&self.device);
        // self.depth_stencil_texture_viewer.print_texture_data_at_pixel(112, 40);

        println!("Total time spent {:?}", first_timer.elapsed());
        // println!("Render method took: {:?}", timer.elapsed());

        Ok(())
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
    /// use winit::event_loop::EventLoop;
    /// use winit::window::WindowBuilder;
    /// use grafo::{MathRect, Renderer, TextAlignment, TextLayout};
    /// use grafo::Shape;
    /// use grafo::Color;
    /// use grafo::Stroke;
    ///
    /// // This is for demonstration purposes only. If you want a working example with winit, please
    /// // refer to the example in the "examples" folder.
    /// let event_loop = EventLoop::new().expect("To create the event loop");
    /// let window_surface = Arc::new(WindowBuilder::new().build(&event_loop).unwrap());
    /// let physical_size = (800, 600);
    /// let scale_factor = 1.0;
    ///
    /// // Initialize the renderer
    /// let mut renderer = block_on(Renderer::new(window_surface, physical_size, scale_factor));
    ///
    /// // Add shapes, images, and text...
    ///
    /// // Clear the draw queue
    /// renderer.clear_draw_queue();
    /// ```
    pub fn clear_draw_queue(&mut self) {
        self.draw_tree.clear();
        self.text_instances.clear();
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
    /// use winit::event_loop::EventLoop;
    /// use winit::window::WindowBuilder;
    /// use grafo::{MathRect, Renderer, TextAlignment, TextLayout};
    /// use grafo::Shape;
    /// use grafo::Color;
    /// use grafo::Stroke;
    ///
    /// // This is for demonstration purposes only. If you want a working example with winit, please
    /// // refer to the example in the "examples" folder.
    /// let event_loop = EventLoop::new().expect("To create the event loop");
    /// let window_surface = Arc::new(WindowBuilder::new().build(&event_loop).unwrap());
    /// let physical_size = (800, 600);
    /// let scale_factor = 1.0;
    ///
    /// // Initialize the renderer
    /// let mut renderer = block_on(Renderer::new(window_surface, physical_size, scale_factor));
    ///
    /// let size = renderer.size();
    /// println!("Rendering surface size: {}x{}", size.0, size.1);
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
    /// use winit::event_loop::EventLoop;
    /// use winit::window::WindowBuilder;
    /// use grafo::{MathRect, Renderer, TextAlignment, TextLayout};
    /// use grafo::Shape;
    /// use grafo::Color;
    /// use grafo::Stroke;
    ///
    /// // This is for demonstration purposes only. If you want a working example with winit, please
    /// // refer to the example in the "examples" folder.
    /// let event_loop = EventLoop::new().expect("To create the event loop");
    /// let window_surface = Arc::new(WindowBuilder::new().build(&event_loop).unwrap());
    /// let physical_size = (800, 600);
    /// let scale_factor = 1.0;
    ///
    /// // Initialize the renderer
    /// let mut renderer = block_on(Renderer::new(window_surface, physical_size, scale_factor));
    ///
    /// // Change the scale factor to 2.0 for high-DPI rendering
    /// renderer.change_scale_factor(2.0);
    /// ```
    pub fn change_scale_factor(&mut self, new_scale_factor: f64) {
        self.scale_factor = new_scale_factor;
        self.resize(self.physical_size)
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
    /// use winit::event_loop::EventLoop;
    /// use winit::window::WindowBuilder;
    /// use grafo::{MathRect, Renderer, TextAlignment, TextLayout};
    /// use grafo::Shape;
    /// use grafo::Color;
    /// use grafo::Stroke;
    ///
    /// // This is for demonstration purposes only. If you want a working example with winit, please
    /// // refer to the example in the "examples" folder.
    /// let event_loop = EventLoop::new().expect("To create the event loop");
    /// let window_surface = Arc::new(WindowBuilder::new().build(&event_loop).unwrap());
    /// let physical_size = (800, 600);
    /// let scale_factor = 1.0;
    ///
    /// // Initialize the renderer
    /// let mut renderer = block_on(Renderer::new(window_surface, physical_size, scale_factor));
    ///
    /// // Resize the renderer to 1024x768 pixels
    /// renderer.resize((1024, 768));
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
        self.texture_bind_group_layout = texture_bind_group_layout;
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
    /// use winit::event_loop::EventLoop;
    /// use winit::window::WindowBuilder;
    /// use grafo::{MathRect, Renderer, TextAlignment, TextLayout};
    /// use grafo::Shape;
    /// use grafo::Color;
    /// use grafo::Stroke;
    /// use grafo::fontdb;
    ///
    /// // This is for demonstration purposes only. If you want a working example with winit, please
    /// // refer to the example in the "examples" folder.
    /// let event_loop = EventLoop::new().expect("To create the event loop");
    /// let window_surface = Arc::new(WindowBuilder::new().build(&event_loop).unwrap());
    /// let physical_size = (800, 600);
    /// let scale_factor = 1.0;
    ///
    /// // Initialize the renderer
    /// let mut renderer = block_on(Renderer::new(window_surface, physical_size, scale_factor));
    ///
    /// let roboto_font_ttf = include_bytes!("../examples/assets/Roboto-Regular.ttf").to_vec();
    /// let roboto_font_source = fontdb::Source::Binary(Arc::new(roboto_font_ttf));
    /// renderer.load_fonts([roboto_font_source].into_iter());
    /// ```
    pub fn load_fonts(&mut self, fonts: impl Iterator<Item = fontdb::Source>) {
        let db = self.text_renderer_wrapper.font_system.db_mut();

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
    /// use winit::event_loop::EventLoop;
    /// use winit::window::WindowBuilder;
    /// use grafo::{MathRect, Renderer, TextAlignment, TextLayout};
    /// use grafo::Shape;
    /// use grafo::Color;
    /// use grafo::Stroke;
    /// use grafo::fontdb;
    ///
    /// // This is for demonstration purposes only. If you want a working example with winit, please
    /// // refer to the example in the "examples" folder.
    /// let event_loop = EventLoop::new().expect("To create the event loop");
    /// let window_surface = Arc::new(WindowBuilder::new().build(&event_loop).unwrap());
    /// let physical_size = (800, 600);
    /// let scale_factor = 1.0;
    ///
    /// // Initialize the renderer
    /// let mut renderer = block_on(Renderer::new(window_surface, physical_size, scale_factor));
    ///
    /// let roboto_font_ttf = include_bytes!("../examples/assets/Roboto-Regular.ttf");
    /// renderer.load_font_from_bytes(roboto_font_ttf);
    /// ```
    pub fn load_font_from_bytes(&mut self, font_bytes: &[u8]) {
        let db = self.text_renderer_wrapper.font_system.db_mut();
        let source = fontdb::Source::Binary(Arc::new(font_bytes.to_vec()));
        db.load_font_source(source);
    }
}
