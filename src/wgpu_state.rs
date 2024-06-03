use lyon::lyon_tessellation::{
    BuffersBuilder, FillOptions, FillTessellator, FillVertex, VertexBuffers,
};
use std::sync::Arc;
use std::time::Instant;
use wgpu::util::DeviceExt;

use easy_tree::rayon::iter::ParallelIterator;

type Rect = lyon::math::Box2D;

use ahash::{HashMap, HashMapExt};
use crate::pipeline::{
    create_and_depth_texture, create_and_pass,
    create_depth_stencil_state_for_text, create_pipeline, create_uniform_bind_group_layout,
    render_buffer_to_texture2, PipelineType, Uniforms,
};
use crate::util::{normalize_rgba_color, to_logical};
use crate::vertex::CustomVertex;
use glyphon::cosmic_text::Align;
use glyphon::{
    Attrs, Buffer as TextBuffer, Color as TextColor, Family, FontSystem, Metrics, Resolution, Shaping,
    SwashCache, TextArea, TextAtlas, TextBounds, TextRenderer,
};

use wgpu::{BindGroup, CompositeAlphaMode, Device, InstanceDescriptor, MultisampleState, RenderPipeline, SurfaceTarget};
use crate::debug_tools::DepthStencilTextureViewer;
use crate::stroke::Stroke;

#[inline(always)]
pub fn depth(draw_command_id: usize, draw_commands_total: usize) -> f32 {
    (1.0 - (draw_command_id as f32 / draw_commands_total as f32)).clamp(0.0000000001, 0.9999999999)
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum TextAlignment {
    Start,
    End,
    Center,
}

pub fn create_render_pipeline(
    window_logical_size: (f32, f32),
    device: &Device,
    config: &wgpu::SurfaceConfiguration,
    depth_stencil_state: Option<wgpu::DepthStencilState>,
) -> (Uniforms, BindGroup, RenderPipeline) {
    // TODO: Experiment with the uniforms
    let uniforms = Uniforms::new(window_logical_size.0, window_logical_size.1);

    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Uniform Buffer"),
        contents: bytemuck::cast_slice(&[uniforms]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // Bind group for uniforms
    let bind_group_layout = create_uniform_bind_group_layout(device);

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform_buffer.as_entire_binding(),
        }],
        label: Some("uniform_bind_group"),
    });

    // Create the render pipeline
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("./shader.wgsl").into()),
    });

    let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Render Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Render Pipeline"),
        layout: Some(&render_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[CustomVertex::desc()],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_main",
            targets: &[Some(wgpu::ColorTargetState {
                format: config.format,
                blend: Some(wgpu::BlendState {
                    color: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::SrcAlpha,
                        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        operation: wgpu::BlendOperation::Add,
                    },
                    alpha: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        operation: wgpu::BlendOperation::Add,
                    },
                }),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: depth_stencil_state,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    });

    (uniforms, bind_group, render_pipeline)
}

fn create_stencil_state(stencil_reference: u32) -> wgpu::StencilState {
    wgpu::StencilState {
        // For front facing polygons
        front: wgpu::StencilFaceState {
            // Compare function: compares what to what?
            // compare: wgpu::CompareFunction::Equal, // Only draw where the stencil value matches the reference
            compare: wgpu::CompareFunction::Always, // Only draw where the stencil value matches the reference
            // Stencil test fail
            fail_op: wgpu::StencilOperation::Keep,
            // Depth test fail, but stencil test pass
            depth_fail_op: wgpu::StencilOperation::Keep,
            // Depth test pass and stencil test pass.
            // Replace stencil value with value provided in most recent call to set_stencil_reference.
            pass_op: wgpu::StencilOperation::Keep,
        },
        back: wgpu::StencilFaceState {
            compare: wgpu::CompareFunction::Equal,
            fail_op: wgpu::StencilOperation::Keep,
            depth_fail_op: wgpu::StencilOperation::Keep,
            pass_op: wgpu::StencilOperation::Replace,
        },
        read_mask: 0xff,
        write_mask: 0xff,
    }
}

pub fn create_depth_stencil_state() -> wgpu::DepthStencilState {
    wgpu::DepthStencilState {
        format: wgpu::TextureFormat::Depth24PlusStencil8,
        depth_write_enabled: true,
        depth_compare: wgpu::CompareFunction::Always,
        stencil: create_stencil_state(0),
        bias: wgpu::DepthBiasState::default(),
    }
}

pub struct TextRendererWrapper {
    text_renderer: TextRenderer,
    atlas: TextAtlas,
    font_system: FontSystem,
    swash_cache: SwashCache,
}

impl TextRendererWrapper {
    pub fn new(
        device: &Device,
        queue: &wgpu::Queue,
        swapchain_format: wgpu::TextureFormat,
        depth_stencil_state: Option<wgpu::DepthStencilState>,
    ) -> Self {
        let mut font_system = FontSystem::new();
        let mut swash_cache = SwashCache::new();
        let mut atlas = TextAtlas::new(device, queue, swapchain_format);
        let text_renderer = TextRenderer::new(
            &mut atlas,
            device,
            MultisampleState::default(),
            depth_stencil_state,
        );

        Self {
            text_renderer,
            atlas,
            font_system,
            swash_cache,
        }
    }
}

#[derive(Debug)]
struct TextDrawData {
    text_buffer: TextBuffer,
    area: Rect,
    vertical_alignment: TextAlignment,
    data: String,
    font_size: f32,
    top: f32,
    clip_to_shape: Option<usize>,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum Visibility {
    Visible,
    Hidden,
}

#[derive(Clone, Debug)]
pub struct Shape {
    pub path: lyon::path::Path,
    pub fill: Color,
    pub stroke: Stroke,
}

impl Shape {
    pub fn new(path: lyon::path::Path, fill: Color, stroke: Stroke) -> Self {
        Self { path, fill, stroke }
    }
}

#[derive(Debug)]
struct ShapeDrawData {
    vertex_buffer: Option<wgpu::Buffer>,
    index_buffer: Option<wgpu::Buffer>,
    num_indices: Option<u32>,
    clip_to_shape: Option<usize>,
    shape: Shape,
}

fn create_rect(canvas_size: (u32, u32), device: &wgpu::Device) -> wgpu::Buffer {
    // Two triangles covering whole screen, 6 vertices total
    let quad = [
        CustomVertex {
            position: [0.0, 0.0],
            color: [0.0, 0.0, 0.0, 0.0],
            depth: 0.0,
        },
        CustomVertex {
            position: [canvas_size.0 as f32, 0.0],
            color: [0.0, 0.0, 0.0, 0.0],
            depth: 0.0,
        },
        CustomVertex {
            position: [0.0, canvas_size.1 as f32],
            color: [0.0, 0.0, 0.0, 0.0],
            depth: 0.0,
        },
        CustomVertex {
            position: [0.0, canvas_size.1 as f32],
            color: [0.0, 0.0, 0.0, 0.0],
            depth: 0.0,
        },
        CustomVertex {
            position: [canvas_size.0 as f32, 0.0],
            color: [0.0, 0.0, 0.0, 0.0],
            depth: 0.0,
        },
        CustomVertex {
            position: [canvas_size.0 as f32, canvas_size.1 as f32],
            color: [0.0, 0.0, 0.0, 0.0],
            depth: 0.0,
        },
    ];

    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Rect Buffer"),
        contents: bytemuck::cast_slice(&quad),
        usage: wgpu::BufferUsages::VERTEX,
    });

    vertex_buffer
}

impl ShapeDrawData {
    pub fn prepare_buffers_not_set(
        &self,
        device: &wgpu::Device,
        depth: f32,
    ) -> (wgpu::Buffer, wgpu::Buffer, u32) {
        let mut vertex_buffers = State::tessellate_shape(&self.shape, depth);

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertex_buffers.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(&vertex_buffers.indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        (
            vertex_buffer,
            index_buffer,
            vertex_buffers.indices.len() as u32,
        )
    }
    pub fn prepare_buffers(&mut self, device: &wgpu::Device, shape_id: usize, max_shape_id: usize) {
        let depth = depth(shape_id, max_shape_id);
        let (
            vertex_buffer,
            index_buffer,
            num_indices
        ) = self.prepare_buffers_not_set(device, depth);

        self.vertex_buffer = Some(vertex_buffer);
        self.index_buffer = Some(index_buffer);
        self.num_indices = Some(num_indices);
    }
}

enum DrawCommand {
    Shape(ShapeDrawData),
    Text(TextDrawData),
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Color(pub [u8; 4]);

impl Color {
    pub const TRANSPARENT: Self = Self([0, 0, 0, 0]);

    pub fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self([r, g, b, 255])
    }

    pub fn rgba(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self([r, g, b, a])
    }

    pub fn normalize(&self) -> [f32; 4] {
        normalize_rgba_color(&self.0)
    }

    pub fn to_array(&self) -> [u8; 4] {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TextLayout {
    pub font_size: f32,
    pub line_height: f32,
    pub color: Color,
    pub area: Rect,
    pub horizontal_alignment: TextAlignment,
    pub vertical_alignment: TextAlignment,
}

pub(crate) struct State<'a> {
    // Window information
    /// Size of the window
    pub(crate) physical_size: (u32, u32),
    /// Scale factor of the window
    scale_factor: f64,

    // WGPU components
    surface: wgpu::Surface<'a>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    /// Uniforms for the render pipeline. They are used to normalize the coordinates of the shapes
    ///  inside the shaders
    uniforms: Uniforms,
    /// Bind group for the uniforms
    bind_group: BindGroup,
    render_pipeline: Arc<wgpu::RenderPipeline>,

    depth_stencil_state: wgpu::DepthStencilState,

    /// Text instances to be rendered
    text_instances: Vec<TextDrawData>,

    text_renderer_wrapper: TextRendererWrapper,
    flat_draw_queue: Vec<DrawCommand>,

    /// Shapes to be rendered
    draw_tree: easy_tree::Tree<DrawCommand>,

    and_uniforms: Uniforms,
    and_bind_group: BindGroup,
    and_pipeline: Arc<wgpu::RenderPipeline>,

    always_uniforms: Uniforms,
    always_bind_group: BindGroup,
    always_pipeline: Arc<wgpu::RenderPipeline>,

    always_decrement_uniforms: Uniforms,
    always_decrement_bind_group: BindGroup,
    always_decrement_pipeline: Arc<wgpu::RenderPipeline>,

    canvas_sized_quad: wgpu::Buffer,

    decrementing_pipeline: Arc<wgpu::RenderPipeline>,
    decrementing_uniforms: Uniforms,
    decrementing_bind_group: BindGroup,

    #[cfg(feature = "performance_measurement")]
    performance_query_set: wgpu::QuerySet,
    #[cfg(feature = "performance_measurement")]
    adapter: wgpu::Adapter,

    // TODO: remove later
    depth_stencil_texture_viewer: DepthStencilTextureViewer,
}

impl State<'_> {
    pub(crate) async fn new(
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
                    required_features: wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::DEPTH32FLOAT_STENCIL8,
                    #[cfg(not(feature = "performance_measurement"))]
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .unwrap();

        #[cfg(feature = "performance_measurement")]
        let frametime_query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: wgpu::Label::Some("Frametime Query Set"),
            count: 2, // We will use two timestamps
            ty: wgpu::QueryType::Timestamp,
        });

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

        let depth_stencil_state = create_depth_stencil_state();

        let (uniforms, bind_group, render_pipeline) = create_render_pipeline(
            canvas_logical_size,
            &device,
            &config,
            Some(depth_stencil_state.clone()),
        );

        let text_instances = Vec::new();

        let (and_uniforms, and_bind_group, and_pipeline) = create_pipeline(
            canvas_logical_size,
            &device,
            &config,
            PipelineType::EqualIncrementStencil,
        );

        let (always_uniforms, always_bind_group, always_pipeline) = create_pipeline(
            canvas_logical_size,
            &device,
            &config,
            PipelineType::AlwaysReplaceStencil,
        );

        let (always_decrement_uniforms, always_decrement_bind_group, always_decrement_pipeline) =
            create_pipeline(
                canvas_logical_size,
                &device,
                &config,
                PipelineType::AlwaysDecrementStencil,
            );

        let canvas_sized_quad = create_rect(size, &device);

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

        let depth_stencil_texture_viewer = DepthStencilTextureViewer::new(&device, size);

        Self {
            surface,
            device,
            queue,
            config,
            physical_size: size,
            render_pipeline: Arc::new(render_pipeline),
            // font_system,
            // swash_cache,
            // atlas,
            // text_renderer,
            uniforms,
            bind_group,
            scale_factor,
            text_instances,
            // draw_queue: BTreeMap::new(),
            text_renderer_wrapper,
            depth_stencil_state,
            flat_draw_queue: Vec::new(),

            and_pipeline: Arc::new(and_pipeline),
            and_uniforms,
            and_bind_group,

            always_pipeline: Arc::new(always_pipeline),
            always_uniforms,
            always_bind_group,

            always_decrement_pipeline: Arc::new(always_decrement_pipeline),
            always_decrement_uniforms,
            always_decrement_bind_group,

            canvas_sized_quad,

            decrementing_pipeline: Arc::new(decrementing_pipeline),
            decrementing_uniforms,
            decrementing_bind_group,

            draw_tree: easy_tree::Tree::new(),

            #[cfg(feature = "performance_measurement")]
            performance_query_set: frametime_query_set,
            #[cfg(feature = "performance_measurement")]
            adapter,

            depth_stencil_texture_viewer
        }
    }

    pub(crate) fn resize(&mut self, new_physical_size: (u32, u32)) {
        self.physical_size = new_physical_size;
        self.config.width = new_physical_size.0;
        self.config.height = new_physical_size.1;
        self.surface.configure(&self.device, &self.config);
        // Update the render pipeline to match the new size with new uniforms. Uniforms are
        // needed to normalize the coordinates of the shapes
        let (uniforms, bind_group, render_pipeline) = create_render_pipeline(
            to_logical(new_physical_size, self.scale_factor),
            &self.device,
            &self.config,
            Some(self.depth_stencil_state.clone()),
        );
        self.uniforms = uniforms;
        self.bind_group = bind_group;
        self.render_pipeline = Arc::new(render_pipeline);

        // Update the and pipeline
        let (and_uniforms, and_bind_group, and_pipeline) = create_pipeline(
            to_logical(new_physical_size, self.scale_factor),
            &self.device,
            &self.config,
            PipelineType::EqualIncrementStencil,
        );
        self.and_uniforms = and_uniforms;
        self.and_bind_group = and_bind_group;
        self.and_pipeline = Arc::new(and_pipeline);

        // Update the always pipeline
        let (always_uniforms, always_bind_group, always_pipeline) = create_pipeline(
            to_logical(new_physical_size, self.scale_factor),
            &self.device,
            &self.config,
            PipelineType::AlwaysReplaceStencil,
        );
        self.always_uniforms = always_uniforms;
        self.always_bind_group = always_bind_group;
        self.always_pipeline = Arc::new(always_pipeline);

        // Update the always decrement pipeline
        let (always_decrement_uniforms, always_decrement_bind_group, always_decrement_pipeline) =
            create_pipeline(
                to_logical(new_physical_size, self.scale_factor),
                &self.device,
                &self.config,
                PipelineType::AlwaysDecrementStencil,
            );
        self.always_decrement_uniforms = always_decrement_uniforms;
        self.always_decrement_bind_group = always_decrement_bind_group;
        self.always_decrement_pipeline = Arc::new(always_decrement_pipeline);

        // Update the always decrement pipeline
        let (always_decrement_uniforms, always_decrement_bind_group, always_decrement_pipeline) =
            create_pipeline(
                to_logical(new_physical_size, self.scale_factor),
                &self.device,
                &self.config,
                PipelineType::EqualDecrementStencil,
            );
        self.decrementing_uniforms = always_decrement_uniforms;
        self.decrementing_bind_group = always_decrement_bind_group;
        self.decrementing_pipeline = Arc::new(always_decrement_pipeline);
    }

    pub(crate) fn update(&mut self) {}

    pub(crate) fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        println!("===============");
        let first_timer = Instant::now();
        let draw_tree_size = self.draw_tree.len();
        let iter = self.draw_tree.par_iter_mut();

        iter.for_each(|draw_command| {
            match draw_command.1 {
                DrawCommand::Shape(ref mut shape) => {
                    shape.prepare_buffers(&self.device, draw_command.0, draw_tree_size);
                }
                DrawCommand::Text(text_instance) => {
                    // text_instance.text_buffer.shape_until_scroll(&mut self.text_renderer_wrapper.font_system);
                }
            }
        });

        println!(
            "Iterating over the draw tree of {} elements took: {:?}",
            self.draw_tree.len(),
            first_timer.elapsed()
        );

        let query_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Query Buffer"),
            size: 16,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let read_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Read Buffer"),
            size: 16,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        println!("Creating buffers took: {:?}", first_timer.elapsed());

        let encoder_timer = Instant::now();
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("And Command Encoder"),
            });
        #[cfg(feature = "performance_measurement")]
        let timer = Instant::now();
        #[cfg(feature = "performance_measurement")]
        encoder.write_timestamp(&self.performance_query_set, 0);
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
            let pass_timer = Instant::now();
            let depth_texture_view =
                depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

            let mut incrementing_pass =
                create_and_pass(&mut encoder, &output_texture_view, &depth_texture_view);

            println!("Creating pass took: {:?}", first_timer.elapsed());

            let mut shape_ids_to_stencil_references = HashMap::<usize, u32>::new();

            let mut data = (incrementing_pass, shape_ids_to_stencil_references);

            let walk_timer = Instant::now();
            self.draw_tree.traverse(
                |shape_id, draw_command, data| {
                    match draw_command {
                        DrawCommand::Shape(shape) => {
                            if let (Some(vertex_buffer), Some(index_buffer)) =
                                (shape.vertex_buffer.as_ref(), shape.index_buffer.as_ref())
                            {
                                if let Some(clip_to_shape) = shape.clip_to_shape {
                                    data.0.set_pipeline(&self.and_pipeline);
                                    data.0.set_bind_group(0, &self.and_bind_group, &[]);

                                    let parent_stencil = *data.1.get(&clip_to_shape).unwrap_or(&0);

                                    render_buffer_to_texture2(
                                        vertex_buffer,
                                        index_buffer,
                                        shape.num_indices.unwrap(),
                                        &mut data.0,
                                        parent_stencil,
                                    );

                                    data.1.insert(shape_id, parent_stencil + 1);
                                } else {
                                    // TODO: every no clip should have its own tree, and before
                                    //  rendering them we need to reset stencil texture, and
                                    //  they should be rendered in a separate step
                                    data.0.set_pipeline(&self.and_pipeline);
                                    data.0.set_bind_group(0, &self.and_bind_group, &[]);

                                    render_buffer_to_texture2(
                                        vertex_buffer,
                                        index_buffer,
                                        shape.num_indices.unwrap(),
                                        &mut data.0,
                                        0,
                                    );

                                    data.1.insert(shape_id, 1);
                                }
                            } else {
                                // println!("Missing vertex or index buffer for shape with stencil reference {}", stencil_reference);
                            }
                        }
                        DrawCommand::Text(text_instance) => {
                            // {
                            //     text_instance.prepare(
                            //         &self.device,
                            //         &self.queue,
                            //         (self.size.width, self.size.height),
                            //         self.scale_factor as f32,
                            //         &mut self.text_renderer_wrapper,
                            //     );
                            // }
                            //
                            // {
                            //     self.text_renderer_wrapper.text_renderer.render(&self.text_renderer_wrapper.atlas, &mut data.0).unwrap()
                            // }
                        }
                    }
                },
                |shape_id, draw_command, data| {
                    match draw_command {
                        DrawCommand::Shape(shape) => {
                            if let (Some(vertex_buffer), Some(index_buffer)) =
                                (shape.vertex_buffer.as_ref(), shape.index_buffer.as_ref())
                            {
                                data.0.set_pipeline(&self.decrementing_pipeline);
                                data.0.set_bind_group(0, &self.decrementing_bind_group, &[]);

                                let this_shape_stencil = { *data.1.get(&shape_id).unwrap_or(&0) };

                                let result = render_buffer_to_texture2(
                                    vertex_buffer,
                                    index_buffer,
                                    shape.num_indices.unwrap(),
                                    &mut data.0,
                                    this_shape_stencil,
                                );
                            } else {
                                // TODO: no vertex or index buffer found - it is an error,
                                //  so probably should panic
                            }
                        }
                        DrawCommand::Text(text) => {
                            // TODO: add support
                        }
                    }
                },
                &mut data,
            );
            println!("Walking the tree took: {:?}", first_timer.elapsed());

            // println!("{}", self.text_instances.len());
            // TODO: cache the text rendering
            let text_instances = std::mem::take(&mut self.text_instances);
            let text_areas = text_instances.iter().map(|text_instance| {
                let area = text_instance.area;
                let top = text_instance.top;

                TextArea {
                    buffer: &text_instance.text_buffer,
                    left: area.min.x * self.scale_factor as f32,
                    top: top * self.scale_factor as f32,
                    scale: self.scale_factor as f32,
                    bounds: TextBounds {
                        left: area.min.x as i32 * self.scale_factor as i32,
                        top: area.min.y as i32 * self.scale_factor as i32,
                        right: area.max.x as i32 * self.scale_factor as i32,
                        bottom: area.max.y as i32 * self.scale_factor as i32,
                    },
                    default_color: TextColor::rgb(255, 255, 255),
                }
            });

            self.text_renderer_wrapper
                .text_renderer
                .prepare_with_depth(
                    &self.device,
                    &self.queue,
                    &mut self.text_renderer_wrapper.font_system,
                    &mut self.text_renderer_wrapper.atlas,
                    Resolution {
                        width: self.physical_size.0,
                        height: self.physical_size.1,
                    },
                    text_areas,
                    &mut self.text_renderer_wrapper.swash_cache,
                    |index| {
                        // println!("Index: {}", index);
                        let text_depth = depth(index, draw_tree_size);
                        // println!("Text depth: {}", text_depth);
                        text_depth
                        // let index = index as f32 / draw_commands_len as f32;
                        // (1.0 - index).clamp(0.0000001, 0.9999999)
                    },
                )
                .unwrap();

            self.text_renderer_wrapper
                .text_renderer
                .render(&self.text_renderer_wrapper.atlas, &mut data.0)
                .unwrap();
        }

        // /// TODO: this is for debugging purposes
        // self.depth_stencil_texture_viewer.copy_depth_stencil_texture_to_buffer(&mut encoder, &depth_texture);

        #[cfg(feature = "performance_measurement")]
        {
            encoder.write_timestamp(&self.performance_query_set, 1);

            encoder.resolve_query_set(&self.performance_query_set, 0..2, &query_buffer, 0);
            // Copy the query buffer data to the read buffer
            encoder.copy_buffer_to_buffer(&query_buffer, 0, &read_buffer, 0, 16);
        };

        let queue_timer = Instant::now();
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        println!("Queue submit took: {:?}", first_timer.elapsed());

        #[cfg(feature = "performance_measurement")]
        {
            self.device.poll(wgpu::Maintain::Poll);

            let buffer_slice = read_buffer.slice(..);
            buffer_slice.map_async(wgpu::MapMode::Read, |result| {
                assert!(result.is_ok());
            });

            self.device.poll(wgpu::Maintain::Wait);

            let data = buffer_slice.get_mapped_range();
            let timestamps: &[u64] = bytemuck::cast_slice(&data);
            let timestamp_start = timestamps[0];
            let timestamp_end = timestamps[1];
            drop(data);
            read_buffer.unmap();

            let timestamp_period = self.queue.get_timestamp_period(); // Duration of a single timestamp in nanoseconds
                                                                      // println!("Timestamp end: {:?}", timestamp_end);
                                                                      // println!("Timestamp start: {:?}", timestamp_start);
                                                                      // if (timestamp_end != 0) {
                                                                      //     let frame_time_ns = (timestamp_end - timestamp_start) as f32 * timestamp_period;
                                                                      //     println!("Frame time: {} ns", frame_time_ns);
                                                                      // }
            if timestamp_start > timestamp_end {
                // let frame_time_ns = (timestamp_start - timestamp_end) as f32 * timestamp_period;
                // println!("Frame time: {} ns", frame_time_ns);
            } else {
                // let frame_time_ns = (timestamp_end - timestamp_start) as f32 * timestamp_period;
                // println!("Frame time: {} ns", frame_time_ns);
            }
        }

        // self.depth_stencil_texture_viewer.save_depth_stencil_texture(&self.device);
        // self.depth_stencil_texture_viewer.print_texture_data_at_pixel(112, 40);

        println!("Total time spent {:?}", first_timer.elapsed());
        // println!("Render method took: {:?}", timer.elapsed());

        Ok(())
    }

    pub fn commands_len(&self) -> usize {
        self.flat_draw_queue.len() + self.text_instances.len()
    }
}

/// Implementation of the methods related to shape rendering
impl State<'_> {
    pub fn clear_draw_queue(&mut self) {
        // self.draw_queue.clear();
        self.flat_draw_queue.clear();
        self.draw_tree.clear();
        self.text_instances.clear();
    }

    pub fn tessellate_shape(shape: &Shape, depth: f32) -> VertexBuffers<CustomVertex, u16> {
        // Define a shape (triangle for now)
        let mut buffers: VertexBuffers<CustomVertex, u16> = VertexBuffers::new();
        let mut tessellator = FillTessellator::new();
        let options = FillOptions::default().with_tolerance(0.01);

        let fill = shape.fill;

        // TODO: make the library to accept colors as &[u8; 4]
        // Normalizing the color
        let color = fill.normalize();

        tessellator
            .tessellate_path(
                &shape.path,
                &options,
                &mut BuffersBuilder::new(&mut buffers, |vertex: FillVertex| CustomVertex {
                    position: vertex.position().to_array(),
                    depth,
                    color,
                }),
            )
            .unwrap();

        buffers
    }

    fn convert_shape_to_vertex_and_add_schedule_for_rendering(
        &mut self,
        shape: Shape,
        clip_to_shape: Option<usize>,
    ) -> usize {
        // let color = normalize_rgba_color(&shape.fill.to_array());
        //
        // if let Some(clip_to_shape) = clip_to_shape {
        //     // TODO: we actually don't need to do that if the shape is fully visible, or not visible at all.
        //     //  Meaning, it's fully withing the clipping shape or fully outside of it.
        //     let clipper_visibility = self
        //         .flat_draw_queue
        //         .get(clip_to_shape)
        //         .map(|draw_command| {
        //             if let DrawCommand::Shape(clipper_shape) = draw_command {
        //                 clipper_shape.visibility
        //             } else {
        //                 Visibility::Visible
        //             }
        //         })
        //         .unwrap_or(Visibility::Visible);
        //
        //     // Parent is not visible; Any children won't be visible; No need to tessellate
        //     if clipper_visibility == Visibility::Hidden {
        //         self.flat_draw_queue.push(DrawCommand::Shape(ShapeDrawData {
        //             vertex_buffers: VertexBuffers::new(),
        //             index,
        //             vertex_buffer: None,
        //             index_buffer: None,
        //             num_indices: None,
        //             stencil_value: stencil_reference,
        //             clipping_polygons: None,
        //             visibility: Visibility::Hidden,
        //             // Not visible
        //             bounding_rect: Rect::new((0.0, 0.0).into(), (0.0, 0.0).into()),
        //         }));
        //         return index;
        //     }
        //
        //     let bounding_rect = shape.bounding_rect();
        //
        //     let parent_bounding_rect = self
        //         .flat_draw_queue
        //         .get(clip_to_shape)
        //         .map(|draw_command| {
        //             if let DrawCommand::Shape(clipper_shape) = draw_command {
        //                 clipper_shape.bounding_rect
        //             } else {
        //                 Rect::new((0.0, 0.0).into(), (0.0, 0.0).into())
        //             }
        //         })
        //         .unwrap_or(Rect::new((0.0, 0.0).into(), (0.0, 0.0).into()));
        //
        //     // Parent box doesn't have any intersection with the child box, meaning the child is not visible
        //     let does_not_fit_at_all = parent_bounding_rect
        //         .intersection_unchecked(&bounding_rect)
        //         .is_empty();
        //
        //     if does_not_fit_at_all {
        //         self.flat_draw_queue.push(DrawCommand::Shape(ShapeDrawData {
        //             vertex_buffers: VertexBuffers::new(),
        //             index,
        //             vertex_buffer: None,
        //             index_buffer: None,
        //             num_indices: None,
        //             stencil_value: stencil_reference,
        //             clipping_polygons: None,
        //             visibility: Visibility::Hidden,
        //             // Not visible, so set the box to zero just in case
        //             bounding_rect: Rect::new((0.0, 0.0).into(), (0.0, 0.0).into()),
        //         }));
        //         return index;
        //     }
        //
        //     {
        //         let clipper_shape = self.flat_draw_queue.get_mut(clip_to_shape);
        //         if let Some(clipper_shape) = clipper_shape {
        //             if let DrawCommand::Shape(ref mut clipper_shape) = clipper_shape {
        //                 if clipper_shape.clipping_polygons.is_none() {
        //                     clipper_shape.clipping_polygons = Some(MultiPolygon::new(
        //                         vertex_buffers_to_polygons(&clipper_shape.vertex_buffers),
        //                     ));
        //                 }
        //             }
        //         }
        //     }
        //
        //     let clipper_shape = self.flat_draw_queue.get(clip_to_shape);
        //     if let Some(clipper_shape) = clipper_shape {
        //         if let DrawCommand::Shape(clipper_shape) = clipper_shape {
        //             let clipper_polygons = clipper_shape.clipping_polygons.as_ref();
        //             if let Some(clipper_polygons) = clipper_polygons {
        //                 // First, we need to check if the bounding box fully fits into the parent clipper polygons
        //                 //  or doesn't fit at all, even partially
        //                 let bounding_polygon = geo::Polygon::new(
        //                     vec![
        //                         (bounding_rect.min.x as f64, bounding_rect.min.y as f64),
        //                         (bounding_rect.max.x as f64, bounding_rect.min.y as f64),
        //                         (bounding_rect.max.x as f64, bounding_rect.max.y as f64),
        //                         (bounding_rect.min.x as f64, bounding_rect.max.y as f64),
        //                     ]
        //                     .into(),
        //                     vec![],
        //                 );
        //                 let does_not_fit_at_all = clipper_polygons.iter().all(|clipper_polygon| {
        //                     let clipper_polygon = clipper_polygon;
        //
        //                     clipper_polygon.intersection(&bounding_polygon).is_empty()
        //                 });
        //
        //                 if does_not_fit_at_all {
        //                     println!(
        //                         "Doesn't fit at all: {:?} into {:?}",
        //                         bounding_rect,
        //                         clipper_polygons.bounding_rect()
        //                     );
        //                     self.flat_draw_queue.push(DrawCommand::Shape(ShapeDrawData {
        //                         vertex_buffers: VertexBuffers::new(),
        //                         index,
        //                         vertex_buffer: None,
        //                         index_buffer: None,
        //                         num_indices: None,
        //                         stencil_value: stencil_reference,
        //                         clipping_polygons: None,
        //                         visibility: Visibility::Hidden,
        //                         // Not visible, so set the box to zero just in case
        //                         bounding_rect: Rect::new((0.0, 0.0).into(), (0.0, 0.0).into()),
        //                     }));
        //                     return index;
        //                 }
        //
        //                 let fully_fits = clipper_polygons.contains(&bounding_polygon);
        //
        //                 if !fully_fits {
        //                     println!(
        //                         "Doesn't fit fully: {:?} into {:?}",
        //                         bounding_rect,
        //                         clipper_polygons.bounding_rect()
        //                     );
        //                     let vertex_buffers = Self::tessellate_shape(shape);
        //
        //                     let polygons_to_clip = vertex_buffers_to_polygons(&vertex_buffers);
        //                     let clipped_polygons =
        //                         crate::geo_utils::clip(polygons_to_clip, clipper_polygons);
        //                     let clipped_buffers =
        //                         polygons_to_vertex_buffers(&clipped_polygons, color);
        //                     let multi_clipped_polygons = MultiPolygon::new(clipped_polygons);
        //                     let clipped_box = multi_clipped_polygons
        //                         .bounding_rect()
        //                         .map(|rect| {
        //                             Rect::new(
        //                                 (rect.min().x as f32, rect.min().y as f32).into(),
        //                                 (rect.max().x as f32, rect.max().y as f32).into(),
        //                             )
        //                         })
        //                         .unwrap_or_else(|| Rect::new((0.0, 0.0).into(), (0.0, 0.0).into()));
        //
        //                     println!("Parent box: ");
        //                     println!("Clipped box: {:?}", clipped_box);
        //
        //                     self.flat_draw_queue.push(DrawCommand::Shape(ShapeDrawData {
        //                         vertex_buffers: clipped_buffers,
        //                         index,
        //                         vertex_buffer: None,
        //                         index_buffer: None,
        //                         num_indices: None,
        //                         stencil_value: stencil_reference,
        //                         clipping_polygons: None,
        //                         // TODO
        //                         visibility: Visibility::Visible,
        //                         // TODO
        //                         bounding_rect: clipped_box,
        //                     }));
        //                 } else {
        //                     println!(
        //                         "Fully fits: {:?} into {:?}",
        //                         bounding_rect,
        //                         clipper_polygons.bounding_rect()
        //                     );
        //                     let vertex_buffers = Self::tessellate_shape(shape);
        //
        //                     self.flat_draw_queue.push(DrawCommand::Shape(ShapeDrawData {
        //                         vertex_buffers,
        //                         index,
        //                         vertex_buffer: None,
        //                         index_buffer: None,
        //                         num_indices: None,
        //                         stencil_value: stencil_reference,
        //                         clipping_polygons: None,
        //                         visibility: Visibility::Visible,
        //                         bounding_rect,
        //                     }));
        //                 }
        //             } else {
        //                 let vertex_buffers = Self::tessellate_shape(shape);
        //
        //                 self.flat_draw_queue.push(DrawCommand::Shape(ShapeDrawData {
        //                     vertex_buffers,
        //                     index,
        //                     vertex_buffer: None,
        //                     index_buffer: None,
        //                     num_indices: None,
        //                     stencil_value: stencil_reference,
        //                     clipping_polygons: None,
        //                     visibility: Visibility::Visible,
        //                     bounding_rect,
        //                 }));
        //             }
        //         } else {
        //             let vertex_buffers = Self::tessellate_shape(shape);
        //
        //             self.flat_draw_queue.push(DrawCommand::Shape(ShapeDrawData {
        //                 vertex_buffers,
        //                 index,
        //                 vertex_buffer: None,
        //                 index_buffer: None,
        //                 num_indices: None,
        //                 stencil_value: stencil_reference,
        //                 clipping_polygons: None,
        //                 visibility: Visibility::Visible,
        //                 bounding_rect,
        //             }));
        //         }
        //     } else {
        //         let vertex_buffers = Self::tessellate_shape(shape);
        //
        //         self.flat_draw_queue.push(DrawCommand::Shape(ShapeDrawData {
        //             vertex_buffers,
        //             index,
        //             vertex_buffer: None,
        //             index_buffer: None,
        //             num_indices: None,
        //             stencil_value: stencil_reference,
        //             clipping_polygons: None,
        //             visibility: Visibility::Visible,
        //             bounding_rect,
        //         }));
        //     }
        // } else {
        //     // self.draw_queue
        //     //     .entry(stencil_reference)
        //     //     .or_insert_with(Vec::new)
        //     //     .push(DrawCommand::Shape(ShapeDrawData {
        //     //         vertex_buffers: vertex_buffers.clone(),
        //     //         index,
        //     //         vertex_buffer: None,
        //     //         index_buffer: None,
        //     //         num_indices: None,
        //     //         stencil_value: stencil_reference,
        //     //         clipping_polygons: None,
        //     //         visibility: Visibility::Visible
        //     //     }));
        //     let vertex_buffers = Self::tessellate_shape(shape);
        //
        //     self.flat_draw_queue.push(DrawCommand::Shape(ShapeDrawData {
        //         vertex_buffers,
        //         index,
        //         vertex_buffer: None,
        //         index_buffer: None,
        //         num_indices: None,
        //         stencil_value: stencil_reference,
        //         clipping_polygons: None,
        //         visibility: Visibility::Visible,
        //         bounding_rect: shape.bounding_rect(),
        //     }));
        // }

        let draw_command = DrawCommand::Shape(ShapeDrawData {
            vertex_buffer: None,
            index_buffer: None,
            num_indices: None,
            clip_to_shape,
            shape,
        });

        let shape_id = if self.draw_tree.is_empty() {
            self.draw_tree.add_node(draw_command)
        } else {
            if let Some(clip_to_shape) = clip_to_shape {
                self.draw_tree.add_child(clip_to_shape, draw_command)
            } else {
                self.draw_tree.add_child_to_root(draw_command)
            }
        };

        shape_id
    }

    pub fn add_shape_to_rendering_queue(
        &mut self,
        path: Shape,
        clip_to_shape: Option<usize>,
    ) -> usize {
        self.convert_shape_to_vertex_and_add_schedule_for_rendering(path, clip_to_shape)
    }
}

/// Implementation of text rendering methods
impl State<'_> {
    pub fn add_text_instance(
        &mut self,
        text: &str,
        layout: TextLayout,
        clip_to_shape: Option<usize>,
    ) {
        let mut buffer = TextBuffer::new(
            &mut self.text_renderer_wrapper.font_system,
            Metrics::new(layout.font_size, layout.line_height),
        );

        let text_area_size = layout.area.size();

        buffer.set_size(
            &mut self.text_renderer_wrapper.font_system,
            text_area_size.width,
            text_area_size.height,
        );

        // TODO: it's set text that causes performance issues
        buffer.set_text(
            &mut self.text_renderer_wrapper.font_system,
            text,
            Attrs::new().family(Family::SansSerif).metadata(clip_to_shape.unwrap()),
            Shaping::Advanced,
        );

        let align = match layout.horizontal_alignment {
            // None is equal to start of the line - left or right, depending on the language
            TextAlignment::Start => None,
            TextAlignment::End => Some(Align::End),
            TextAlignment::Center => Some(Align::Center),
        };

        for line in buffer.lines.iter_mut() {
            line.set_align(align);
        }

        let area = layout.area;

        let mut min_y = f32::INFINITY;
        let mut max_y = f32::NEG_INFINITY;

        buffer.shape_until_scroll(&mut self.text_renderer_wrapper.font_system);

        for layout_run in buffer.layout_runs() {
            for glyph in layout_run.glyphs.iter() {
                let physical_glyph = glyph.physical((0.0, 0.0), self.scale_factor as f32);
                min_y = min_y.min(physical_glyph.y as f32 + layout_run.line_y);
                max_y = max_y.max(physical_glyph.y as f32 + layout_run.line_y);
            }
        }

        // TODO: that should be a line height
        let buffer_height = if max_y > min_y {
            max_y - min_y + layout.font_size
        } else {
            layout.font_size // for a single line
        };

        let top = match layout.vertical_alignment {
            TextAlignment::Start => area.min.y,
            TextAlignment::End => area.max.y - buffer_height,
            TextAlignment::Center => area.min.y + (area.height() - buffer_height) / 2.0,
        };

        // println!("Text {} is clipped to shape {}", text, clip_to_shape.unwrap_or(0));

        self.text_instances.push(TextDrawData {
            top,
            text_buffer: buffer,
            area: layout.area,
            vertical_alignment: layout.vertical_alignment,
            data: text.to_string(),
            font_size: layout.font_size,
            clip_to_shape,
        });
    }
}
