use std::sync::Arc;
use std::time::Instant;

use easy_tree::rayon::iter::ParallelIterator;

type Rect = lyon::math::Box2D;

use crate::pipeline::{
    create_and_depth_texture, create_and_pass, create_depth_stencil_state_for_text,
    create_pipeline, render_buffer_to_texture2, PipelineType, Uniforms,
};
use crate::util::to_logical;
use ahash::{HashMap, HashMapExt};
use glyphon::cosmic_text::Align;
use glyphon::{
    Attrs, Buffer as TextBuffer, Color as TextColor, Family, FontSystem, Metrics, Resolution,
    Shaping, SwashCache, TextArea, TextAtlas, TextBounds, TextRenderer,
};

use crate::debug_tools::DepthStencilTextureViewer;
use crate::shape::ShapeDrawData;
use crate::{Color, Shape};
use wgpu::{
    BindGroup, CompositeAlphaMode, Device, InstanceDescriptor, MultisampleState, SurfaceTarget,
};

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

enum DrawCommand {
    Shape(ShapeDrawData),
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

pub struct Renderer<'a> {
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

    /// Text instances to be rendered
    text_instances: Vec<TextDrawData>,

    text_renderer_wrapper: TextRendererWrapper,

    /// Shapes to be rendered
    draw_tree: easy_tree::Tree<DrawCommand>,

    and_uniforms: Uniforms,
    and_bind_group: BindGroup,
    and_pipeline: Arc<wgpu::RenderPipeline>,

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

impl Renderer<'_> {
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

        let depth_stencil_texture_viewer = DepthStencilTextureViewer::new(&device, size);

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

            and_pipeline: Arc::new(and_pipeline),
            and_uniforms,
            and_bind_group,

            decrementing_pipeline: Arc::new(decrementing_pipeline),
            decrementing_uniforms,
            decrementing_bind_group,

            draw_tree: easy_tree::Tree::new(),

            #[cfg(feature = "performance_measurement")]
            performance_query_set: frametime_query_set,
            #[cfg(feature = "performance_measurement")]
            adapter,

            depth_stencil_texture_viewer,
        }
    }

    pub fn size(&self) -> (u32, u32) {
        self.physical_size
    }

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
    }

    pub fn update(&mut self) {}

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        println!("===============");
        let first_timer = Instant::now();
        let draw_tree_size = self.draw_tree.len();
        let iter = self.draw_tree.par_iter_mut();

        iter.for_each(|draw_command| {
            match draw_command.1 {
                DrawCommand::Shape(ref mut shape) => {
                    shape.prepare_buffers(&self.device, draw_command.0, draw_tree_size);
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

                                render_buffer_to_texture2(
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
}

/// Implementation of the methods related to shape rendering
impl Renderer<'_> {
    pub fn clear_draw_queue(&mut self) {
        self.draw_tree.clear();
        self.text_instances.clear();
    }

    pub fn add_shape(&mut self, shape: impl Into<Shape>, clip_to_shape: Option<usize>) -> usize {
        let shape = shape.into();

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
}

/// Implementation of text rendering methods
impl Renderer<'_> {
    pub fn add_text(
        &mut self,
        text: &str,
        layout: impl Into<TextLayout>,
        clip_to_shape: Option<usize>,
    ) {
        let layout = layout.into();

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
            Attrs::new()
                .family(Family::SansSerif)
                .metadata(clip_to_shape.unwrap()),
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
