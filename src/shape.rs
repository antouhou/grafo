//! The `shape` module provides structures and methods for creating and managing graphical shapes
//! within the Grafo library. It supports both simple and complex shapes, including rectangles and
//! custom paths with fill and stroke properties.
//!
//! # Examples
//!
//! Creating and using different shapes:
//!
//! ```rust
//! use grafo::Color;
//! use grafo::Stroke;
//! use grafo::{Shape, ShapeBuilder, BorderRadii};
//!
//! // Create a simple rectangle
//! let rect = Shape::rect(
//!     [(0.0, 0.0), (100.0, 50.0)],
//!     Color::rgb(255, 0, 0), // Red fill
//!     Stroke::new(2.0, Color::BLACK), // Black stroke with width 2.0
//! );
//!
//! // Create a rounded rectangle
//! let rounded_rect = Shape::rounded_rect(
//!     [(0.0, 0.0), (100.0, 50.0)],
//!     BorderRadii::new(10.0),
//!     Color::rgba(0, 255, 0, 128), // Semi-transparent green fill
//!     Stroke::new(1.5, Color::BLACK), // Black stroke with width 1.5
//! );
//!
//! // Build a custom shape using ShapeBuilder
//! let custom_shape = Shape::builder()
//!     .fill(Color::rgb(0, 0, 255)) // Blue fill
//!     .stroke(Stroke::new(3.0, Color::BLACK)) // Black stroke with width 3.0
//!     .begin((0.0, 0.0))
//!     .line_to((50.0, 10.0))
//!     .line_to((50.0, 50.0))
//!     .close()
//!     .build();
//! ```

use crate::vertex::CustomVertex;
use crate::{Color, Stroke};
use lyon::lyon_tessellation::{
    BuffersBuilder, FillOptions, FillTessellator, FillVertex, VertexBuffers,
};
use lyon::path::Winding;
use lyon::tessellation::FillVertexConstructor;
use wgpu::{Buffer, Queue};
use crate::util::{BufferPool, LyonVertexBuffersPool};

/// Represents a graphical shape, which can be either a custom path or a simple rectangle.
///
/// # Variants
///
/// - `Path(PathShape)`: A custom path shape defined using Bézier curves and lines.
/// - `Rect(RectShape)`: A simple rectangular shape with optional rounded corners.
///
/// # Examples
///
/// ```rust
/// use grafo::Color;
/// use grafo::Stroke;
/// use grafo::{Shape, BorderRadii};
///
/// // Create a simple rectangle
/// let rect = Shape::rect(
///     [(0.0, 0.0), (100.0, 50.0)],
///     Color::rgb(255, 0, 0), // Red fill
///     Stroke::new(2.0, Color::BLACK), // Black stroke with width 2.0
/// );
///
/// // Create a custom path shape
/// let custom_path = Shape::builder()
///     .fill(Color::rgb(0, 255, 0))
///     .stroke(Stroke::new(1.0, Color::BLACK))
///     .begin((0.0, 0.0))
///     .line_to((50.0, 10.0))
///     .line_to((50.0, 50.0))
///     .close()
///     .build();
/// ```
#[derive(Debug, Clone)]
pub enum Shape {
    /// A custom path shape defined using Bézier curves and lines.
    Path(PathShape),
    /// A simple rectangular shape.
    Rect(RectShape),
}

impl Shape {
    /// Creates a new [`ShapeBuilder`] for constructing complex shapes.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use grafo::Shape;
    ///
    /// let builder = Shape::builder();
    /// ```
    pub fn builder() -> ShapeBuilder {
        ShapeBuilder::new()
    }

    /// Creates a simple rectangle shape with the specified coordinates, fill color, and stroke.
    ///
    /// # Parameters
    ///
    /// - `rect`: An array containing two tuples representing the top-left and bottom-right
    ///   coordinates of the rectangle.
    /// - `fill_color`: The fill color of the rectangle.
    /// - `stroke`: The stroke properties of the rectangle.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use grafo::Color;
    /// use grafo::Stroke;
    /// use grafo::Shape;
    ///
    /// let rect = Shape::rect(
    ///     [(0.0, 0.0), (100.0, 50.0)],
    ///     Color::rgb(255, 0, 0), // Red fill
    ///     Stroke::new(2.0, Color::BLACK), // Black stroke with width 2.0
    /// );
    /// ```
    pub fn rect(rect: [(f32, f32); 2], fill_color: Color, stroke: Stroke) -> Shape {
        let rect_shape = RectShape::new(rect, fill_color, stroke);
        Shape::Rect(rect_shape)
    }

    /// Creates a rectangle shape with rounded corners.
    ///
    /// # Parameters
    ///
    /// - `rect`: An array containing two tuples representing the top-left and bottom-right
    ///   coordinates of the rectangle.
    /// - `border_radii`: The radii for each corner of the rectangle.
    /// - `fill_color`: The fill color of the rectangle.
    /// - `stroke`: The stroke properties of the rectangle.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use grafo::Color;
    /// use grafo::Stroke;
    /// use grafo::{Shape, BorderRadii};
    ///
    /// let rounded_rect = Shape::rounded_rect(
    ///     [(0.0, 0.0), (100.0, 50.0)],
    ///     BorderRadii::new(10.0),
    ///     Color::rgba(0, 255, 0, 128), // Semi-transparent green fill
    ///     Stroke::new(1.5, Color::BLACK), // Black stroke with width 1.5
    /// );
    /// ```
    pub fn rounded_rect(
        rect: [(f32, f32); 2],
        border_radii: BorderRadii,
        fill_color: Color,
        stroke: Stroke,
    ) -> Shape {
        let mut path_builder = lyon::path::Path::builder();
        let box2d = lyon::math::Box2D::new(rect[0].into(), rect[1].into());

        path_builder.add_rounded_rectangle(&box2d, &border_radii.into(), Winding::Positive);
        let path = path_builder.build();

        let path_shape = PathShape {
            path,
            fill: fill_color,
            stroke,
        };
        Shape::Path(path_shape)
    }
}

impl From<PathShape> for Shape {
    fn from(value: PathShape) -> Self {
        Shape::Path(value)
    }
}

impl From<RectShape> for Shape {
    fn from(value: RectShape) -> Self {
        Shape::Rect(value)
    }
}

/// Represents a simple rectangular shape with a fill color and stroke.
///
/// You typically do not need to use `RectShape` directly; instead, use the [`Shape::rect`] method.
///
/// # Fields
///
/// - `rect`: An array containing two tuples representing the top-left and bottom-right
///   coordinates of the rectangle.
/// - `fill`: The fill color of the rectangle.
/// - `stroke`: The stroke properties of the rectangle.
///
/// # Examples
///
/// ```rust
/// use grafo::RectShape;
/// use grafo::Color;
/// use grafo::Stroke;
///
/// let rect_shape = RectShape::new(
///     [(0.0, 0.0), (100.0, 50.0)],
///     Color::rgb(255, 0, 0), // Red fill
///     Stroke::new(2.0, Color::BLACK), // Black stroke with width 2.0
/// );
/// ```
#[derive(Debug, Clone)]
pub struct RectShape {
    /// An array containing two tuples representing the top-left and bottom-right coordinates
    /// of the rectangle.
    pub(crate) rect: [(f32, f32); 2],
    /// The fill color of the rectangle.
    pub(crate) fill: Color,
    /// The stroke properties of the rectangle.
    #[allow(unused)]
    pub(crate) stroke: Stroke,
}

impl RectShape {
    /// Creates a new `RectShape` with the specified coordinates, fill color, and stroke.
    ///
    /// # Parameters
    ///
    /// - `rect`: An array containing two tuples representing the top-left and bottom-right
    ///   coordinates of the rectangle.
    /// - `fill`: The fill color of the rectangle.
    /// - `stroke`: The stroke properties of the rectangle.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use grafo::RectShape;
    /// use grafo::Color;
    /// use grafo::Stroke;
    ///
    /// let rect_shape = RectShape::new(
    ///     [(0.0, 0.0), (100.0, 50.0)],
    ///     Color::rgb(255, 0, 0), // Red fill
    ///     Stroke::new(2.0, Color::BLACK), // Black stroke with width 2.0
    /// );
    /// ```
    pub fn new(rect: [(f32, f32); 2], fill: Color, stroke: Stroke) -> Self {
        Self { rect, fill, stroke }
    }
}

/// Represents a custom path shape with a fill color and stroke.
///
/// You typically do not need to use `PathShape` directly; instead, use the [`Shape::builder`]
/// method to construct complex shapes.
///
/// # Fields
///
/// - `path`: The geometric path defining the shape.
/// - `fill`: The fill color of the shape.
/// - `stroke`: The stroke properties of the shape.
///
/// # Examples
///
/// ```rust
/// use grafo::{Shape, PathShape};
/// use grafo::Color;
/// use grafo::Stroke;
///
/// // Replace this with your own path
/// let path = lyon::path::Path::builder().build();
///
/// let path_shape = PathShape::new(
///     path,
///     Color::rgb(0, 255, 0), // Green fill
///     Stroke::new(1.0, Color::BLACK), // Black stroke with width 1.0
/// );
///
/// let shape = Shape::Path(path_shape);
/// ```
#[derive(Clone, Debug)]
pub struct PathShape {
    /// The geometric path defining the shape.
    pub(crate) path: lyon::path::Path,
    /// The fill color of the shape.
    pub(crate) fill: Color,
    /// The stroke properties of the shape.
    #[allow(unused)]
    pub(crate) stroke: Stroke,
}

struct VertexConverter {
    depth: f32,
    color: [f32; 4],
}

impl VertexConverter {
    fn new(depth: f32, color: [f32; 4]) -> Self {
        Self { depth, color }
    }
}

impl FillVertexConstructor<CustomVertex> for VertexConverter {
    fn new_vertex(&mut self, vertex: FillVertex) -> CustomVertex {
        CustomVertex {
            position: vertex.position().to_array(),
            depth: self.depth,
            color: self.color,
        }
    }
}

impl PathShape {
    /// Creates a new `PathShape` with the specified path, fill color, and stroke.
    ///
    /// # Parameters
    ///
    /// - `path`: The geometric path defining the shape.
    /// - `fill`: The fill color of the shape.
    /// - `stroke`: The stroke properties of the shape.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use grafo::PathShape;
    /// use grafo::Color;
    /// use grafo::Stroke;
    /// use lyon::path::Path;
    ///
    /// let path = Path::builder().build();
    /// let path_shape = PathShape::new(path, Color::rgb(0, 255, 0), Stroke::default());
    /// ```
    pub fn new(path: lyon::path::Path, fill: Color, stroke: Stroke) -> Self {
        Self { path, fill, stroke }
    }

    /// Tessellates the path shape into vertex and index buffers for rendering.
    ///
    /// # Parameters
    ///
    /// - `depth`: The depth value used for rendering order.
    ///
    /// # Returns
    ///
    /// A `VertexBuffers` structure containing the tessellated vertices and indices.
    /// ```
    pub(crate) fn tessellate(
        &self,
        depth: f32,
        tessellator: &mut FillTessellator,
        lyon_vertex_buffers_pool: &mut LyonVertexBuffersPool,
    ) -> VertexBuffers<CustomVertex, u16> {
        let mut buffers: VertexBuffers<CustomVertex, u16> = lyon_vertex_buffers_pool.get_vertex_buffers();

        let options = FillOptions::default();

        let color = self.fill.normalize();
        let vertex_converter = VertexConverter::new(depth, color);

        tessellator
            .tessellate_path(
                &self.path,
                &options,
                &mut BuffersBuilder::new(&mut buffers, vertex_converter),
            )
            .unwrap();

        buffers
    }
}

/// Contains the data required to draw a shape, including vertex and index buffers.
///
/// This struct is used internally by the renderer and typically does not need to be used
/// directly by library users.
#[derive(Debug)]
pub(crate) struct ShapeDrawData {
    /// The vertex buffer containing the shape's vertices.
    pub(crate) vertex_buffer: Option<wgpu::Buffer>,
    /// The index buffer containing the shape's indices.
    pub(crate) index_buffer: Option<wgpu::Buffer>,
    /// The number of indices in the index buffer.
    pub(crate) num_indices: Option<u32>,
    /// An optional index of another shape to clip to.
    pub(crate) clip_to_shape: Option<usize>,
    /// The shape associated with this draw data.
    pub(crate) shape: Shape,

    pub(crate) vertex_buffers: Option<VertexBuffers<CustomVertex, u16>>,
}

impl ShapeDrawData {
    pub fn new(shape: impl Into<Shape>, clip_to_shape: Option<usize>) -> Self {
        let shape = shape.into();

        ShapeDrawData {
            vertex_buffer: None,
            index_buffer: None,
            num_indices: None,
            vertex_buffers: None,
            clip_to_shape,
            shape,
        }
    }

    /// Tesselates complex shapes and stores the resulting buffers.
    #[inline(always)]
    pub(crate) fn tessellate(&mut self, depth: f32, tessellator: &mut FillTessellator, lyon_vertex_buffers_pool: &mut LyonVertexBuffersPool) {
        match &self.shape {
            Shape::Path(path_shape) => {
                let mut vertex_buffers = path_shape.tessellate(depth, tessellator, lyon_vertex_buffers_pool);
                if vertex_buffers.indices.len() % 2 != 0 {
                    vertex_buffers.indices.push(0);
                }
                self.vertex_buffers = Some(vertex_buffers);
            }
            Shape::Rect(rect_shape) => {
                let min_width = rect_shape.rect[0].0;
                let min_height = rect_shape.rect[0].1;
                let max_width = rect_shape.rect[1].0;
                let max_height = rect_shape.rect[1].1;

                let color = rect_shape.fill.normalize();

                let quad = [
                    CustomVertex {
                        position: [min_width, min_height],
                        color,
                        depth,
                    },
                    CustomVertex {
                        position: [max_width, min_height],
                        color,
                        depth,
                    },
                    CustomVertex {
                        position: [min_width, max_height],
                        color,
                        depth,
                    },
                    CustomVertex {
                        position: [min_width, max_height],
                        color,
                        depth,
                    },
                    CustomVertex {
                        position: [max_width, min_height],
                        color,
                        depth,
                    },
                    CustomVertex {
                        position: [max_width, max_height],
                        color,
                        depth,
                    },
                ];
                let indices = [0u16, 1, 2, 3, 4, 5];

                let mut vertex_buffers = lyon_vertex_buffers_pool.get_vertex_buffers();

                vertex_buffers.vertices.extend(quad);
                vertex_buffers.indices.extend(indices);

                self.vertex_buffers = Some(vertex_buffers);
            }
        }
    }

    #[inline(always)]
    fn shape_data_to_buffers(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        vertex_buffer_pool: &mut BufferPool,
        index_buffer_pool: &mut BufferPool,
    ) -> (wgpu::Buffer, wgpu::Buffer, u32) {
        let vertex_buffers = self
            .vertex_buffers
            .as_ref()
            .expect("To be tesselated at this point");
        // TODO: better buffer allocation
        let vertex_buffer = vertex_buffer_pool.get_buffer(device, 100000);
        let index_buffer = index_buffer_pool.get_buffer(device, 1000);

        queue.write_buffer(
            &vertex_buffer,
            0,
            bytemuck::cast_slice(&vertex_buffers.vertices),
        );
        queue.write_buffer(
            &index_buffer,
            0,
            bytemuck::cast_slice(&vertex_buffers.indices),
        );

        (
            vertex_buffer,
            index_buffer,
            vertex_buffers.indices.len() as u32,
        )
    }

    pub(crate) fn return_buffers_to_pool(
        &mut self,
        vertex_buffer_pool: &mut BufferPool,
        index_buffer_pool: &mut BufferPool,
    ) {
        let vertex_buffer = self.vertex_buffer.take();
        if let Some(vertex_buffer) = vertex_buffer {
            vertex_buffer_pool.return_buffer(vertex_buffer);
        }

        let index_buffer = self.index_buffer.take();
        if let Some(index_buffer) = index_buffer {
            index_buffer_pool.return_buffer(index_buffer);
        }
    }

    pub(crate) fn return_lyon_vertex_buffers_to_pool(
        &mut self,
        vertex_buffer_pool: &mut LyonVertexBuffersPool,
    ) {
        let vertex_buffers = self.vertex_buffers.take();
        if let Some(mut vertex_buffers) = vertex_buffers {
            vertex_buffers.clear();
            vertex_buffer_pool.return_vertex_buffers(vertex_buffers);
        }
    }

    /// Prepares the GPU buffers for rendering the shape.
    ///
    /// # Parameters
    ///
    /// - `device`: The WGPU device used to create the buffers.
    /// - `shape_id`: The identifier for the shape, used to calculate depth.
    /// - `max_shape_id`: The maximum number of shapes, used to normalize depth.
    ///
    /// # Examples
    ///
    /// ```rust
    /// // Assuming `device` is a valid wgpu::Device instance and shape_id/max_shape_id are set
    /// // shape_draw_data.prepare_buffers(&device, shape_id, max_shape_id);
    /// ```
    #[inline(always)]
    pub fn prepare_buffers(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        vertex_buffer_pool: &mut BufferPool,
        index_buffer_pool: &mut BufferPool,
    ) {
        let (vertex_buffer, index_buffer, num_indices) =
            self.shape_data_to_buffers(device, queue, vertex_buffer_pool, index_buffer_pool);

        self.vertex_buffer = Some(vertex_buffer);
        self.index_buffer = Some(index_buffer);
        self.num_indices = Some(num_indices);
    }

    /// Copies tessellated data into WGPU buffers and returns num_indices
    pub fn copy_data_to_buffers(
        &self,
        vertex_buffer: &Buffer,
        index_buffer: &Buffer,
        queue: &Queue,
    ) -> u32 {
        let vertex_buffers = self
            .vertex_buffers
            .as_ref()
            .expect("To be tesselated at this point");
        let vertex_bytes = bytemuck::cast_slice(&vertex_buffers.vertices);
        let index_bytes = bytemuck::cast_slice(&vertex_buffers.indices);

        // // Map the buffer for writing
        // vertex_buffer.slice(..).map_async(wgpu::MapMode::Write, |_| {});

        // Wait for the mapping to complete
        // device.poll(wgpu::Maintain::Wait);
        //
        // // Write data to the buffer
        // {
        //     let mut mapped_range = buffer.slice(..).get_mapped_range_mut();
        //     mapped_range.copy_from_slice(data);
        // }
        //
        // // Unmap the buffer
        // buffer.unmap();

        // if vertex_bytes.len() == 42 {
        //     println!("Vertex buffer: is 42");
        // }
        //
        // if index_bytes.len() == 42 {
        //     println!("Index buffer: is 42");
        //     println!("{:?}", vertex_buffers.indices.len())
        // }

        queue.write_buffer(vertex_buffer, 0, vertex_bytes);
        queue.write_buffer(index_buffer, 0, index_bytes);

        self.vertex_buffers.as_ref().unwrap().indices.len() as u32
    }
}

/// A builder for creating complex shapes using a fluent interface.
///
/// The `ShapeBuilder` allows you to define the fill color, stroke, and path of a shape
/// using method chaining. You also can get it from the [`Shape::builder`] method.
///
/// # Examples
///
/// ```rust
/// use grafo::Color;
/// use grafo::Stroke;
/// use grafo::ShapeBuilder;
///
/// let custom_shape = ShapeBuilder::new()
///     .fill(Color::rgb(0, 0, 255)) // Blue fill
///     .stroke(Stroke::new(3.0, Color::BLACK)) // Black stroke with width 3.0
///     .begin((0.0, 0.0))
///     .line_to((50.0, 10.0))
///     .line_to((50.0, 50.0))
///     .close()
///     .build();
/// ```
#[derive(Clone)]
pub struct ShapeBuilder {
    /// The fill color of the shape.
    color: Color,
    /// The stroke properties of the shape.
    stroke: Stroke,
    /// The path builder used to construct the shape's geometric path.
    path_builder: lyon::path::Builder,
}

impl Default for ShapeBuilder {
    /// Creates a default `ShapeBuilder` with black fill and stroke.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use grafo::ShapeBuilder;
    ///
    /// let builder = ShapeBuilder::default();
    /// ```
    fn default() -> Self {
        Self::new()
    }
}

impl ShapeBuilder {
    /// Creates a new `ShapeBuilder` with default fill color (black) and stroke.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use grafo::ShapeBuilder;
    ///
    /// let builder = ShapeBuilder::new();
    /// ```
    pub fn new() -> Self {
        Self {
            color: Color::rgb(0, 0, 0),
            stroke: Stroke::new(1.0, Color::rgb(0, 0, 0)),
            path_builder: lyon::path::Path::builder(),
        }
    }

    /// Sets the fill color of the shape.
    ///
    /// # Parameters
    ///
    /// - `color`: The desired fill color.
    ///
    /// # Returns
    ///
    /// The updated `ShapeBuilder` instance.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use grafo::Color;
    /// use grafo::ShapeBuilder;
    ///
    /// let builder = ShapeBuilder::new().fill(Color::rgb(255, 0, 0)); // Red fill
    /// ```
    pub fn fill(mut self, color: Color) -> Self {
        self.color = color;
        self
    }

    /// Sets the stroke properties of the shape.
    ///
    /// # Parameters
    ///
    /// - `stroke`: The desired stroke properties.
    ///
    /// # Returns
    ///
    /// The updated `ShapeBuilder` instance.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use grafo::Stroke;
    /// use grafo::Color;
    /// use grafo::ShapeBuilder;
    ///
    /// let builder = ShapeBuilder::new().stroke(Stroke::new(2.0, Color::BLACK)); // Black stroke with width 2.0
    /// ```
    pub fn stroke(mut self, stroke: Stroke) -> Self {
        self.stroke = stroke;
        self
    }

    /// Begin path at point
    ///
    /// # Parameters
    ///
    /// - `point`: The start point of the shape.
    ///
    /// # Returns
    ///
    /// The updated `ShapeBuilder` instance.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use grafo::ShapeBuilder;
    ///
    /// let builder = ShapeBuilder::new().begin((0.0, 0.0));
    /// ```
    pub fn begin(mut self, point: (f32, f32)) -> Self {
        self.path_builder.begin(point.into());
        self
    }

    /// Draws a line from the current point to the specified point.
    ///
    /// # Parameters
    ///
    /// - `point`: The end point of the line.
    ///
    /// # Returns
    ///
    /// The updated `ShapeBuilder` instance.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use grafo::ShapeBuilder;
    ///
    /// let builder = ShapeBuilder::new().begin((0.0, 0.0)).line_to((50.0, 10.0));
    /// ```
    pub fn line_to(mut self, point: (f32, f32)) -> Self {
        self.path_builder.line_to(point.into());
        self
    }

    /// Draws a cubic Bézier curve from the current point to the specified end point.
    ///
    /// # Parameters
    ///
    /// - `ctrl`: The first control point.
    /// - `ctrl2`: The second control point.
    /// - `to`: The end point of the curve.
    ///
    /// # Returns
    ///
    /// The updated `ShapeBuilder` instance.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use grafo::ShapeBuilder;
    ///
    /// let builder = ShapeBuilder::new()
    ///     .begin((0.0, 0.0))
    ///     .cubic_bezier_to((20.0, 30.0), (40.0, 30.0), (50.0, 10.0));
    /// ```
    pub fn cubic_bezier_to(mut self, ctrl: (f32, f32), ctrl2: (f32, f32), to: (f32, f32)) -> Self {
        self.path_builder
            .cubic_bezier_to(ctrl.into(), ctrl2.into(), to.into());
        self
    }

    /// Draws a quadratic Bézier curve from the current point to the specified end point.
    ///
    /// # Parameters
    ///
    /// - `ctrl`: The control point.
    /// - `to`: The end point of the curve.
    ///
    /// # Returns
    ///
    /// The updated `ShapeBuilder` instance.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use grafo::ShapeBuilder;
    ///
    /// let builder = ShapeBuilder::new()
    ///     .begin((0.0, 0.0))
    ///     .quadratic_bezier_to((25.0, 40.0), (50.0, 10.0));
    /// ```
    pub fn quadratic_bezier_to(mut self, ctrl: (f32, f32), to: (f32, f32)) -> Self {
        self.path_builder
            .quadratic_bezier_to(ctrl.into(), to.into());
        self
    }

    /// Closes the current sub-path by drawing a line back to the starting point.
    ///
    /// # Returns
    ///
    /// The updated `ShapeBuilder` instance.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use grafo::ShapeBuilder;
    ///
    /// let builder = ShapeBuilder::new().begin((0.0, 0.0)).close();
    /// ```
    pub fn close(mut self) -> Self {
        self.path_builder.close();
        self
    }

    /// Builds the [`Shape`] from the accumulated path, fill color, and stroke.
    ///
    /// # Returns
    ///
    /// A `Shape` instance representing the constructed shape.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use grafo::ShapeBuilder;
    ///
    /// let shape = ShapeBuilder::new()
    ///     .begin((0.0, 0.0))
    ///     .line_to((50.0, 10.0))
    ///     .line_to((50.0, 50.0))
    ///     .close()
    ///     .build();
    /// ```
    pub fn build(self) -> Shape {
        let path = self.path_builder.build();
        Shape::Path(PathShape {
            path,
            fill: self.color,
            stroke: self.stroke,
        })
    }
}

impl From<ShapeBuilder> for Shape {
    fn from(value: ShapeBuilder) -> Self {
        value.build()
    }
}

/// A set of border radii for a rounded rectangle
#[derive(Copy, Clone, PartialEq, PartialOrd, Debug, Default)]
pub struct BorderRadii {
    pub top_left: f32,
    pub top_right: f32,
    pub bottom_left: f32,
    pub bottom_right: f32,
}

/// Represents the radii of each corner for a rounded rectangle.
///
/// # Fields
///
/// - `top_left`: Radius of the top-left corner.
/// - `top_right`: Radius of the top-right corner.
/// - `bottom_left`: Radius of the bottom-left corner.
/// - `bottom_right`: Radius of the bottom-right corner.
///
/// # Examples
///
/// Creating uniform and non-uniform border radii:
///
/// ```rust
/// use grafo::BorderRadii;
///
/// // Uniform border radii
/// let uniform_radii = BorderRadii::new(10.0);
///
/// // Custom border radii
/// let custom_radii = BorderRadii {
///     top_left: 5.0,
///     top_right: 10.0,
///     bottom_left: 15.0,
///     bottom_right: 20.0,
/// };
/// ```
impl BorderRadii {
    /// Creates a new `BorderRadii` with the same radius for all corners.
    ///
    /// # Parameters
    ///
    /// - `radius`: The radius to apply to all corners.
    ///
    /// # Returns
    ///
    /// A `BorderRadii` instance with uniform corner radii.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use grafo::BorderRadii;
    ///
    /// let radii = BorderRadii::new(10.0);
    /// ```
    pub fn new(radius: f32) -> Self {
        let r = radius.abs();
        BorderRadii {
            top_left: r,
            top_right: r,
            bottom_left: r,
            bottom_right: r,
        }
    }
}

impl core::fmt::Display for BorderRadii {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // In the order of a well known convention (CSS) clockwise from top left
        write!(
            f,
            "BorderRadii({}, {}, {}, {})",
            self.top_left, self.top_right, self.bottom_left, self.bottom_right
        )
    }
}

impl From<BorderRadii> for lyon::path::builder::BorderRadii {
    fn from(val: BorderRadii) -> Self {
        lyon::path::builder::BorderRadii {
            top_left: val.top_left,
            top_right: val.top_right,
            bottom_left: val.bottom_left,
            bottom_right: val.bottom_right,
        }
    }
}
