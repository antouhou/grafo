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

use crate::util::PoolManager;
use crate::vertex::{CustomVertex, InstanceTransform};
use crate::{Color, Stroke};
use lyon::lyon_tessellation::{
    BuffersBuilder, FillOptions, FillTessellator, FillVertex, VertexBuffers,
};
use lyon::path::Winding;
use lyon::tessellation::FillVertexConstructor;

pub(crate) struct CachedShape {
    pub depth: f32,
    pub vertex_buffers: VertexBuffers<CustomVertex, u16>,
}

impl CachedShape {
    /// Creates a new `CachedShape` with the specified shape and depth.
    /// Note that tessellator_cache_key is different from the shape cache key; a Shape cache key is
    /// the shape identifier, while tesselator_cache_key is used to cache the tessellation of the
    /// shape and should be based on the shape properties, and not the shape identifier
    pub fn new(
        shape: &Shape,
        depth: f32,
        tessellator: &mut FillTessellator,
        pool: &mut PoolManager,
        tessellator_cache_key: Option<u64>,
    ) -> Self {
        let vertices = shape.tessellate(depth, tessellator, pool, tessellator_cache_key);
        Self {
            depth,
            vertex_buffers: vertices,
        }
    }

    pub fn set_depth(&mut self, depth: f32) {
        self.depth = depth;
        for vertex in self.vertex_buffers.vertices.iter_mut() {
            vertex.order = depth;
        }
    }
}
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

    pub(crate) fn tessellate(
        &self,
        depth: f32,
        tessellator: &mut FillTessellator,
        buffers_pool: &mut PoolManager,
        tesselation_cache_key: Option<u64>,
    ) -> VertexBuffers<CustomVertex, u16> {
        match &self {
            Shape::Path(path_shape) => {
                path_shape.tessellate(depth, tessellator, buffers_pool, tesselation_cache_key)
            }
            Shape::Rect(rect_shape) => {
                let min_width = rect_shape.rect[0].0;
                let min_height = rect_shape.rect[0].1;
                let max_width = rect_shape.rect[1].0;
                let max_height = rect_shape.rect[1].1;

                let color = rect_shape.fill.normalize();

                // Compute UVs mapping the rectangle to [0,1] in local space
                let w = (max_width - min_width).max(1e-6);
                let h = (max_height - min_height).max(1e-6);
                let uv =
                    |x: f32, y: f32| -> [f32; 2] { [(x - min_width) / w, (y - min_height) / h] };

                let quad = [
                    CustomVertex {
                        position: [min_width, min_height],
                        color,
                        tex_coords: uv(min_width, min_height),
                        order: depth,
                    },
                    CustomVertex {
                        position: [max_width, min_height],
                        color,
                        tex_coords: uv(max_width, min_height),
                        order: depth,
                    },
                    CustomVertex {
                        position: [min_width, max_height],
                        color,
                        tex_coords: uv(min_width, max_height),
                        order: depth,
                    },
                    CustomVertex {
                        position: [min_width, max_height],
                        color,
                        tex_coords: uv(min_width, max_height),
                        order: depth,
                    },
                    CustomVertex {
                        position: [max_width, min_height],
                        color,
                        tex_coords: uv(max_width, min_height),
                        order: depth,
                    },
                    CustomVertex {
                        position: [max_width, max_height],
                        color,
                        tex_coords: uv(max_width, max_height),
                        order: depth,
                    },
                ];
                let indices = [0u16, 1, 2, 3, 4, 5];

                let mut vertex_buffers = buffers_pool.lyon_vertex_buffers_pool.get_vertex_buffers();

                vertex_buffers.vertices.extend(quad);
                vertex_buffers.indices.extend(indices);

                vertex_buffers
            }
        }
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
            order: self.depth,
            color: self.color,
            tex_coords: [0.0, 0.0],
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
        buffers_pool: &mut PoolManager,
        tesselation_cache_key: Option<u64>,
    ) -> VertexBuffers<CustomVertex, u16> {
        let mut buffers = if let Some(cache_key) = tesselation_cache_key {
            if let Some(buffers) = buffers_pool
                .tessellation_cache
                .get_vertex_buffers(&cache_key)
            {
                buffers
            } else {
                let mut buffers: VertexBuffers<CustomVertex, u16> =
                    buffers_pool.lyon_vertex_buffers_pool.get_vertex_buffers();

                self.tesselate_into_buffers(&mut buffers, depth, tessellator);
                buffers_pool
                    .tessellation_cache
                    .insert_vertex_buffers(cache_key, buffers.clone());

                buffers
            }
        } else {
            let mut buffers: VertexBuffers<CustomVertex, u16> =
                buffers_pool.lyon_vertex_buffers_pool.get_vertex_buffers();

            self.tesselate_into_buffers(&mut buffers, depth, tessellator);

            buffers
        };

        if buffers.indices.len() % 2 != 0 {
            buffers.indices.push(0);
        }

        buffers
    }

    fn tesselate_into_buffers(
        &self,
        buffers: &mut VertexBuffers<CustomVertex, u16>,
        depth: f32,
        tessellator: &mut FillTessellator,
    ) {
        let options = FillOptions::default();

        let color = self.fill.normalize();
        let vertex_converter = VertexConverter::new(depth, color);

        tessellator
            .tessellate_path(
                &self.path,
                &options,
                &mut BuffersBuilder::new(buffers, vertex_converter),
            )
            .unwrap();

        // Generate UVs for the tessellated path using its axis-aligned bounding box in local space
        if !buffers.vertices.is_empty() {
            // Compute AABB of positions before offset translation
            let mut min_x = f32::INFINITY;
            let mut min_y = f32::INFINITY;
            let mut max_x = f32::NEG_INFINITY;
            let mut max_y = f32::NEG_INFINITY;
            for v in buffers.vertices.iter() {
                let x = v.position[0];
                let y = v.position[1];
                if x < min_x {
                    min_x = x;
                }
                if y < min_y {
                    min_y = y;
                }
                if x > max_x {
                    max_x = x;
                }
                if y > max_y {
                    max_y = y;
                }
            }
            let w = (max_x - min_x).max(1e-6);
            let h = (max_y - min_y).max(1e-6);
            for v in buffers.vertices.iter_mut() {
                let u = (v.position[0] - min_x) / w;
                let vcoord = (v.position[1] - min_y) / h;
                v.tex_coords = [u, vcoord];
            }
        }
    }
}

/// Contains the data required to draw a shape, including vertex and index buffers.
///
/// This struct is used internally by the renderer and typically does not need to be used
/// directly by library users.
#[derive(Debug)]
pub(crate) struct ShapeDrawData {
    /// The shape associated with this draw data.
    pub(crate) shape: Shape,
    /// Optional cache key for the shape, used for caching tessellated buffers.
    pub(crate) cache_key: Option<u64>,
    /// Range in the aggregated index buffer (start_index, count)  
    pub(crate) index_buffer_range: Option<(usize, usize)>,
    /// Indicates whether the shape is empty (no vertices or indices).
    pub(crate) is_empty: bool,
    /// Stencil reference assigned during render traversal (parent + 1). Cleared after frame.
    pub(crate) stencil_ref: Option<u32>,
    /// Index into the per-frame instance transform buffer
    pub(crate) instance_index: Option<usize>,
    /// Optional per-shape transform applied in clip-space (post-normalization)
    pub(crate) transform: Option<InstanceTransform>,
    /// Optional texture ids associated with this shape for multi-texturing layers.
    /// Layer 0: background/base
    /// Layer 1: foreground/overlay (e.g. text or decals) blended on top
    pub(crate) texture_ids: [Option<u64>; 2],
}

impl ShapeDrawData {
    pub fn new(shape: impl Into<Shape>, cache_key: Option<u64>) -> Self {
        let shape = shape.into();

        ShapeDrawData {
            shape,
            cache_key,
            index_buffer_range: None,
            is_empty: false,
            stencil_ref: None,
            instance_index: None,
            transform: None,
            texture_ids: [None, None],
        }
    }

    /// Tessellates complex shapes and stores the resulting buffers.
    #[inline(always)]
    pub(crate) fn tessellate(
        &mut self,
        depth: f32,
        tessellator: &mut FillTessellator,
        buffers_pool: &mut PoolManager,
    ) -> VertexBuffers<CustomVertex, u16> {
        self.shape
            .tessellate(depth, tessellator, buffers_pool, self.cache_key)
    }
}

pub(crate) struct CachedShapeDrawData {
    pub(crate) id: u64,
    pub(crate) index_buffer_range: Option<(usize, usize)>,
    pub(crate) is_empty: bool,
    /// Stencil reference assigned during render traversal (parent + 1). Cleared after frame.
    pub(crate) stencil_ref: Option<u32>,
    /// Index into the per-frame instance transform buffer
    pub(crate) instance_index: Option<usize>,
    /// Optional per-shape transform applied in clip-space (post-normalization)
    pub(crate) transform: Option<InstanceTransform>,
    /// Optional texture ids associated with this cached shape
    pub(crate) texture_ids: [Option<u64>; 2],
}

impl CachedShapeDrawData {
    pub fn new(id: u64) -> Self {
        Self {
            id,
            index_buffer_range: None,
            is_empty: false,
            stencil_ref: None,
            instance_index: None,
            transform: None,
            texture_ids: [None, None],
        }
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
    /// use grafo::ShapeBuilder;    ///
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

pub(crate) trait DrawShapeCommand {
    fn index_buffer_range(&self) -> Option<(usize, usize)>; // (start_index, index_count)
    fn is_empty(&self) -> bool;
    fn stencil_ref_mut(&mut self) -> &mut Option<u32>;
    fn instance_index_mut(&mut self) -> &mut Option<usize>;
    fn instance_index(&self) -> Option<usize>;
    fn transform(&self) -> Option<InstanceTransform>;
    fn set_transform(&mut self, t: InstanceTransform);
    fn texture_id(&self, layer: usize) -> Option<u64>;
    fn set_texture_id(&mut self, layer: usize, id: Option<u64>);
}

impl DrawShapeCommand for ShapeDrawData {
    #[inline]
    fn index_buffer_range(&self) -> Option<(usize, usize)> {
        self.index_buffer_range
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.is_empty
    }

    #[inline]
    fn stencil_ref_mut(&mut self) -> &mut Option<u32> {
        &mut self.stencil_ref
    }

    #[inline]
    fn instance_index_mut(&mut self) -> &mut Option<usize> {
        &mut self.instance_index
    }

    #[inline]
    fn instance_index(&self) -> Option<usize> {
        self.instance_index
    }

    #[inline]
    fn transform(&self) -> Option<InstanceTransform> {
        self.transform
    }

    #[inline]
    fn set_transform(&mut self, t: InstanceTransform) {
        self.transform = Some(t);
    }

    #[inline]
    fn texture_id(&self, layer: usize) -> Option<u64> {
        self.texture_ids.get(layer).copied().unwrap_or(None)
    }

    #[inline]
    fn set_texture_id(&mut self, layer: usize, id: Option<u64>) {
        if let Some(slot) = self.texture_ids.get_mut(layer) {
            *slot = id;
        }
    }
}

impl DrawShapeCommand for CachedShapeDrawData {
    #[inline]
    fn index_buffer_range(&self) -> Option<(usize, usize)> {
        self.index_buffer_range
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.is_empty
    }

    #[inline]
    fn stencil_ref_mut(&mut self) -> &mut Option<u32> {
        &mut self.stencil_ref
    }

    #[inline]
    fn instance_index_mut(&mut self) -> &mut Option<usize> {
        &mut self.instance_index
    }

    #[inline]
    fn instance_index(&self) -> Option<usize> {
        self.instance_index
    }

    #[inline]
    fn transform(&self) -> Option<InstanceTransform> {
        self.transform
    }

    #[inline]
    fn set_transform(&mut self, t: InstanceTransform) {
        self.transform = Some(t);
    }

    #[inline]
    fn texture_id(&self, layer: usize) -> Option<u64> {
        self.texture_ids.get(layer).copied().unwrap_or(None)
    }

    #[inline]
    fn set_texture_id(&mut self, layer: usize, id: Option<u64>) {
        if let Some(slot) = self.texture_ids.get_mut(layer) {
            *slot = id;
        }
    }
}
