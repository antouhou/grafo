use crate::renderer::depth;
use crate::vertex::CustomVertex;
use crate::{Color, Stroke};
use lyon::lyon_tessellation::{
    BuffersBuilder, FillOptions, FillTessellator, FillVertex, VertexBuffers,
};
use lyon::path::Winding;
use lyon::tessellation::FillVertexConstructor;
use wgpu::util::DeviceExt;

#[derive(Debug, Clone)]
pub enum Shape {
    Path(PathShape),
    Rect(RectShape),
}

impl Shape {
    /// Create a builder for creating shapes
    pub fn builder() -> ShapeBuilder {
        ShapeBuilder::new()
    }

    /// Create a simple rectangle shape
    pub fn rect(rect: [(f32, f32); 2], fill_color: Color, stroke: Stroke) -> Shape {
        let rect_shape = RectShape::new(rect, fill_color, stroke);
        Shape::Rect(rect_shape)
    }

    /// Create a rectangle shape with rounded corners
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

#[derive(Debug, Clone)]
pub struct RectShape {
    pub(crate) rect: [(f32, f32); 2],
    pub(crate) fill: Color,
    pub(crate) stroke: Stroke,
}

impl RectShape {
    pub fn new(rect: [(f32, f32); 2], fill: Color, stroke: Stroke) -> Self {
        Self { rect, fill, stroke }
    }
}

#[derive(Clone, Debug)]
pub struct PathShape {
    pub(crate) path: lyon::path::Path,
    pub(crate) fill: Color,
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
    pub fn new(path: lyon::path::Path, fill: Color, stroke: Stroke) -> Self {
        Self { path, fill, stroke }
    }

    pub(crate) fn tessellate(&self, depth: f32) -> VertexBuffers<CustomVertex, u16> {
        let mut buffers: VertexBuffers<CustomVertex, u16> = VertexBuffers::new();
        let mut tessellator = FillTessellator::new();
        let options = FillOptions::default().with_tolerance(0.01);

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

#[derive(Debug)]
pub struct ShapeDrawData {
    pub(crate) vertex_buffer: Option<wgpu::Buffer>,
    pub(crate) index_buffer: Option<wgpu::Buffer>,
    pub(crate) num_indices: Option<u32>,
    pub(crate) clip_to_shape: Option<usize>,
    pub(crate) shape: Shape,
}

impl ShapeDrawData {
    pub fn new(shape: impl Into<Shape>, clip_to_shape: Option<usize>) -> Self {
        let shape = shape.into();

        ShapeDrawData {
            vertex_buffer: None,
            index_buffer: None,
            num_indices: None,
            clip_to_shape,
            shape,
        }
    }

    fn shape_data_to_buffers(
        &self,
        device: &wgpu::Device,
        depth: f32,
    ) -> (wgpu::Buffer, wgpu::Buffer, u32) {
        match &self.shape {
            Shape::Path(path_shape) => {
                let vertex_buffers = path_shape.tessellate(depth);

                let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(&vertex_buffers.vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                });

                let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(&vertex_buffers.indices),
                    usage: wgpu::BufferUsages::INDEX,
                });

                (
                    vertex_buffer,
                    index_buffer,
                    vertex_buffers.indices.len() as u32,
                )
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

                let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(&quad),
                    usage: wgpu::BufferUsages::VERTEX,
                });

                let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(&[0u16, 1, 2, 3, 4, 5]),
                    usage: wgpu::BufferUsages::INDEX,
                });

                (vertex_buffer, index_buffer, 6)
            }
        }
    }
    pub fn prepare_buffers(&mut self, device: &wgpu::Device, shape_id: usize, max_shape_id: usize) {
        let depth = depth(shape_id, max_shape_id);
        let (vertex_buffer, index_buffer, num_indices) = self.shape_data_to_buffers(device, depth);

        self.vertex_buffer = Some(vertex_buffer);
        self.index_buffer = Some(index_buffer);
        self.num_indices = Some(num_indices);
    }
}

/// A builder for creating shapes
#[derive(Clone)]
pub struct ShapeBuilder {
    color: Color,
    stroke: Stroke,
    path_builder: lyon::path::Builder,
}

impl Default for ShapeBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ShapeBuilder {
    /// Create a new shape builder
    pub fn new() -> Self {
        Self {
            color: Color::rgb(0, 0, 0),
            stroke: Stroke::new(1.0, Color::rgb(0, 0, 0)),
            path_builder: lyon::path::Path::builder(),
        }
    }

    /// Set the fill color of the shape
    pub fn fill(mut self, color: Color) -> Self {
        self.color = color;
        self
    }

    /// Set the stroke of the shape
    pub fn stroke(mut self, stroke: Stroke) -> Self {
        self.stroke = stroke;
        self
    }

    /// Draw a line from the current point to the given point
    pub fn line_to(mut self, point: (f32, f32)) -> Self {
        self.path_builder.line_to(point.into());
        self
    }

    /// Draw a cubic bezier curve from the current point to the given point
    pub fn cubic_bezier_to(mut self, ctrl: (f32, f32), ctrl2: (f32, f32), to: (f32, f32)) -> Self {
        self.path_builder
            .cubic_bezier_to(ctrl.into(), ctrl2.into(), to.into());
        self
    }

    /// Draw a quadratic bezier curve from the current point to the given point
    pub fn quadratic_bezier_to(mut self, ctrl: (f32, f32), to: (f32, f32)) -> Self {
        self.path_builder
            .quadratic_bezier_to(ctrl.into(), to.into());
        self
    }

    /// Close the current sub-path
    pub fn close(mut self) -> Self {
        self.path_builder.close();
        self
    }

    /// Build the shape
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

impl BorderRadii {
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
