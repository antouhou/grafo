use crate::renderer::depth;
use crate::vertex::CustomVertex;
use crate::{Color, Stroke};
use lyon::lyon_tessellation::{
    BuffersBuilder, FillOptions, FillTessellator, FillVertex, VertexBuffers,
};
use lyon::tessellation::FillVertexConstructor;
use wgpu::util::DeviceExt;

#[derive(Debug, Clone)]
pub enum Shape {
    Path(Path),
    Rect(Rect),
}

impl From<Path> for Shape {
    fn from(value: Path) -> Self {
        Shape::Path(value)
    }
}

impl From<Rect> for Shape {
    fn from(value: Rect) -> Self {
        Shape::Rect(value)
    }
}

#[derive(Debug, Clone)]
pub struct Rect {
    pub rect: [(f32, f32); 2],
    pub fill: Color,
    pub stroke: Stroke,
}

impl Rect {
    pub fn new(rect: [(f32, f32); 2], fill: Color, stroke: Stroke) -> Self {
        Self { rect, fill, stroke }
    }
}

#[derive(Clone, Debug)]
pub struct Path {
    pub path: lyon::path::Path,
    pub fill: Color,
    pub stroke: Stroke,
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

impl Path {
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