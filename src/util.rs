use crate::renderer::MathRect;
use crate::vertex::CustomVertex;
use glyphon::{Buffer as TextBuffer, FontSystem, Metrics};
use lyon::geom::euclid::Point2D;
use lyon::tessellation::VertexBuffers;
use wgpu::util::DeviceExt;
use wgpu::{Buffer, BufferUsages};

pub fn normalize_rgba_color(color: &[u8; 4]) -> [f32; 4] {
    [
        color[0] as f32 / 255.0,
        color[1] as f32 / 255.0,
        color[2] as f32 / 255.0,
        color[3] as f32 / 255.0,
    ]
}

#[inline(always)]
pub fn normalize_rect(
    logical_rect: &MathRect,
    canvas_physical_size: (u32, u32),
    scale_factor: f32,
) -> MathRect {
    let ndc_min_x = 2.0 * logical_rect.min.x * scale_factor / canvas_physical_size.0 as f32 - 1.0;
    let ndc_min_y = 1.0 - 2.0 * logical_rect.min.y * scale_factor / canvas_physical_size.1 as f32;
    let ndc_max_x = 2.0 * logical_rect.max.x * scale_factor / canvas_physical_size.0 as f32 - 1.0;
    let ndc_max_y = 1.0 - 2.0 * logical_rect.max.y * scale_factor / canvas_physical_size.1 as f32;

    MathRect {
        min: Point2D::new(ndc_min_x, ndc_min_y),
        max: Point2D::new(ndc_max_x, ndc_max_y),
    }
}

pub(crate) struct BufferPool {
    buffer_usages: BufferUsages,
    buffers: Vec<Buffer>,
}

impl BufferPool {
    pub(crate) fn new(buffer_usages: BufferUsages) -> Self {
        Self {
            buffers: Vec::new(),
            buffer_usages,
        }
    }

    pub(crate) fn get_buffer(&mut self, device: &wgpu::Device, size: usize) -> Buffer {
        if let Some(buffer) = self.buffers.pop() {
            buffer
        } else {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: size as u64,
                usage: self.buffer_usages,
                mapped_at_creation: false,
            })
        }
    }

    pub(crate) fn return_buffer(&mut self, buffer: Buffer) {
        self.buffers.push(buffer);
    }

    pub(crate) fn size(&self) -> usize {
        self.buffers.len()
    }
}

pub struct LyonVertexBuffersPool {
    vertex_buffers: Vec<VertexBuffers<CustomVertex, u16>>,
}

impl LyonVertexBuffersPool {
    pub fn new() -> Self {
        Self {
            vertex_buffers: Vec::new(),
        }
    }

    pub fn get_vertex_buffers(&mut self) -> VertexBuffers<CustomVertex, u16> {
        if let Some(vertex_buffers) = self.vertex_buffers.pop() {
            vertex_buffers
        } else {
            VertexBuffers::new()
        }
    }

    pub fn return_vertex_buffers(&mut self, vertex_buffers: VertexBuffers<CustomVertex, u16>) {
        self.vertex_buffers.push(vertex_buffers);
    }

    pub fn size(&self) -> usize {
        self.vertex_buffers.len()
    }
}

pub struct TextBuffersPool {
    buffers: Vec<TextBuffer>,
}

impl TextBuffersPool {
    pub fn new() -> Self {
        Self {
            buffers: Vec::new(),
        }
    }

    pub fn get_text_buffer(
        &mut self,
        font_system: &mut FontSystem,
        metrics: Metrics,
    ) -> TextBuffer {
        if let Some(mut buffer) = self.buffers.pop() {
            buffer.set_metrics(font_system, metrics);
            buffer
        } else {
            TextBuffer::new(font_system, metrics)
        }
    }

    pub fn return_text_buffer(&mut self, buffer: TextBuffer) {
        self.buffers.push(buffer);
    }

    pub fn size(&self) -> usize {
        self.buffers.len()
    }
}

pub struct ImageBuffersPool {
    vertex_buffers: Vec<wgpu::Buffer>,
    index_buffers: Vec<wgpu::Buffer>,
}

impl ImageBuffersPool {
    pub fn new() -> Self {
        Self {
            vertex_buffers: Vec::new(),
            index_buffers: Vec::new(),
        }
    }

    pub fn get_vertex_buffer(&mut self, device: &wgpu::Device) -> wgpu::Buffer {
        if let Some(buffer) = self.vertex_buffers.pop() {
            buffer
        } else {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                // To draw an image, we always need 4 quad vertices, so the size is always the same
                size: 64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        }
    }

    pub fn get_index_buffer(&mut self, device: &wgpu::Device) -> wgpu::Buffer {
        if let Some(buffer) = self.index_buffers.pop() {
            buffer
        } else {
            // To draw an image we need to use only two triangles, so indices are always
            //  going to be the same
            let indices: &[u16] = &[
                0, 1, 2, // first triangle
                2, 3, 0, // second triangle
            ];

            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(indices),
                usage: wgpu::BufferUsages::INDEX,
            })
        }
    }

    pub fn return_vertex_buffer(&mut self, buffer: wgpu::Buffer) {
        self.vertex_buffers.push(buffer);
    }

    pub fn return_index_buffer(&mut self, buffer: wgpu::Buffer) {
        self.index_buffers.push(buffer);
    }

    pub fn size(&self) -> usize {
        self.vertex_buffers.len() + self.index_buffers.len()
    }
}

// #[inline(always)]
// fn srgb_to_linear(value: f32) -> f32 {
//     if value <= 0.04045 {
//         value / 12.92
//     } else {
//         ((value + 0.055) / 1.055).powf(2.4)
//     }
// }

// pub fn rgba_to_linear(rgba: [f32; 4]) -> [f32; 4] {
//     let [r, g, b, a] = rgba;
//     [
//         srgb_to_linear(r),
//         srgb_to_linear(g),
//         srgb_to_linear(b),
//         a, // Alpha channel remains unchanged
//     ]
// }

#[inline(always)]
pub fn to_logical(physical_size: (u32, u32), scale_factor: f64) -> (f32, f32) {
    let (physical_width, physical_height) = physical_size;
    let logical_width = physical_width as f64 / scale_factor;
    let logical_height = physical_height as f64 / scale_factor;
    (logical_width as f32, logical_height as f32)
}
