use crate::cache::Cache;
use crate::vertex::CustomVertex;
use lyon::tessellation::VertexBuffers;

pub fn normalize_rgba_color(color: &[u8; 4]) -> [f32; 4] {
    [
        color[0] as f32 / 255.0,
        color[1] as f32 / 255.0,
        color[2] as f32 / 255.0,
        color[3] as f32 / 255.0,
    ]
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
}

pub(crate) struct PoolManager {
    pub lyon_vertex_buffers_pool: LyonVertexBuffersPool,
    pub tessellation_cache: Cache,
}

impl PoolManager {
    pub(crate) fn new() -> Self {
        Self {
            lyon_vertex_buffers_pool: LyonVertexBuffersPool::new(),
            tessellation_cache: Cache::new(),
        }
    }

    // pub fn print_sizes(&self) {
    //     println!("Vertex buffers: {}", self.vertex_buffer_pool.buffers.len());
    //     println!("Index buffers: {}", self.index_buffer_pool.buffers.len());
    //     println!("Lyon vertex buffers: {}", self.lyon_vertex_buffers_pool.vertex_buffers.len());
    //     // println!("Tessellation cache: {}", self.tessellation_cache.len());
    // }
}

#[inline(always)]
pub fn to_logical(physical_size: (u32, u32), scale_factor: f64) -> (f32, f32) {
    let (physical_width, physical_height) = physical_size;
    let logical_width = physical_width as f64 / scale_factor;
    let logical_height = physical_height as f64 / scale_factor;
    (logical_width as f32, logical_height as f32)
}
