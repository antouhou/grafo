use crate::cache::Cache;
use crate::vertex::CustomVertex;
use lyon::tessellation::VertexBuffers;
use std::num::NonZeroUsize;

const MAX_LYON_VERTEX_BUFFER_POOL_SIZE: usize = 256;

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
    #[cfg(test)]
    stats: PoolReuseStats,
}

#[cfg(test)]
#[derive(Default, Clone, Copy)]
pub(crate) struct PoolReuseStats {
    pub(crate) reused: usize,
    pub(crate) created: usize,
    pub(crate) returned: usize,
}

impl LyonVertexBuffersPool {
    pub fn new() -> Self {
        Self {
            vertex_buffers: Vec::new(),
            #[cfg(test)]
            stats: PoolReuseStats::default(),
        }
    }

    pub fn len(&self) -> usize {
        self.vertex_buffers.len()
    }

    pub fn get_vertex_buffers(&mut self) -> VertexBuffers<CustomVertex, u16> {
        if let Some(mut vertex_buffers) = self.vertex_buffers.pop() {
            vertex_buffers.vertices.clear();
            vertex_buffers.indices.clear();
            #[cfg(test)]
            {
                self.stats.reused += 1;
            }
            vertex_buffers
        } else {
            #[cfg(test)]
            {
                self.stats.created += 1;
            }
            VertexBuffers::new()
        }
    }

    pub fn return_vertex_buffers(&mut self, mut vertex_buffers: VertexBuffers<CustomVertex, u16>) {
        vertex_buffers.vertices.clear();
        vertex_buffers.indices.clear();
        if self.vertex_buffers.len() < MAX_LYON_VERTEX_BUFFER_POOL_SIZE {
            self.vertex_buffers.push(vertex_buffers);
            #[cfg(test)]
            {
                self.stats.returned += 1;
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn stats(&self) -> PoolReuseStats {
        self.stats
    }
}

pub(crate) struct PoolManager {
    pub lyon_vertex_buffers_pool: LyonVertexBuffersPool,
    pub tessellation_cache: Cache,
}

impl PoolManager {
    pub(crate) fn new(tesselation_cache_size: NonZeroUsize) -> Self {
        Self {
            lyon_vertex_buffers_pool: LyonVertexBuffersPool::new(),
            tessellation_cache: Cache::new(tesselation_cache_size),
        }
    }

    pub fn print_sizes(&self) {
        println!("Pool sizes:");
        println!("Vertex buffers: {}", self.lyon_vertex_buffers_pool.len());
        println!("Index buffers: {}", self.tessellation_cache.len());
    }
}

#[inline(always)]
pub fn to_logical(physical_size: (u32, u32), scale_factor: f64) -> (f32, f32) {
    let (physical_width, physical_height) = physical_size;
    let logical_width = physical_width as f64 / scale_factor;
    let logical_height = physical_height as f64 / scale_factor;
    (logical_width as f32, logical_height as f32)
}

#[cfg(test)]
mod tests {
    use super::LyonVertexBuffersPool;

    #[test]
    fn vertex_buffer_pool_reuses_returned_buffers() {
        let mut pool = LyonVertexBuffersPool::new();
        let mut first = pool.get_vertex_buffers();
        first.vertices.reserve(128);
        first.indices.reserve(256);
        pool.return_vertex_buffers(first);

        let second = pool.get_vertex_buffers();
        assert!(second.vertices.capacity() >= 128);
        assert!(second.indices.capacity() >= 256);

        let stats = pool.stats();
        assert_eq!(stats.created, 1);
        assert_eq!(stats.returned, 1);
        assert_eq!(stats.reused, 1);
    }
}
