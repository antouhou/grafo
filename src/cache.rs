use crate::vertex::CustomVertex;
use lru::LruCache;
use lyon::tessellation::VertexBuffers;

pub struct Cache {
    tessellation_cache: LruCache<u64, VertexBuffers<CustomVertex, u16>>,
}

impl Default for Cache {
    fn default() -> Self {
        Self::new()
    }
}

impl Cache {
    pub fn new() -> Self {
        Self {
            tessellation_cache: LruCache::unbounded(),
        }
    }

    pub fn get_vertex_buffers(
        &mut self,
        cache_key: &u64,
    ) -> Option<VertexBuffers<CustomVertex, u16>> {
        self.tessellation_cache.get(cache_key).cloned()
    }

    pub fn insert_vertex_buffers(
        &mut self,
        cache_key: u64,
        vertex_buffers: VertexBuffers<CustomVertex, u16>,
    ) {
        self.tessellation_cache.put(cache_key, vertex_buffers);
    }
}
