use crate::vertex::CustomVertex;
use lru::LruCache;
use lyon::tessellation::VertexBuffers;

pub(crate) struct Cache {
    tessellation_cache: LruCache<u64, VertexBuffers<CustomVertex, u16>>,
}

impl Cache {
    pub(crate) fn new() -> Self {
        Self {
            tessellation_cache: LruCache::unbounded(),
        }
    }

    pub(crate) fn get_vertex_buffers(
        &mut self,
        cache_key: &u64,
    ) -> Option<VertexBuffers<CustomVertex, u16>> {
        self.tessellation_cache.get(cache_key).cloned()
    }

    pub(crate) fn insert_vertex_buffers(
        &mut self,
        cache_key: u64,
        vertex_buffers: VertexBuffers<CustomVertex, u16>,
    ) {
        self.tessellation_cache.put(cache_key, vertex_buffers);
    }
}
