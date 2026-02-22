use crate::vertex::CustomVertex;
use lru::LruCache;
use lyon::tessellation::VertexBuffers;
use std::num::NonZeroUsize;
use std::sync::Arc;

pub(crate) struct Cache {
    tessellation_cache: LruCache<u64, Arc<VertexBuffers<CustomVertex, u16>>>,
}

impl Cache {
    pub(crate) fn new(size: NonZeroUsize) -> Self {
        Self {
            tessellation_cache: LruCache::new(size),
        }
    }

    pub fn len(&self) -> usize {
        self.tessellation_cache.len()
    }

    pub(crate) fn get_vertex_buffers(
        &mut self,
        cache_key: &u64,
    ) -> Option<Arc<VertexBuffers<CustomVertex, u16>>> {
        self.tessellation_cache.get(cache_key).cloned()
    }

    pub(crate) fn insert_vertex_buffers(
        &mut self,
        cache_key: u64,
        vertex_buffers: Arc<VertexBuffers<CustomVertex, u16>>,
    ) {
        self.tessellation_cache.put(cache_key, vertex_buffers);
    }
}

#[cfg(test)]
mod tests {
    use super::Cache;
    use crate::vertex::CustomVertex;
    use lyon::tessellation::VertexBuffers;
    use std::num::NonZeroUsize;
    use std::sync::Arc;

    #[test]
    fn cache_returns_shared_arc_without_cloning_vertex_buffers() {
        let mut cache = Cache::new(NonZeroUsize::new(4).unwrap());
        let mut vertex_buffers = VertexBuffers::<CustomVertex, u16>::new();
        vertex_buffers.vertices.push(CustomVertex {
            position: [0.0, 0.0],
            tex_coords: [0.0, 0.0],
            normal: [0.0, 0.0],
            coverage: 1.0,
        });
        vertex_buffers.indices.push(0);

        let shared_vertex_buffers = Arc::new(vertex_buffers);
        cache.insert_vertex_buffers(7, shared_vertex_buffers.clone());

        let cached_vertex_buffers = cache.get_vertex_buffers(&7).unwrap();
        assert!(Arc::ptr_eq(&shared_vertex_buffers, &cached_vertex_buffers));
    }
}
