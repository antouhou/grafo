use crate::vertex::CustomVertex;
use ahash::{HashMap, HashMapExt};
use lyon::tessellation::VertexBuffers;
use std::num::NonZeroUsize;
use std::sync::Arc;

pub(crate) struct Cache {
    previous_frame: HashMap<u64, Arc<VertexBuffers<CustomVertex, u16>>>,
    current_frame: HashMap<u64, Arc<VertexBuffers<CustomVertex, u16>>>,
}

impl Cache {
    pub(crate) fn new(_size: NonZeroUsize) -> Self {
        Self {
            previous_frame: HashMap::new(),
            current_frame: HashMap::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.previous_frame.len() + self.current_frame.len()
    }

    pub(crate) fn get_vertex_buffers(
        &mut self,
        cache_key: &u64,
    ) -> Option<Arc<VertexBuffers<CustomVertex, u16>>> {
        if let Some(vertex_buffers) = self.current_frame.get(cache_key) {
            return Some(Arc::clone(vertex_buffers));
        }

        let vertex_buffers = self.previous_frame.get(cache_key)?.clone();
        self.current_frame
            .entry(*cache_key)
            .or_insert_with(|| Arc::clone(&vertex_buffers));
        Some(vertex_buffers)
    }

    pub(crate) fn insert_vertex_buffers(
        &mut self,
        cache_key: u64,
        vertex_buffers: Arc<VertexBuffers<CustomVertex, u16>>,
    ) {
        self.current_frame.insert(cache_key, vertex_buffers);
    }

    pub(crate) fn refresh_vertex_buffers(
        &mut self,
        cache_key: u64,
        vertex_buffers: &Arc<VertexBuffers<CustomVertex, u16>>,
    ) {
        self.current_frame
            .entry(cache_key)
            .or_insert_with(|| Arc::clone(vertex_buffers));
    }

    pub(crate) fn end_frame(&mut self) {
        std::mem::swap(&mut self.previous_frame, &mut self.current_frame);
        self.current_frame.clear();
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

    #[test]
    fn cache_promotes_previous_frame_hits_into_current_frame() {
        let mut cache = Cache::new(NonZeroUsize::new(4).unwrap());
        let shared_vertex_buffers = Arc::new(VertexBuffers::<CustomVertex, u16>::new());
        cache.insert_vertex_buffers(7, Arc::clone(&shared_vertex_buffers));

        cache.end_frame();

        let cached_vertex_buffers = cache.get_vertex_buffers(&7).unwrap();
        assert!(Arc::ptr_eq(&shared_vertex_buffers, &cached_vertex_buffers));

        cache.end_frame();

        let cached_vertex_buffers = cache.get_vertex_buffers(&7).unwrap();
        assert!(Arc::ptr_eq(&shared_vertex_buffers, &cached_vertex_buffers));
    }

    #[test]
    fn cache_drops_entries_not_used_for_a_frame() {
        let mut cache = Cache::new(NonZeroUsize::new(4).unwrap());
        let shared_vertex_buffers = Arc::new(VertexBuffers::<CustomVertex, u16>::new());
        cache.insert_vertex_buffers(7, shared_vertex_buffers);

        cache.end_frame();
        cache.end_frame();

        assert!(cache.get_vertex_buffers(&7).is_none());
    }

    #[test]
    fn cache_refresh_keeps_rendered_geometry_available_next_frame() {
        let mut cache = Cache::new(NonZeroUsize::new(4).unwrap());
        let shared_vertex_buffers = Arc::new(VertexBuffers::<CustomVertex, u16>::new());

        cache.refresh_vertex_buffers(7, &shared_vertex_buffers);
        cache.end_frame();

        let cached_vertex_buffers = cache.get_vertex_buffers(&7).unwrap();
        assert!(Arc::ptr_eq(&shared_vertex_buffers, &cached_vertex_buffers));
    }
}
