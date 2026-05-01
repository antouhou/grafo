use crate::vertex::CustomVertex;
use ahash::{HashMap, HashMapExt};
use lyon::tessellation::VertexBuffers;
use std::num::NonZeroUsize;
use std::sync::Arc;

#[derive(Debug)]
pub(crate) struct CachedTessellation {
    pub(crate) vertex_buffers: Arc<VertexBuffers<CustomVertex, u16>>,
    pub(crate) local_bounds: [(f32, f32); 2],
    pub(crate) texture_mapping_size: [f32; 2],
}

pub(crate) struct Cache {
    previous_frame: HashMap<u64, Arc<CachedTessellation>>,
    current_frame: HashMap<u64, Arc<CachedTessellation>>,
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
    ) -> Option<Arc<CachedTessellation>> {
        if let Some(tessellation) = self.current_frame.get(cache_key) {
            return Some(Arc::clone(tessellation));
        }

        let tessellation = Arc::clone(self.previous_frame.get(cache_key)?);
        self.current_frame
            .entry(*cache_key)
            .or_insert_with(|| Arc::clone(&tessellation));
        Some(tessellation)
    }

    pub(crate) fn insert_vertex_buffers(
        &mut self,
        cache_key: u64,
        tessellation: Arc<CachedTessellation>,
    ) {
        self.current_frame.insert(cache_key, tessellation);
    }

    pub(crate) fn refresh_vertex_buffers(
        &mut self,
        cache_key: u64,
        tessellation: &Arc<CachedTessellation>,
    ) {
        self.current_frame
            .entry(cache_key)
            .or_insert_with(|| Arc::clone(tessellation));
    }

    pub(crate) fn end_frame(&mut self) {
        std::mem::swap(&mut self.previous_frame, &mut self.current_frame);
        self.current_frame.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::{Cache, CachedTessellation};
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
        cache.insert_vertex_buffers(
            7,
            Arc::new(CachedTessellation {
                vertex_buffers: shared_vertex_buffers.clone(),
                local_bounds: [(0.0, 0.0), (1.0, 1.0)],
                texture_mapping_size: [1.0, 1.0],
            }),
        );

        let cached_vertex_buffers = cache.get_vertex_buffers(&7).unwrap();
        assert!(Arc::ptr_eq(
            &shared_vertex_buffers,
            &cached_vertex_buffers.vertex_buffers
        ));
    }

    #[test]
    fn cache_promotes_previous_frame_hits_into_current_frame() {
        let mut cache = Cache::new(NonZeroUsize::new(4).unwrap());
        let shared_vertex_buffers = Arc::new(VertexBuffers::<CustomVertex, u16>::new());
        cache.insert_vertex_buffers(
            7,
            Arc::new(CachedTessellation {
                vertex_buffers: Arc::clone(&shared_vertex_buffers),
                local_bounds: [(0.0, 0.0), (1.0, 1.0)],
                texture_mapping_size: [1.0, 1.0],
            }),
        );

        cache.end_frame();

        let cached_vertex_buffers = cache.get_vertex_buffers(&7).unwrap();
        assert!(Arc::ptr_eq(
            &shared_vertex_buffers,
            &cached_vertex_buffers.vertex_buffers
        ));

        cache.end_frame();

        let cached_vertex_buffers = cache.get_vertex_buffers(&7).unwrap();
        assert!(Arc::ptr_eq(
            &shared_vertex_buffers,
            &cached_vertex_buffers.vertex_buffers
        ));
    }

    #[test]
    fn cache_drops_entries_not_used_for_a_frame() {
        let mut cache = Cache::new(NonZeroUsize::new(4).unwrap());
        let shared_vertex_buffers = Arc::new(VertexBuffers::<CustomVertex, u16>::new());
        cache.insert_vertex_buffers(
            7,
            Arc::new(CachedTessellation {
                vertex_buffers: shared_vertex_buffers,
                local_bounds: [(0.0, 0.0), (1.0, 1.0)],
                texture_mapping_size: [1.0, 1.0],
            }),
        );

        cache.end_frame();
        cache.end_frame();

        assert!(cache.get_vertex_buffers(&7).is_none());
    }

    #[test]
    fn cache_refresh_keeps_rendered_geometry_available_next_frame() {
        let mut cache = Cache::new(NonZeroUsize::new(4).unwrap());
        let shared_vertex_buffers = Arc::new(VertexBuffers::<CustomVertex, u16>::new());

        cache.refresh_vertex_buffers(
            7,
            &Arc::new(CachedTessellation {
                vertex_buffers: shared_vertex_buffers.clone(),
                local_bounds: [(0.0, 0.0), (1.0, 1.0)],
                texture_mapping_size: [1.0, 1.0],
            }),
        );
        cache.end_frame();

        let cached_vertex_buffers = cache.get_vertex_buffers(&7).unwrap();
        assert!(Arc::ptr_eq(
            &shared_vertex_buffers,
            &cached_vertex_buffers.vertex_buffers
        ));
    }
}
