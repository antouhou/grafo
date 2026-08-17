use crate::vertex::CustomVertex;
use ahash::{HashMap, HashMapExt};
use lyon::tessellation::VertexBuffers;
use std::hash::Hash;
use std::num::NonZeroUsize;
use std::sync::Arc;

#[derive(Debug)]
pub(crate) struct CachedTessellation {
    pub(crate) vertex_buffers: Arc<VertexBuffers<CustomVertex, u16>>,
    pub(crate) local_bounds: [(f32, f32); 2],
    pub(crate) texture_mapping_size: [f32; 2],
}

pub(crate) struct FrameCache<K, V> {
    previous_frame: HashMap<K, V>,
    current_frame: HashMap<K, V>,
}

impl<K, V> FrameCache<K, V>
where
    K: Eq + Hash + Clone,
    V: Clone,
{
    pub(crate) fn new() -> Self {
        Self {
            previous_frame: HashMap::new(),
            current_frame: HashMap::new(),
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.previous_frame.len() + self.current_frame.len()
    }

    pub(crate) fn get(&mut self, cache_key: &K) -> Option<V> {
        if let Some(value) = self.current_frame.get(cache_key) {
            return Some(value.clone());
        }

        let value = self.previous_frame.get(cache_key)?.clone();
        self.current_frame
            .entry(cache_key.clone())
            .or_insert_with(|| value.clone());
        Some(value)
    }

    pub(crate) fn insert(&mut self, cache_key: K, value: V) {
        self.current_frame.insert(cache_key, value);
    }

    pub(crate) fn retain(&mut self, mut predicate: impl FnMut(&K, &mut V) -> bool) {
        self.previous_frame
            .retain(|cache_key, value| predicate(cache_key, value));
        self.current_frame
            .retain(|cache_key, value| predicate(cache_key, value));
    }

    pub(crate) fn end_frame(&mut self) -> usize {
        let collected_entry_count = self
            .previous_frame
            .keys()
            .filter(|cache_key| !self.current_frame.contains_key(*cache_key))
            .count();
        std::mem::swap(&mut self.previous_frame, &mut self.current_frame);
        self.current_frame.clear();
        collected_entry_count
    }
}

pub(crate) struct Cache {
    entries: FrameCache<u64, Arc<CachedTessellation>>,
}

impl Cache {
    pub(crate) fn new(_size: NonZeroUsize) -> Self {
        Self {
            entries: FrameCache::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub(crate) fn get_vertex_buffers(
        &mut self,
        cache_key: &u64,
    ) -> Option<Arc<CachedTessellation>> {
        self.entries.get(cache_key)
    }

    pub(crate) fn insert_vertex_buffers(
        &mut self,
        cache_key: u64,
        tessellation: Arc<CachedTessellation>,
    ) {
        self.entries.insert(cache_key, tessellation);
    }

    pub(crate) fn refresh_vertex_buffers(
        &mut self,
        cache_key: u64,
        tessellation: &Arc<CachedTessellation>,
    ) {
        if self.entries.get(&cache_key).is_none() {
            self.entries.insert(cache_key, Arc::clone(tessellation));
        }
    }

    pub(crate) fn end_frame(&mut self) {
        self.entries.end_frame();
    }
}

#[cfg(test)]
mod tests {
    use super::{Cache, CachedTessellation, FrameCache};
    use crate::vertex::CustomVertex;
    use lyon::tessellation::VertexBuffers;
    use std::num::NonZeroUsize;
    use std::sync::{Arc, Mutex};

    struct DropCounter(Arc<Mutex<usize>>);

    impl Drop for DropCounter {
        fn drop(&mut self) {
            *self.0.lock().unwrap() += 1;
        }
    }

    #[test]
    fn frame_cache_promotes_live_values_and_collects_unreferenced_values() {
        let drops = Arc::new(Mutex::new(0));
        let mut cache = FrameCache::new();
        let value = Arc::new(DropCounter(Arc::clone(&drops)));
        cache.insert(7, Arc::clone(&value));
        drop(value);

        assert!(cache.get(&7).is_some());
        cache.end_frame();
        assert!(cache.get(&7).is_some());
        cache.end_frame();
        assert!(cache.get(&7).is_some());
        cache.end_frame();

        assert_eq!(cache.len(), 1);
        cache.end_frame();
        assert_eq!(cache.len(), 0);
        assert_eq!(*drops.lock().unwrap(), 1);
    }

    #[test]
    fn frame_cache_retain_filters_both_generations() {
        let mut cache = FrameCache::new();
        cache.insert(1, "one");
        cache.end_frame();
        cache.insert(2, "two");

        cache.retain(|cache_key, _| *cache_key == 2);

        assert!(cache.get(&1).is_none());
        assert_eq!(cache.get(&2), Some("two"));
    }

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
