use super::{Cache, CachedTessellation, FrameCache};
use crate::core::vertex::CustomVertex;
use lyon::tessellation::VertexBuffers;
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
    drop(cache.end_frame());
    assert!(cache.get(&7).is_some());
    drop(cache.end_frame());
    assert!(cache.get(&7).is_some());
    drop(cache.end_frame());

    assert_eq!(cache.len(), 1);
    drop(cache.end_frame());
    assert_eq!(cache.len(), 0);
    assert_eq!(*drops.lock().unwrap(), 1);
}

#[test]
fn frame_cache_retain_filters_both_generations() {
    let mut cache = FrameCache::new();
    cache.insert(1, "one");
    drop(cache.end_frame());
    cache.insert(2, "two");

    cache.retain(|cache_key, _| *cache_key == 2);

    assert!(cache.get(&1).is_none());
    assert_eq!(cache.get(&2), Some("two"));
}

#[test]
fn eviction_yields_only_unused_entries_once() {
    let mut cache = FrameCache::new();
    cache.insert("shared", 1);
    cache.insert("unused", 2);
    assert!(cache.end_frame().eq([]));

    assert_eq!(cache.get(&"shared"), Some(1));
    assert_eq!(cache.get(&"shared"), Some(1));
    assert!(cache.end_frame().eq([("unused", 2)]));

    assert_eq!(cache.get(&"shared"), Some(1));
    assert!(cache.end_frame().eq([]));
    assert!(cache.end_frame().eq([("shared", 1)]));
    assert!(cache.end_frame().eq([]));
}

#[test]
fn dropping_partial_eviction_keeps_live_entries_and_reuses_storage() {
    let drops = Arc::new(Mutex::new(0));
    let mut cache = FrameCache::new();
    for key in 0..3 {
        cache.insert(key, Arc::new(DropCounter(Arc::clone(&drops))));
    }
    let initial_capacity = cache.current_frame.capacity();
    drop(cache.end_frame());
    assert!(cache.get(&0).is_some());
    let retained_capacity = cache.current_frame.capacity();

    let mut evicted = cache.end_frame();
    let (key, value) = evicted.next().unwrap();
    assert_ne!(key, 0);
    drop(value);
    assert_eq!(*drops.lock().unwrap(), 1);
    drop(evicted);

    assert_eq!(*drops.lock().unwrap(), 2);
    assert_eq!(cache.len(), 1);
    assert_eq!(cache.previous_frame.capacity(), retained_capacity);
    assert_eq!(cache.current_frame.capacity(), initial_capacity);
    assert!(cache.get(&0).is_some());
    assert!(cache.get(&1).is_none());
    assert!(cache.get(&2).is_none());

    drop(cache.end_frame());
    drop(cache.end_frame());
    assert_eq!(*drops.lock().unwrap(), 3);
}

#[test]
fn cache_returns_shared_arc_without_cloning_vertex_buffers() {
    let mut cache = Cache::new();
    let mut vertex_buffers = VertexBuffers::<CustomVertex, u16>::new();
    vertex_buffers.vertices.push(CustomVertex {
        position: [0.0, 0.0],
        tex_coords: [0.0, 0.0],
        normal: [0.0, 0.0],
        coverage: 1.0,
    });
    vertex_buffers.indices.push(0);

    let shared_vertex_buffers = Arc::new(vertex_buffers);
    cache.insert_tessellation(
        7,
        Arc::new(CachedTessellation {
            vertex_buffers: shared_vertex_buffers.clone(),
            local_bounds: [(0.0, 0.0), (1.0, 1.0)],
            texture_mapping_size: [1.0, 1.0],
        }),
    );

    let cached_tessellation = cache.get_tessellation(&7).unwrap();
    assert!(Arc::ptr_eq(
        &shared_vertex_buffers,
        &cached_tessellation.vertex_buffers
    ));
}

#[test]
fn cache_promotes_previous_frame_hits_into_current_frame() {
    let mut cache = Cache::new();
    let shared_vertex_buffers = Arc::new(VertexBuffers::<CustomVertex, u16>::new());
    cache.insert_tessellation(
        7,
        Arc::new(CachedTessellation {
            vertex_buffers: Arc::clone(&shared_vertex_buffers),
            local_bounds: [(0.0, 0.0), (1.0, 1.0)],
            texture_mapping_size: [1.0, 1.0],
        }),
    );

    cache.end_frame();

    let cached_tessellation = cache.get_tessellation(&7).unwrap();
    assert!(Arc::ptr_eq(
        &shared_vertex_buffers,
        &cached_tessellation.vertex_buffers
    ));

    cache.end_frame();

    let cached_tessellation = cache.get_tessellation(&7).unwrap();
    assert!(Arc::ptr_eq(
        &shared_vertex_buffers,
        &cached_tessellation.vertex_buffers
    ));
}

#[test]
fn cache_drops_entries_not_used_for_a_frame() {
    let mut cache = Cache::new();
    let shared_vertex_buffers = Arc::new(VertexBuffers::<CustomVertex, u16>::new());
    cache.insert_tessellation(
        7,
        Arc::new(CachedTessellation {
            vertex_buffers: shared_vertex_buffers,
            local_bounds: [(0.0, 0.0), (1.0, 1.0)],
            texture_mapping_size: [1.0, 1.0],
        }),
    );

    cache.end_frame();
    cache.end_frame();

    assert!(cache.get_tessellation(&7).is_none());
}

#[test]
fn cache_refresh_keeps_rendered_geometry_available_next_frame() {
    let mut cache = Cache::new();
    let shared_vertex_buffers = Arc::new(VertexBuffers::<CustomVertex, u16>::new());

    cache.refresh_tessellation(
        7,
        &Arc::new(CachedTessellation {
            vertex_buffers: shared_vertex_buffers.clone(),
            local_bounds: [(0.0, 0.0), (1.0, 1.0)],
            texture_mapping_size: [1.0, 1.0],
        }),
    );
    cache.end_frame();

    let cached_tessellation = cache.get_tessellation(&7).unwrap();
    assert!(Arc::ptr_eq(
        &shared_vertex_buffers,
        &cached_tessellation.vertex_buffers
    ));
}
