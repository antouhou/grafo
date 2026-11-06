use super::{Cache, CachedTessellation, FrameCache};
use crate::core::shape::{BorderRadii, LineCap, Shape, StrokeOptions};
use crate::core::util::ShapeResources;
use crate::core::vertex::CustomVertex;
use lyon::tessellation::{GeometryBuilderError, TessellationError};
use std::mem;
use std::sync::{Arc, Mutex};

struct DropCounter(Arc<Mutex<usize>>);

impl Drop for DropCounter {
    fn drop(&mut self) {
        *self.0.lock().unwrap() += 1;
    }
}

fn tessellation() -> Arc<CachedTessellation> {
    let mut tessellation = CachedTessellation::empty();
    tessellation.vertex_buffers.vertices.push(CustomVertex {
        position: [0.0, 0.0],
        tex_coords: [0.0, 0.0],
    });
    tessellation.vertex_buffers.indices.push(0);
    Arc::new(tessellation)
}

fn allocation_addresses(tessellation: &CachedTessellation) -> [usize; 3] {
    [
        tessellation.vertex_buffers.vertices.as_ptr() as usize,
        tessellation.vertex_buffers.indices.as_ptr() as usize,
        tessellation.boundary_edges.as_ptr() as usize,
    ]
}

fn capacities(tessellation: &CachedTessellation) -> [usize; 3] {
    [
        tessellation.vertex_buffers.vertices.capacity(),
        tessellation.vertex_buffers.indices.capacity(),
        tessellation.boundary_edges.capacity(),
    ]
}

fn geometry_variants() -> [Shape; 4] {
    [
        Shape::builder()
            .begin((0.0, 0.0))
            .line_to((12.0, 0.0))
            .quadratic_bezier_to((15.0, 12.0), (0.0, 8.0))
            .close()
            .build(),
        Shape::rect([(20.0, 30.0), (50.0, 60.0)]),
        Shape::rounded_rect([(70.0, 80.0), (120.0, 110.0)], BorderRadii::new(8.0)),
        Shape::builder()
            .begin((130.0, 140.0))
            .cubic_bezier_to((150.0, 110.0), (160.0, 180.0), (180.0, 140.0))
            .end()
            .build()
            .stroke(StrokeOptions::new(4.0).line_cap(LineCap::Round)),
    ]
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

    assert_eq!(cache.previous_frame.len(), 1);
    assert!(cache.current_frame.is_empty());
    drop(cache.end_frame());
    assert!(cache.previous_frame.is_empty());
    assert!(cache.current_frame.is_empty());
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
    assert_eq!(cache.previous_frame.len(), 1);
    assert!(cache.current_frame.is_empty());
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
    let shared_tessellation = tessellation();
    cache.insert_tessellation(7, Arc::clone(&shared_tessellation));

    let cached_tessellation = cache.get_tessellation(&7).unwrap();
    assert!(Arc::ptr_eq(&shared_tessellation, &cached_tessellation));
}

#[test]
fn cache_promotes_previous_frame_hits_into_current_frame() {
    let mut cache = Cache::new();
    let shared_tessellation = tessellation();
    cache.insert_tessellation(7, Arc::clone(&shared_tessellation));

    cache.end_frame();

    let cached_tessellation = cache.get_tessellation(&7).unwrap();
    assert!(Arc::ptr_eq(&shared_tessellation, &cached_tessellation));

    cache.end_frame();

    let cached_tessellation = cache.get_tessellation(&7).unwrap();
    assert!(Arc::ptr_eq(&shared_tessellation, &cached_tessellation));
}

#[test]
fn cache_drops_entries_not_used_for_a_frame() {
    let mut cache = Cache::new();
    cache.insert_tessellation(7, tessellation());

    cache.end_frame();
    cache.end_frame();

    assert!(cache.get_tessellation(&7).is_none());
}

#[test]
fn cache_refresh_keeps_rendered_geometry_available_next_frame() {
    let mut cache = Cache::new();
    let shared_tessellation = tessellation();
    cache.refresh_tessellation(7, &shared_tessellation);
    cache.end_frame();

    let cached_tessellation = cache.get_tessellation(&7).unwrap();
    assert!(Arc::ptr_eq(&shared_tessellation, &cached_tessellation));
}

#[test]
fn failed_stroke_returns_storage_for_the_next_attempt() {
    let mut resources = ShapeResources::new();
    let valid = Shape::builder()
        .begin((0.0, 0.0))
        .line_to((10.0, 0.0))
        .end()
        .build()
        .stroke(StrokeOptions::new(4.0).line_cap(LineCap::Square));
    let geometry = valid.tessellate(&mut resources, None).unwrap();
    let original_addresses = allocation_addresses(&geometry);
    let original_capacities = capacities(&geometry);
    resources.tessellation_cache.release_tessellation(geometry);

    let overflowing = Shape::builder()
        .begin((f32::MAX, 0.0))
        .line_to((f32::MAX, 0.0))
        .end()
        .build()
        .stroke(StrokeOptions::new(f32::MAX).line_cap(LineCap::Square));
    assert!(matches!(
        overflowing.tessellate(&mut resources, None),
        Err(TessellationError::GeometryBuilder(
            GeometryBuilderError::InvalidVertex
        ))
    ));
    let reused = valid.tessellate(&mut resources, None).unwrap();
    assert_eq!(allocation_addresses(&reused), original_addresses);
    assert_eq!(capacities(&reused), original_capacities);
    assert!(!reused.vertex_buffers.indices.is_empty());
    assert!(reused.vertex_buffers.vertices.iter().all(|vertex| {
        vertex
            .position
            .into_iter()
            .chain(vertex.tex_coords)
            .all(f32::is_finite)
    }));
}

#[test]
fn uncached_replacements_reuse_geometry_storage_after_warmup() {
    let mut resources = ShapeResources::new();
    let shapes = geometry_variants();
    let mut current = shapes[0].tessellate(&mut resources, None).unwrap();
    for _ in 0..2 {
        for shape in &shapes {
            let replacement = shape.tessellate(&mut resources, None).unwrap();
            resources
                .tessellation_cache
                .release_tessellation(mem::replace(&mut current, replacement));
        }
    }
    let spare = resources.tessellation_cache.acquire_tessellation();
    let mut expected_current = (allocation_addresses(&current), capacities(&current));
    let mut expected_spare = (allocation_addresses(&spare), capacities(&spare));
    resources.tessellation_cache.recycle_tessellation(spare);
    let return_capacity = resources
        .tessellation_cache
        .available_tessellations
        .capacity();

    for _ in 0..4 {
        for shape in &shapes {
            let previous_vertex = current.vertex_buffers.vertices[0].position;
            let replacement = shape.tessellate(&mut resources, None).unwrap();
            assert_eq!(
                (allocation_addresses(&replacement), capacities(&replacement)),
                expected_spare
            );
            assert_eq!(allocation_addresses(&current), expected_current.0);
            assert_eq!(current.vertex_buffers.vertices[0].position, previous_vertex);
            resources
                .tessellation_cache
                .release_tessellation(mem::replace(&mut current, replacement));
            mem::swap(&mut expected_current, &mut expected_spare);
            assert_eq!(
                resources.tessellation_cache.available_tessellations.len(),
                1
            );
            assert_eq!(
                resources
                    .tessellation_cache
                    .available_tessellations
                    .capacity(),
                return_capacity
            );
        }
    }
    resources.tessellation_cache.release_tessellation(current);
    assert_eq!(
        resources.tessellation_cache.available_tessellations.len(),
        2
    );
    assert_eq!(
        resources
            .tessellation_cache
            .available_tessellations
            .capacity(),
        return_capacity
    );
}

#[test]
fn cache_eviction_preserves_shared_geometry_until_explicit_release() {
    let mut resources = ShapeResources::new();
    let shape = Shape::rect([(0.0, 0.0), (20.0, 10.0)]).stroke(StrokeOptions::new(2.0));
    let external = shape.tessellate(&mut resources, Some(91)).unwrap();
    let original_addresses = allocation_addresses(&external);
    let original_capacities = capacities(&external);
    resources.tessellation_cache.end_frame();
    resources.tessellation_cache.end_frame();
    assert!(resources.tessellation_cache.get_tessellation(&91).is_none());
    assert!(resources
        .tessellation_cache
        .available_tessellations
        .is_empty());
    assert!(!external.vertex_buffers.indices.is_empty());
    assert_eq!(allocation_addresses(&external), original_addresses);
    resources.tessellation_cache.release_tessellation(external);
    assert_eq!(
        resources.tessellation_cache.available_tessellations.len(),
        1
    );
    let reused = resources.tessellation_cache.acquire_tessellation();
    assert_eq!(allocation_addresses(&reused), original_addresses);
    assert_eq!(capacities(&reused), original_capacities);
    assert!(reused.vertex_buffers.vertices.is_empty());
    assert!(reused.vertex_buffers.indices.is_empty());
    assert!(reused.boundary_edges.is_empty());
}
