use crate::vertex::CustomVertex;
use ahash::{HashMap, HashMapExt};
use lyon::tessellation::VertexBuffers;
use std::hash::Hash;
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

    pub(crate) fn clear(&mut self) {
        self.previous_frame.clear();
        self.current_frame.clear();
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
    pub(crate) fn new() -> Self {
        Self {
            entries: FrameCache::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub(crate) fn get_tessellation(&mut self, cache_key: &u64) -> Option<Arc<CachedTessellation>> {
        self.entries.get(cache_key)
    }

    pub(crate) fn insert_tessellation(
        &mut self,
        cache_key: u64,
        tessellation: Arc<CachedTessellation>,
    ) {
        self.entries.insert(cache_key, tessellation);
    }

    pub(crate) fn refresh_tessellation(
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
mod tests;
