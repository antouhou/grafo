use crate::core::vertex::CustomVertex;
use ahash::{HashMap, HashMapExt};
use lyon::tessellation::VertexBuffers;
use std::collections::hash_map::Entry;
use std::hash::Hash;
use std::mem;
use std::sync::Arc;

#[derive(Debug)]
pub struct CachedTessellation {
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
{
    pub(crate) fn new() -> Self {
        Self {
            previous_frame: HashMap::new(),
            current_frame: HashMap::new(),
        }
    }

    #[cfg(any(feature = "render_metrics", test))]
    pub(crate) fn len(&self) -> usize {
        self.previous_frame.len() + self.current_frame.len()
    }

    pub(crate) fn get(&mut self, cache_key: &K) -> Option<V>
    where
        V: Clone,
    {
        self.get_mut(cache_key).cloned()
    }

    pub(crate) fn get_mut(&mut self, cache_key: &K) -> Option<&mut V> {
        match self.current_frame.entry(cache_key.clone()) {
            Entry::Occupied(entry) => Some(entry.into_mut()),
            Entry::Vacant(entry) => {
                let value = self.previous_frame.remove(entry.key())?;
                Some(entry.insert(value))
            }
        }
    }

    pub(crate) fn get_or_insert_with(
        &mut self,
        key: K,
        create: impl FnOnce() -> V,
    ) -> (&mut V, bool) {
        match self.current_frame.entry(key) {
            Entry::Occupied(entry) => (entry.into_mut(), true),
            Entry::Vacant(entry) => {
                let previous = self.previous_frame.remove(entry.key());
                let was_cached = previous.is_some();
                (entry.insert(previous.unwrap_or_else(create)), was_cached)
            }
        }
    }

    pub(crate) fn insert(&mut self, cache_key: K, value: V) {
        self.previous_frame.remove(&cache_key);
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

    /// Yields unused entries. Dropping the iterator discards the remainder.
    pub(crate) fn end_frame(&mut self) -> impl Iterator<Item = (K, V)> + '_ {
        mem::swap(&mut self.previous_frame, &mut self.current_frame);
        self.current_frame.drain()
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

    #[cfg(feature = "render_metrics")]
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
        drop(self.entries.end_frame());
    }
}

#[cfg(test)]
mod tests;
