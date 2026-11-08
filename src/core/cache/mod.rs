use crate::core::shape::boundary::{PathBoundaryEdge, RoundedRectBoundary};
use crate::core::shape::{LineCap, LineJoin, StrokeOptions};
use crate::core::vertex::CustomVertex;
use ahash::{HashMap, HashMapExt};
use lyon::tessellation::VertexBuffers;
use std::collections::hash_map::Entry;
use std::hash::{Hash, Hasher};
use std::mem;
use std::sync::{Arc, Weak};

#[derive(Debug)]
pub struct CachedTessellation {
    pub(crate) is_rect: bool,
    pub(crate) boundary_edges: Vec<PathBoundaryEdge>,
    pub(crate) vertex_buffers: VertexBuffers<CustomVertex, u32>,
    pub(crate) rounded_rect_boundary: Option<RoundedRectBoundary>,
    pub(crate) local_bounds: [(f32, f32); 2],
    pub(crate) texture_mapping_size: [f32; 2],
}

impl CachedTessellation {
    pub(crate) fn empty() -> Self {
        Self {
            is_rect: false,
            boundary_edges: Vec::new(),
            vertex_buffers: VertexBuffers {
                vertices: Vec::new(),
                indices: Vec::new(),
            },
            rounded_rect_boundary: None,
            local_bounds: [(0.0, 0.0), (1.0, 1.0)],
            texture_mapping_size: [1.0, 1.0],
        }
    }

    fn clear(&mut self) {
        self.vertex_buffers.clear();
        self.boundary_edges.clear();
        self.is_rect = false;
        self.rounded_rect_boundary = None;
        self.local_bounds = [(0.0, 0.0), (1.0, 1.0)];
        self.texture_mapping_size = [1.0, 1.0];
    }
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

    pub(crate) fn remove(&mut self, cache_key: &K) {
        self.current_frame.remove(cache_key);
        self.previous_frame.remove(cache_key);
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

#[derive(Clone, Debug)]
pub(crate) struct StrokeTessellationKey {
    shape: Weak<CachedTessellation>,
    width: u32,
    line_cap: LineCap,
    line_join: LineJoin,
    miter_limit: u32,
}

impl StrokeTessellationKey {
    pub(crate) fn new(shape: &Arc<CachedTessellation>, options: StrokeOptions) -> Self {
        Self {
            shape: Arc::downgrade(shape),
            width: options.width.to_bits(),
            line_cap: options.line_cap,
            line_join: options.line_join,
            miter_limit: options.miter_limit.to_bits(),
        }
    }
}

impl PartialEq for StrokeTessellationKey {
    fn eq(&self, other: &Self) -> bool {
        Weak::ptr_eq(&self.shape, &other.shape)
            && self.width == other.width
            && self.line_cap == other.line_cap
            && self.line_join == other.line_join
            && self.miter_limit == other.miter_limit
    }
}

impl Eq for StrokeTessellationKey {}

impl Hash for StrokeTessellationKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.shape.as_ptr().hash(state);
        self.width.hash(state);
        self.line_cap.hash(state);
        self.line_join.hash(state);
        self.miter_limit.hash(state);
    }
}

pub(crate) struct Cache {
    entries: FrameCache<u64, Arc<CachedTessellation>>,
    strokes: FrameCache<StrokeTessellationKey, Arc<CachedTessellation>>,
    available_tessellations: Vec<CachedTessellation>,
}

impl Cache {
    pub(crate) fn new() -> Self {
        Self {
            entries: FrameCache::new(),
            strokes: FrameCache::new(),
            available_tessellations: Vec::new(),
        }
    }

    pub(crate) fn acquire_tessellation(&mut self) -> CachedTessellation {
        self.available_tessellations
            .pop()
            .unwrap_or_else(CachedTessellation::empty)
    }

    pub(crate) fn recycle_tessellation(&mut self, mut tessellation: CachedTessellation) {
        tessellation.clear();
        self.available_tessellations.push(tessellation);
    }

    pub(crate) fn release_tessellation(&mut self, tessellation: Arc<CachedTessellation>) {
        if let Some(tessellation) = Arc::into_inner(tessellation) {
            self.recycle_tessellation(tessellation);
        }
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

    pub(crate) fn get_stroke(
        &mut self,
        key: &StrokeTessellationKey,
    ) -> Option<Arc<CachedTessellation>> {
        self.strokes.get(key)
    }

    pub(crate) fn insert_stroke(
        &mut self,
        key: StrokeTessellationKey,
        tessellation: Arc<CachedTessellation>,
    ) {
        self.strokes.insert(key, tessellation);
    }

    pub(crate) fn end_frame(&mut self) {
        let fills = self
            .entries
            .end_frame()
            .map(|(_, tessellation)| tessellation);
        let strokes = self
            .strokes
            .end_frame()
            .map(|(_, tessellation)| tessellation);
        for tessellation in fills.chain(strokes) {
            if let Some(mut tessellation) = Arc::into_inner(tessellation) {
                tessellation.clear();
                self.available_tessellations.push(tessellation);
            }
        }
    }
}

#[cfg(test)]
mod tests;
