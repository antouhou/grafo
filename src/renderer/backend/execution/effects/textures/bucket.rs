use std::cmp::Ordering;
use std::collections::BinaryHeap;

struct TextureEntry<T> {
    texture_id: u64,
    texture: T,
}

impl<T> PartialEq for TextureEntry<T> {
    fn eq(&self, other: &Self) -> bool {
        self.texture_id == other.texture_id
    }
}

impl<T> Eq for TextureEntry<T> {}

impl<T> PartialOrd for TextureEntry<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for TextureEntry<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        other.texture_id.cmp(&self.texture_id)
    }
}

/// Reuse textures in creation order, independent of their return order.
pub(super) struct TextureBucket<T> {
    available: BinaryHeap<TextureEntry<T>>,
}

impl<T> Default for TextureBucket<T> {
    fn default() -> Self {
        Self {
            available: BinaryHeap::new(),
        }
    }
}

impl<T> TextureBucket<T> {
    pub(super) fn acquire(&mut self) -> Option<T> {
        self.available.pop().map(|entry| entry.texture)
    }

    pub(super) fn recycle(&mut self, texture_id: u64, texture: T) {
        self.available.push(TextureEntry {
            texture_id,
            texture,
        });
    }

    pub(super) fn is_empty(&self) -> bool {
        self.available.is_empty()
    }

    pub(super) fn discard_unused(&mut self) {
        self.available.clear();
    }
}
