use super::types::DrawTreeNode;
use super::SceneState;
use crate::core::{CachedShapeHandle, GlyphContent, PathShape, TextSource};
use ahash::{HashMap, HashSet};
use lyon::path::FillRule;
use std::any;

#[derive(Default)]
pub(super) struct GlyphUsage {
    counts: HashMap<u64, usize>,
    glyphs_to_remove: HashSet<u64>,
}

impl GlyphUsage {
    fn insert(&mut self, glyph_key: u64) {
        *self.counts.entry(glyph_key).or_default() += 1;
        self.glyphs_to_remove.remove(&glyph_key);
    }

    fn remove(&mut self, glyph_key: u64) {
        let Some(count) = self.counts.get_mut(&glyph_key) else {
            return;
        };
        *count -= 1;
        if *count == 0 {
            self.glyphs_to_remove.insert(glyph_key);
        }
    }

    pub(super) fn remove_node(&mut self, node: &DrawTreeNode) {
        if let DrawTreeNode::CachedShape(shape) = node {
            if let Some(glyph_key) = shape.instance.cached_shape.geometry_id {
                self.remove(glyph_key);
            }
        }
    }

    pub(super) fn forget(&mut self, cache_key: u64) {
        self.counts.remove(&cache_key);
        self.glyphs_to_remove.remove(&cache_key);
    }
}

impl SceneState {
    pub(crate) fn load_glyph<Source: TextSource>(
        &mut self,
        source: &mut Source,
        key: &Source::GlyphKey,
    ) -> Option<u64> {
        let geometry_id = self
            .context
            .glyph_hasher
            .hash_one((any::type_name::<Source::GlyphKey>(), key));
        let mut glyph_usage = self
            .context
            .glyph_usage
            .write()
            .expect("shared glyph usage lock poisoned");
        if let Ok(shape) = self.loaded_shape(geometry_id) {
            if shape.vertex_buffers().vertices.is_empty() {
                return None;
            }
            glyph_usage.insert(geometry_id);
            return Some(geometry_id);
        }
        drop(glyph_usage);
        let Some(GlyphContent::Outline(events)) = source.glyph_content(key) else {
            return None;
        };
        let shape = CachedShapeHandle {
            tessellation: PathShape::tessellate_events(
                events.iter().copied(),
                FillRule::NonZero,
                false,
                &mut self.tessellator,
                &mut self.shape_resources,
                Some(geometry_id),
            ),
            geometry_id: Some(geometry_id),
        };
        let has_geometry = !shape.vertex_buffers().vertices.is_empty();
        let mut glyph_usage = self
            .context
            .glyph_usage
            .write()
            .expect("shared glyph usage lock poisoned");
        self.context
            .loaded_shapes
            .write()
            .expect("shared shape cache lock poisoned")
            .entry(geometry_id)
            .or_insert(shape);
        if has_geometry {
            glyph_usage.insert(geometry_id);
        } else {
            glyph_usage.counts.entry(geometry_id).or_default();
            glyph_usage.glyphs_to_remove.insert(geometry_id);
        }
        has_geometry.then_some(geometry_id)
    }

    pub(crate) fn release_glyph(&mut self, glyph_key: u64) {
        self.context
            .glyph_usage
            .write()
            .expect("shared glyph usage lock poisoned")
            .remove(glyph_key);
    }

    pub(super) fn unload_unused_glyphs(&mut self) {
        let mut glyph_usage = self
            .context
            .glyph_usage
            .write()
            .expect("shared glyph usage lock poisoned");
        if glyph_usage.glyphs_to_remove.is_empty() {
            return;
        }
        let mut loaded_shapes = self
            .context
            .loaded_shapes
            .write()
            .expect("shared shape cache lock poisoned");
        let GlyphUsage {
            counts,
            glyphs_to_remove,
        } = &mut *glyph_usage;
        for glyph_key in glyphs_to_remove.drain() {
            counts.remove(&glyph_key);
            loaded_shapes.remove(&glyph_key);
        }
    }
}

impl Drop for SceneState {
    fn drop(&mut self) {
        if self.text_glyphs.is_empty() {
            return;
        }
        let mut glyph_usage = self
            .context
            .glyph_usage
            .write()
            .expect("shared glyph usage lock poisoned");
        for &node_id in &self.text_glyphs {
            glyph_usage.remove_node(self.draw_tree.get_unchecked(node_id));
        }
    }
}
