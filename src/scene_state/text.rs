use super::SceneState;
use crate::core::{CachedShapeHandle, GlyphContent, PathShape, TextSource};
use lyon::path::FillRule;
use std::any;

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
        if let Ok(shape) = self.loaded_shape(geometry_id) {
            return (!shape.vertex_buffers().vertices.is_empty()).then_some(geometry_id);
        }
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
        self.context
            .loaded_shapes
            .write()
            .expect("shared shape cache lock poisoned")
            .insert(geometry_id, shape);
        has_geometry.then_some(geometry_id)
    }
}
