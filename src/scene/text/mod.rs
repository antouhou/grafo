use super::{
    draw_commands::{has_child_clip, NodeTarget},
    DrawCommandError, Scene,
};
use crate::commands::ShapeDrawId;
use crate::core::{
    geometry, MathRect, ShapeDrawCommandOptions, Size, TextSource, TransformInstance,
};
use crate::render_backend::RenderBackend;
use crate::scene_state::effects::EffectAttachments;
use crate::scene_state::types::{DrawTreeNode, GroupDrawData};
use crate::scene_state::SceneError;
use std::mem;

#[derive(Default)]
pub(super) struct TextScratch {
    glyph_ids: Vec<usize>,
    removed_glyph_ids: Vec<usize>,
}

impl<B: RenderBackend> Scene<B> {
    pub(super) fn remove_text_glyphs(&mut self, node_id: usize) {
        let mut glyph_ids = mem::take(&mut self.text_scratch.removed_glyph_ids);
        glyph_ids.extend(
            self.state
                .draw_tree
                .children(node_id)
                .iter()
                .copied()
                .filter(|id| self.state.text_glyphs.contains(id)),
        );
        self.remove_subtrees(glyph_ids.iter().copied(), |_| {});
        glyph_ids.clear();
        self.text_scratch.removed_glyph_ids = glyph_ids;
    }

    fn text_visible_bounds(
        &self,
        target: &NodeTarget,
        transform: Option<TransformInstance>,
    ) -> Option<MathRect> {
        let viewport = self.state.viewport();
        let scale = viewport.scale_factor as f32;
        if !scale.is_finite() || scale <= 0.0 {
            return None;
        }
        let mut bounds = MathRect::from_size(Size::from(viewport.physical_size).to_f32()) / scale;
        let replaced = match *target {
            NodeTarget::Insert { .. } => None,
            NodeTarget::Replace { node_id } => Some(node_id),
        };
        let mut ancestor = match *target {
            NodeTarget::Insert { parent } => parent,
            NodeTarget::Replace { node_id } => Some(node_id),
        };
        // TODO: cache that, this is not good
        while let Some(node_id) = ancestor {
            let node = self.state.draw_tree.get(node_id)?;
            if Some(node_id) != replaced
                && has_child_clip(node)
                && node.logical_screen_bounds().is_finite()
            {
                bounds = bounds
                    .intersection(&node.logical_screen_bounds())
                    .unwrap_or_else(MathRect::zero);
            }
            ancestor = self.state.draw_tree.parent_index_unchecked(node_id);
        }
        if bounds.is_empty() {
            return Some(MathRect::zero());
        }
        // Keep geometry contributing antialiasing coverage along the boundary.
        let guard = (1.0 / viewport.scale_factor) as f32;
        geometry::logical_screen_rect_to_local_bounds(bounds.inflate(guard, guard), transform)
    }

    fn queue_glyphs(
        &mut self,
        group_id: usize,
        source: &mut impl TextSource,
        options: ShapeDrawCommandOptions,
        effects: EffectAttachments,
        scratch: &mut TextScratch,
    ) -> Result<(), DrawCommandError<B::Error>> {
        while let Some(glyph) = source.next_glyph() {
            let Some(cache_key) = self.state.load_glyph(source, &glyph.key) else {
                continue;
            };
            let offset = TransformInstance::translation(glyph.position.0, glyph.position.1);
            let transform = options
                .transform
                .map_or(offset, |transform| offset.then(&transform));
            let glyph_id = self.add_cached_shape(
                cache_key,
                Some(group_id),
                options.clone().transform(transform),
            )?;
            scratch.glyph_ids.push(glyph_id);
            self.apply_effects(glyph_id, effects)?;
        }
        Ok(())
    }

    pub(super) fn queue_text(
        &mut self,
        target: NodeTarget,
        mut source: impl TextSource,
        options: ShapeDrawCommandOptions,
    ) -> Result<usize, DrawCommandError<B::Error>> {
        match target {
            NodeTarget::Insert { parent } => self.state.validate_parent(parent)?,
            NodeTarget::Replace { node_id } => {
                self.state
                    .draw_tree
                    .get(node_id)
                    .ok_or(SceneError::NodeNotFound(node_id))?;
            }
        }
        source.set_visible_bounds(self.text_visible_bounds(&target, options.transform));
        let previous_dirty = self.dirty_bounds;
        let group_id = match target {
            NodeTarget::Insert { parent } => self.add_group(parent)?,
            NodeTarget::Replace { node_id } => node_id,
        };
        let mut effects = self.state.node_effects(group_id);
        let glyph_effects = if matches!(
            self.state.draw_tree.get(group_id),
            Some(DrawTreeNode::Group(_))
        ) {
            EffectAttachments::default()
        } else {
            effects
        };
        let mut scratch = mem::take(&mut self.text_scratch);
        scratch.removed_glyph_ids.extend(
            self.state
                .draw_tree
                .children(group_id)
                .iter()
                .copied()
                .filter(|id| self.state.text_glyphs.contains(id)),
        );
        let result = self.queue_glyphs(group_id, &mut source, options, glyph_effects, &mut scratch);
        if result.is_ok() {
            self.remove_subtrees(scratch.removed_glyph_ids.iter().copied(), |_| {});
            if matches!(target, NodeTarget::Replace { .. }) {
                effects.remove_inherited();
                self.backend.unregister_shapes(&[ShapeDrawId(group_id)]);
                self.replace_node(
                    group_id,
                    DrawTreeNode::Group(GroupDrawData {
                        effects,
                        ..Default::default()
                    }),
                    None,
                );
            }
            self.state
                .text_glyphs
                .extend(scratch.glyph_ids.iter().copied());
        } else {
            match target {
                NodeTarget::Insert { .. } => self.remove_subtrees([group_id], |_| {}),
                NodeTarget::Replace { .. } => {
                    self.remove_subtrees(scratch.glyph_ids.iter().copied(), |_| {})
                }
            }
            self.dirty_bounds = previous_dirty;
        }
        scratch.glyph_ids.clear();
        scratch.removed_glyph_ids.clear();
        self.text_scratch = scratch;
        result.map(|()| group_id)
    }
}

#[cfg(all(test, feature = "cosmic-text"))]
mod tests;
