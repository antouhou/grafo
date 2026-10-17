use super::damage::{mark_dirty, mark_physical_dirty, mark_shape_mask_effect_dirty};
use super::{EffectError, RenderBackend, Renderer};
use crate::commands::ShapeDrawId;
use crate::core::effect::{
    BackdropEffectConfig, ContentEffectConfig, ShapeMaskEffectBounds, ShapeMaskEffectConfig,
};
use crate::scene::effects::EffectAttachmentKind;
use crate::scene::types::DrawTreeNode;

impl<B: RenderBackend> Renderer<B> {
    /// Loads shader passes. Changed sources detach old instances and invalidate cached results.
    pub fn load_effect(
        &mut self,
        effect_id: u64,
        pass_sources: &[&str],
    ) -> Result<(), EffectError<B::Error>> {
        if self
            .backend
            .load_effect(effect_id, pass_sources)
            .map_err(EffectError::Backend)?
        {
            self.remove_effect_attachments(effect_id);
        }
        Ok(())
    }

    /// Sets an effect applied to the node and its descendants.
    ///
    /// Captures all descendants rendered within the shape's bounds and uses that as an input for
    /// the effect. Overflowing children are not included into the capture. The output is drawn in
    /// a place of the shape. You can reserve additional space around the shape for the output via
    /// the effect config, if the output needs to expand original shape.
    pub fn set_content_effect(
        &mut self,
        node_id: usize,
        effect_id: u64,
        params: &[u8],
        config: ContentEffectConfig,
    ) -> Result<(), EffectError<B::Error>> {
        self.scene.shape(node_id)?;
        self.backend
            .validate_effect_params(effect_id, params)
            .map_err(EffectError::Backend)?;
        let previous_output = self.scene.content_effect_output(node_id);
        self.scene.set_content_effect(
            node_id,
            effect_id,
            params,
            config,
            self.backend.maximum_texture_dimension(),
        )?;
        mark_physical_dirty(&mut self.dirty_bounds, previous_output);
        mark_physical_dirty(
            &mut self.dirty_bounds,
            self.scene.content_effect_output(node_id),
        );
        self.mark_shape_dirty(node_id);
        Ok(())
    }

    /// Removes the content effect, restoring the node's ordinary rendering.
    pub fn remove_content_effect(&mut self, node_id: usize) {
        let previous_output = self.scene.content_effect_output(node_id);
        if self.scene.remove_content_effect(node_id) {
            mark_physical_dirty(&mut self.dirty_bounds, previous_output);
            self.mark_shape_dirty(node_id);
        }
    }

    /// A backdrop filter: Filters previously rendered scene pixels behind the node. Perfect fit
    /// for frosted glass background and similar effects.
    ///
    /// Effect config can be used to control capture area and padding around the input, independently
    /// of the node's own config
    pub fn set_backdrop_effect(
        &mut self,
        node_id: usize,
        effect_id: u64,
        params: &[u8],
        config: BackdropEffectConfig,
    ) -> Result<(), EffectError<B::Error>> {
        self.scene.shape(node_id)?;
        self.backend
            .validate_effect_params(effect_id, params)
            .map_err(EffectError::Backend)?;
        self.scene.set_backdrop_effect(
            node_id,
            effect_id,
            params,
            config,
            self.backend.maximum_texture_dimension(),
        )?;
        // Damage expansion performed before planning will include the capture region
        self.mark_shape_dirty(node_id);
        Ok(())
    }

    /// Removes the node's backdrop effect.
    pub fn remove_backdrop_effect(&mut self, node_id: usize) {
        if self.scene.remove_backdrop_effect(node_id) {
            self.mark_shape_dirty(node_id);
        }
        self.backend.remove_backdrop_effect(ShapeDrawId(node_id));
    }

    /// Applies an effect to the shape's white geometric coverage alpha mask. Perfect for things
    /// like box shadows.
    ///
    /// The input is an alpha mask of the shape it is attached to. It excludes the shape's fill,
    /// textures, and children. The result is drawn behind the shape.
    pub fn set_shape_mask_effect(
        &mut self,
        node_id: usize,
        effect_id: u64,
        params: &[u8],
        config: ShapeMaskEffectConfig,
    ) -> Result<(), EffectError<B::Error>> {
        self.scene.shape(node_id)?;
        self.backend
            .validate_effect_params(effect_id, params)
            .map_err(EffectError::Backend)?;
        let old_bounds = self
            .scene
            .shape_mask_effects
            .get(&node_id)
            .map(|effect| effect.bounds);
        let previous_content_output = self.scene.content_effect_output(node_id);
        self.scene.set_shape_mask_effect(
            node_id,
            effect_id,
            params,
            config,
            self.backend.maximum_texture_dimension(),
        )?;
        self.backend.set_shape_mask_effect_geometry(
            ShapeDrawId(node_id),
            &self.scene.shape(node_id)?.cached_shape,
        );
        if let Some(bounds) = old_bounds {
            self.mark_shape_mask_effect_dirty(bounds);
        }
        self.mark_shape_mask_effect_dirty(self.scene.shape_mask_effect(node_id)?.bounds);
        mark_physical_dirty(&mut self.dirty_bounds, previous_content_output);
        mark_physical_dirty(
            &mut self.dirty_bounds,
            self.scene.content_effect_output(node_id),
        );
        Ok(())
    }

    /// Removes the shape's mask effect while keeping its fill and children.
    pub fn remove_shape_mask_effect(&mut self, node_id: usize) {
        let previous_content_output = self.scene.content_effect_output(node_id);
        let bounds = self
            .scene
            .shape_mask_effects
            .get(&node_id)
            .map(|effect| effect.bounds);
        self.scene
            .remove_shape_mask_effect(node_id, self.backend.maximum_texture_dimension());
        if let Some(bounds) = bounds {
            self.mark_shape_mask_effect_dirty(bounds);
        }
        self.backend.remove_shape_mask_effect(ShapeDrawId(node_id));
        if bounds.is_some() {
            mark_physical_dirty(&mut self.dirty_bounds, previous_content_output);
            mark_physical_dirty(
                &mut self.dirty_bounds,
                self.scene.content_effect_output(node_id),
            );
        }
    }

    pub fn unload_effect(&mut self, effect_id: u64) {
        self.remove_effect_attachments(effect_id);
        self.backend.unload_effect(effect_id);
    }

    fn remove_effect_attachments(&mut self, effect_id: u64) {
        self.scene.remove_effect_attachments(
            effect_id,
            self.backend.maximum_texture_dimension(),
            |attachment| {
                mark_physical_dirty(&mut self.dirty_bounds, attachment.damage_bounds);
                match attachment.kind {
                    EffectAttachmentKind::Content => {}
                    EffectAttachmentKind::Backdrop => {
                        self.backend
                            .remove_backdrop_effect(ShapeDrawId(attachment.node_id));
                    }
                    EffectAttachmentKind::ShapeMask => {
                        self.backend
                            .remove_shape_mask_effect(ShapeDrawId(attachment.node_id));
                    }
                }
            },
        );
        self.backend.invalidate_effect(effect_id);
    }

    fn mark_shape_dirty(&mut self, node_id: usize) {
        if let Some(DrawTreeNode::CachedShape(shape)) = self.scene.draw_tree.get(node_id) {
            mark_dirty(
                &mut self.dirty_bounds,
                shape.logical_screen_bounds,
                self.scene.viewport(),
                self.scene.fringe_width(),
            );
        }
    }

    fn mark_shape_mask_effect_dirty(&mut self, bounds: ShapeMaskEffectBounds) {
        mark_shape_mask_effect_dirty(&mut self.dirty_bounds, bounds, self.scene.viewport());
    }
}
