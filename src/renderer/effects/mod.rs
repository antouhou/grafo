use super::damage::{mark_dirty, mark_physical_dirty, mark_shape_effect_dirty};
use super::{EffectError, RenderBackend, Renderer};
use crate::commands::ShapeDrawId;
use crate::core::effect::{
    BackdropEffectConfig, GroupEffectConfig, ShapeEffectBounds, ShapeEffectConfig,
};
use crate::scene::effects::{EffectAttachmentKind, EffectInstance};
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

    /// Processes the rendered group within the attached shape's bounds.
    /// Includes any attached shape-effect output in the input bounds.
    /// Outsets reserve transparent space for output beyond those bounds.
    pub fn set_group_effect(
        &mut self,
        node_id: usize,
        effect_id: u64,
        params: &[u8],
        config: GroupEffectConfig,
    ) -> Result<(), EffectError<B::Error>> {
        self.scene.shape(node_id)?;
        self.backend
            .validate_effect_params(effect_id, params)
            .map_err(EffectError::Backend)?;
        let parameters = self.planner.store_effect_parameters(params);
        let previous_output = self.scene.group_effect_output(node_id);
        self.scene.set_group_effect(
            node_id,
            effect_id,
            parameters,
            config,
            self.backend.maximum_texture_dimension(),
        )?;
        mark_physical_dirty(&mut self.dirty_bounds, previous_output);
        mark_physical_dirty(
            &mut self.dirty_bounds,
            self.scene.group_effect_output(node_id),
        );
        self.mark_shape_dirty(node_id);
        Ok(())
    }

    pub fn remove_group_effect(&mut self, node_id: usize) {
        let previous_output = self.scene.group_effect_output(node_id);
        if self.scene.remove_group_effect(node_id) {
            mark_physical_dirty(&mut self.dirty_bounds, previous_output);
            self.mark_shape_dirty(node_id);
        }
    }

    pub fn set_shape_backdrop_effect(
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
        let parameters = self.planner.store_effect_parameters(params);
        self.scene.set_shape_backdrop_effect(
            node_id,
            EffectInstance {
                effect_id,
                parameters,
            },
            config,
            self.backend.maximum_texture_dimension(),
        )?;
        // Damage expansion performed before planning will include the capture region
        self.mark_shape_dirty(node_id);
        Ok(())
    }

    pub fn remove_backdrop_effect(&mut self, node_id: usize) {
        if self.scene.remove_backdrop_effect(node_id) {
            self.mark_shape_dirty(node_id);
        }
        self.backend.remove_backdrop_effect(ShapeDrawId(node_id));
    }
    /// Uses shared CPU geometry to identify and prepare the node's coverage mask.
    pub fn set_shape_effect(
        &mut self,
        node_id: usize,
        effect_id: u64,
        params: &[u8],
        config: ShapeEffectConfig,
    ) -> Result<(), EffectError<B::Error>> {
        self.scene.shape(node_id)?;
        self.backend
            .validate_effect_params(effect_id, params)
            .map_err(EffectError::Backend)?;
        let parameters = self.planner.store_effect_parameters(params);
        let old_bounds = self
            .scene
            .shape_effects
            .get(&node_id)
            .map(|effect| effect.bounds);
        let previous_group_output = self.scene.group_effect_output(node_id);
        self.scene.set_shape_effect(
            node_id,
            effect_id,
            parameters,
            config,
            self.backend.maximum_texture_dimension(),
        )?;
        self.backend.set_shape_effect_geometry(
            ShapeDrawId(node_id),
            &self.scene.shape(node_id)?.cached_shape,
        );
        if let Some(bounds) = old_bounds {
            self.mark_shape_effect_dirty(bounds);
        }
        self.mark_shape_effect_dirty(self.scene.shape_effect(node_id)?.bounds);
        mark_physical_dirty(&mut self.dirty_bounds, previous_group_output);
        mark_physical_dirty(
            &mut self.dirty_bounds,
            self.scene.group_effect_output(node_id),
        );
        Ok(())
    }

    pub fn remove_shape_effect(&mut self, node_id: usize) {
        let previous_group_output = self.scene.group_effect_output(node_id);
        let bounds = self
            .scene
            .shape_effects
            .get(&node_id)
            .map(|effect| effect.bounds);
        self.scene
            .remove_shape_effect(node_id, self.backend.maximum_texture_dimension());
        if let Some(bounds) = bounds {
            self.mark_shape_effect_dirty(bounds);
        }
        self.backend.remove_shape_effect(ShapeDrawId(node_id));
        if bounds.is_some() {
            mark_physical_dirty(&mut self.dirty_bounds, previous_group_output);
            mark_physical_dirty(
                &mut self.dirty_bounds,
                self.scene.group_effect_output(node_id),
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
                    EffectAttachmentKind::Group => {}
                    EffectAttachmentKind::Backdrop => {
                        self.backend
                            .remove_backdrop_effect(ShapeDrawId(attachment.node_id));
                    }
                    EffectAttachmentKind::Shape => {
                        self.backend
                            .remove_shape_effect(ShapeDrawId(attachment.node_id));
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

    fn mark_shape_effect_dirty(&mut self, bounds: ShapeEffectBounds) {
        mark_shape_effect_dirty(&mut self.dirty_bounds, bounds, self.scene.viewport());
    }
}
