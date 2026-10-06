use super::damage::{mark_dirty, mark_shape_effect_dirty};
use super::{EffectError, RenderBackend, Renderer};
use crate::commands::ShapeDrawId;
use crate::core::effect::{BackdropEffectConfig, ShapeEffectBounds, ShapeEffectConfig};
use crate::scene::effects::{EffectAttachment, EffectInstance};
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

    pub fn set_group_effect(
        &mut self,
        node_id: usize,
        effect_id: u64,
        params: &[u8],
    ) -> Result<(), EffectError<B::Error>> {
        self.scene.shape(node_id)?;
        self.backend
            .validate_effect_params(effect_id, params)
            .map_err(EffectError::Backend)?;
        let parameters = self.planner.store_effect_parameters(params);
        Ok(self
            .scene
            .set_group_effect(node_id, effect_id, parameters)?)
    }

    pub fn update_group_effect_params(
        &mut self,
        node_id: usize,
        params: &[u8],
    ) -> Result<(), EffectError<B::Error>> {
        let effect = self.scene.group_effect(node_id)?;
        self.backend
            .validate_effect_params(effect.effect_id, params)
            .map_err(EffectError::Backend)?;
        let parameters = self
            .planner
            .update_effect_parameters(effect.parameters, params);
        Ok(self.scene.update_group_effect_params(node_id, parameters)?)
    }

    pub fn remove_group_effect(&mut self, node_id: usize) {
        self.scene.remove_group_effect(node_id);
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

    pub fn update_backdrop_effect_config(
        &mut self,
        node_id: usize,
        config: BackdropEffectConfig,
    ) -> Result<(), EffectError<B::Error>> {
        self.scene.update_backdrop_effect_config(
            node_id,
            config,
            self.backend.maximum_texture_dimension(),
        )?;
        // Damage expansion performed before planning will include the capture region
        self.mark_shape_dirty(node_id);
        Ok(())
    }

    pub fn update_backdrop_effect_params(
        &mut self,
        node_id: usize,
        params: &[u8],
    ) -> Result<(), EffectError<B::Error>> {
        let effect = self.scene.backdrop_effect(node_id)?;
        self.backend
            .validate_effect_params(effect.effect_id, params)
            .map_err(EffectError::Backend)?;
        let parameters = self
            .planner
            .update_effect_parameters(effect.parameters, params);
        self.scene
            .update_backdrop_effect_params(node_id, parameters)?;
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
        self.scene
            .set_shape_effect(node_id, effect_id, parameters, config)?;
        self.backend.set_shape_effect_geometry(
            ShapeDrawId(node_id),
            &self.scene.shape(node_id)?.cached_shape,
        );
        if let Some(bounds) = old_bounds {
            self.mark_shape_effect_dirty(bounds);
        }
        self.mark_shape_effect_dirty(self.scene.shape_effect(node_id)?.bounds);
        Ok(())
    }

    pub fn update_shape_effect_params(
        &mut self,
        node_id: usize,
        params: &[u8],
    ) -> Result<(), EffectError<B::Error>> {
        let effect = self.scene.shape_effect(node_id)?;
        let bounds = effect.bounds;
        self.backend
            .validate_effect_params(effect.effect_id, params)
            .map_err(EffectError::Backend)?;
        let parameters = self
            .planner
            .update_effect_parameters(effect.parameters, params);
        self.scene.update_shape_effect_params(node_id, parameters)?;
        self.mark_shape_effect_dirty(bounds);
        Ok(())
    }

    pub fn update_shape_effect_config(
        &mut self,
        node_id: usize,
        config: ShapeEffectConfig,
    ) -> Result<(), EffectError<B::Error>> {
        let old_bounds = self.scene.shape_effect(node_id)?.bounds;
        self.scene.update_shape_effect_config(node_id, config)?;
        self.mark_shape_effect_dirty(old_bounds);
        self.mark_shape_effect_dirty(self.scene.shape_effect(node_id)?.bounds);
        Ok(())
    }

    pub fn remove_shape_effect(&mut self, node_id: usize) {
        let bounds = self
            .scene
            .shape_effects
            .get(&node_id)
            .map(|effect| effect.bounds);
        self.scene.remove_shape_effect(node_id);
        if let Some(bounds) = bounds {
            self.mark_shape_effect_dirty(bounds);
        }
        self.backend.remove_shape_effect(ShapeDrawId(node_id));
    }

    pub fn unload_effect(&mut self, effect_id: u64) {
        self.backend.unload_effect(effect_id);
        self.remove_effect_attachments(effect_id);
    }

    fn remove_effect_attachments(&mut self, effect_id: u64) {
        let viewport = self.scene.viewport();
        let fringe_width = self.scene.fringe_width();
        self.scene
            .remove_effect_attachments(effect_id, |node_id, attachment| match attachment {
                EffectAttachment::Backdrop { shape_bounds } => {
                    if let Some(shape_bounds) = shape_bounds {
                        mark_dirty(&mut self.dirty_bounds, shape_bounds, viewport, fringe_width);
                    }
                    self.backend.remove_backdrop_effect(ShapeDrawId(node_id));
                }
                EffectAttachment::Shape(bounds) => {
                    mark_shape_effect_dirty(&mut self.dirty_bounds, bounds, viewport);
                    self.backend.remove_shape_effect(ShapeDrawId(node_id));
                }
            });
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
