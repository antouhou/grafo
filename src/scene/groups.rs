use super::{EffectError, Scene};
use crate::core::effect::ShapeMaskEffectBounds;
use crate::render_backend::RenderBackend;
use crate::scene_state::effects::{
    EffectAttachment, EffectAttachmentKind, EffectConfig, EffectInstance,
};
use crate::scene_state::types::DrawTreeNode;
use crate::scene_state::SceneError;
use std::mem;

#[derive(Default)]
pub(super) struct GroupEffectScratch {
    members: Vec<usize>,
    pub(super) parameters: Vec<u8>,
    pub(super) removed_effects: Vec<(usize, EffectAttachmentKind)>,
}

impl<B: RenderBackend> Scene<B> {
    fn set_group_member_effects(
        &mut self,
        members: &[usize],
        effect_id: u64,
        params: &[u8],
        config: EffectConfig,
    ) -> Result<(), EffectError<B::Error>> {
        if let EffectConfig::ShapeMask(config) = config {
            for &member_id in members {
                let shape = self.state.shape(member_id)?;
                ShapeMaskEffectBounds::new(
                    shape.cached_shape.local_bounds(),
                    config,
                    shape.transform,
                    self.state.viewport().scale_factor,
                )
                .ok_or(SceneError::InvalidShapeMaskEffectBounds(member_id))?;
            }
        }
        for &member_id in members {
            self.set_effect(member_id, effect_id, params, config)?;
        }
        Ok(())
    }

    pub(super) fn set_group_effect(
        &mut self,
        node_id: usize,
        effect_id: u64,
        params: &[u8],
        config: EffectConfig,
    ) -> Result<(), EffectError<B::Error>> {
        self.backend
            .validate_effect_params(effect_id, params)
            .map_err(EffectError::Backend)?;
        config.validate()?;
        let mut scratch = mem::take(&mut self.group_effect_scratch);
        self.state
            .collect_group_members(node_id, config.kind(), &mut scratch.members);
        let result = self.set_group_member_effects(&scratch.members, effect_id, params, config);
        scratch.members.clear();
        self.group_effect_scratch = scratch;
        result?;

        let parameters = self.state.effect_parameters.store(params);
        let DrawTreeNode::Group(group) = self.state.draw_tree.get_unchecked_mut(node_id) else {
            unreachable!("effect target was checked before group dispatch");
        };
        *group.effects.get_mut(config.kind()) = Some(EffectAttachment {
            instance: EffectInstance {
                effect_id,
                parameters,
            },
            config,
        });
        self.state.effect_parameters.request_compaction();
        Ok(())
    }

    pub(super) fn remove_group_effect(&mut self, node_id: usize, kind: EffectAttachmentKind) {
        let DrawTreeNode::Group(group) = self.state.draw_tree.get_unchecked_mut(node_id) else {
            unreachable!("effect target was checked before group dispatch");
        };
        if group.effects.get_mut(kind).take().is_none() {
            return;
        }
        self.state.effect_parameters.request_compaction();
        let mut scratch = mem::take(&mut self.group_effect_scratch);
        self.state
            .collect_group_members(node_id, kind, &mut scratch.members);
        for &member_id in &scratch.members {
            match kind {
                EffectAttachmentKind::Content => self.remove_content_effect(member_id),
                EffectAttachmentKind::Backdrop => self.remove_backdrop_effect(member_id),
                EffectAttachmentKind::ShapeMask => self.remove_shape_mask_effect(member_id),
            }
            self.restore_group_effect(member_id, kind, &mut scratch.parameters);
        }
        scratch.members.clear();
        scratch.parameters.clear();
        self.group_effect_scratch = scratch;
    }

    fn restore_group_effect(
        &mut self,
        node_id: usize,
        kind: EffectAttachmentKind,
        parameters: &mut Vec<u8>,
    ) {
        if let Some(effect) = self.state.inherited_effects(node_id).get(kind) {
            if self
                .apply_stored_effect(node_id, effect, parameters)
                .is_err()
            {
                tracing::warn!(
                    node_id,
                    effect_id = effect.instance.effect_id,
                    "Could not restore inherited group effect"
                );
            }
        }
    }

    pub(super) fn restore_removed_group_effects(&mut self) {
        let mut scratch = mem::take(&mut self.group_effect_scratch);
        for &(node_id, kind) in &scratch.removed_effects {
            self.restore_group_effect(node_id, kind, &mut scratch.parameters);
        }
        scratch.removed_effects.clear();
        scratch.parameters.clear();
        self.group_effect_scratch = scratch;
    }
}
