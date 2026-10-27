use super::{DrawCommandError, EffectError, Scene};
use crate::core::effect::ShapeMaskEffectBounds;
use crate::render_backend::RenderBackend;
use crate::scene_state::effects::{
    EffectAttachment, EffectAttachmentKind, EffectAttachments, EffectConfig,
};
use crate::scene_state::SceneError;
use std::mem;

pub(super) struct PendingMemberEffectAttachment {
    node_id: usize,
    attachment: EffectAttachment,
}

#[derive(Default)]
pub(super) struct GroupEffectScratch {
    members: Vec<usize>,
    pending_member_effects: Vec<PendingMemberEffectAttachment>,
    pub(super) parameters: Vec<u8>,
    pub(super) removed_effects: Vec<(usize, EffectAttachmentKind)>,
}

impl<B: RenderBackend> Scene<B> {
    fn collect_member_effect_attachments(
        &self,
        node_id: usize,
        effects: EffectAttachments,
        members: &mut Vec<usize>,
        pending: &mut Vec<PendingMemberEffectAttachment>,
    ) -> Result<(), DrawCommandError<B::Error>> {
        for &attachment in effects.iter() {
            self.backend
                .validate_effect_params(
                    attachment.instance.effect_id,
                    self.state
                        .effect_parameters
                        .get(attachment.instance.parameters)
                        .bytes,
                )
                .map_err(DrawCommandError::Backend)?;
            let kind = attachment.config.kind();
            self.state.collect_group_members(node_id, kind, members);
            for &member_id in members.iter() {
                if self
                    .state
                    .node_effects(member_id)
                    .get(kind)
                    .is_some_and(|effect| effect.source_group_id.is_none())
                {
                    continue;
                }
                if let EffectConfig::ShapeMask(config) = attachment.config {
                    let shape = self.state.shape(member_id)?;
                    ShapeMaskEffectBounds::new(
                        shape.cached_shape.local_bounds(),
                        config,
                        shape.transform,
                        self.state.viewport().scale_factor,
                    )
                    .ok_or(SceneError::InvalidShapeMaskEffectBounds(member_id))?;
                }
                pending.push(PendingMemberEffectAttachment {
                    node_id: member_id,
                    attachment,
                });
            }
        }
        Ok(())
    }

    pub(super) fn prepare_inherited_member_effects(
        &mut self,
        node_id: usize,
        effects: EffectAttachments,
    ) -> Result<Vec<PendingMemberEffectAttachment>, DrawCommandError<B::Error>> {
        let mut scratch = mem::take(&mut self.group_effect_scratch);
        let mut pending = mem::take(&mut scratch.pending_member_effects);
        let result = self.collect_member_effect_attachments(
            node_id,
            effects,
            &mut scratch.members,
            &mut pending,
        );
        scratch.members.clear();
        self.group_effect_scratch = scratch;
        if let Err(error) = result {
            self.discard_member_effects(pending);
            return Err(error);
        }
        Ok(pending)
    }

    fn discard_member_effects(&mut self, mut pending: Vec<PendingMemberEffectAttachment>) {
        pending.clear();
        self.group_effect_scratch.pending_member_effects = pending;
    }

    pub(super) fn apply_member_effects(
        &mut self,
        pending: Vec<PendingMemberEffectAttachment>,
    ) -> Result<(), DrawCommandError<B::Error>> {
        let result = pending.iter().try_for_each(|member| {
            let mut effects = EffectAttachments::default();
            *effects.get_mut(member.attachment.config.kind()) = Some(member.attachment);
            self.apply_effects(member.node_id, effects)
        });
        self.discard_member_effects(pending);
        result
    }

    fn set_group_member_effects(
        &mut self,
        group_id: usize,
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
            self.state
                .set_shape_effect_source(member_id, config.kind(), Some(group_id));
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
        scratch.members.retain(|&member_id| {
            self.state
                .node_effects(member_id)
                .get(config.kind())
                .is_none_or(|effect| effect.source_group_id.is_some())
        });
        let result =
            self.set_group_member_effects(node_id, &scratch.members, effect_id, params, config);
        scratch.members.clear();
        self.group_effect_scratch = scratch;
        result?;

        self.state
            .set_group_effect(node_id, effect_id, params, config);
        Ok(())
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

    pub(super) fn remove_group_effect(&mut self, node_id: usize, kind: EffectAttachmentKind) {
        if !self.state.remove_group_effect(node_id, kind) {
            return;
        }
        let mut scratch = mem::take(&mut self.group_effect_scratch);
        self.state
            .collect_group_members(node_id, kind, &mut scratch.members);
        scratch.members.retain(|&member_id| {
            self.state
                .node_effects(member_id)
                .get(kind)
                .is_some_and(|effect| effect.source_group_id == Some(node_id))
        });
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

    pub(super) fn restore_removed_group_effects(&mut self) {
        let mut scratch = mem::take(&mut self.group_effect_scratch);
        for &(node_id, kind) in &scratch.removed_effects {
            self.restore_group_effect(node_id, kind, &mut scratch.parameters);
        }
        scratch.removed_effects.clear();
        scratch.parameters.clear();
        self.group_effect_scratch = scratch;
    }

    pub(super) fn remove_inherited_member_effects(&mut self, node_id: usize) {
        let mut scratch = mem::take(&mut self.group_effect_scratch);
        for kind in [
            EffectAttachmentKind::Content,
            EffectAttachmentKind::Backdrop,
            EffectAttachmentKind::ShapeMask,
        ] {
            self.state
                .collect_group_members(node_id, kind, &mut scratch.members);
            for &member_id in &scratch.members {
                let is_inherited = self
                    .state
                    .node_effects(member_id)
                    .get(kind)
                    .is_some_and(|effect| effect.source_group_id.is_some());
                if is_inherited {
                    match kind {
                        EffectAttachmentKind::Content => self.remove_content_effect(member_id),
                        EffectAttachmentKind::Backdrop => self.remove_backdrop_effect(member_id),
                        EffectAttachmentKind::ShapeMask => self.remove_shape_mask_effect(member_id),
                    }
                }
            }
        }
        scratch.members.clear();
        self.group_effect_scratch = scratch;
    }
}
