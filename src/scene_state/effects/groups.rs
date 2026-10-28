use super::{
    EffectAttachment, EffectAttachmentKind, EffectAttachments, EffectConfig, EffectInstance,
};
use crate::scene_state::types::{DrawTreeNode, GroupEffectSources};
use crate::scene_state::SceneState;
use std::mem;

impl SceneState {
    pub(crate) fn set_shape_effect_source(
        &mut self,
        node_id: usize,
        kind: EffectAttachmentKind,
        source_group_id: Option<usize>,
    ) {
        if let Some(DrawTreeNode::CachedShape(shape)) = self.draw_tree.get_mut(node_id) {
            *shape.inherited_effect_sources.get_mut(kind) = source_group_id;
        }
    }

    fn parent_group_effect_sources(&self, node_id: usize) -> GroupEffectSources {
        self.draw_tree
            .parent_index_unchecked(node_id)
            .map(|parent_id| {
                self.draw_tree
                    .get_unchecked(parent_id)
                    .group_effect_sources()
            })
            .unwrap_or_default()
    }

    fn inherited_effect(
        &self,
        source_group_id: Option<usize>,
        kind: EffectAttachmentKind,
    ) -> Option<EffectAttachment> {
        let group_id = source_group_id?;
        let mut attachment = self.node_effects(group_id).get(kind)?;
        attachment.source_group_id = Some(group_id);
        Some(attachment)
    }

    pub(crate) fn inherited_effects(&self, node_id: usize) -> EffectAttachments {
        let sources = self.parent_group_effect_sources(node_id);

        EffectAttachments {
            content: self.inherited_effect(sources.content, EffectAttachmentKind::Content),
            backdrop: self.inherited_effect(sources.backdrop, EffectAttachmentKind::Backdrop),
            shape_mask: self.inherited_effect(sources.shape_mask, EffectAttachmentKind::ShapeMask),
        }
    }

    pub(in crate::scene_state) fn refresh_node_group_effect_sources(
        &mut self,
        node_id: usize,
    ) -> bool {
        let mut sources = self.parent_group_effect_sources(node_id);
        let stored = match self.draw_tree.get_unchecked_mut(node_id) {
            DrawTreeNode::Group(group) => {
                if group.effects.content.is_some() {
                    sources.content = Some(node_id);
                }
                if group.effects.backdrop.is_some() {
                    sources.backdrop = Some(node_id);
                }
                if group.effects.shape_mask.is_some() {
                    sources.shape_mask = Some(node_id);
                }
                &mut group.effect_sources
            }
            DrawTreeNode::ClipRect(clip) => &mut clip.effect_sources,
            DrawTreeNode::CachedShape(_) => return false,
        };
        let has_changed = *stored != sources;
        *stored = sources;
        has_changed
    }

    pub(crate) fn refresh_group_effect_sources(&mut self, node_id: usize) {
        self.refresh_node_group_effect_sources(node_id);
        let mut pending = mem::take(&mut self.group_effect_source_scratch);
        pending.extend_from_slice(self.draw_tree.children(node_id));
        while let Some(child_id) = pending.pop() {
            if self.refresh_node_group_effect_sources(child_id) {
                pending.extend_from_slice(self.draw_tree.children(child_id));
            }
        }
        self.group_effect_source_scratch = pending;
    }

    pub(super) fn remove_group_effect_attachments(&mut self, effect_id: u64) {
        let Some(mut pending_groups) = self.group_effect_index.take_groups(effect_id) else {
            return;
        };
        let mut pending = mem::take(&mut self.group_effect_source_scratch);
        pending.extend(pending_groups.iter().copied());
        while let Some(node_id) = pending.pop() {
            if pending_groups.contains(&node_id) {
                let sources = self.parent_group_effect_sources(node_id);
                // Resolve inherited owners before removing a descendant's override.
                if let Some(parent_source) = sources.iter().find(|id| pending_groups.contains(id)) {
                    pending.push(node_id);
                    pending.push(parent_source);
                    continue;
                }
                pending_groups.remove(&node_id);
                let DrawTreeNode::Group(group) = self.draw_tree.get_unchecked_mut(node_id) else {
                    unreachable!("effect index contains only group nodes");
                };
                group.effects.remove_effect(effect_id);
                self.effect_parameters.request_compaction();
            }
            if self.refresh_node_group_effect_sources(node_id) {
                pending.extend_from_slice(self.draw_tree.children(node_id));
            }
        }
        self.group_effect_source_scratch = pending;
    }

    pub(crate) fn set_group_effect(
        &mut self,
        node_id: usize,
        effect_id: u64,
        params: &[u8],
        config: EffectConfig,
    ) {
        let parameters = self.effect_parameters.store(params);
        let DrawTreeNode::Group(group) = self.draw_tree.get_unchecked_mut(node_id) else {
            unreachable!("effect target was checked before group dispatch");
        };
        let previous = group.effects;
        *group.effects.get_mut(config.kind()) = Some(EffectAttachment {
            instance: EffectInstance {
                effect_id,
                parameters,
            },
            config,
            source_group_id: None,
        });
        self.group_effect_index
            .replace(node_id, previous, group.effects);
        self.effect_parameters.request_compaction();
        self.refresh_group_effect_sources(node_id);
    }

    pub(crate) fn remove_group_effect(
        &mut self,
        node_id: usize,
        kind: EffectAttachmentKind,
    ) -> bool {
        let DrawTreeNode::Group(group) = self.draw_tree.get_unchecked_mut(node_id) else {
            unreachable!("effect target was checked before group dispatch");
        };
        let previous = group.effects;
        if group.effects.get_mut(kind).take().is_none() {
            return false;
        }
        self.group_effect_index
            .replace(node_id, previous, group.effects);
        self.effect_parameters.request_compaction();
        self.refresh_group_effect_sources(node_id);
        true
    }

    pub(crate) fn collect_group_members(
        &self,
        node_id: usize,
        kind: EffectAttachmentKind,
        members: &mut Vec<usize>,
    ) {
        members.clear();
        members.extend_from_slice(self.draw_tree.children(node_id));
        let mut index = 0;
        while let Some(&member_id) = members.get(index) {
            let should_visit_children = match self.draw_tree.get_unchecked(member_id) {
                DrawTreeNode::Group(group) => group.effects.get(kind).is_none(),
                DrawTreeNode::ClipRect(_) => true,
                DrawTreeNode::CachedShape(_) => false,
            };
            if should_visit_children {
                members.extend_from_slice(self.draw_tree.children(member_id));
            }
            index += 1;
        }
        members.retain(|&id| {
            matches!(
                self.draw_tree.get_unchecked(id),
                DrawTreeNode::CachedShape(_)
            )
        });
    }
}
