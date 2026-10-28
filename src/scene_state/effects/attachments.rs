use super::parameters::EffectParametersHandle;
use super::{validate_backdrop_config, validate_shape_mask_effect_config};
use crate::core::effect::{BackdropEffectConfig, ContentEffectConfig, ShapeMaskEffectConfig};
use crate::core::UnsignedPhysicalRect;
use crate::scene_state::types::{DrawTreeNode, GroupEffectSources};
use crate::scene_state::{SceneError, SceneState};

/// Parameters shared by content and backdrop effect attachments.
#[derive(Clone, Copy, Debug)]
pub(crate) struct EffectInstance {
    /// The loaded effect's ID.
    pub effect_id: u64,
    pub parameters: EffectParametersHandle,
}

#[derive(Clone, Copy)]
pub(crate) enum EffectAttachmentKind {
    Content,
    Backdrop,
    ShapeMask,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum EffectConfig {
    Content(ContentEffectConfig),
    Backdrop(BackdropEffectConfig),
    ShapeMask(ShapeMaskEffectConfig),
}

impl EffectConfig {
    pub(crate) fn kind(self) -> EffectAttachmentKind {
        match self {
            Self::Content(_) => EffectAttachmentKind::Content,
            Self::Backdrop(_) => EffectAttachmentKind::Backdrop,
            Self::ShapeMask(_) => EffectAttachmentKind::ShapeMask,
        }
    }

    pub(crate) fn validate(self) -> Result<(), SceneError> {
        match self {
            Self::Content(config) if !config.is_valid() => Err(SceneError::InvalidParams(
                "invalid content effect config".to_string(),
            )),
            Self::Backdrop(config) => validate_backdrop_config(&config),
            Self::ShapeMask(config) => validate_shape_mask_effect_config(&config),
            _ => Ok(()),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct EffectAttachment {
    pub instance: EffectInstance,
    pub config: EffectConfig,
    pub source_group_id: Option<usize>,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct EffectAttachments {
    pub content: Option<EffectAttachment>,
    pub backdrop: Option<EffectAttachment>,
    pub shape_mask: Option<EffectAttachment>,
}

impl EffectAttachments {
    pub(crate) fn get(&self, kind: EffectAttachmentKind) -> Option<EffectAttachment> {
        match kind {
            EffectAttachmentKind::Content => self.content,
            EffectAttachmentKind::Backdrop => self.backdrop,
            EffectAttachmentKind::ShapeMask => self.shape_mask,
        }
    }

    pub(crate) fn get_mut(&mut self, kind: EffectAttachmentKind) -> &mut Option<EffectAttachment> {
        match kind {
            EffectAttachmentKind::Content => &mut self.content,
            EffectAttachmentKind::Backdrop => &mut self.backdrop,
            EffectAttachmentKind::ShapeMask => &mut self.shape_mask,
        }
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = &EffectAttachment> {
        self.content
            .iter()
            .chain(self.backdrop.iter())
            .chain(self.shape_mask.iter())
    }

    pub(super) fn iter_mut(&mut self) -> impl Iterator<Item = &mut EffectAttachment> {
        self.content
            .iter_mut()
            .chain(self.backdrop.iter_mut())
            .chain(self.shape_mask.iter_mut())
    }

    pub(crate) fn for_group_members(mut self, group_id: usize) -> Self {
        for effect in self.iter_mut() {
            effect.source_group_id.get_or_insert(group_id);
        }
        self
    }

    pub(crate) fn inherit_from(&mut self, parent: Self) {
        self.content = self.content.or(parent.content);
        self.backdrop = self.backdrop.or(parent.backdrop);
        self.shape_mask = self.shape_mask.or(parent.shape_mask);
    }

    pub(crate) fn remove_inherited(&mut self) {
        for effect in [&mut self.content, &mut self.backdrop, &mut self.shape_mask] {
            if effect.is_some_and(|effect| effect.source_group_id.is_some()) {
                *effect = None;
            }
        }
    }

    pub(super) fn remove_effect(&mut self, effect_id: u64) {
        for effect in [&mut self.content, &mut self.backdrop, &mut self.shape_mask] {
            if effect.is_some_and(|effect| effect.instance.effect_id == effect_id) {
                *effect = None;
            }
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct RemovedEffectAttachment {
    pub node_id: usize,
    pub kind: EffectAttachmentKind,
    pub damage_bounds: Option<UnsignedPhysicalRect>,
}

impl SceneState {
    pub(crate) fn node_effects(&self, node_id: usize) -> EffectAttachments {
        if let Some(DrawTreeNode::Group(group)) = self.draw_tree.get(node_id) {
            return group.effects;
        }
        let sources = match self.draw_tree.get(node_id) {
            Some(DrawTreeNode::CachedShape(shape)) => shape.inherited_effect_sources,
            _ => GroupEffectSources::default(),
        };
        EffectAttachments {
            content: self
                .content_effects
                .get(&node_id)
                .map(|effect| EffectAttachment {
                    instance: EffectInstance {
                        effect_id: effect.effect_id,
                        parameters: effect.parameters,
                    },
                    config: EffectConfig::Content(effect.config),
                    source_group_id: sources.content,
                }),
            backdrop: self
                .backdrop_effects
                .get(&node_id)
                .map(|effect| EffectAttachment {
                    instance: effect.effect,
                    config: EffectConfig::Backdrop(effect.config),
                    source_group_id: sources.backdrop,
                }),
            shape_mask: self
                .shape_mask_effects
                .get(&node_id)
                .map(|effect| EffectAttachment {
                    instance: EffectInstance {
                        effect_id: effect.effect_id,
                        parameters: effect.parameters,
                    },
                    config: EffectConfig::ShapeMask(effect.config),
                    source_group_id: sources.shape_mask,
                }),
        }
    }
}
