use super::effect_damage::{EffectDamageEntry, EffectDamageKind};
use super::types::{DrawTreeNode, GroupEffectSources};
use super::{SceneError, SceneState};
use crate::core::effect::{
    BackdropEffectConfig, ContentEffectBounds, ContentEffectConfig, ShapeMaskEffectBounds,
    ShapeMaskEffectConfig,
};
use crate::core::{geometry, UnsignedPhysicalRect, Viewport};
pub(crate) use backdrops::validate_backdrop_config;
pub(crate) use backdrops::BackdropEffectInstance;
pub(super) use group_index::GroupEffectIndex;
use parameters::EffectParametersHandle;
use std::mem;

mod backdrops;
mod group_index;
pub(crate) mod parameters;

/// A cached shape mask effect attachment. GPU parameter resources are created only on cache misses.
#[derive(Clone, Copy)]
pub(crate) struct ShapeMaskEffectInstance {
    pub effect_id: u64,
    pub parameters: EffectParametersHandle,
    pub config: ShapeMaskEffectConfig,
    pub bounds: ShapeMaskEffectBounds,
}

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

pub(crate) fn validate_shape_mask_effect_config(
    config: &ShapeMaskEffectConfig,
) -> Result<(), SceneError> {
    if !(config.downsample > 0.0 && config.downsample <= 1.0) {
        return Err(SceneError::InvalidParams(format!(
            "shape mask effect downsample must be in the range (0.0, 1.0], got {}",
            config.downsample
        )));
    }

    let outsets = [
        config.left_outset,
        config.top_outset,
        config.right_outset,
        config.bottom_outset,
    ];
    if outsets
        .iter()
        .any(|outset| !outset.is_finite() || *outset < 0.0)
    {
        return Err(SceneError::InvalidParams(
            "shape mask effect outsets must be finite and non-negative".to_string(),
        ));
    }

    Ok(())
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

    fn iter_mut(&mut self) -> impl Iterator<Item = &mut EffectAttachment> {
        self.content
            .iter_mut()
            .chain(self.backdrop.iter_mut())
            .chain(self.shape_mask.iter_mut())
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

    fn remove_effect(&mut self, effect_id: u64) {
        for effect in [&mut self.content, &mut self.backdrop, &mut self.shape_mask] {
            if effect.is_some_and(|effect| effect.instance.effect_id == effect_id) {
                *effect = None;
            }
        }
    }
}

/// A content attachment with its resolved input and output bounds.
#[derive(Clone, Copy)]
pub(crate) struct ContentEffectInstance {
    pub effect_id: u64,
    pub parameters: EffectParametersHandle,
    pub config: ContentEffectConfig,
    pub bounds: Option<ContentEffectBounds>,
}

impl ContentEffectInstance {
    pub(crate) fn new(
        effect: EffectInstance,
        config: ContentEffectConfig,
        node: &DrawTreeNode,
        shape_mask_effect: Option<&ShapeMaskEffectInstance>,
        viewport: Viewport,
        maximum_texture_dimension: u32,
    ) -> Self {
        let shape_mask_effect_bounds = shape_mask_effect
            .filter(|_| matches!(node, DrawTreeNode::CachedShape(shape) if shape.has_geometry()))
            .map(|effect| effect.bounds);
        Self {
            effect_id: effect.effect_id,
            parameters: effect.parameters,
            config,
            bounds: ContentEffectBounds::new(
                node.logical_screen_bounds(),
                shape_mask_effect_bounds,
                config,
                viewport,
                maximum_texture_dimension,
            ),
        }
    }
}

fn resolve_content_effect(
    node_id: usize,
    node: &DrawTreeNode,
    effect: EffectInstance,
    config: ContentEffectConfig,
    shape_mask_effect: Option<&ShapeMaskEffectInstance>,
    viewport: Viewport,
    maximum_texture_dimension: u32,
) -> (ContentEffectInstance, Option<EffectDamageEntry>) {
    let instance = ContentEffectInstance::new(
        effect,
        config,
        node,
        shape_mask_effect,
        viewport,
        maximum_texture_dimension,
    );
    let entry = EffectDamageEntry::content(node_id, instance.bounds, viewport.physical_size.into());
    (instance, entry)
}

pub(super) fn resolve_backdrop_effect(
    node_id: usize,
    node: &DrawTreeNode,
    effect: EffectInstance,
    config: BackdropEffectConfig,
    viewport: Viewport,
    maximum_texture_dimension: u32,
) -> (BackdropEffectInstance, Option<EffectDamageEntry>) {
    let instance = BackdropEffectInstance::new(
        effect,
        config,
        node.logical_screen_bounds(),
        viewport,
        maximum_texture_dimension,
    );
    let entry = EffectDamageEntry::backdrop(node_id, node, instance.capture_region, viewport);
    (instance, entry)
}

#[derive(Clone, Copy)]
pub(crate) struct RemovedEffectAttachment {
    pub node_id: usize,
    pub kind: EffectAttachmentKind,
    pub damage_bounds: Option<UnsignedPhysicalRect>,
}

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

    fn inherited(
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
            content: self.inherited(sources.content, EffectAttachmentKind::Content),
            backdrop: self.inherited(sources.backdrop, EffectAttachmentKind::Backdrop),
            shape_mask: self.inherited(sources.shape_mask, EffectAttachmentKind::ShapeMask),
        }
    }

    pub(super) fn refresh_node_group_effect_sources(&mut self, node_id: usize) -> bool {
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

    fn remove_group_effect_attachments(&mut self, effect_id: u64) {
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

    pub(crate) fn compact_effect_parameters(&mut self) {
        let parameters = self
            .content_effects
            .values_mut()
            .map(|effect| &mut effect.parameters)
            .chain(
                self.backdrop_effects
                    .values_mut()
                    .map(|effect| &mut effect.effect.parameters),
            )
            .chain(
                self.shape_mask_effects
                    .values_mut()
                    .map(|effect| &mut effect.parameters),
            )
            .chain(
                self.draw_tree
                    .iter_mut()
                    .filter_map(|(_, node)| match node {
                        DrawTreeNode::Group(group) => Some(&mut group.effects),
                        _ => None,
                    })
                    .flat_map(|effects| {
                        effects
                            .iter_mut()
                            .map(|effect| &mut effect.instance.parameters)
                    }),
            );
        self.effect_parameters.compact(parameters);
    }

    pub(crate) fn shape_mask_effect(
        &self,
        node_id: usize,
    ) -> Result<&ShapeMaskEffectInstance, SceneError> {
        self.shape_mask_effects
            .get(&node_id)
            .ok_or(SceneError::NodeNotFound(node_id))
    }

    pub(crate) fn set_content_effect(
        &mut self,
        node_id: usize,
        effect_id: u64,
        parameters: &[u8],
        config: ContentEffectConfig,
        maximum_texture_dimension: u32,
    ) -> Result<(), SceneError> {
        self.shape(node_id)?;
        if !config.is_valid() {
            return Err(SceneError::InvalidParams(
                "content effect outsets must be finite and non-negative, and downsample must be in the range (0.0, 1.0]".to_string(),
            ));
        }
        let parameters = self.effect_parameters.store(parameters);
        if self.content_effects.contains_key(&node_id) {
            self.effect_parameters.request_compaction();
        }
        let (instance, entry) = resolve_content_effect(
            node_id,
            self.draw_tree.get_unchecked(node_id),
            EffectInstance {
                effect_id,
                parameters,
            },
            config,
            self.shape_mask_effects.get(&node_id),
            self.viewport,
            maximum_texture_dimension,
        );
        self.replace_content_effect(node_id, instance, entry);
        self.set_shape_effect_source(node_id, EffectAttachmentKind::Content, None);
        Ok(())
    }

    fn replace_content_effect(
        &mut self,
        node_id: usize,
        instance: ContentEffectInstance,
        entry: Option<EffectDamageEntry>,
    ) {
        self.effect_damage
            .replace(EffectDamageKind::Content, node_id, entry);
        self.content_effects.insert(node_id, instance);
    }

    pub(crate) fn remove_content_effect(&mut self, node_id: usize) -> bool {
        if self.content_effects.remove(&node_id).is_none() {
            return false;
        }
        self.set_shape_effect_source(node_id, EffectAttachmentKind::Content, None);
        self.effect_parameters.request_compaction();
        self.effect_damage
            .remove(EffectDamageKind::Content, node_id);
        true
    }

    pub(super) fn replace_backdrop_effect(
        &mut self,
        node_id: usize,
        instance: BackdropEffectInstance,
        entry: Option<EffectDamageEntry>,
    ) {
        self.effect_damage
            .replace(EffectDamageKind::Backdrop, node_id, entry);
        self.backdrop_effects.insert(node_id, instance);
    }

    pub(crate) fn set_backdrop_effect(
        &mut self,
        node_id: usize,
        effect_id: u64,
        parameters: &[u8],
        config: BackdropEffectConfig,
        maximum_texture_dimension: u32,
    ) -> Result<(), SceneError> {
        let node = self
            .draw_tree
            .get(node_id)
            .ok_or(SceneError::NodeNotFound(node_id))?;
        let DrawTreeNode::CachedShape(_) = node else {
            return Err(SceneError::UnsupportedNodeOperation(node_id, "effects"));
        };
        validate_backdrop_config(&config)?;
        let parameters = self.effect_parameters.store(parameters);
        if self.backdrop_effects.contains_key(&node_id) {
            self.effect_parameters.request_compaction();
        }
        let effect = EffectInstance {
            effect_id,
            parameters,
        };
        let (instance, entry) = resolve_backdrop_effect(
            node_id,
            node,
            effect,
            config,
            self.viewport,
            maximum_texture_dimension,
        );
        self.replace_backdrop_effect(node_id, instance, entry);
        self.set_shape_effect_source(node_id, EffectAttachmentKind::Backdrop, None);
        Ok(())
    }

    /// Refreshes capture bounds, viewport overlap and allocation limits after viewport changes.
    pub(super) fn refresh_backdrop_capture_regions(&mut self, maximum_texture_dimension: u32) {
        let entries = self
            .backdrop_effects
            .iter_mut()
            .filter_map(|(&node_id, instance)| {
                let node = self.draw_tree.get(node_id)?;
                let (updated, entry) = resolve_backdrop_effect(
                    node_id,
                    node,
                    instance.effect,
                    instance.config,
                    self.viewport,
                    maximum_texture_dimension,
                );
                *instance = updated;
                entry
            });
        self.effect_damage.rebuild(entries);
    }

    pub(crate) fn remove_backdrop_effect(&mut self, node_id: usize) -> bool {
        if self.backdrop_effects.remove(&node_id).is_none() {
            return false;
        }
        self.set_shape_effect_source(node_id, EffectAttachmentKind::Backdrop, None);
        self.effect_parameters.request_compaction();
        self.effect_damage
            .remove(EffectDamageKind::Backdrop, node_id);
        true
    }

    /// Attaches an effect using the scene's rasterization settings.
    /// Leaves the previous attachment unchanged if its bounds cannot be calculated.
    pub(crate) fn set_shape_mask_effect(
        &mut self,
        node_id: usize,
        effect_id: u64,
        parameters: &[u8],
        config: ShapeMaskEffectConfig,
        maximum_texture_dimension: u32,
    ) -> Result<(), SceneError> {
        let shape = self.shape(node_id)?;
        validate_shape_mask_effect_config(&config)?;
        let bounds = ShapeMaskEffectBounds::new(
            shape.cached_shape.tessellation.local_bounds,
            config,
            shape.transform,
            self.viewport.scale_factor,
        )
        .ok_or(SceneError::InvalidShapeMaskEffectBounds(node_id))?;
        let parameters = self.effect_parameters.store(parameters);
        if self.shape_mask_effects.contains_key(&node_id) {
            self.effect_parameters.request_compaction();
        }
        self.shape_mask_effects.insert(
            node_id,
            ShapeMaskEffectInstance {
                effect_id,
                parameters,
                config,
                bounds,
            },
        );
        self.set_shape_effect_source(node_id, EffectAttachmentKind::ShapeMask, None);
        self.refresh_content_effect_bounds(node_id, maximum_texture_dimension);
        Ok(())
    }

    /// Refreshes cached rectangles using the supplied rasterization settings.
    pub(super) fn refresh_shape_mask_effect_bounds(
        &mut self,
        scale_factor: f64,
    ) -> Result<(), SceneError> {
        for (&node_id, effect) in &mut self.shape_mask_effects {
            let node = self
                .draw_tree
                .get(node_id)
                .ok_or(SceneError::NodeNotFound(node_id))?;
            let DrawTreeNode::CachedShape(shape) = node else {
                return Err(SceneError::UnsupportedNodeOperation(node_id, "effects"));
            };
            effect.bounds = ShapeMaskEffectBounds::new(
                shape.instance.cached_shape.tessellation.local_bounds,
                effect.config,
                shape.instance.transform,
                scale_factor,
            )
            .ok_or(SceneError::InvalidShapeMaskEffectBounds(node_id))?;
        }
        Ok(())
    }

    pub(crate) fn remove_shape_mask_effect(
        &mut self,
        node_id: usize,
        maximum_texture_dimension: u32,
    ) {
        if self.shape_mask_effects.remove(&node_id).is_some() {
            self.set_shape_effect_source(node_id, EffectAttachmentKind::ShapeMask, None);
            self.effect_parameters.request_compaction();
            self.refresh_content_effect_bounds(node_id, maximum_texture_dimension);
        }
    }

    pub(crate) fn remove_effect_attachments(
        &mut self,
        effect_id: u64,
        maximum_texture_dimension: u32,
        mut removed: impl FnMut(RemovedEffectAttachment),
    ) {
        self.remove_group_effect_attachments(effect_id);
        self.content_effects.retain(|node_id, instance| {
            if instance.effect_id != effect_id {
                return true;
            }
            if let Some(DrawTreeNode::CachedShape(shape)) = self.draw_tree.get_mut(*node_id) {
                shape.inherited_effect_sources.content = None;
            }
            let damage_bounds = self
                .effect_damage
                .output_region(EffectDamageKind::Content, *node_id);
            self.effect_damage
                .remove(EffectDamageKind::Content, *node_id);
            self.effect_parameters.request_compaction();
            removed(RemovedEffectAttachment {
                node_id: *node_id,
                kind: EffectAttachmentKind::Content,
                damage_bounds,
            });
            false
        });
        self.backdrop_effects.retain(|node_id, instance| {
            if instance.effect.effect_id != effect_id {
                return true;
            }
            if let Some(DrawTreeNode::CachedShape(shape)) = self.draw_tree.get_mut(*node_id) {
                shape.inherited_effect_sources.backdrop = None;
            }
            self.effect_damage
                .remove(EffectDamageKind::Backdrop, *node_id);
            let damage_bounds = self.draw_tree.get(*node_id).and_then(|node| {
                geometry::logical_bounds_to_viewport_rect(
                    node.logical_screen_bounds(),
                    self.viewport,
                )
            });
            self.effect_parameters.request_compaction();
            removed(RemovedEffectAttachment {
                node_id: *node_id,
                kind: EffectAttachmentKind::Backdrop,
                damage_bounds,
            });
            false
        });
        self.shape_mask_effects.retain(|node_id, instance| {
            if instance.effect_id != effect_id {
                return true;
            }
            if let Some(DrawTreeNode::CachedShape(shape)) = self.draw_tree.get_mut(*node_id) {
                shape.inherited_effect_sources.shape_mask = None;
            }
            let mut damage_bounds = geometry::logical_bounds_to_viewport_rect(
                instance.bounds.logical_screen_bounds,
                self.viewport,
            );
            if let (Some(group), Some(node)) = (
                self.content_effects.get_mut(node_id),
                self.draw_tree.get(*node_id),
            ) {
                if let Some(output) = self
                    .effect_damage
                    .output_region(EffectDamageKind::Content, *node_id)
                {
                    damage_bounds =
                        Some(damage_bounds.map_or(output, |bounds| bounds.union(&output)));
                }
                let (updated, entry) = resolve_content_effect(
                    *node_id,
                    node,
                    EffectInstance {
                        effect_id: group.effect_id,
                        parameters: group.parameters,
                    },
                    group.config,
                    None,
                    self.viewport,
                    maximum_texture_dimension,
                );
                *group = updated;
                if let Some(output) = entry.and_then(|entry| entry.target_region) {
                    damage_bounds =
                        Some(damage_bounds.map_or(output, |bounds| bounds.union(&output)));
                }
                self.effect_damage
                    .replace(EffectDamageKind::Content, *node_id, entry);
            }
            self.effect_parameters.request_compaction();
            removed(RemovedEffectAttachment {
                node_id: *node_id,
                kind: EffectAttachmentKind::ShapeMask,
                damage_bounds,
            });
            false
        });
    }

    pub(super) fn refresh_content_effect_bounds(
        &mut self,
        node_id: usize,
        maximum_texture_dimension: u32,
    ) {
        let Some(instance) = self.content_effects.get(&node_id) else {
            return;
        };
        let Some(node) = self.draw_tree.get(node_id) else {
            return;
        };
        let (updated, entry) = resolve_content_effect(
            node_id,
            node,
            EffectInstance {
                effect_id: instance.effect_id,
                parameters: instance.parameters,
            },
            instance.config,
            self.shape_mask_effects.get(&node_id),
            self.viewport,
            maximum_texture_dimension,
        );
        self.replace_content_effect(node_id, updated, entry);
    }

    pub(super) fn refresh_all_content_effect_bounds(&mut self, maximum_texture_dimension: u32) {
        for (&node_id, instance) in &mut self.content_effects {
            let Some(node) = self.draw_tree.get(node_id) else {
                continue;
            };
            let (updated, entry) = resolve_content_effect(
                node_id,
                node,
                EffectInstance {
                    effect_id: instance.effect_id,
                    parameters: instance.parameters,
                },
                instance.config,
                self.shape_mask_effects.get(&node_id),
                self.viewport,
                maximum_texture_dimension,
            );
            *instance = updated;
            self.effect_damage
                .replace(EffectDamageKind::Content, node_id, entry);
        }
    }
}

#[cfg(test)]
mod tests;
