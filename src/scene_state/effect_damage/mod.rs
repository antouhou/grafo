use super::types::DrawTreeNode;
use super::SceneState;
use crate::core::effect::{CaptureRegion, ContentEffectBounds};
use crate::core::geometry;
use crate::core::{Size, UnsignedPhysicalRect, Viewport};
use ahash::{HashMap, HashSet};
use rstar::{RTree, RTreeObject, AABB};
use std::ops::ControlFlow;
use traversal::DamageTraversal;

mod removal;
mod traversal;

fn envelope(bounds: UnsignedPhysicalRect) -> AABB<[f64; 2]> {
    AABB::from_corners(
        [f64::from(bounds.min.x), f64::from(bounds.min.y)],
        [f64::from(bounds.max.x), f64::from(bounds.max.y)],
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum EffectDamageKind {
    Backdrop,
    Content,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct EffectDamageEntry {
    pub node_id: usize,
    pub kind: EffectDamageKind,
    pub capture_region: Option<UnsignedPhysicalRect>,
    pub target_region: Option<UnsignedPhysicalRect>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct EffectDamageTrigger {
    entry: EffectDamageEntry,
    region: UnsignedPhysicalRect,
}

impl EffectDamageEntry {
    fn triggers(self) -> impl Iterator<Item = EffectDamageTrigger> {
        let target = (self.kind == EffectDamageKind::Backdrop
            && self.target_region != self.capture_region)
            .then_some(self.target_region)
            .flatten();
        [self.capture_region, target]
            .into_iter()
            .flatten()
            .filter(|region| !region.is_empty())
            .map(move |region| EffectDamageTrigger {
                entry: self,
                region,
            })
    }

    pub(super) fn backdrop(
        node_id: usize,
        node: &DrawTreeNode,
        capture_region: Option<CaptureRegion>,
        viewport: Viewport,
    ) -> Option<Self> {
        let capture_region = capture_region?.source_rect;
        let DrawTreeNode::CachedShape(shape) = node else {
            return None;
        };
        if !shape.has_geometry() {
            return None;
        }
        let target_region =
            geometry::logical_bounds_to_viewport_rect(shape.logical_screen_bounds, viewport);
        if capture_region.is_none() && target_region.is_none() {
            return None;
        }
        Some(Self {
            node_id,
            kind: EffectDamageKind::Backdrop,
            capture_region,
            target_region,
        })
    }

    pub(super) fn content(
        node_id: usize,
        bounds: Option<ContentEffectBounds>,
        physical_size: Size,
    ) -> Option<Self> {
        let bounds = bounds?;
        let capture_region = bounds.capture_region.source_rect;
        let target_region = bounds
            .capture_region
            .bounds
            .to_i64()
            .intersection(&UnsignedPhysicalRect::from_size(physical_size).to_i64())
            .map(|bounds| bounds.to_u32());
        if capture_region.is_none() && target_region.is_none() {
            return None;
        }
        Some(Self {
            node_id,
            kind: EffectDamageKind::Content,
            capture_region,
            target_region,
        })
    }
}

impl RTreeObject for EffectDamageTrigger {
    type Envelope = AABB<[f64; 2]>;

    fn envelope(&self) -> Self::Envelope {
        envelope(self.region)
    }
}

/// Includes effect output when damage overlaps its input, and backdrop capture when needed.
#[derive(Default)]
pub(super) struct EffectDamage {
    entries: HashMap<(EffectDamageKind, usize), EffectDamageEntry>,
    tree: RTree<EffectDamageTrigger>,
    traversal: DamageTraversal,
    processed_effects: HashSet<(EffectDamageKind, usize)>,
}

impl EffectDamage {
    pub(super) fn output_region(
        &self,
        kind: EffectDamageKind,
        node_id: usize,
    ) -> Option<UnsignedPhysicalRect> {
        self.entries
            .get(&(kind, node_id))
            .and_then(|entry| entry.target_region)
    }

    pub(super) fn replace(
        &mut self,
        kind: EffectDamageKind,
        node_id: usize,
        entry: Option<EffectDamageEntry>,
    ) {
        let key = (kind, node_id);
        if self.entries.get(&key).copied() == entry {
            return;
        }
        self.remove(kind, node_id);
        if let Some(entry) = entry {
            for trigger in entry.triggers() {
                self.tree.insert(trigger);
            }
            self.entries.insert(key, entry);
        }
    }

    pub(super) fn remove(&mut self, kind: EffectDamageKind, node_id: usize) {
        if let Some(entry) = self.entries.remove(&(kind, node_id)) {
            for trigger in entry.triggers() {
                self.tree.remove(&trigger);
            }
        }
    }

    pub(super) fn rebuild(&mut self, entries: impl IntoIterator<Item = EffectDamageEntry>) {
        self.clear();
        self.entries.extend(
            entries
                .into_iter()
                .map(|entry| ((entry.kind, entry.node_id), entry)),
        );
        for entry in self.entries.values() {
            for trigger in entry.triggers() {
                self.tree.insert(trigger);
            }
        }
    }

    pub(super) fn clear(&mut self) {
        self.traversal.clear();
        self.processed_effects.clear();
        self.entries.clear();
        if self.tree.size() != 0 {
            self.tree.drain().for_each(drop);
        }
    }

    fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Unions affected group outputs and backdrop capture/target rectangles with damage.
    pub(super) fn expand(
        &mut self,
        physical_size: Size,
        dirty_bounds: Option<UnsignedPhysicalRect>,
    ) -> Option<UnsignedPhysicalRect> {
        self.processed_effects.clear();
        let viewport = UnsignedPhysicalRect::from_size(physical_size);
        let bounds = dirty_bounds?.intersection(&viewport)?;
        if self.tree.size() == 0 || bounds == viewport {
            return Some(bounds);
        }
        let has_trigger = self
            .tree
            .locate_in_envelope_intersecting_int(envelope(bounds), |trigger| {
                if trigger.region.intersects(&bounds) {
                    ControlFlow::Break(())
                } else {
                    ControlFlow::Continue(())
                }
            })
            .is_break();
        if !has_trigger {
            return Some(bounds);
        }
        Some(
            self.traversal
                .expand(&self.tree, viewport, bounds, |trigger, bounds| {
                    let entry = trigger.entry;
                    if !self.processed_effects.insert((entry.kind, entry.node_id)) {
                        return;
                    }
                    let capture = (entry.kind == EffectDamageKind::Backdrop)
                        .then_some(entry.capture_region)
                        .flatten();
                    for region in [capture, entry.target_region].into_iter().flatten() {
                        *bounds = bounds.union(&region);
                    }
                }),
        )
    }
}

impl SceneState {
    pub(crate) fn content_effect_output(&self, node_id: usize) -> Option<UnsignedPhysicalRect> {
        self.effect_damage
            .output_region(EffectDamageKind::Content, node_id)
    }
}

#[cfg(test)]
mod tests;
