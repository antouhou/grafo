use super::types::DrawTreeNode;
use super::Scene;
use crate::core::effect::{CaptureRegion, GroupEffectBounds};
use crate::core::geometry;
use crate::core::{Size, UnsignedPhysicalRect, Viewport};
use ahash::{HashMap, HashSet};
use rstar::{RTree, RTreeObject, AABB};
use std::ops::ControlFlow;

fn envelope(bounds: UnsignedPhysicalRect) -> AABB<[f64; 2]> {
    AABB::from_corners(
        [f64::from(bounds.min.x), f64::from(bounds.min.y)],
        [f64::from(bounds.max.x), f64::from(bounds.max.y)],
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum EffectDamageKind {
    Backdrop,
    Group,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct EffectDamageEntry {
    pub node_id: usize,
    pub kind: EffectDamageKind,
    pub capture_region: Option<UnsignedPhysicalRect>,
    pub target_region: Option<UnsignedPhysicalRect>,
}

impl EffectDamageEntry {
    pub(super) fn backdrop(
        node_id: usize,
        node: &DrawTreeNode,
        capture_region: Option<CaptureRegion>,
        viewport: Viewport,
        fringe_width: f32,
    ) -> Option<Self> {
        let capture_region = capture_region?.source_rect;
        let DrawTreeNode::CachedShape(shape) = node else {
            return None;
        };
        if !shape.has_geometry() {
            return None;
        }
        let target_region = geometry::logical_bounds_to_viewport_rect(
            shape.logical_screen_bounds,
            viewport,
            fringe_width,
        );
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

    pub(super) fn group(
        node_id: usize,
        bounds: Option<GroupEffectBounds>,
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
            kind: EffectDamageKind::Group,
            capture_region,
            target_region,
        })
    }
}

impl RTreeObject for EffectDamageEntry {
    type Envelope = AABB<[f64; 2]>;

    fn envelope(&self) -> Self::Envelope {
        let bounds = match (self.capture_region, self.target_region) {
            (Some(capture), Some(target)) => capture.union(&target),
            (Some(region), None) | (None, Some(region)) => region,
            (None, None) => UnsignedPhysicalRect::zero(),
        };
        envelope(bounds)
    }
}

/// Includes effect output when damage overlaps its input, and backdrop capture when needed.
#[derive(Default)]
pub(super) struct EffectDamage {
    entries: HashMap<(EffectDamageKind, usize), EffectDamageEntry>,
    tree: RTree<EffectDamageEntry>,
    query_regions: Vec<UnsignedPhysicalRect>,
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
            self.tree.insert(entry);
            self.entries.insert(key, entry);
        }
    }

    pub(super) fn remove(&mut self, kind: EffectDamageKind, node_id: usize) {
        if let Some(entry) = self.entries.remove(&(kind, node_id)) {
            self.tree.remove(&entry);
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
            self.tree.insert(*entry);
        }
    }

    pub(super) fn clear(&mut self) {
        self.query_regions.clear();
        self.processed_effects.clear();
        if self.entries.is_empty() {
            return;
        }
        self.entries.clear();
        self.tree.drain().for_each(drop);
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
        self.query_regions.clear();
        self.processed_effects.clear();
        let viewport = UnsignedPhysicalRect::from_size(physical_size);
        let mut bounds = dirty_bounds?.intersection(&viewport)?;
        if self.is_empty() || bounds == viewport {
            return Some(bounds);
        }
        self.query_regions.push(bounds);
        let mut next_region = 0;
        while let Some(&query) = self.query_regions.get(next_region) {
            next_region += 1;
            let previous = bounds;
            let _ = self
                .tree
                .locate_in_envelope_intersecting_int(envelope(query), |entry| {
                    let target_trigger = (entry.kind == EffectDamageKind::Backdrop)
                        .then_some(entry.target_region)
                        .flatten();
                    let is_affected = [entry.capture_region, target_trigger]
                        .into_iter()
                        .flatten()
                        .any(|region| region.intersects(&bounds));
                    if is_affected && self.processed_effects.insert((entry.kind, entry.node_id)) {
                        let capture = (entry.kind == EffectDamageKind::Backdrop)
                            .then_some(entry.capture_region)
                            .flatten();
                        for region in [capture, entry.target_region].into_iter().flatten() {
                            bounds = bounds.union(&region);
                        }
                    }
                    ControlFlow::<()>::Continue(())
                });
            if bounds == viewport {
                break;
            }
            if bounds != previous {
                self.enqueue_added_regions(previous, bounds);
            }
        }
        Some(bounds)
    }

    fn enqueue_added_regions(
        &mut self,
        previous: UnsignedPhysicalRect,
        current: UnsignedPhysicalRect,
    ) {
        // Query all newly covered pixels, including gaps filled by the bounding union.
        for strip in [
            UnsignedPhysicalRect::new(current.min, (current.max.x, previous.min.y).into()),
            UnsignedPhysicalRect::new((current.min.x, previous.max.y).into(), current.max),
            UnsignedPhysicalRect::new(
                (current.min.x, previous.min.y).into(),
                (previous.min.x, previous.max.y).into(),
            ),
            UnsignedPhysicalRect::new(
                (previous.max.x, previous.min.y).into(),
                (current.max.x, previous.max.y).into(),
            ),
        ] {
            if !strip.is_empty() {
                self.query_regions.push(strip);
            }
        }
    }
}

impl Scene {
    pub(crate) fn group_effect_output(&self, node_id: usize) -> Option<UnsignedPhysicalRect> {
        self.effect_damage
            .output_region(EffectDamageKind::Group, node_id)
    }
}

#[cfg(test)]
mod tests;
