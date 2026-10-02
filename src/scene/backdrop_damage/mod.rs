use super::types::DrawTreeNode;
use crate::core::effect::BackdropCaptureRegion;
use crate::core::geometry;
use crate::core::{UnsignedPhysicalRect, Viewport};
use ahash::HashMap;
use rstar::{RTree, RTreeObject, AABB};
use std::ops::ControlFlow;

fn envelope(bounds: UnsignedPhysicalRect) -> AABB<[f64; 2]> {
    AABB::from_corners(
        [f64::from(bounds.min.x), f64::from(bounds.min.y)],
        [f64::from(bounds.max.x), f64::from(bounds.max.y)],
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BackdropDamageEntry {
    pub node_id: usize,
    pub capture_region: Option<UnsignedPhysicalRect>,
    pub target_region: Option<UnsignedPhysicalRect>,
}

impl BackdropDamageEntry {
    pub(super) fn new(
        node_id: usize,
        node: &DrawTreeNode,
        capture_region: Option<BackdropCaptureRegion>,
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
            capture_region,
            target_region,
        })
    }
}

impl RTreeObject for BackdropDamageEntry {
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

/// Keeps backdrop capture and target regions synchronized with spatial queries.
#[derive(Default)]
pub(crate) struct BackdropDamageIndex {
    entries: HashMap<usize, BackdropDamageEntry>,
    tree: RTree<BackdropDamageEntry>,
}

impl BackdropDamageIndex {
    pub(super) fn replace(&mut self, node_id: usize, entry: Option<BackdropDamageEntry>) {
        if self.entries.get(&node_id).copied() == entry {
            return;
        }
        self.remove(node_id);
        if let Some(entry) = entry {
            self.tree.insert(entry);
            self.entries.insert(node_id, entry);
        }
    }

    pub(super) fn remove(&mut self, node_id: usize) {
        if let Some(entry) = self.entries.remove(&node_id) {
            self.tree.remove(&entry);
        }
    }

    pub(crate) fn rebuild(&mut self, entries: impl IntoIterator<Item = BackdropDamageEntry>) {
        self.clear();
        self.entries
            .extend(entries.into_iter().map(|entry| (entry.node_id, entry)));
        for entry in self.entries.values() {
            self.tree.insert(*entry);
        }
    }

    pub(super) fn clear(&mut self) {
        if self.entries.is_empty() {
            return;
        }
        self.entries.clear();
        self.tree.drain().for_each(drop);
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub(crate) fn for_each_intersecting(
        &self,
        bounds: UnsignedPhysicalRect,
        mut visit: impl FnMut(&BackdropDamageEntry),
    ) {
        let _ = self
            .tree
            .locate_in_envelope_intersecting_int(envelope(bounds), |entry| {
                visit(entry);
                ControlFlow::<()>::Continue(())
            });
    }
}

#[cfg(test)]
mod tests;
