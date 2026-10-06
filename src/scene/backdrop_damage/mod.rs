use super::types::DrawTreeNode;
use crate::core::effect::BackdropCaptureRegion;
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct BackdropDamageEntry {
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

/// Tracks backdrop dependencies and expands damage through their capture and target regions.
#[derive(Default)]
pub(super) struct BackdropDamage {
    entries: HashMap<usize, BackdropDamageEntry>,
    tree: RTree<BackdropDamageEntry>,
    query_regions: Vec<UnsignedPhysicalRect>,
    processed_backdrops: HashSet<usize>,
}

impl BackdropDamage {
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

    pub(super) fn rebuild(&mut self, entries: impl IntoIterator<Item = BackdropDamageEntry>) {
        self.clear();
        self.entries
            .extend(entries.into_iter().map(|entry| (entry.node_id, entry)));
        for entry in self.entries.values() {
            self.tree.insert(*entry);
        }
    }

    pub(super) fn clear(&mut self) {
        self.query_regions.clear();
        self.processed_backdrops.clear();
        if self.entries.is_empty() {
            return;
        }
        self.entries.clear();
        self.tree.drain().for_each(drop);
    }

    fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Includes capture and target regions of affected backdrop effects.
    pub(super) fn expand(
        &mut self,
        physical_size: Size,
        dirty_bounds: Option<UnsignedPhysicalRect>,
    ) -> Option<UnsignedPhysicalRect> {
        self.query_regions.clear();
        self.processed_backdrops.clear();
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
                    let is_affected = [entry.capture_region, entry.target_region]
                        .into_iter()
                        .flatten()
                        .any(|region| region.intersects(&bounds));
                    if is_affected && self.processed_backdrops.insert(entry.node_id) {
                        for region in [entry.capture_region, entry.target_region]
                            .into_iter()
                            .flatten()
                        {
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

#[cfg(test)]
mod tests;
