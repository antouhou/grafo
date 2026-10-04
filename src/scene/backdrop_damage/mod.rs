use super::types::DrawTreeNode;
use crate::core::effect::BackdropCaptureRegion;
use crate::core::geometry;
use crate::core::{Size, UnsignedPhysicalRect, Viewport};
use ahash::{HashMap, HashSet};
use rstar::{RTree, RTreeObject, AABB};
use std::ops::ControlFlow;

const CELL_SIZE: u32 = 128;

fn envelope(bounds: UnsignedPhysicalRect) -> AABB<[f64; 2]> {
    AABB::from_corners(
        [f64::from(bounds.min.x), f64::from(bounds.min.y)],
        [f64::from(bounds.max.x), f64::from(bounds.max.y)],
    )
}

fn cell_range(bounds: UnsignedPhysicalRect) -> UnsignedPhysicalRect {
    UnsignedPhysicalRect::new(
        (bounds.min.x / CELL_SIZE, bounds.min.y / CELL_SIZE).into(),
        (
            bounds.max.x.div_ceil(CELL_SIZE),
            bounds.max.y.div_ceil(CELL_SIZE),
        )
            .into(),
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
    cells: Vec<usize>,
    visited_cells: Vec<bool>,
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
        self.cells.clear();
        self.visited_cells.clear();
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
        self.cells.clear();
        self.processed_backdrops.clear();
        let viewport = UnsignedPhysicalRect::from_size(physical_size);
        let mut bounds = dirty_bounds?.intersection(&viewport)?;
        if self.is_empty() || bounds == viewport {
            return Some(bounds);
        }
        let columns = physical_size.width.div_ceil(CELL_SIZE) as usize;
        let rows = physical_size.height.div_ceil(CELL_SIZE) as usize;
        self.visited_cells.resize(columns * rows, false);
        self.visited_cells.fill(false);
        self.enqueue(cell_range(bounds), columns);
        let mut next_cell = 0;
        while let Some(&cell) = self.cells.get(next_cell) {
            next_cell += 1;
            let x = (cell % columns) as u32 * CELL_SIZE;
            let y = (cell / columns) as u32 * CELL_SIZE;
            let query = UnsignedPhysicalRect::new(
                (x, y).into(),
                (
                    x.saturating_add(CELL_SIZE).min(physical_size.width),
                    y.saturating_add(CELL_SIZE).min(physical_size.height),
                )
                    .into(),
            );
            let previous = cell_range(bounds);
            let _ = self
                .tree
                .locate_in_envelope_intersecting_int(envelope(query), |entry| {
                    if self.processed_backdrops.insert(entry.node_id) {
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
            self.enqueue_added_cells(previous, cell_range(bounds), columns);
        }
        Some(bounds)
    }

    fn enqueue(&mut self, cells: UnsignedPhysicalRect, columns: usize) {
        for y in cells.min.y..cells.max.y {
            for x in cells.min.x..cells.max.x {
                let cell = y as usize * columns + x as usize;
                if !self.visited_cells[cell] {
                    self.visited_cells[cell] = true;
                    self.cells.push(cell);
                }
            }
        }
    }

    fn enqueue_added_cells(
        &mut self,
        previous: UnsignedPhysicalRect,
        current: UnsignedPhysicalRect,
        columns: usize,
    ) {
        // Differences in cell coordinates avoid rechecking edges that grow within a cell.
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
            self.enqueue(strip, columns);
        }
    }
}

#[cfg(test)]
mod tests;
