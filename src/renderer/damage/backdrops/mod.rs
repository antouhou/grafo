use crate::core::{Size, UnsignedPhysicalRect};
use crate::scene::backdrop_damage::BackdropDamageIndex;
use ahash::HashSet;

const CELL_SIZE: u32 = 128;

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

/// Expands damage through backdrop captures, querying each viewport cell at most once.
#[derive(Default)]
pub(in crate::renderer) struct BackdropDamage {
    cells: Vec<usize>,
    visited_cells: Vec<bool>,
    processed_backdrops: HashSet<usize>,
}

impl BackdropDamage {
    pub(in crate::renderer) fn expand(
        &mut self,
        index: &BackdropDamageIndex,
        physical_size: Size,
        dirty_bounds: Option<UnsignedPhysicalRect>,
    ) -> Option<UnsignedPhysicalRect> {
        self.cells.clear();
        self.processed_backdrops.clear();
        let viewport = UnsignedPhysicalRect::from_size(physical_size);
        let mut bounds = dirty_bounds?.intersection(&viewport)?;
        if index.is_empty() || bounds == viewport {
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
            index.for_each_intersecting(query, |entry| {
                if self.processed_backdrops.insert(entry.node_id) {
                    for region in [entry.capture_region, entry.target_region]
                        .into_iter()
                        .flatten()
                    {
                        bounds = bounds.union(&region);
                    }
                }
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
