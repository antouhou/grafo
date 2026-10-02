use super::BackdropDamage;
use crate::core::{Size, UnsignedPhysicalRect};
use crate::scene::backdrop_damage::{BackdropDamageEntry, BackdropDamageIndex};

fn rect(min: (u32, u32), max: (u32, u32)) -> UnsignedPhysicalRect {
    UnsignedPhysicalRect::new(min.into(), max.into())
}

fn entry(node_id: usize, bounds: UnsignedPhysicalRect) -> BackdropDamageEntry {
    BackdropDamageEntry {
        node_id,
        capture_region: Some(bounds),
        target_region: Some(bounds),
    }
}

fn assert_unique_cells(cells: &[usize]) {
    let mut visited = [false; 256];
    for &cell in cells {
        assert!(!visited[cell]);
        visited[cell] = true;
    }
}

#[test]
fn long_chains_query_each_cell_once_and_reuse_worklist_storage() {
    let mut index = BackdropDamageIndex::default();
    index.rebuild((0..64).rev().map(|node| {
        entry(
            node as usize,
            rect((node * 64, 140), ((node + 1) * 64, 180)),
        )
    }));
    let mut damage = BackdropDamage::default();
    let initial = Some(rect((0, 150), (1, 151)));
    let expected = Some(rect((0, 140), (4096, 180)));
    assert_eq!(
        damage.expand(&index, Size::new(8192, 512), initial),
        expected
    );
    assert_eq!(damage.processed_backdrops.len(), 64);
    assert_eq!(damage.cells.len(), 32);
    assert_unique_cells(&damage.cells);
    let capacities = (
        damage.cells.capacity(),
        damage.visited_cells.capacity(),
        damage.processed_backdrops.capacity(),
    );
    for _ in 0..3 {
        assert_eq!(damage.expand(&index, Size::new(8192, 512), None), None);
        assert_eq!(
            damage.expand(&index, Size::new(8192, 512), initial),
            expected
        );
        assert_eq!(
            (
                damage.cells.capacity(),
                damage.visited_cells.capacity(),
                damage.processed_backdrops.capacity()
            ),
            capacities
        );
        assert_eq!(damage.processed_backdrops.len(), 64);
        assert_eq!(damage.cells.len(), 32);
    }
}

#[test]
fn expansion_in_all_directions_includes_newly_covered_cells_without_duplicates() {
    let mut index = BackdropDamageIndex::default();
    index.rebuild([
        entry(1, rect((128, 128), (600, 600))),
        entry(2, rect((0, 350), (160, 400))),
        entry(3, rect((200, 0), (260, 160))),
        entry(4, rect((600, 100), (900, 129))),
    ]);
    let mut damage = BackdropDamage::default();
    assert_eq!(
        damage.expand(
            &index,
            Size::new(1024, 1024),
            Some(rect((150, 150), (151, 151)))
        ),
        Some(rect((0, 0), (900, 600)))
    );
    assert_eq!(damage.processed_backdrops.len(), 4);
    assert_eq!(damage.cells.len(), 40);
    assert_unique_cells(&damage.cells);
}

#[test]
fn expansion_propagates_through_separate_capture_and_target_regions() {
    let mut index = BackdropDamageIndex::default();
    index.rebuild([
        BackdropDamageEntry {
            node_id: 1,
            capture_region: Some(rect((0, 0), (64, 64))),
            target_region: Some(rect((256, 8), (288, 32))),
        },
        BackdropDamageEntry {
            node_id: 2,
            capture_region: Some(rect((256, 0), (320, 64))),
            target_region: Some(rect((512, 8), (544, 32))),
        },
    ]);
    let mut damage = BackdropDamage::default();
    assert_eq!(
        damage.expand(&index, Size::new(1024, 512), Some(rect((8, 8), (16, 16)))),
        Some(rect((0, 0), (544, 64)))
    );
    assert_eq!(damage.processed_backdrops.len(), 2);
    assert_eq!(damage.expand(&index, Size::new(1024, 512), None), None);
}
