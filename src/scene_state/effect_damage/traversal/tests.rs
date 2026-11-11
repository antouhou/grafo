use super::DamageTraversal;
use crate::core::{Size, UnsignedPhysicalRect};
use crate::scene_state::effect_damage::{EffectDamageEntry, EffectDamageKind, EffectDamageTrigger};
use ahash::HashSet;
use rstar::{RTree, RTreeObject, AABB};
use std::cell::Cell;

struct CountedTrigger {
    trigger: EffectDamageTrigger,
    envelope_queries: Cell<usize>,
    callbacks: Cell<usize>,
}

impl RTreeObject for CountedTrigger {
    type Envelope = AABB<[f64; 2]>;

    fn envelope(&self) -> Self::Envelope {
        self.envelope_queries.set(self.envelope_queries.get() + 1);
        self.trigger.envelope()
    }
}

fn rect(min: (u32, u32), max: (u32, u32)) -> UnsignedPhysicalRect {
    UnsignedPhysicalRect::new(min.into(), max.into())
}

fn entry(
    node_id: usize,
    capture_region: UnsignedPhysicalRect,
    target_region: UnsignedPhysicalRect,
) -> EffectDamageEntry {
    EffectDamageEntry {
        node_id,
        kind: EffectDamageKind::Backdrop,
        capture_region: Some(capture_region),
        target_region: Some(target_region),
    }
}

fn tree(entries: impl IntoIterator<Item = EffectDamageEntry>) -> RTree<CountedTrigger> {
    let mut tree = RTree::new();
    for entry in entries {
        for trigger in entry.triggers() {
            tree.insert(CountedTrigger {
                trigger,
                envelope_queries: Cell::new(0),
                callbacks: Cell::new(0),
            });
        }
    }
    tree
}

fn expand(
    traversal: &mut DamageTraversal,
    tree: &RTree<CountedTrigger>,
    size: Size,
    initial: UnsignedPhysicalRect,
) -> (UnsignedPhysicalRect, usize) {
    for trigger in tree {
        trigger.envelope_queries.set(0);
        trigger.callbacks.set(0);
    }
    let mut processed_effects = HashSet::default();
    let bounds = traversal.expand(
        tree,
        UnsignedPhysicalRect::from_size(size),
        initial,
        |trigger, bounds| {
            trigger.callbacks.set(trigger.callbacks.get() + 1);
            let entry = trigger.trigger.entry;
            if processed_effects.insert((entry.kind, entry.node_id)) {
                let capture = (entry.kind == EffectDamageKind::Backdrop)
                    .then_some(entry.capture_region)
                    .flatten();
                for region in [capture, entry.target_region].into_iter().flatten() {
                    *bounds = bounds.union(&region);
                }
            }
        },
    );
    for trigger in tree {
        assert!(trigger.callbacks.get() <= 1);
        assert!(trigger.envelope_queries.get() <= 5);
    }
    (bounds, processed_effects.len())
}

fn storage_capacities(traversal: &DamageTraversal) -> [usize; 6] {
    let mut capacities = [0; 6];
    capacities[0] = traversal.node_locations.capacity();
    capacities[1] = traversal.path.capacity();
    for (capacity, waiting) in capacities[2..].iter_mut().zip(&traversal.waiting_nodes) {
        *capacity = waiting.capacity();
    }
    capacities
}

#[test]
fn long_chains_do_not_revisit_unrelated_triggers_and_reuse_traversal_storage() {
    for length in [64, 512, 2048] {
        let chain = (0..length).rev().map(|node| {
            let region = rect((node * 4, 40), ((node + 1) * 4 + 1, 60));
            entry(node as usize, region, region)
        });
        let unrelated = (length..length * 2).map(|node| {
            entry(
                node as usize,
                rect((0, 0), (length * 4 + 1, 10)),
                rect((0, 90), (length * 4 + 1, 100)),
            )
        });
        let tree = tree(chain.chain(unrelated));
        let size = Size::new(length * 4 + 100, 128);
        let initial = rect((0, 50), (1, 51));
        let expected = (rect((0, 40), (length * 4 + 1, 60)), length as usize);
        let mut traversal = DamageTraversal::default();
        assert_eq!(expand(&mut traversal, &tree, size, initial), expected);
        let capacities = storage_capacities(&traversal);
        for _ in 0..3 {
            assert_eq!(expand(&mut traversal, &tree, size, initial), expected);
            assert_eq!(storage_capacities(&traversal), capacities);
        }
        assert!(tree
            .iter()
            .filter(|trigger| trigger.trigger.entry.node_id >= length as usize)
            .all(|trigger| trigger.callbacks.get() == 0));
    }
}

#[test]
fn a_branch_waits_on_another_edge_after_its_first_edge_is_reached() {
    let tree = tree(
        [
            entry(0, rect((150, 200), (160, 210)), rect((0, 0), (300, 300))),
            entry(
                1,
                rect((200, 100), (210, 110)),
                rect((100, 100), (210, 205)),
            ),
            entry(
                2,
                rect((150, 100), (160, 110)),
                rect((100, 100), (205, 110)),
            ),
            entry(
                3,
                rect((100, 100), (110, 110)),
                rect((100, 100), (160, 110)),
            ),
        ]
        .map(|mut entry| {
            entry.kind = EffectDamageKind::Content;
            entry
        }),
    );
    let mut traversal = DamageTraversal::default();
    assert_eq!(
        expand(
            &mut traversal,
            &tree,
            Size::new(512, 512),
            rect((100, 100), (101, 101)),
        ),
        (rect((0, 0), (300, 300)), 4)
    );
    let corner = tree
        .iter()
        .find(|trigger| trigger.trigger.entry.node_id == 0)
        .unwrap();
    assert_eq!(corner.envelope_queries.get(), 3);
}

#[test]
fn touching_edges_near_maximum_coordinates_wait_for_actual_overlap() {
    let maximum = u32::MAX;
    let trigger = rect((maximum - 8, maximum - 8), (maximum - 4, maximum - 4));
    let output = rect((maximum - 10, maximum - 10), (maximum - 4, maximum - 4));
    let tree = tree([entry(0, trigger, output)]);
    let size = Size::new(maximum, maximum);
    let mut traversal = DamageTraversal::default();
    let touching = rect((maximum - 4, maximum - 8), (maximum - 2, maximum - 4));
    assert_eq!(expand(&mut traversal, &tree, size, touching), (touching, 0));
    let overlapping = rect((maximum - 5, maximum - 8), (maximum - 2, maximum - 4));
    assert_eq!(
        expand(&mut traversal, &tree, size, overlapping),
        (output.union(&overlapping), 1)
    );
}
