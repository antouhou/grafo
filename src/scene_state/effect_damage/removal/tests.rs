use super::super::{
    envelope, EffectDamage, EffectDamageEntry, EffectDamageKind, EffectDamageTrigger,
};
use super::RemovedEffectSelection;
use crate::core::{Size, UnsignedPhysicalRect};
use ahash::HashSet;
use rstar::{SelectionFunction, AABB};
use std::cell::RefCell;
use std::panic::{self, AssertUnwindSafe};

struct CountedSelection<'a> {
    selection: RemovedEffectSelection<'a>,
    visited_triggers: &'a RefCell<HashSet<(usize, u32)>>,
}

impl SelectionFunction<EffectDamageTrigger> for CountedSelection<'_> {
    fn should_unpack_parent(&self, envelope: &AABB<[f64; 2]>) -> bool {
        self.selection.should_unpack_parent(envelope)
    }

    fn should_unpack_leaf(&self, trigger: &EffectDamageTrigger) -> bool {
        assert!(self
            .visited_triggers
            .borrow_mut()
            .insert((trigger.entry.node_id, trigger.region.min.x)));
        self.selection.should_unpack_leaf(trigger)
    }
}

fn rect(min: (u32, u32), max: (u32, u32)) -> UnsignedPhysicalRect {
    UnsignedPhysicalRect::new(min.into(), max.into())
}

fn backdrop(node_id: usize) -> EffectDamageEntry {
    EffectDamageEntry {
        node_id,
        kind: EffectDamageKind::Backdrop,
        capture_region: Some(rect((0, 0), (64, 64))),
        target_region: Some(rect((96, 0), (128, 32))),
    }
}

#[test]
fn clearing_after_map_removals_also_clears_spatial_triggers() {
    let mut damage = EffectDamage::default();
    damage.rebuild([backdrop(1)]);
    damage.entries.remove(&(EffectDamageKind::Backdrop, 1));
    damage.clear();
    assert_eq!(damage.tree.size(), 0);
    let initial = Some(rect((1, 1), (2, 2)));
    assert_eq!(damage.expand(Size::new(256, 128), initial), initial);
}

#[test]
fn shared_trigger_batches_preserve_survivors_outputs_and_id_reuse() {
    for count in [64, 1000] {
        let mut damage = EffectDamage::default();
        let content = EffectDamageEntry {
            node_id: 1,
            kind: EffectDamageKind::Content,
            capture_region: Some(rect((160, 0), (176, 16))),
            target_region: Some(rect((160, 0), (192, 32))),
        };
        damage.rebuild((0..count).map(backdrop).chain([content]));
        let trigger_count = damage.tree.size();
        {
            let mut removals = damage.removal_batch();
            for node_id in (0..count).step_by(2) {
                assert_eq!(
                    removals.remove(EffectDamageKind::Backdrop, node_id),
                    backdrop(node_id).target_region
                );
                assert_eq!(removals.remove(EffectDamageKind::Backdrop, node_id), None);
            }
            assert_eq!(
                removals.remove(EffectDamageKind::Content, 1),
                content.target_region
            );
            assert_eq!(
                removals.remove(EffectDamageKind::Backdrop, usize::MAX),
                None
            );
            assert_eq!(removals.damage.tree.size(), trigger_count);
        }
        assert_eq!(damage.entries.len(), count / 2);
        assert_eq!(damage.tree.size(), count);
        assert!(damage.tree.iter().all(|trigger| {
            trigger.entry.kind == EffectDamageKind::Backdrop && trigger.entry.node_id % 2 == 1
        }));
        assert_eq!(
            damage.expand(Size::new(256, 128), content.capture_region),
            content.capture_region
        );
        assert_eq!(
            damage.expand(Size::new(256, 128), Some(rect((1, 1), (2, 2)))),
            Some(rect((0, 0), (128, 64)))
        );
        {
            let mut removals = damage.removal_batch();
            for node_id in (1..count).step_by(2) {
                removals.remove(EffectDamageKind::Backdrop, node_id);
            }
        }
        assert!(damage.entries.is_empty());
        assert_eq!(damage.tree.size(), 0);
        let initial = Some(rect((1, 1), (2, 2)));
        assert_eq!(damage.expand(Size::new(256, 128), initial), initial);
        damage.replace(EffectDamageKind::Content, 1, Some(content));
        assert_eq!(
            damage.expand(Size::new(256, 128), content.capture_region),
            content.target_region
        );
        assert_eq!(damage.expand(Size::new(256, 128), initial), initial);
    }
}

#[test]
fn batches_without_triggers_preserve_unrelated_effects() {
    let mut damage = EffectDamage::default();
    let content = EffectDamageEntry {
        node_id: 1,
        kind: EffectDamageKind::Content,
        capture_region: None,
        target_region: Some(rect((160, 0), (192, 32))),
    };
    damage.rebuild([content, backdrop(1)]);
    drop(damage.removal_batch());
    {
        let mut removals = damage.removal_batch();
        assert_eq!(removals.remove(EffectDamageKind::Content, 2), None);
        assert_eq!(
            removals.remove(EffectDamageKind::Content, 1),
            content.target_region
        );
    }
    assert_eq!(damage.entries.len(), 1);
    assert_eq!(damage.tree.size(), 2);
    assert!(damage
        .tree
        .iter()
        .all(|trigger| trigger.entry == backdrop(1)));
    damage.remove(EffectDamageKind::Backdrop, 1);
    damage.replace(EffectDamageKind::Content, 1, Some(content));
    {
        let mut removals = damage.removal_batch();
        removals.remove(EffectDamageKind::Content, 1);
    }
    assert!(damage.entries.is_empty());
    assert_eq!(damage.tree.size(), 0);
}

#[test]
fn batch_drop_synchronizes_triggers_during_unwinding() {
    for remove_all in [false, true] {
        let mut damage = EffectDamage::default();
        damage.rebuild([backdrop(1), backdrop(2)]);
        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            let mut removals = damage.removal_batch();
            removals.remove(EffectDamageKind::Backdrop, 1);
            if remove_all {
                removals.remove(EffectDamageKind::Backdrop, 2);
            }
            panic!("removal callback failed");
        }));
        assert!(result.is_err());
        assert_eq!(damage.tree.size(), if remove_all { 0 } else { 2 });
        assert!(damage.tree.iter().all(|trigger| {
            damage
                .entries
                .get(&(trigger.entry.kind, trigger.entry.node_id))
                == Some(&trigger.entry)
        }));
        let initial = Some(rect((1, 1), (2, 2)));
        assert_eq!(
            damage.expand(Size::new(256, 128), initial),
            if remove_all {
                initial
            } else {
                Some(rect((0, 0), (128, 64)))
            }
        );
    }
}

#[test]
fn selection_visits_shared_triggers_once_and_prunes_distant_branches() {
    for shared in [false, true] {
        let count = 1000;
        let mut damage = EffectDamage::default();
        damage.rebuild((0..count).map(|node_id| {
            if shared || node_id < 8 {
                return backdrop(node_id);
            }
            let x = 256 + node_id as u32 * 4;
            let region = rect((x, 128), (x + 2, 144));
            EffectDamageEntry {
                capture_region: Some(region),
                target_region: Some(region),
                ..backdrop(node_id)
            }
        }));
        for node_id in 0..4 {
            damage
                .entries
                .remove(&(EffectDamageKind::Backdrop, node_id));
        }
        let trigger_count = damage.tree.size();
        let visited_triggers = RefCell::new(HashSet::default());
        damage
            .tree
            .drain_with_selection_function(CountedSelection {
                selection: RemovedEffectSelection {
                    entries: &damage.entries,
                    bounds: envelope(rect((0, 0), (128, 64))),
                },
                visited_triggers: &visited_triggers,
            })
            .for_each(drop);
        assert_eq!(damage.tree.size(), trigger_count - 8);
        assert!(damage.tree.iter().all(|trigger| trigger.entry.node_id >= 4));
        if shared {
            assert_eq!(visited_triggers.borrow().len(), trigger_count);
        } else {
            assert!(visited_triggers.borrow().len() < trigger_count);
        }
    }
}
