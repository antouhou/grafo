use super::{envelope, EffectDamage, EffectDamageEntry, EffectDamageKind, EffectDamageTrigger};
use crate::core::UnsignedPhysicalRect;
use ahash::HashMap;
use rstar::{Envelope, SelectionFunction, AABB};

struct RemovedEffectSelection<'a> {
    entries: &'a HashMap<(EffectDamageKind, usize), EffectDamageEntry>,
    bounds: AABB<[f64; 2]>,
}

impl SelectionFunction<EffectDamageTrigger> for RemovedEffectSelection<'_> {
    fn should_unpack_parent(&self, envelope: &AABB<[f64; 2]>) -> bool {
        self.bounds.intersects(envelope)
    }

    fn should_unpack_leaf(&self, trigger: &EffectDamageTrigger) -> bool {
        let entry = trigger.entry;
        !self.entries.contains_key(&(entry.kind, entry.node_id))
    }
}

/// Removes effects as one batch, preserving their output regions for damage tracking.
pub(in crate::scene_state) struct EffectDamageRemovalBatch<'a> {
    damage: &'a mut EffectDamage,
    removed_bounds: Option<UnsignedPhysicalRect>,
}

impl EffectDamageRemovalBatch<'_> {
    /// Returns the removed effect's output region.
    pub(in crate::scene_state) fn remove(
        &mut self,
        kind: EffectDamageKind,
        node_id: usize,
    ) -> Option<UnsignedPhysicalRect> {
        let entry = self.damage.entries.remove(&(kind, node_id))?;
        for trigger in entry.triggers() {
            self.removed_bounds = Some(
                self.removed_bounds
                    .map_or(trigger.region, |bounds| bounds.union(&trigger.region)),
            );
        }
        entry.target_region
    }
}

impl Drop for EffectDamageRemovalBatch<'_> {
    fn drop(&mut self) {
        let Some(bounds) = self.removed_bounds else {
            return;
        };
        if self.damage.is_empty() {
            self.damage.clear();
            return;
        }
        self.damage
            .tree
            .drain_with_selection_function(RemovedEffectSelection {
                entries: &self.damage.entries,
                bounds: envelope(bounds),
            })
            .for_each(drop);
    }
}

impl EffectDamage {
    /// Defers spatial removals until the batch is dropped.
    pub(in crate::scene_state) fn removal_batch(&mut self) -> EffectDamageRemovalBatch<'_> {
        EffectDamageRemovalBatch {
            damage: self,
            removed_bounds: None,
        }
    }
}

#[cfg(test)]
mod tests;
