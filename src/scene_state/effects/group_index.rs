use super::EffectAttachments;
use ahash::{HashMap, HashSet};
use std::collections::hash_map::Entry;

#[derive(Default)]
pub(in crate::scene_state) struct GroupEffectIndex {
    groups_by_effect: HashMap<u64, HashSet<usize>>,
    group_set_pool: Vec<HashSet<usize>>,
}

impl GroupEffectIndex {
    pub(in crate::scene_state) fn replace(
        &mut self,
        node_id: usize,
        previous: EffectAttachments,
        current: EffectAttachments,
    ) {
        for effect in previous.iter() {
            let effect_id = effect.instance.effect_id;
            if current
                .iter()
                .any(|effect| effect.instance.effect_id == effect_id)
            {
                continue;
            }
            if let Entry::Occupied(mut entry) = self.groups_by_effect.entry(effect_id) {
                let groups = entry.get_mut();
                groups.remove(&node_id);
                if groups.is_empty() {
                    self.group_set_pool.push(entry.remove());
                }
            }
        }
        for effect in current.iter() {
            let effect_id = effect.instance.effect_id;
            if !previous
                .iter()
                .any(|effect| effect.instance.effect_id == effect_id)
            {
                self.groups_by_effect
                    .entry(effect_id)
                    .or_insert_with(|| self.group_set_pool.pop().unwrap_or_default())
                    .insert(node_id);
            }
        }
    }

    pub(in crate::scene_state) fn take_groups(&mut self, effect_id: u64) -> Option<HashSet<usize>> {
        self.groups_by_effect.remove(&effect_id)
    }

    pub(in crate::scene_state) fn recycle_groups(&mut self, mut groups: HashSet<usize>) {
        groups.clear();
        self.group_set_pool.push(groups);
    }
}
