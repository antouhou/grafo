use super::EffectAttachments;
use ahash::{HashMap, HashSet};

#[derive(Default)]
pub(in crate::scene_state) struct GroupEffectIndex {
    groups_by_effect: HashMap<u64, HashSet<usize>>,
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
            if let Some(groups) = self.groups_by_effect.get_mut(&effect_id) {
                groups.remove(&node_id);
                if groups.is_empty() {
                    self.groups_by_effect.remove(&effect_id);
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
                    .or_default()
                    .insert(node_id);
            }
        }
    }

    pub(in crate::scene_state) fn take_groups(&mut self, effect_id: u64) -> Option<HashSet<usize>> {
        self.groups_by_effect.remove(&effect_id)
    }
}
