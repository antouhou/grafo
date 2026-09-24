use super::effects::{OffscreenTexturePool, PooledTexture};
use super::shape_effects::CompletedMask;
use super::targets::ActiveTarget;
use crate::commands::IntermediateTextureId;
use crate::core::cache::FrameCache;
use ahash::{HashMap, HashMapExt};
pub(crate) use shape_effects::{
    CachedShapeEffectMask, ShapeEffectCacheKey, ShapeEffectMaskCache, ShapeEffectMaskCacheKey,
};
use wgpu::{BindGroup, Texture};

mod bindings;
mod shape_effects;

/// The texture and its sampling binding have one owner in execution storage.
pub(crate) struct IntermediateTexture {
    pub(crate) texture: PooledTexture,
    pub(crate) bind_group: Option<BindGroup>,
}

#[derive(Clone, Copy)]
pub(super) enum PlannedTexture {
    Pending,
    Work(usize),
    Cached(IntermediateTextureId),
    Mask(usize),
}

/// Persistent cached textures and transient resources retained through submission.
pub(crate) struct IntermediateTextureResources {
    pub(super) active_targets: Vec<ActiveTarget>,
    pub(super) masks: Vec<CompletedMask>,
    sampled_textures: HashMap<IntermediateTextureId, IntermediateTexture>,
    /// Resolves command-local IDs without copying their textures.
    pub(super) planned: Vec<PlannedTexture>,
    pub(crate) pool: OffscreenTexturePool,
    pub(crate) work_textures: Vec<PooledTexture>,
    pub(crate) shape_effect_results: FrameCache<ShapeEffectCacheKey, IntermediateTextureId>,
    pub(crate) shape_effect_masks: ShapeEffectMaskCache,
}

impl IntermediateTextureResources {
    pub(crate) fn new() -> Self {
        Self {
            active_targets: Vec::new(),
            masks: Vec::new(),
            sampled_textures: HashMap::new(),
            planned: Vec::new(),
            pool: OffscreenTexturePool::new(),
            work_textures: Vec::new(),
            shape_effect_results: FrameCache::new(),
            shape_effect_masks: FrameCache::new(),
        }
    }

    /// Called only on a cache miss. Cache hits keep the same texture ID and binding.
    pub(crate) fn insert_cached(
        &mut self,
        cache_key: ShapeEffectCacheKey,
        sampled_texture: IntermediateTexture,
    ) -> IntermediateTextureId {
        let texture_id = IntermediateTextureId::Registered(sampled_texture.texture.texture_id);
        self.sampled_textures.insert(texture_id, sampled_texture);
        self.shape_effect_results.insert(cache_key, texture_id);
        texture_id
    }

    pub(in crate::renderer::backend::execution) fn resolve_id(
        &self,
        texture: IntermediateTextureId,
    ) -> IntermediateTextureId {
        match texture {
            IntermediateTextureId::Planned(index) => match self.planned[index] {
                PlannedTexture::Cached(texture) => texture,
                _ => texture,
            },
            _ => texture,
        }
    }

    pub(crate) fn bind_group(&self, texture_id: IntermediateTextureId) -> &BindGroup {
        match self.resolve_id(texture_id) {
            IntermediateTextureId::Registered(texture) => self.sampled_textures
                [&IntermediateTextureId::Registered(texture)]
                .bind_group
                .as_ref()
                .expect("texture has a sampling binding"),
            IntermediateTextureId::Planned(index) => {
                let PlannedTexture::Work(index) = self.planned[index] else {
                    unreachable!("composites reference completed textures");
                };
                self.work_textures[index].prepared_composite_bind_group()
            }
        }
    }

    pub(crate) fn texture(&self, texture_id: IntermediateTextureId) -> &Texture {
        let texture = match self.resolve_id(texture_id) {
            IntermediateTextureId::Registered(id) => {
                &self.sampled_textures[&IntermediateTextureId::Registered(id)].texture
            }
            IntermediateTextureId::Planned(index) => match self.planned[index] {
                PlannedTexture::Work(index) => &self.work_textures[index],
                _ => unreachable!("texture must be produced before sampling"),
            },
        };
        texture
            .resolve_texture
            .as_ref()
            .unwrap_or(&texture.color_texture)
    }

    pub(in crate::renderer::backend::execution) fn insert_planned(
        &mut self,
        texture_id: IntermediateTextureId,
        texture: PooledTexture,
    ) {
        let IntermediateTextureId::Planned(index) = texture_id else {
            unreachable!("planned textures have command-local IDs");
        };
        if index == self.planned.len() {
            self.reserve_planned(texture_id);
        }
        let slot = &mut self.planned[index];
        assert!(
            matches!(*slot, PlannedTexture::Pending),
            "planned textures are produced once"
        );
        *slot = PlannedTexture::Work(self.work_textures.len());
        self.work_textures.push(texture);
    }

    pub(super) fn insert_mask(&mut self, mask: CompletedMask) {
        let IntermediateTextureId::Planned(index) = mask.texture else {
            unreachable!("mask outputs are command-local");
        };
        assert!(matches!(self.planned[index], PlannedTexture::Pending));
        self.planned[index] = PlannedTexture::Mask(self.masks.len());
        self.masks.push(mask);
    }

    pub(super) fn insert_cached_output(
        &mut self,
        output: IntermediateTextureId,
        texture: IntermediateTextureId,
    ) {
        self.reserve_planned(output);
        let IntermediateTextureId::Planned(index) = output else {
            unreachable!()
        };
        self.planned[index] = PlannedTexture::Cached(texture);
    }

    /// A target reserves its slot before captures create later logical outputs.
    pub(in crate::renderer::backend::execution) fn reserve_planned(
        &mut self,
        texture: IntermediateTextureId,
    ) {
        assert_eq!(texture, IntermediateTextureId::Planned(self.planned.len()));
        self.planned.push(PlannedTexture::Pending);
    }

    /// Drops logical references while retaining storage and execution-owned textures.
    pub(in crate::renderer::backend::execution) fn finish_plan(&mut self) {
        self.planned.clear();
        self.masks.clear();
        debug_assert!(self.active_targets.is_empty());
    }

    /// All transient textures remain live until the commands that sample them are submitted.
    pub(crate) fn recycle_submitted(&mut self) {
        self.pool.recycle(&mut self.work_textures);
    }

    pub(crate) fn collect_unused_shape_effects(&mut self) -> (usize, usize) {
        let mut collected_results = 0;
        for (_, texture_id) in self.shape_effect_results.end_frame() {
            if let Some(texture) = self.sampled_textures.remove(&texture_id) {
                self.work_textures.push(texture.texture);
            }
            collected_results += 1;
        }
        let mut collected_masks = 0;
        for (_, mask) in self.shape_effect_masks.end_frame() {
            self.work_textures.push(mask.texture);
            collected_masks += 1;
        }
        (collected_results, collected_masks)
    }

    pub(crate) fn invalidate_shape_effect(&mut self, effect_id: u64) {
        self.shape_effect_results.retain(|cache_key, texture_id| {
            if cache_key.effect_id == effect_id {
                if let Some(texture) = self.sampled_textures.remove(texture_id) {
                    self.work_textures.push(texture.texture);
                }
                return false;
            }
            true
        });
    }

    pub(crate) fn clear_shape_effects(&mut self) {
        self.shape_effect_results.retain(|_, texture_id| {
            self.sampled_textures.remove(texture_id);
            false
        });
        self.shape_effect_masks.clear();
    }
}

#[cfg(test)]
mod tests;
