use super::effects::{OffscreenTexturePool, PooledTexture};
use crate::cache::FrameCache;
use crate::renderer::types::{trim_hash_map_if_needed, trim_vector_if_needed};
use crate::renderer::IntermediateTextureId;
use ahash::{HashMap, HashMapExt};
pub(crate) use shape_effects::{
    CachedShapeEffectMask, ShapeEffectCacheKey, ShapeEffectMaskCache, ShapeEffectMaskCacheKey,
};
use wgpu::BindGroup;

mod bindings;
mod shape_effects;

const MAX_SAMPLED_TEXTURES_CAPACITY: usize = 4_096;
const MAX_WORK_TEXTURES_CAPACITY: usize = 2_048;

/// The texture and its sampling binding have one owner in execution storage.
pub(crate) struct SampledTexture {
    pub(crate) texture: PooledTexture,
    pub(crate) bind_group: BindGroup,
}

/// Persistent cached textures and transient resources retained through submission.
pub(crate) struct IntermediateTextureResources {
    sampled_textures: HashMap<IntermediateTextureId, SampledTexture>,
    pub(crate) pool: OffscreenTexturePool,
    pub(crate) work_textures: Vec<PooledTexture>,
    pub(crate) shape_effect_results: FrameCache<ShapeEffectCacheKey, IntermediateTextureId>,
    pub(crate) shape_effect_masks: ShapeEffectMaskCache,
}

impl IntermediateTextureResources {
    pub(crate) fn new() -> Self {
        Self {
            sampled_textures: HashMap::new(),
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
        sampled_texture: SampledTexture,
    ) -> IntermediateTextureId {
        let texture_id = IntermediateTextureId(sampled_texture.texture.texture_id);
        self.sampled_textures.insert(texture_id, sampled_texture);
        self.shape_effect_results.insert(cache_key, texture_id);
        texture_id
    }

    pub(crate) fn insert_transient(
        &mut self,
        sampled_texture: SampledTexture,
    ) -> IntermediateTextureId {
        let texture_id = IntermediateTextureId(sampled_texture.texture.texture_id);
        self.sampled_textures.insert(texture_id, sampled_texture);
        texture_id
    }

    pub(crate) fn bind_group(&self, texture_id: IntermediateTextureId) -> &BindGroup {
        &self.sampled_textures[&texture_id].bind_group
    }

    /// All transient textures remain live until the commands that sample them are submitted.
    pub(crate) fn recycle_submitted(
        &mut self,
        transient_texture_ids: impl Iterator<Item = IntermediateTextureId>,
    ) {
        for texture_id in transient_texture_ids {
            let sampled_texture = self
                .sampled_textures
                .remove(&texture_id)
                .expect("transient textures remain registered until submission");
            self.work_textures.push(sampled_texture.texture);
        }
        self.pool.recycle(&mut self.work_textures);
    }

    pub(crate) fn collect_unused_shape_effects(&mut self) -> (usize, usize) {
        let mut collected_results = 0;
        for (_, texture_id) in self.shape_effect_results.end_frame() {
            self.sampled_textures.remove(&texture_id);
            collected_results += 1;
        }
        let collected_masks = self.shape_effect_masks.end_frame().count();
        (collected_results, collected_masks)
    }

    pub(crate) fn invalidate_shape_effect(&mut self, effect_id: u64) {
        self.shape_effect_results.retain(|cache_key, texture_id| {
            if cache_key.effect_id == effect_id {
                self.sampled_textures.remove(texture_id);
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

    pub(crate) fn trim_to_policy(&mut self) {
        trim_hash_map_if_needed(&mut self.sampled_textures, MAX_SAMPLED_TEXTURES_CAPACITY);
        trim_vector_if_needed(&mut self.work_textures, MAX_WORK_TEXTURES_CAPACITY);
    }
}

#[cfg(test)]
mod tests;
