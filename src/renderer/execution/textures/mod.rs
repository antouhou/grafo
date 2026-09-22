use super::effects::{OffscreenTexturePool, PooledTexture};
use crate::cache::FrameCache;
use crate::renderer::IntermediateTextureId;
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

/// Persistent cached textures and transient resources retained through submission.
pub(crate) struct IntermediateTextureResources {
    sampled_textures: HashMap<IntermediateTextureId, IntermediateTexture>,
    /// Maps each planner-assigned texture ID to its index in work_textures.
    pub(in crate::renderer::execution) texture_id_to_work_textures_index: Vec<usize>,
    pub(crate) pool: OffscreenTexturePool,
    pub(crate) work_textures: Vec<PooledTexture>,
    pub(crate) shape_effect_results: FrameCache<ShapeEffectCacheKey, IntermediateTextureId>,
    pub(crate) shape_effect_masks: ShapeEffectMaskCache,
}

impl IntermediateTextureResources {
    pub(crate) fn new() -> Self {
        Self {
            sampled_textures: HashMap::new(),
            texture_id_to_work_textures_index: Vec::new(),
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

    pub(crate) fn insert_transient(
        &mut self,
        sampled_texture: IntermediateTexture,
    ) -> IntermediateTextureId {
        let texture_id = IntermediateTextureId::Registered(sampled_texture.texture.texture_id);
        self.sampled_textures.insert(texture_id, sampled_texture);
        texture_id
    }

    pub(crate) fn bind_group(&self, texture_id: IntermediateTextureId) -> &BindGroup {
        match texture_id {
            IntermediateTextureId::Registered(_) => self.sampled_textures[&texture_id]
                .bind_group
                .as_ref()
                .expect("this texture was prepared for direct sampling"),
            IntermediateTextureId::Planned(index) => self.work_textures
                [self.texture_id_to_work_textures_index[index]]
                .prepared_composite_bind_group(),
        }
    }

    pub(crate) fn texture(&self, texture_id: IntermediateTextureId) -> &Texture {
        let texture = match texture_id {
            IntermediateTextureId::Registered(_) => &self.sampled_textures[&texture_id].texture,
            IntermediateTextureId::Planned(index) => {
                &self.work_textures[self.texture_id_to_work_textures_index[index]]
            }
        };
        texture
            .resolve_texture
            .as_ref()
            .unwrap_or(&texture.color_texture)
    }

    pub(in crate::renderer::execution) fn insert_planned(
        &mut self,
        texture_id: IntermediateTextureId,
        texture: PooledTexture,
    ) {
        assert_eq!(
            texture_id,
            IntermediateTextureId::Planned(self.texture_id_to_work_textures_index.len())
        );
        self.texture_id_to_work_textures_index
            .push(self.work_textures.len());
        self.work_textures.push(texture);
    }

    /// Clears logical references in constant time; allocations survive through submission.
    pub(in crate::renderer::execution) fn finish_plan(&mut self) {
        self.texture_id_to_work_textures_index.clear();
    }

    /// Ends a command reference while keeping the GPU allocation alive through submission.
    pub(crate) fn finish_transient(&mut self, texture_id: IntermediateTextureId) {
        let texture = self
            .sampled_textures
            .remove(&texture_id)
            .expect("transient texture references remain registered until their last draw");
        self.work_textures.push(texture.texture);
    }

    /// All transient textures remain live until the commands that sample them are submitted.
    pub(crate) fn recycle_submitted(
        &mut self,
        transient_texture_ids: impl Iterator<Item = IntermediateTextureId>,
    ) {
        for texture_id in transient_texture_ids {
            self.finish_transient(texture_id);
        }
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
