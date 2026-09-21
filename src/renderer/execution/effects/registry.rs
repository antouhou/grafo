use super::parameters::EffectParameterResources;
use super::shaders::{compile_effect_pipeline, LoadedEffect};
use crate::effect::EffectError;
use ahash::{HashMap, HashMapExt};
use naga::valid::{Capabilities, ValidationFlags, Validator};
use wgpu::{BindGroupLayout, Device, TextureFormat};

/// Registered shaders and reusable validation storage, owned by execution.
pub(crate) struct EffectRegistry {
    pub(super) loaded: HashMap<u64, LoadedEffect>,
    validator: Validator,
}

impl EffectRegistry {
    pub(crate) fn new() -> Self {
        Self {
            loaded: HashMap::new(),
            validator: Validator::new(ValidationFlags::all(), Capabilities::default()),
        }
    }

    /// Returns whether compilation replaced the registration. Failure preserves it.
    pub(crate) fn load(
        &mut self,
        device: &Device,
        format: TextureFormat,
        effect_id: u64,
        pass_sources: &[&str],
    ) -> Result<bool, EffectError> {
        if self.loaded.get(&effect_id).is_some_and(|effect| {
            effect.pass_sources.len() == pass_sources.len()
                && effect
                    .pass_sources
                    .iter()
                    .zip(pass_sources)
                    .all(|(stored, requested)| stored.as_ref() == *requested)
        }) {
            return Ok(false);
        }
        let effect = compile_effect_pipeline(device, pass_sources, format, &mut self.validator)?;
        self.loaded.insert(effect_id, effect);
        Ok(true)
    }

    pub(crate) fn unload(&mut self, effect_id: u64) {
        self.loaded.remove(&effect_id);
    }

    #[cfg(feature = "render_metrics")]
    pub(crate) fn pass_count(&self, effect_id: u64) -> usize {
        self.loaded[&effect_id].passes.len()
    }

    pub(crate) fn input_bind_group_layout(&self, effect_id: u64) -> &BindGroupLayout {
        &self.loaded[&effect_id].input_bind_group_layout
    }

    pub(crate) fn validate_params(&self, effect_id: u64, params: &[u8]) -> Result<(), EffectError> {
        let effect = self
            .loaded
            .get(&effect_id)
            .ok_or(EffectError::EffectNotLoaded(effect_id))?;
        let expects_params = effect.params_bind_group_layout.is_some();
        if expects_params && params.is_empty() {
            return Err(EffectError::InvalidParams(format!(
                "effect {effect_id} expects parameters but none were provided"
            )));
        }
        if !expects_params && !params.is_empty() {
            return Err(EffectError::InvalidParams(format!(
                "effect {effect_id} does not accept parameters but {} bytes were provided",
                params.len()
            )));
        }
        Ok(())
    }

    pub(crate) fn create_parameters(
        &self,
        device: &Device,
        effect_id: u64,
        params: &[u8],
    ) -> Option<EffectParameterResources> {
        self.loaded[&effect_id]
            .params_bind_group_layout
            .as_ref()
            .map(|layout| EffectParameterResources::new(device, layout, params))
    }
}
