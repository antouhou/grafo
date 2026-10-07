use super::{EffectInstance, SceneError};
use crate::core::effect::{
    backdrops, BackdropCaptureArea, BackdropCaptureRegion, BackdropEffectConfig,
};
use crate::core::{MathRect, Viewport};

/// A backdrop attachment with its resolved capture region.
#[derive(Clone, Copy)]
pub(crate) struct BackdropEffectInstance {
    pub effect: EffectInstance,
    pub config: BackdropEffectConfig,
    pub capture_region: Option<BackdropCaptureRegion>,
}

impl BackdropEffectInstance {
    pub(crate) fn new(
        effect: EffectInstance,
        config: BackdropEffectConfig,
        logical_screen_bounds: MathRect,
        viewport: Viewport,
        maximum_texture_dimension: u32,
    ) -> Self {
        Self {
            effect,
            config,
            capture_region: backdrops::compute_backdrop_capture_region(
                logical_screen_bounds,
                config,
                viewport.scale_factor,
                viewport.physical_size.into(),
                maximum_texture_dimension,
            ),
        }
    }
}

pub(super) fn validate_backdrop_config(config: &BackdropEffectConfig) -> Result<(), SceneError> {
    if !(config.downsample > 0.0 && config.downsample <= 1.0) {
        return Err(SceneError::InvalidParams(format!(
            "backdrop downsample must be in the range (0.0, 1.0], got {}",
            config.downsample
        )));
    }
    if !config.padding.is_finite() || config.padding < 0.0 {
        return Err(SceneError::InvalidParams(format!(
            "backdrop padding must be finite and non-negative, got {}",
            config.padding
        )));
    }
    if let BackdropCaptureArea::ScreenRect([(x0, y0), (x1, y1)]) = config.capture_area {
        if !(x0.is_finite() && y0.is_finite() && x1.is_finite() && y1.is_finite()) {
            return Err(SceneError::InvalidParams(
                "backdrop screen capture rectangles must use only finite coordinates".to_string(),
            ));
        }
        if !(x1 > x0 && y1 > y0) {
            return Err(SceneError::InvalidParams(
                "backdrop screen capture rectangles must have positive width and height"
                    .to_string(),
            ));
        }
    }
    Ok(())
}
